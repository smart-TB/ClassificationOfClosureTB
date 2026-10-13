"""One fan-control lease for the whole worker group, including failures.

No credentials are stored. sudo -n is attempted first; an authenticated desktop
owner can use the same NV-CONTROL assignments without privilege escalation.
SIGKILL/power loss cannot execute cleanup; the event log records incomplete leases.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import threading
from datetime import datetime, timezone


class GPUFanLease:
    def __init__(self, log_path, *, display=None, xauthority=None, expected_gpu_count=2):
        if not isinstance(expected_gpu_count, int) or expected_gpu_count < 1:
            raise ValueError("Positive physical GPU count required")
        self.expected_gpu_count = expected_gpu_count
        self.log_path = Path(log_path)
        self.env = os.environ.copy()
        if display:
            self.env["DISPLAY"] = display
        if xauthority:
            self.env["XAUTHORITY"] = xauthority
        self.lock = None
        self.old_handlers = {}
        self.command_prefix = ["sudo", "-n"]
        self.adjustment_attempted = False
        self.stop_refresh = threading.Event()
        self.refresh_thread = None

    def _refresh_sudo(self):
        # Refresh the already-authorized terminal ticket; never supply or store a
        # password. This keeps restoration possible after a long training run.
        while not self.stop_refresh.wait(60):
            try:
                result = subprocess.run(["sudo", "-n", "-v"], capture_output=True,
                                        text=True, timeout=15)
                if result.returncode:
                    self._log("sudo_ticket_refresh_failed")
                    return
            except (OSError, subprocess.TimeoutExpired):
                self._log("sudo_ticket_refresh_failed")
                return

    def _log(self, event, **fields):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a") as f:
            f.write(json.dumps({"timestamp": datetime.now(timezone.utc).isoformat(),
                                "event": event, "pid": os.getpid(), **fields}) + "\n")

    def _settings(self, args):
        command = ["nvidia-settings", *args]
        result = subprocess.run(self.command_prefix + command, env=self.env,
                                text=True, capture_output=True, timeout=30)
        if result.returncode and self.command_prefix:
            # Same assignments, no root needed when epigeo owns the X session.
            result = subprocess.run(command, env=self.env, text=True,
                                    capture_output=True, timeout=30)
            if result.returncode == 0:
                self.command_prefix = []
                self._log("desktop_owner_control_without_sudo")
        if result.returncode or "ERROR" in result.stderr.upper():
            raise RuntimeError(f"Fan control failed: {result.stderr.strip()}")
        return result.stdout

    def _active_compute(self):
        result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid",
                                 "--format=csv,noheader,nounits"], text=True,
                                capture_output=True, timeout=15, check=True)
        return [s.strip() for s in result.stdout.splitlines() if s.strip().isdigit()]

    def _set(self, speed):
        self._settings(["-a", "[gpu]/GPUFanControlState=1",
                        "-a", f"[fan]/GPUTargetFanSpeed={speed}"])
        states = self._settings(["-t", "-q", "[gpu]/GPUFanControlState"]).split()
        targets = self._settings(["-t", "-q", "[fan]/GPUTargetFanSpeed"]).split()
        if len(states) != self.expected_gpu_count or any(s != "1" for s in states):
            raise RuntimeError(f"Manual fan state not confirmed on {self.expected_gpu_count} GPUs: {states}")
        if len(targets) < self.expected_gpu_count or any(t != str(speed) for t in targets):
            raise RuntimeError(f"Fan target {speed} not confirmed on all fans: {targets}")
        self._log("fan_target_verified", target=speed, states=states, targets=targets)

    @staticmethod
    def _interrupted(signum, frame):
        raise KeyboardInterrupt(f"GPU coordinator received signal {signum}")

    def __enter__(self):
        import fcntl

        # Shared across run directories: two coordinators must not lower fans
        # while the other still has active workers.
        self.lock = Path(f"/tmp/smart-tb-gpu-fans-{os.getuid()}.lock").open("a")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            active = self._active_compute()
            if active:
                raise RuntimeError(f"GPUs already have compute processes: {active}")
            self.adjustment_attempted = True
            self._set(80)
            for sig in [signal.SIGTERM, signal.SIGINT]:
                self.old_handlers[sig] = signal.signal(sig, self._interrupted)
            if self.command_prefix:
                self.refresh_thread = threading.Thread(target=self._refresh_sudo, daemon=True)
                self.refresh_thread.start()
            self._log("lease_started")
            return self
        except BaseException:
            try:
                if self.adjustment_attempted and not self._active_compute():
                    self._set(30)
            finally:
                for sig, handler in self.old_handlers.items():
                    signal.signal(sig, handler)
                self.lock.close()
                self.lock = None
            raise

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop_refresh.set()
        if self.refresh_thread:
            self.refresh_thread.join(timeout=20)
        try:
            # Called only AFTER schedule() has waited for/killed all workers.
            active = self._active_compute()
            if active:
                self._log("restore_blocked_active_compute", processes=active)
                raise RuntimeError("Cannot lower fans while GPU compute processes remain")
            self._set(30)
            self._log("lease_closed", worker_error=str(exc_value) if exc_value else None)
        finally:
            for sig, handler in self.old_handlers.items():
                signal.signal(sig, handler)
            if self.lock:
                self.lock.close()
