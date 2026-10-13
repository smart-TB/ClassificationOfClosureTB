"""Audit supplement call order, EN/PT parity, captions, assets and generator order.

Run from any directory with: python scripts/check_supplement_references.py
The index identifies content by title and image path, not only by its S number.
"""
from pathlib import Path
import ast
import json
import re

IDS = r'S\d+(?:\s*,?\s*(?:and|e)?\s*S\d+)*'
REF = re.compile(rf'(?P<en>{IDS}\s+(?P<ek>Tables?|Figs?))|(?P<pt>(?P<pk>Tabelas?|Figuras?)\s+{IDS})')

def references(text):
    # Commented material must not count as a call in the rendered manuscript.
    text = re.sub(r'(?<!\\)%[^\n]*', '', text)
    refs = {'table': [], 'figure': []}
    for match in REF.finditer(text):
        kind = 'table' if (match['ek'] or match['pk']).startswith('Tab') else 'figure'
        refs[kind].extend(int(n) for n in re.findall(r'S(\d+)', match[0]))
    return refs

def require(condition, message):
    if not condition:
        raise AssertionError(message)

def audit(root):
    m = root/'manuscrito/PLOS_DigitalHealth'
    index = json.loads((m/'supplement_index.json').read_text())
    tex = (m/'S1_File.tex').read_text()
    parts = tex.split(r'\clearpage')
    texts = {n:(m/(n+'.tex')).read_text() for n in ['manuscript','manuscrito_ptbr','S1_Checklist','S1_File']}
    refs = {}
    for name, text in texts.items():
        refs[name] = references(text.split(r'\section*{Supporting information}')[0])
        for kind in ['table','figure']:
            expected = list(range(1,len(index[kind+'s'])+1))
            require(set(refs[name][kind]) <= set(expected), f'{name}: unresolved {kind} call')
            if name in ['manuscript','manuscrito_ptbr']:
                require(list(dict.fromkeys(refs[name][kind])) == expected,
                        f'{name}: missing or out-of-order first {kind} calls')
    require(refs['manuscript']==refs['manuscrito_ptbr'], 'EN/PT call sequences differ')
    # Detect S references that the grammar did not recognise, excluding S1 File/Checklist.
    for name,text in texts.items():
        residue=REF.sub('',text)
        residue=re.sub(r'(?:S1\s+(?:File|Checklist)|(?:Arquivo|Checklist)\s+S1)','',residue)
        require(not re.search(r'\bS\d+\b',residue),f'{name}: unparsed S-number; inspect manually')
    for kind, prefix in [('table','tab'),('figure','fig')]:
        found=[]
        for block in parts:
            label=re.search(r'\\label\{'+prefix+r':s(\d+)\}',block)
            if not label: continue
            n=int(label[1]);found.append(n);entry=index[kind+'s'][n-1]
            title=re.search(r'\\caption\{\{\\bf (.*?)\}',block,re.S)[1]
            require(title==entry['title'],f'{kind} S{n}: caption/content mismatch')
            if kind=='figure':
                asset=re.search(r'\{(figures_c7/[^}]+)\}',block)[1]
                require(asset==entry['image'],f'figure S{n}: wrong image')
                require((m/asset).is_file(),f'figure S{n}: image absent')
        require(found==list(range(1,len(index[kind+'s'])+1)),f'{kind}: labels/order mismatch')
    # Verify the English inventory is sequential and names the same items.
    inventory=re.findall(r'\\paragraph\*\{S(\d+) (Table|Fig)\.\}\s*\{\\bf ([^}]+)\}',texts['manuscript'])
    expected_inventory=[(str(e['number']),k,e['title'])
                        for k,entries in (('Table',index['tables']),('Fig',index['figures']))
                        for e in entries]
    require(inventory==expected_inventory,'Supporting-information inventory mismatch')
    # Verify the generator's declared order and captions, without running scientific work.
    tree=ast.parse((root/'scripts/build_supplementary.py').read_text())
    main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    titles=[ast.literal_eval(n.value.args[0])+'.' for n in main.body
            if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call)
            and isinstance(n.value.func,ast.Name) and n.value.func.id=='tabela']
    require(titles==[e['title'] for e in index['tables']],'Generator table order mismatch')
    figs=next(n.value for n in tree.body if isinstance(n,ast.Assign)
              and any(isinstance(t,ast.Name) and t.id=='FIGURAS' for t in n.targets))
    for values,entry in zip(ast.literal_eval(figs),index['figures']):
        require('figures_c7/'+values[0]+'.png'==entry['image'],'Generator figure order mismatch')
        require(values[1]+'.'==entry['title'],'Generator figure title mismatch')
    # The Table 2 caption must survive future regeneration with the correct coverage-table call.
    generator=(root/'scripts/editorial/tabelas_c7.py').read_text()
    coverage=next(e['number'] for e in index['tables'] if e['title'].startswith('Coverage of candidate'))
    require(f'S{coverage} Table' in generator and f'Tabela S{coverage}' in generator,'Table 2 generator cross-reference mismatch')
    result={'tables':len(index['tables']),'figures':len(index['figures']),
            'first_occurrence_order':'consecutive in EN/PT','all_calls_en_pt_equal':True,
            'all_calls_resolved':True,'inventory_captions_assets_and_generator':'consistent',
            'call_counts':{n:{k:len(v) for k,v in r.items()} for n,r in refs.items()}}
    return result

if __name__=='__main__':
    print(json.dumps(audit(Path(__file__).resolve().parents[1]),indent=2))
