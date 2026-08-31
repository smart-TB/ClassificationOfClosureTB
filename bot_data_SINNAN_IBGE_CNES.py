"""
## Processamento de Dados SINAN e CNES

Este script reúne diversas funções destinadas a:
1. **Baixar** arquivos DBC do SINAN (Tuberculose) a partir do FTP do DataSUS.
2. **Baixar** indicadores do IBGE e transformá-los em JSON/CSV.
3. **Baixar e converter** dados de estabelecimentos e profissionais do CNES (DataSUS).
4. **Converter** arquivos DBC em DBF e depois para Parquet.
5. **Unificar** vários arquivos Parquet (ex.: TUBEBR…) em um único.
6. **Tratar** e **ajustar** códigos de municípios (ID_MUNICIP, ID_MN_RESI), 
   inserindo dados de latitude/longitude e ajustando colunas para uso final.
7. **Fundir** dados do SINAN com CNES e IBGE em um único DataFrame.
8. **(Opcional)** Analisar um DataFrame Parquet do SINAN quanto à qualidade dos dados.

As etapas se complementam para gerar um *pipeline* de ingestão, limpeza, transformação e unificação dos dados epidemiológicos do SINAN e dados complementares do CNES e do IBGE.

## Observação Importante
- A última função, **`analisar_sinan_parquet`**, é meramente para análise e não será chamada na execução principal.
- A ordem de execução recomendada será feita por meio de uma função `main` que chama cada etapa consecutivamente, 
  com tratamento de exceções para indicar em qual etapa ocorreu um possível erro.

"""

import os
import io
import re
import logging
import traceback
import requests
import json
import pandas as pd
import dbfread
import glob
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pyreaddbc import dbc2dbf
from dbfread import DBF
from ftplib import FTP
from urllib.parse import urlparse
from datetime import datetime
# Renomeado para não sombrear o ConnectionError nativo do Python (builtins.ConnectionError),
# que é usado por outras bibliotecas (ex.: ftplib, sockets) para erros de conexão de baixo nível.
from requests.exceptions import ConnectionError as RequestsConnectionError, Timeout, RequestException
from bs4 import BeautifulSoup
from unidecode import unidecode  # Para remover acentos das strings

# -----------------------------------------------------------------------------
# Configuração de logging (console + arquivo). Handlers são adicionados por
# configurar_logging(), chamada em main(); se o módulo for importado sem
# chamar main(), o logger funciona normalmente via handler padrão do Python.
# -----------------------------------------------------------------------------
logger = logging.getLogger("bot_data_SINNAN_IBGE_CNES")


# -----------------------------------------------------------------------------
# Geografia de análise
# -----------------------------------------------------------------------------
# Este pipeline é de ingestão bruta: baixa e processa toda a série disponível.
# Nenhum recorte de coorte (janela de anos, elegibilidade, desfecho) acontece
# aqui — isso é decisão científica versionada e vive no código de análise.
#
# O SINAN traz dois municípios por notificação: o de residência (ID_MN_RESI) e o
# de notificação (ID_MUNICIP), que divergem em ~10% dos registros. A versão
# anterior deste pipeline usava os dois ao mesmo tempo: lat/long e nomes vinham
# da residência, enquanto IBGE/IPEA/CNES vinham da notificação. O resultado era
# que ~130 mil registros carregavam o contexto socioeconômico de um município
# diferente daquele que define suas coordenadas — e, portanto, seu cluster
# espacial. Isso quebra o bloqueio espacial em que a validação aninhada se apoia.
#
# A correção é resolver UMA vez o município de análise (proc10, coluna
# ID_MUNIC_ANALISE) e usá-lo em todos os enriquecimentos geográficos.
# Padrão: residência, que é a convenção do DATASUS para indicadores municipais e
# é o território cujo contexto socioeconômico atua sobre o paciente. Para a
# análise de sensibilidade por município de notificação, troque para
# "ID_MUNICIP" e regenere o parquet num caminho distinto.
CHAVE_MUNICIPIO_ANALISE = "ID_MN_RESI"
CHAVE_MUNICIPIO_FALLBACK = "ID_MUNICIP"
COLUNA_MUNICIPIO_RESOLVIDO = "ID_MUNIC_ANALISE"


def configurar_logging(log_dir="data/logs"):
    """Configura handlers de console e arquivo (com timestamp) para o logger do pipeline."""
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"pipeline_{datetime.now():%Y%m%d_%H%M%S}.log")

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)

    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.propagate = False

    return log_path


# -----------------------------------------------------------------------------
# Cookies de sessão do TabNet/DataSUS. São tokens efêmeros de sessão de navegador
# e expiram periodicamente — quando o raspador passar a não encontrar o <select
# id="A"> esperado, é sinal de que precisam ser renovados (inspecionar uma
# requisição real no navegador e atualizar as variáveis de ambiente abaixo).
# -----------------------------------------------------------------------------
COOKIE_FTP_DATASUS = os.environ.get("SINAN_COOKIE_FTP_DATASUS", "")
COOKIE_TABNET_CNES = os.environ.get(
    "SINAN_COOKIE_TABNET_CNES",
    "TS014879da=01e046ca4c3d32569119f7a09810ef467569a17e59f9d49e01edff835050ce3808b82ef6b5e378b9639ebe09e1bb2bf08093ae0475",
)


def _requisicao_com_retry(method, url, max_retries=5, timeout=120, **kwargs):
    """
    Faz uma requisição HTTP (GET/POST) com retry e backoff exponencial.

    Parâmetros:
    -----------
    method : str
        'GET' ou 'POST'.
    url : str
        URL de destino.
    max_retries : int
        Número máximo de tentativas antes de desistir.
    timeout : int
        Timeout (segundos) de cada tentativa individual.
    **kwargs :
        Repassado para requests.request (data, headers, etc.).

    Retorno:
    --------
    requests.Response

    Exceção:
    --------
    RequestsConnectionError:
        Lançada se esgotar as tentativas sem sucesso.
    """
    tentativa = 0
    while tentativa < max_retries:
        try:
            response = requests.request(method, url, timeout=timeout, **kwargs)
            if response.status_code == 200:
                return response
            logger.error(f"Falha na requisição (Status {response.status_code}): {url}")
        except (RequestsConnectionError, Timeout) as e:
            tentativa += 1
            logger.error(f"Conexão falhou (tentativa {tentativa}/{max_retries}): {e}")
            if tentativa < max_retries:
                time.sleep(2 ** tentativa)

    raise RequestsConnectionError(f"Falha após {max_retries} tentativas ao acessar {url}")


# -----------------------------------------------------------------------------
# Helpers compartilhados de raspagem do TabNet/CNES (usados por proc03 e proc04,
# que baixam formatos de tabela quase idênticos — estabelecimentos e
# profissionais — diferindo apenas em qual coluna de contagem é gerada e em
# como a coluna "Total" da resposta é tratada).
# -----------------------------------------------------------------------------

def _cnes_html_to_dataframe(html_content, remover_linha_total=False, remover_coluna_total=False):
    """
    Converte o HTML de resposta do TabNet (bloco <PRE> em formato CSV) em DataFrame.

    Parâmetros:
    -----------
    html_content : str
        Conteúdo HTML retornado pelo POST ao TabNet.
    remover_linha_total : bool
        Se True, descarta a linha de totais (aquela cujo primeiro campo, antes do
        primeiro ';', é exatamente "Total"). Checar apenas o primeiro campo (em vez
        de checar se a substring "Total" aparece em qualquer lugar da linha) evita
        remover por engano a própria linha de cabeçalho quando ela também tem uma
        coluna chamada "Total".
    remover_coluna_total : bool
        Se True, remove a coluna 'Total' (se existir) após a leitura do CSV.
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    pre_text = soup.find('pre').get_text()

    if remover_linha_total:
        pre_text = "\n".join(
            line for line in pre_text.splitlines()
            if line.split(';', 1)[0].strip() != "Total"
        )

    df = pd.read_csv(io.StringIO(pre_text), sep=';', encoding='iso-8859-1', low_memory=False)

    if remover_coluna_total and 'Total' in df.columns:
        df = df.drop(columns=['Total'])

    return df


def _cnes_extrair_valores_options(html):
    """Extrai os valores ('value') de cada <option> dentro de um <select id="A">."""
    soup = BeautifulSoup(html, 'html.parser')
    select = soup.find('select', {'id': 'A'})
    if not select:
        return []
    return [option['value'] for option in select.find_all('option')]


def _cnes_inverte_col_data(df, value_name):
    """
    Converte colunas de data no formato 'YYYY/MesAbrev' (ex.: '2022/Jan') para 'YYYY-MM'
    e empilha essas colunas em linhas via melt, gerando a coluna 'value_name'.
    """
    meses_str = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun',
                 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']

    def muda_data(data):
        parte_ano, parte_mes = data.split('/')
        idx_mes = meses_str.index(parte_mes) + 1
        if idx_mes < 10:
            idx_mes = f'0{idx_mes}'
        return f'{parte_ano}-{idx_mes}'

    # Tenta renomear cada coluna (exceto a primeira, 'ID_MUNICIP') para o formato YYYY-MM.
    # Colunas que não seguem o padrão de data (ex.: 'Total' remanescente) são
    # mantidas com o nome original e acabam no 'melt' abaixo mesmo assim.
    for col in df.columns[1:]:
        try:
            df = df.rename(columns={col: muda_data(col)})
        except (ValueError, TypeError):
            pass

    df = df.melt(id_vars=["ID_MUNICIP"], var_name="ANOMES_DIAG", value_name=value_name)
    return df


def _cnes_tratamento(df, value_name):
    """
    Padroniza o DataFrame retornado pelo TabNet (CNES):
    1) Extrai o código do município (primeiro token da coluna 'Município').
    2) Renomeia essa coluna para 'ID_MUNICIP'.
    3) Empilha as colunas de data via '_cnes_inverte_col_data'.
    4) Converte a coluna de contagem ('value_name') para numérico.
    """
    df['Município'] = df['Município'].str.split(' ').str[0]
    df = df.rename(columns={'Município': 'ID_MUNICIP'})
    df = _cnes_inverte_col_data(df, value_name)
    df[value_name] = pd.to_numeric(df[value_name], errors='coerce')
    return df

def proc01_download_SINNANDBC(ano_inicial=2001, replace=True):
    """
    Faz o download de arquivos DBC do FTP do DataSUS para os anos entre ano_inicial e o ano atual.
    
    Parâmetros:
    -----------
    ano_inicial : int
        Define o primeiro ano a ser baixado. Por padrão, 2001.
    replace : bool
        Se True, apaga todos os arquivos .dbc existentes no diretório local 
        antes de fazer os downloads. Se False, baixa apenas arquivos ainda não existentes.
    
    Retorna:
    --------
    bool
        True se todo o processo ocorrer sem exceções, False caso ocorra algum erro.
    """
    
    # Determina o ano final baseado na data atual
    ano_final = datetime.now().year

    logger.info(f"Iniciando download dos arquivos DBC do SINAN para os anos de {ano_inicial} a {ano_final}...")
    
    # Cria a lista de anos de acordo com os parâmetros definidos
    years = range(ano_inicial, ano_final + 1)

    # Variáveis de configuração
    ftp_host = 'ftp.datasus.gov.br'
    local_directory = 'data/arquivos_dbc'
    
    # Payload e cabeçalho para a requisição POST
    payload_template = (
        'tipo_arquivo%5B%5D=TUBE&modalidade%5B%5D=1&fonte%5B%5D=SINAN->INPUT<-&uf%5B%5D=BR'
    )
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
    }
    if COOKIE_FTP_DATASUS:
        headers['Cookie'] = COOKIE_FTP_DATASUS
    url_post = 'https://datasus.saude.gov.br/wp-content/ftp.php'
    
    # Monta o payload com os anos
    try:
        p1, p2 = payload_template.split('->INPUT<-')
        for year in years:
            p1 += f'&ano%5B%5D={year}'
        payload = p1 + p2
    except ValueError:
        logger.error("Falha ao montar o payload. Verifique o placeholder '->INPUT<-'.")
        return False
    
    # Faz a requisição POST para obter os links dos arquivos DBC (com retry/backoff)
    try:
        response_post = _requisicao_com_retry('POST', url_post, data=payload, headers=headers)
        response_post.raise_for_status()
    except requests.exceptions.RequestException as e:
        logger.error(f"Falha na requisição POST: {e}")
        return False

    # Converte a resposta em JSON
    try:
        response_json = response_post.json()
    except ValueError as e:
        logger.error(f"Não foi possível converter a resposta em JSON: {e}")
        return False
    
    # Garante que o diretório local exista
    os.makedirs(local_directory, exist_ok=True)
    
    # Se replace=True, apaga todos os arquivos .dbc existentes no diretório
    if replace:
        for file_name in os.listdir(local_directory):
            if file_name.lower().endswith('.dbc'):
                file_path = os.path.join(local_directory, file_name)
                try:
                    os.remove(file_path)
                    logger.info(f"Arquivo removido: {file_name}")
                except OSError as err:
                    logger.error(f"Não foi possível remover {file_name}: {err}")

    # Conecta ao FTP e faz o download
    try:
        with FTP(ftp_host) as ftp:
            ftp.login()  # Login anônimo
            
            for data_item in response_json:
                url = data_item.get('endereco')
                remote_filename = data_item.get('arquivo')
                
                # Caso não venha algum dos campos, continue
                if not url or not remote_filename:
                    logger.warning("Registro com campos incompletos, ignorando...")
                    continue

                parsed_url = urlparse(url)
                remote_directory = os.path.dirname(parsed_url.path)
                local_filename = remote_filename
                local_filepath = os.path.join(local_directory, local_filename)

                # Se replace=False, pula download se o arquivo já existir
                if not replace and os.path.exists(local_filepath):
                    logger.info(f"O arquivo {local_filename} já existe. Pulando download...")
                    continue

                logger.info(f"Baixando o arquivo {local_filename}...")

                # Muda o diretório remoto e faz o download binário
                ftp.cwd(remote_directory)
                with open(local_filepath, 'wb') as local_file:
                    ftp.retrbinary(f"RETR {remote_filename}", local_file.write)

                logger.info(f"Arquivo {local_filename} baixado com sucesso!")
                
    except Exception as e:
        logger.error(f"Ocorreu um problema ao baixar os arquivos do FTP: {e}")
        return False
    
    return True


def proc02_download_IndicadoresIBGE(ano_inicial=2000):
    """
    Função para obter indicadores do IBGE de todos os municípios brasileiros e 
    salvar os dados em um arquivo JSON no formato desejado.
    
    Parâmetros:
    -----------
    ano_inicial : int
        Define o primeiro ano que será considerado no preenchimento de dados. 
        O ano final é definido com base no ano atual da máquina.
    
    Fluxo Geral:
    ------------
    1) Define as URLs-base para obter informações de municípios e indicadores do IBGE.
    2) Cria um intervalo de anos de 'ano_inicial' até o ano atual.
    3) Define funções auxiliares:
       - 'preencher_anos': para preencher dados faltantes em cada indicador,
         com base nos anos disponíveis e no intervalo de anos definido.
       - 'fazer_requisicao': para fazer requisições GET com retry (tentativas repetidas).
    4) Busca a lista de todos os municípios no IBGE e itera sobre cada um,
       solicitando os indicadores via API do IBGE.
    5) Reestrutura os dados no formato desejado e salva tudo em um arquivo JSON.
    """

    # 1) URLs de referência para buscar dados
    # ---------------------------------------------------------------------
    # URL para a lista de municípios (retorna JSON com informações detalhadas de cada município)
    url_municipios = "https://servicodados.ibge.gov.br/api/v1/localidades/municipios"

    # URL para indicadores IBGE. A sequência de parâmetros 'params1' e 'params2' 
    # serão concatenados para formar a URL final.
    url_indicadores = "https://servicodados.ibge.gov.br/api/v1/pesquisas/indicadores/"
    params1 = "96386%7C143558%7C143514%7C60036%7C60037%7C60045%7C78187%7C78192%7C5950%7C5955%7C47001%7C30255%7C30279%7C60032%7C28242%7C95335%7C60030%7C60029%7C60031%7C77861%7C82270"
    params2 = "/resultados/"

    # 2) Intervalo de anos para preenchimento e análise
    # ---------------------------------------------------------------------
    # Obtém o ano atual (ex.: 2025) para definir o fim do intervalo.
    ano_final = datetime.now().year
    # Cria a lista de anos do ano_inicial até o ano_final.
    anos = list(range(ano_inicial, ano_final + 1))
    
    # 3) Função auxiliar para preencher anos faltantes
    # ---------------------------------------------------------------------
    def preencher_anos(dados, anos_lista):
        """
        Preenche os anos faltantes em 'dados' com base nos valores mais próximos
        dentro do intervalo fornecido por 'anos_lista'.

        Parâmetros:
        -----------
        dados : dict
            Dicionário onde cada chave é um ano (string) e cada valor é
            o dado correspondente àquele ano.
        anos_lista : list
            Lista de anos inteiros a serem preenchidos (ex.: [2000, 2001, 2002, ...]).

        Retorno:
        --------
        dict
            Dicionário com todos os anos de 'anos_lista' preenchidos,
            ordenado por ano. Se 'dados' vier vazio (indicador sem nenhum valor
            retornado pela API), retorna um dicionário vazio em vez de lançar erro.
        """
        if not dados:
            return {}

        # Transforma as chaves existentes em inteiros para fácil ordenação
        anos_disponiveis = sorted(int(ano) for ano in dados.keys())

        # Identifica o primeiro e último ano efetivamente disponíveis em 'dados'
        primeiro_ano = anos_disponiveis[0]
        ultimo_ano = anos_disponiveis[-1]

        # Anos anteriores ao primeiro disponível ficam AUSENTES.
        #
        # A versão anterior copiava para trás o valor do primeiro ano disponível.
        # Um indicador cuja série começa em 2019 era assim atribuído a
        # notificações de 2015-2018 — informação que ainda não existia no momento
        # da notificação. O briefing proíbe usar valor futuro (§8.4). Usamos None
        # (e não a omissão da chave) para que o preenchimento intermediário
        # abaixo não tente reconstruí-los.
        for ano in range(ano_inicial, primeiro_ano):
            dados[str(ano)] = None

        # Anos posteriores ao último disponível carregam adiante o último valor
        # conhecido — este é passado, não futuro, e portanto é legítimo.
        for ano in range(ultimo_ano + 1, ano_final + 1):
            dados[str(ano)] = dados[str(ultimo_ano)]

        # Preenche os anos intermediários que faltam
        for ano in anos_lista:
            if str(ano) not in dados:
                # Encontra o ano mais próximo (para trás) que exista em dados
                ano_mais_proximo = max(a for a in anos_disponiveis if a <= ano)
                dados[str(ano)] = dados[str(ano_mais_proximo)]

        # Ordena por ano e retorna o dicionário
        dados_ordenados = {str(ano): dados[str(ano)] for ano in sorted(int(x) for x in dados.keys())}
        return dados_ordenados

    # 4) Obter a lista de municípios
    # ---------------------------------------------------------------------
    try:
        response_municipios = _requisicao_com_retry('GET', url_municipios, timeout=10)
        municipios_data = response_municipios.json()
    except RequestsConnectionError as e:
        logger.error(f"Não foi possível obter a lista de municípios: {e}")
        return False

    # 5) Função que processa um único município (usada em paralelo abaixo)
    # ---------------------------------------------------------------------
    def processar_municipio(indice_municipio):
        i, municipio = indice_municipio

        if (
            municipio.get('microrregiao') and
            municipio['microrregiao'].get('mesorregiao') and
            municipio['microrregiao']['mesorregiao'].get('UF')
            ):
            uf = municipio['microrregiao']['mesorregiao']['UF']['sigla']
            cod_municipio = str(municipio['id'])
            nome_municipio = municipio['nome']
        else:
            logger.warning(f"Estrutura incompleta para município: {municipio.get('nome')} — pulando.")
            return None

        logger.info(f"{i} - Processando município: {nome_municipio} - {uf} (Código: {cod_municipio})")

        # Monta a URL final para indicadores deste município
        indicadores_url = url_indicadores + params1 + params2 + cod_municipio

        try:
            # Faz a requisição dos indicadores
            response_indicadores = _requisicao_com_retry('GET', indicadores_url, timeout=10)
            original_data = response_indicadores.json()

            # Estrutura de dados base para este município
            formatted_data = {
                "localidade": {
                    "codigo": cod_municipio,
                    "nome": nome_municipio,
                    "uf": uf
                },
                "indicadores": []
            }

            # Percorre cada indicador retornado e ajusta o dicionário de anos
            for item in original_data:
                for localidade_data in item['res']:
                    # Preenche anos faltantes neste localidade_data
                    res_completo = preencher_anos(localidade_data['res'], anos)

                    # Formata o dicionário final para cada indicador
                    formatted_item = {
                        "id": item['id'],
                        "res": res_completo
                    }
                    formatted_data["indicadores"].append(formatted_item)

            return formatted_data

        except RequestsConnectionError as e:
            logger.error(f"Falha ao processar município {nome_municipio}: {e}")
        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Dados inesperados da API para o município {nome_municipio}: {e}")
        finally:
            # Pausa para evitar sobrecarregar a API com muitas requisições seguidas
            time.sleep(0.1)

        return None

    # 6) Loop principal (paralelizado) + checkpoint incremental em disco
    # ---------------------------------------------------------------------
    os.makedirs("data", exist_ok=True)
    output_path = 'data/indicadores_todos_municipios.json'
    checkpoint_every = 200
    max_workers = 8

    resultados_municipios = []
    total = len(municipios_data)
    logger.info(f"Iniciando coleta de indicadores para {total} municípios (max_workers={max_workers})...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futuros = {
            executor.submit(processar_municipio, (i, municipio)): i
            for i, municipio in enumerate(municipios_data, start=1)
        }

        processados = 0
        for futuro in as_completed(futuros):
            processados += 1
            try:
                resultado = futuro.result()
            except Exception as e:
                # Qualquer erro inesperado num único município não deve derrubar
                # o processamento dos demais nem descartar o progresso já salvo.
                logger.error(f"Erro inesperado ao processar município (índice {futuros[futuro]}): {e}")
                resultado = None

            if resultado is not None:
                resultados_municipios.append(resultado)

            if processados % checkpoint_every == 0 or processados == total:
                logger.info(f"Checkpoint: {processados}/{total} municípios processados. Salvando progresso parcial em '{output_path}'...")
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(resultados_municipios, f, ensure_ascii=False, indent=4)

    logger.info(f"Processamento concluído e resultados salvos em '{output_path}'.")
    return True


def proc03_download_IndicadoresIPEA(ano_inicial=2000):
    """
    Baixa indicadores sociais do Atlas do Desenvolvimento Humano no Brasil
    (parceria PNUD + IPEA + Fundação João Pinheiro), via a API pública e
    aberta do Ipeadata (http://www.ipeadata.gov.br/api/odata4/), para todos
    os municípios brasileiros.

    Motivação: o indicador de IDHM que a API de indicadores do IBGE oferecia
    (ver proc02_download_IndicadoresIBGE, indicador 30255) parou de retornar
    dados — confirmado testando a API diretamente (nenhum resultado, mesmo em
    nível de Brasil inteiro). O Ipeadata republica a mesma base do Atlas do
    Desenvolvimento Humano (atlasbrasil.org.br) com uma API estável e sem
    autenticação. Além do IDHM, foram selecionados outros determinantes
    sociais com relevância epidemiológica conhecida para tuberculose:
    aglomeração domiciliar (>2 pessoas/dormitório), saneamento inadequado,
    pobreza, desigualdade de renda, renda per capita, analfabetismo e
    desemprego.

    Como os dados do Atlas DH só existem nos anos de Censo (1991, 2000, 2010,
    e para alguns indicadores 2022), cada série é preenchida (forward/backward
    fill a partir do ano de Censo mais próximo) para todos os anos entre
    'ano_inicial' e o ano atual — mesmo princípio de preenchimento usado em
    'preencher_anos' dentro de proc02_download_IndicadoresIBGE — permitindo o
    merge posterior por [ID_MUNICIP, ANO_DIAG], igual ao que já é feito com
    os indicadores do IBGE.

    Parâmetros:
    -----------
    ano_inicial : int
        Primeiro ano a preencher no intervalo. O ano final é o ano atual.

    Retorna:
    --------
    bool
        True se o processo ocorrer sem exceções, False caso ocorra algum erro.
    """
    IPEADATA_URL = "http://www.ipeadata.gov.br/api/odata4/ValoresSerie(SERCODIGO='{codigo}')"

    # Indicadores do Atlas do Desenvolvimento Humano (Censo) selecionados por
    # relevância como determinantes sociais de tuberculose. Todos em nível de
    # município inteiro (sem desagregação por raça — o Atlas só desagrega por
    # raça/cor para 2010 e em apenas ~111 municípios metropolitanos, o que
    # inviabilizaria cobertura nacional consistente).
    indicadores_ipea = {
        "ADH_IDHM": "IPEA_IDHM",
        "ADH_T_DENS": "IPEA_PCT_ADENSAMENTO_DOMICILIAR",
        "ADH_T_AGUA_ESGOTO": "IPEA_PCT_AGUA_ESGOTO_INADEQUADO",
        "ADH_PPOB": "IPEA_PCT_VULNERAVEIS_POBREZA",
        "ADH_PIND": "IPEA_PCT_EXTREMAMENTE_POBRES",
        "ADH_GINI": "IPEA_INDICE_GINI",
        "ADH_RDPC": "IPEA_RENDA_PER_CAPITA",
        "ADH_T_ANALF15M": "IPEA_TAXA_ANALFABETISMO_15MAIS",
        "ADH_T_DES18M": "IPEA_TAXA_DESEMPREGO_18MAIS",
        "ADH_ESPVIDA": "IPEA_ESPERANCA_VIDA",
        "ADH_MORT1": "IPEA_MORTALIDADE_ATE_1ANO",
        "ADH_T_PAREDE": "IPEA_PCT_PAREDE_INADEQUADA",
        "ADH_T_AGUA": "IPEA_PCT_AGUA_ENCANADA",
        "ADH_T_LIXO": "IPEA_PCT_COLETA_LIXO",
        "ADH_P_FORMAL": "IPEA_PCT_TRABALHO_FORMAL",
        "ADH_RAZDEP": "IPEA_RAZAO_DEPENDENCIA",
        "ADH_T_ENV": "IPEA_TAXA_ENVELHECIMENTO",
        "ADH_PESOTOT": "IPEA_POPULACAO_TOTAL",
        "ADH_THEIL": "IPEA_INDICE_THEIL",
    }

    ano_final = datetime.now().year
    anos = list(range(ano_inicial, ano_final + 1))

    def preencher_anos_ipea(valores_por_ano):
        """
        Preenche os anos faltantes com o valor do ano de Censo disponível mais
        próximo (mesmo princípio de 'preencher_anos' em proc02, adaptado para
        chaves inteiras em vez de string).

        Retorna dois dicionários {ano: valor} e {ano: ano_censo_usado}. O
        segundo existe porque nem todo indicador tem a mesma cobertura de
        Censos (a maioria só vai até 2010; alguns já têm 2022) — sem essa
        informação, duas colunas na mesma linha podem estar refletindo Censos
        de anos bem diferentes (ex.: uma de 2010, outra de 2022) sem que isso
        fique visível.
        """
        if not valores_por_ano:
            return {}, {}
        anos_disponiveis = sorted(valores_por_ano.keys())
        primeiro = anos_disponiveis[0]

        completo = {}
        ano_censo_usado = {}
        for ano in anos:
            if ano in valores_por_ano:
                completo[ano] = valores_por_ano[ano]
                ano_censo_usado[ano] = ano
            elif ano < primeiro:
                # Nenhum Censo ANTERIOR a este ano: o indicador é desconhecido
                # aqui e assim permanece.
                #
                # A versão anterior copiava para trás o valor do primeiro Censo
                # disponível. Para municípios criados depois de 2010 — que só
                # aparecem no Censo 2022 — isso atribuía dados de 2022 a
                # notificações de 2012-2021, ou seja, informação que ainda não
                # existia no momento da notificação. O briefing proíbe usar
                # valor futuro (§8.4); ausência é a descrição correta.
                completo[ano] = None
                ano_censo_usado[ano] = None
            else:
                # Censo mais recente ANTERIOR OU IGUAL ao ano (cobre também
                # ano > ultimo, que carrega o último Censo conhecido adiante).
                ano_mais_proximo = max(a for a in anos_disponiveis if a <= ano)
                completo[ano] = valores_por_ano[ano_mais_proximo]
                ano_censo_usado[ano] = ano_mais_proximo
        return completo, ano_censo_usado

    logger.info(f"Baixando {len(indicadores_ipea)} indicadores do Atlas do Desenvolvimento Humano (Ipeadata)...")

    # dados_por_municipio[codigo_municipio][nome_coluna] = {ano: valor}
    dados_por_municipio = {}

    for codigo_serie, nome_coluna in indicadores_ipea.items():
        url = IPEADATA_URL.format(codigo=codigo_serie)
        try:
            response = _requisicao_com_retry('GET', url, timeout=60)
            valores = response.json().get('value', [])
        except RequestsConnectionError as e:
            logger.error(f"Falha ao baixar a série '{codigo_serie}' do Ipeadata: {e}")
            continue
        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Resposta inesperada do Ipeadata para a série '{codigo_serie}': {e}")
            continue

        valores_municipio = [v for v in valores if v.get('NIVNOME') == 'Municípios']
        logger.info(f"  {codigo_serie} -> {nome_coluna}: {len(valores_municipio)} valores em nível de município.")

        por_municipio = {}
        for v in valores_municipio:
            cod_municipio = v['TERCODIGO']
            ano = int(v['VALDATA'][:4])
            por_municipio.setdefault(cod_municipio, {})[ano] = v['VALVALOR']

        coluna_ano_censo = f"{nome_coluna}_ANO_CENSO"
        for cod_municipio, valores_por_ano in por_municipio.items():
            completo, ano_censo_usado = preencher_anos_ipea(valores_por_ano)
            dados_por_municipio.setdefault(cod_municipio, {})[nome_coluna] = completo
            dados_por_municipio[cod_municipio][coluna_ano_censo] = ano_censo_usado

        # Pausa leve entre as 9 chamadas para não sobrecarregar a API.
        time.sleep(0.2)

    if not dados_por_municipio:
        logger.error("Nenhum indicador do Ipeadata foi baixado com sucesso.")
        return False

    # Reestrutura em formato longo: uma linha por (município, ano). Para cada
    # indicador, além da coluna de valor, inclui '<coluna>_ANO_CENSO' — o ano
    # do Censo (1991/2000/2010/2022) de onde aquele valor específico veio,
    # já que indicadores diferentes têm cobertura de Censo diferente e podem
    # ficar "desatualizados" em graus diferentes na mesma linha.
    linhas = []
    nomes_colunas = list(indicadores_ipea.values())
    for cod_municipio, colunas in dados_por_municipio.items():
        for ano in anos:
            linha = {"ID_MUNICIP": cod_municipio, "ANO_DIAG": ano}
            for nome_coluna in nomes_colunas:
                linha[nome_coluna] = colunas.get(nome_coluna, {}).get(ano)
                linha[f"{nome_coluna}_ANO_CENSO"] = colunas.get(f"{nome_coluna}_ANO_CENSO", {}).get(ano)
            linhas.append(linha)

    df_ipea = pd.DataFrame(linhas)

    os.makedirs("data", exist_ok=True)
    output_path = "data/indicadores_IPEA.csv"
    df_ipea.to_csv(output_path, index=False)
    logger.info(f"Indicadores do Ipeadata salvos em '{output_path}' ({df_ipea.shape[0]} linhas, {df_ipea.shape[1]} colunas, {len(dados_por_municipio)} municípios).")

    return True


def proc04_download_estabelecimentosCNES():
    """
    Realiza a captura e processamento de dados sobre estabelecimentos do CNES (DataSUS),
    gravando os resultados em um arquivo Parquet no diretório 'data'.

    Fluxo Geral:
    ------------
    1) Obter via GET o documento HTML que contém um elemento <select> com diversas <option>.
    2) Extrair todos os valores das <option> (cada valor corresponde a um conjunto de dados).
    3) Montar um payload para a requisição POST, inserindo esses valores de forma dinâmica.
    4) Fazer o POST, receber a resposta em formato HTML (contendo dados no formato <PRE>).
    5) Converter esse HTML em um DataFrame via 'html_to_dataframe'.
    6) Realizar o tratamento do DataFrame via 'tratamnto_cnes'.
    7) Salvar os dados resultantes em 'data/estabelecimentos_CNES.parquet'.

    Retorno:
    --------
    None
        A função apenas exibe mensagens e salva o arquivo Parquet. 
        Em caso de falha, retorna False (e imprime mensagens de erro).
    """

    # ------------------------------------------------------------------------------
    # Variáveis e Configurações
    # ------------------------------------------------------------------------------
    
    # URLs para requisições
    urlOPT = 'http://tabnet.datasus.gov.br/cgi/deftohtm.exe?cnes/cnv/estabbr.def'  # GET (para extrair <options>)
    urlRSP = 'http://tabnet.datasus.gov.br/cgi/tabcgi.exe?cnes/cnv/estabbr.def'    # POST (para obter dados)

    # Cabeçalhos (Headers) para as requisições HTTP
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                      '(KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
        # Cookie de sessão do TabNet — ver SINAN_COOKIE_TABNET_CNES no topo do arquivo.
        'Cookie': COOKIE_TABNET_CNES,
    }

    # Payload base que será enviado no POST. Observe que há um placeholder "->INPUT<-"
    # onde serão inseridas as <option> (cada valor do select).
    payload = (
        'Linha=Munic%EDpio&Coluna=Ano%2Fm%EAs_compet.&Incremento=Quantidade&'
        '->INPUT<-'
        'SRegi%E3o=TODAS_AS_CATEGORIAS__&pesqmes2=Digite+o+texto+e+ache+f%E1cil&'
        'SUnidade_da_Federa%E7%E3o=TODAS_AS_CATEGORIAS__&pesqmes3=Digite+o+texto+e+ache+f%E1cil&'
        'SMunic%EDpio=TODAS_AS_CATEGORIAS__&pesqmes4=Digite+o+texto+e+ache+f%E1cil&'
        'SMunic%EDpio_gestor=TODAS_AS_CATEGORIAS__&pesqmes5=Digite+o+texto+e+ache+f%E1cil&'
        'SCapital=TODAS_AS_CATEGORIAS__&pesqmes6=Digite+o+texto+e+ache+f%E1cil&'
        'SRegi%E3o_de_Sa%FAde_%28CIR%29=TODAS_AS_CATEGORIAS__&pesqmes7=Digite+o+texto+e+ache+f%E1cil&'
        'SMacrorregi%E3o_de_Sa%FAde=TODAS_AS_CATEGORIAS__&pesqmes8=Digite+o+texto+e+ache+f%E1cil&'
        'SMicrorregi%E3o_IBGE=TODAS_AS_CATEGORIAS__&pesqmes9=Digite+o+texto+e+ache+f%E1cil&'
        'SRegi%E3o_Metropolitana_-_RIDE=TODAS_AS_CATEGORIAS__&pesqmes10=Digite+o+texto+e+ache+f%E1cil&'
        'STerrit%F3rio_da_Cidadania=TODAS_AS_CATEGORIAS__&pesqmes11=Digite+o+texto+e+ache+f%E1cil&'
        'SMesorregi%E3o_PNDR=TODAS_AS_CATEGORIAS__&SAmaz%F4nia_Legal=TODAS_AS_CATEGORIAS__&'
        'SSemi%E1rido=TODAS_AS_CATEGORIAS__&SFaixa_de_Fronteira=TODAS_AS_CATEGORIAS__&'
        'SZona_de_Fronteira=TODAS_AS_CATEGORIAS__&SMunic%EDpio_de_extrema_pobreza=TODAS_AS_CATEGORIAS__&'
        'SEnsino%2FPesquisa=TODAS_AS_CATEGORIAS__&pesqmes18=Digite+o+texto+e+ache+f%E1cil&'
        'SNatureza_Jur%EDdica=TODAS_AS_CATEGORIAS__&pesqmes19=Digite+o+texto+e+ache+f%E1cil&'
        'SEsfera_Jur%EDdica=TODAS_AS_CATEGORIAS__&SEsfera_Administrativa=TODAS_AS_CATEGORIAS__&'
        'pesqmes21=Digite+o+texto+e+ache+f%E1cil&SNatureza=TODAS_AS_CATEGORIAS__&pesqmes22=Digite+o+texto+e+ache+f%E1cil&'
        'STipo_de_Estabelecimento=TODAS_AS_CATEGORIAS__&STipo_de_Gest%E3o=TODAS_AS_CATEGORIAS__&'
        'STipo_de_Prestador=TODAS_AS_CATEGORIAS__&formato=prn&mostre=Mostra'
    )

    # ------------------------------------------------------------------------------
    # Lógica Principal (usa os helpers compartilhados _cnes_* definidos no topo do módulo)
    # ------------------------------------------------------------------------------

    try:
        # 1) Faz a requisição GET para obter o HTML que contém o select com <options>
        logger.info("Fazendo a requisição GET para obter os valores das <options>...")
        response = _requisicao_com_retry('GET', urlOPT, headers=headers, timeout=120)
        response.raise_for_status()  # Levanta exceção em caso de status code de erro
        logger.info(f"Status da resposta: {response.status_code}")

        # 2) Extrair valores das <option> (cada option representa um conjunto de dados)
        logger.info("Extraindo valores das <option>...")
        options = _cnes_extrair_valores_options(response.text)

    except requests.exceptions.Timeout:
        logger.error("A requisição GET para urlOPT demorou muito (timeout).")
        return False
    except requests.exceptions.RequestException as e:
        logger.error(f"Erro ao fazer a requisição GET: {e}")
        return False

    # Verifica se encontrou alguma <option>
    if not options:
        logger.error("Nenhum valor encontrado nas <option>. Processo encerrado.")
        return False

    # 3) Montar o payload substituindo o placeholder "->INPUT<-"
    #    Concatenamos 'Arquivos=valor&' para cada valor encontrado
    str_options = ''
    for option in options:
        str_options += f'Arquivos={option}&'

    # Substitui o '->INPUT<-' pelo conjunto de opções
    payload = payload.replace('->INPUT<-', str_options)

    try:
        # 4) Faz a requisição POST para obter os dados.
        #    Como resposta, teremos um HTML contendo um <PRE> com dados.
        logger.info("Fazendo a requisição POST para obter os dados CNES...")
        response_post = _requisicao_com_retry('POST', urlRSP, data=payload, headers=headers, timeout=600)
        response_post.raise_for_status()  # Levanta exceção se houver erro HTTP

        # 5) Converte o HTML para DataFrame (removendo linhas de "Total")
        df = _cnes_html_to_dataframe(response_post.text, remover_linha_total=True)

        # 6) Aplica o tratamento do DataFrame (renomear colunas, converter datas, etc.)
        df = _cnes_tratamento(df, 'QT_ESTABELECIMENTOS_TOTAL')

        # 7) Garante que o diretório 'data' exista
        os.makedirs('data', exist_ok=True)

        # 8) Salva o resultado em Parquet
        df.to_parquet('data/estabelecimentos_CNES.parquet', index=False)
        logger.info("Dados processados e salvos em 'data/estabelecimentos_CNES.parquet'.")

    except requests.exceptions.Timeout:
        logger.error("A requisição POST demorou muito (timeout).")
        return False
    except requests.exceptions.RequestException as e:
        logger.error(f"Falha ao fazer a requisição POST: {e}")
        return False

    # Se chegou até aqui sem erros, retorno implícito (sucesso).
    return True

def proc05_download_profissionaisCNES():
    """
    Captura dados de profissionais (CNES) a partir do DataSUS, 
    consolida os resultados em um DataFrame e salva em Parquet.

    Fluxo Geral:
    ------------
    1) Faz uma requisição GET (urlOPT) para obter um HTML contendo um select com várias <option>.
    2) Extrai a lista de 'options', cada qual representa um arquivo/tabela diferente a ser baixado.
    3) Para cada option, faz uma requisição POST (urlRSP), obtem dados no formato <PRE> (CSV).
    4) Converte para DataFrame, aplica tratamento (renomeio de colunas, datas), 
       e concatena num DF final.
    5) Salva o resultado em 'data/profissionais_CNES.parquet'.

    Retorno:
    --------
    None
        Em caso de sucesso, imprime logs no console e salva o arquivo Parquet em 'data/'.
        Se ocorrer algum erro grave, retorna False.
    """

    # -------------------------------------------------------------------------
    # Configurações de URL e Headers (funções auxiliares de raspagem CNES são as
    # helpers de módulo _cnes_* — compartilhadas com proc03)
    # -------------------------------------------------------------------------

    urlOPT = 'http://tabnet.datasus.gov.br/cgi/deftohtm.exe?cnes/cnv/prid02br.def'
    urlRSP = 'http://tabnet.datasus.gov.br/cgi/tabcgi.exe?cnes/cnv/prid02br.def'

    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                      'AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/58.0.3029.110 Safari/537.3',
        # Cookie de sessão do TabNet — ver SINAN_COOKIE_TABNET_CNES no topo do arquivo.
        'Cookie': COOKIE_TABNET_CNES,
    }

    # Esse payload possui um placeholder "->INPUT<-"
    # que será substituído pelos valores de cada arquivo (option).
    payload = (
        'Linha=Munic%EDpio&Coluna=Ano%2Fm%EAs_compet.&Incremento=Quantidade&'
        '->INPUT<-'
        'SRegi%E3o=TODAS_AS_CATEGORIAS__&pesqmes2=Digite+o+texto+e+ache+f%E1cil&'
        'SUnidade_da_Federa%E7%E3o=TODAS_AS_CATEGORIAS__&pesqmes3=Digite+o+texto+e+ache+f%E1cil&'
        'SMunic%EDpio=TODAS_AS_CATEGORIAS__&pesqmes4=Digite+o+texto+e+ache+f%E1cil&'
        'SMunic%EDpio_gestor=TODAS_AS_CATEGORIAS__&pesqmes5=Digite+o+texto+e+ache+f%E1cil&'
        'SCapital=TODAS_AS_CATEGORIAS__&pesqmes6=Digite+o+texto+e+ache+f%E1cil&'
        'SRegi%E3o_de_Sa%FAde_%28CIR%29=TODAS_AS_CATEGORIAS__&pesqmes7=Digite+o+texto+e+ache+f%E1cil&'
        'SMacrorregi%E3o_de_Sa%FAde=TODAS_AS_CATEGORIAS__&pesqmes8=Digite+o+texto+e+ache+f%E1cil&'
        'SMicrorregi%E3o_IBGE=TODAS_AS_CATEGORIAS__&pesqmes9=Digite+o+texto+e+ache+f%E1cil&'
        'SRegi%E3o_Metropolitana_-_RIDE=TODAS_AS_CATEGORIAS__&pesqmes10=Digite+o+texto+e+ache+f%E1cil&'
        'STerrit%F3rio_da_Cidadania=TODAS_AS_CATEGORIAS__&pesqmes11=Digite+o+texto+e+ache+f%E1cil&'
        'SMesorregi%E3o_PNDR=TODAS_AS_CATEGORIAS__&SAmaz%F4nia_Legal=TODAS_AS_CATEGORIAS__&'
        'SSemi%E1rido=TODAS_AS_CATEGORIAS__&SFaixa_de_Fronteira=TODAS_AS_CATEGORIAS__&'
        'SZona_de_Fronteira=TODAS_AS_CATEGORIAS__&SMunic%EDpio_de_extrema_pobreza=TODAS_AS_CATEGORIAS__&'
        'SEnsino%2FPesquisa=TODAS_AS_CATEGORIAS__&pesqmes18=Digite+o+texto+e+ache+f%E1cil&'
        'SNatureza_Jur%EDdica=TODAS_AS_CATEGORIAS__&pesqmes19=Digite+o+texto+e+ache+f%E1cil&'
        'SEsfera_Jur%EDdica=TODAS_AS_CATEGORIAS__&SEsfera_Administrativa=TODAS_AS_CATEGORIAS__&'
        'pesqmes21=Digite+o+texto+e+ache+f%E1cil&SNatureza=TODAS_AS_CATEGORIAS__&pesqmes22=Digite+o+texto+e+ache+f%E1cil&'
        'STipo_de_Estabelecimento=TODAS_AS_CATEGORIAS__&STipo_de_Gest%E3o=TODAS_AS_CATEGORIAS__&'
        'STipo_de_Prestador=TODAS_AS_CATEGORIAS__&pesqmes25=Digite+o+texto+e+ache+f%E1cil&'
        'SOcupa%E7%F5es_de_N%EDvel_Superior=TODAS_AS_CATEGORIAS__&pesqmes26=Digite+o+texto+e+ache+f%E1cil&'
        'SOcupa%E7%F5es_de_N%EDvel_T%E9c_Aux=TODAS_AS_CATEGORIAS__&pesqmes27=Digite+o+texto+e+ache+f%E1cil&'
        'SOcupa%E7%F5es_de_N%EDvel_Elementar=TODAS_AS_CATEGORIAS__&pesqmes28=Digite+o+texto+e+ache+f%E1cil&'
        'SOcupa%E7%F5es_Administrativas=TODAS_AS_CATEGORIAS__&pesqmes29=Digite+o+texto+e+ache+f%E1cil&'
        'SOcupa%E7%F5es_em_geral=TODAS_AS_CATEGORIAS__&pesqmes30=Digite+o+texto+e+ache+f%E1cil&'
        'SM%E9dicos=TODAS_AS_CATEGORIAS__&SAtende_no_SUS=TODAS_AS_CATEGORIAS__&formato=prn&mostre=Mostra'
    )

    # -------------------------------------------------------------------------
    # 1) Faz a requisição GET para obter a lista de <option>
    # -------------------------------------------------------------------------
    try:
        logger.info("Fazendo a requisição GET para obter as options...")
        response = _requisicao_com_retry('GET', urlOPT, headers=headers, timeout=120)
        response.raise_for_status()  # Levanta exceção se status >= 400
        logger.info(f"Status da resposta: {response.status_code}")

        # Extrai valores das <option>
        logger.info("Extraindo valores das <option> do HTML...")
        options = _cnes_extrair_valores_options(response.text)

    except Timeout:
        logger.error("A requisição GET excedeu o tempo limite.")
        return False
    except RequestException as e:
        logger.error(f"Falha na requisição GET: {e}")
        return False

    # Se nenhuma <option> for encontrada, encerrar
    if not options:
        logger.error("Nenhum valor encontrado nas <option>. Encerrando processo.")
        return False

    # -------------------------------------------------------------------------
    # 2) Acumula os DataFrames de cada arquivo numa lista (em vez de fazer
    #    pd.concat a cada iteração, que é O(n^2) para dezenas de arquivos)
    # -------------------------------------------------------------------------
    dfs_parciais = []

    # -------------------------------------------------------------------------
    # 3) Para cada option, faz uma requisição POST separada
    # -------------------------------------------------------------------------
    for arquivo in options:
        try:
            logger.info(f"Fazendo a requisição POST para obter dados do arquivo: {arquivo}")

            # Substitui '->INPUT<-' com 'Arquivos=XXX&'
            payload_mod = payload.replace('->INPUT<-', f'Arquivos={arquivo}&')

            # Requisição POST (com retry/backoff)
            response_post = _requisicao_com_retry('POST', urlRSP, data=payload_mod, headers=headers, timeout=120)
            response_post.raise_for_status()

            # Converte HTML para DataFrame (mantém todas as linhas, remove só a coluna 'Total')
            tmp_df = _cnes_html_to_dataframe(response_post.text, remover_coluna_total=True)

            # Aplica tratamento
            tmp_df = _cnes_tratamento(tmp_df, 'QT_PROFISSIONAIS_TOTAL')

            dfs_parciais.append(tmp_df)

        except Timeout:
            logger.error("A requisição POST excedeu o tempo limite.")
            continue
        except RequestException as e:
            logger.error(f"Falha ao fazer requisição POST ({arquivo}): {e}")
            continue

    # -------------------------------------------------------------------------
    # 4) Concatena tudo de uma vez e salva o resultado final em Parquet
    # -------------------------------------------------------------------------
    if dfs_parciais:
        df_final = pd.concat(dfs_parciais, ignore_index=True)
    else:
        df_final = pd.DataFrame(columns=['ID_MUNICIP', 'ANOMES_DIAG', 'QT_PROFISSIONAIS_TOTAL'])

    # Cria a pasta 'data' se não existir
    os.makedirs('data', exist_ok=True)

    output_path = 'data/profissionais_CNES.parquet'
    df_final.to_parquet(output_path, index=False)
    logger.info(f"Processo concluído. Dados salvos em {output_path}")
    return True


def proc06_converter_SINNAN(caminho_data='data/'):
    """
    Converte arquivos DBC (Sinan) para DBF e depois de DBF para Parquet, 
    armazenando cada tabela convertida no diretório final.
    
    Parâmetros:
    -----------
    caminho_data : str
        Caminho base onde serão buscados/salvos os arquivos.
        Subdiretórios padrão utilizados:
          - 'arquivos_dbc/'   : Contém os arquivos .dbc originais
          - 'arquivos_dbf/'   : Pasta onde os arquivos .dbf serão gerados
          - 'arquivos_parquet/': Pasta de destino para os arquivos .parquet
    
    Retorno:
    --------
    bool
        Retorna True se o processo ocorrer sem erros graves. 
        Retorna False se não encontrar a pasta de arquivos DBC.
    """

    logger.info('Convertendo arquivos DBC para DBF e depois para Parquet...')

    # -------------------------------------------------------------------------
    # 1) Definição dos diretórios de entrada e saída
    # -------------------------------------------------------------------------
    caminho_dbc = os.path.join(caminho_data, 'arquivos_dbc')
    caminho_dbf = os.path.join(caminho_data, 'arquivos_dbf')
    caminho_parquet = os.path.join(caminho_data, 'arquivos_parquet')
    
    # Verifica se o diretório de arquivos DBC existe
    if not os.path.exists(caminho_dbc):
        logger.error('Pasta com arquivos DBC não encontrada!')
        return False
    
    # Garante a existência dos diretórios de saída
    os.makedirs(caminho_dbf, exist_ok=True)
    os.makedirs(caminho_parquet, exist_ok=True)

    # -------------------------------------------------------------------------
    # 2) Conversão de DBC para DBF
    # -------------------------------------------------------------------------
    # Para cada arquivo .dbc encontrado em 'caminho_dbc', gera um arquivo .dbf correspondente
    for nome_arquivo in os.listdir(caminho_dbc):
        # Verifica se a extensão do arquivo é .dbc (ignora maiúsculas/minúsculas)
        if nome_arquivo.lower().endswith('.dbc'):
            # Constroi o caminho de entrada e de saída para a conversão
            caminho_entrada = os.path.join(caminho_dbc, nome_arquivo)
            nome_dbf = os.path.splitext(nome_arquivo)[0] + '.dbf'
            caminho_saida = os.path.join(caminho_dbf, nome_dbf)
            
            # Executa a conversão DBC -> DBF
            dbc2dbf(caminho_entrada, caminho_saida)
            logger.info(f"Convertido DBC -> DBF: {nome_arquivo} -> {nome_dbf}")

    # -------------------------------------------------------------------------
    # 3) Conversão de DBF para Parquet
    # -------------------------------------------------------------------------
    # Para cada arquivo .dbf em 'caminho_dbf', lê e converte para Parquet
    for nome_arquivo_dbf in os.listdir(caminho_dbf):
        # Verifica se a extensão é .dbf
        if nome_arquivo_dbf.lower().endswith('.dbf'):
            caminho_dbf_completo = os.path.join(caminho_dbf, nome_arquivo_dbf)
            
            # Lê o arquivo DBF (usando dbfread) com encoding 'latin1'
            try:
                tabela = DBF(caminho_dbf_completo, encoding='latin1')
            except Exception as e:
                logger.error(f"Não foi possível ler o arquivo DBF: {nome_arquivo_dbf}. Detalhes: {e}")
                continue

            # Converte o DBF para um DataFrame
            df = pd.DataFrame(iter(tabela))
            
            # Tenta limpar espaços em branco de cada coluna (caso seja string)
            for col in df.columns:
                try:
                    df[col] = df[col].astype(str).str.strip()
                except Exception as e:
                    logger.error(f"Erro ao converter/limpar coluna '{col}': {e}")

            # Define o caminho de saída no formato Parquet
            nome_parquet = os.path.splitext(nome_arquivo_dbf)[0] + '.parquet'
            caminho_parquet_completo = os.path.join(caminho_parquet, nome_parquet)
            
            # Salva o DataFrame em Parquet
            try:
                df.to_parquet(caminho_parquet_completo, index=False)
                logger.info(f"Convertido DBF -> Parquet: {nome_arquivo_dbf} -> {nome_parquet}")
            except Exception as e:
                logger.error(f"Falha ao salvar Parquet '{nome_parquet}': {e}")

    logger.info('Conversão concluída com sucesso!')
    return True


def proc07_Unir_tabelas(
    caminho_parquet='data/arquivos_parquet/',
    output_file='data/sinnan.parquet'
):
    """
    Unifica todos os arquivos Parquet do diretório especificado em um único arquivo,
    ordenando-os com base no número presente ao final do nome de cada arquivo.
    
    Parâmetros:
    -----------
    caminho_parquet : str
        Caminho da pasta onde estão os arquivos .parquet (por exemplo: "data/arquivos_parquet/").
    output_file : str
        Caminho completo + nome do arquivo de saída (por exemplo: "data/sinnan.parquet").
    
    Retorno:
    --------
    bool
        Retorna True se houver arquivos a unir e o processo concluir com sucesso, 
        ou False caso nenhum arquivo seja encontrado.
    """
   
    logger.info('Iniciando a unificação dos arquivos Parquet...')

    # -------------------------------------------------------------------------
    # 1) Localiza todos os arquivos TUBEBR*.parquet no diretório especificado
    # -------------------------------------------------------------------------
    # Monta o padrão: "TUBEBR*.parquet" – ajustável conforme a convenção de nomes
    pattern = os.path.join(caminho_parquet, 'TUBEBR*.parquet')
    arquivos = glob.glob(pattern)

    # Se nenhum arquivo for encontrado, retorna False
    if not arquivos:
        logger.error("Nenhum arquivo '.parquet' encontrado no diretório especificado.")
        return False

    # -------------------------------------------------------------------------
    # 2) Ordena a lista de arquivos com base no número do final do nome
    # -------------------------------------------------------------------------
    # Exemplo de nome de arquivo: TUBEBR2021.parquet
    # Usa expressão regular para capturar a parte numérica antes de ".parquet"
    # e converte para int a fim de ordenar numericamente (2020 < 2021 < 2022...).
    def extrai_numero_final(filepath):
        match = re.search(r'(\d+)\.parquet$', filepath)
        # Garante que se não encontrar nada, retorne um número grande para ir ao final
        return int(match.group(1)) if match else 9999999999
    
    arquivos.sort(key=extrai_numero_final)

    # -------------------------------------------------------------------------
    # 3) Lê cada arquivo e concatena os DataFrames em um único DF
    # -------------------------------------------------------------------------
    df_list = []
    for arquivo in arquivos:
        logger.info(f"Lendo arquivo: {arquivo}")
        df_temp = pd.read_parquet(arquivo)
        df_list.append(df_temp)

    df_final = pd.concat(df_list, ignore_index=True)
    logger.info(f"Total de registros após concatenação: {len(df_final)}")

    # -------------------------------------------------------------------------
    # 4) Salva o DataFrame resultante no arquivo de saída (Parquet)
    # -------------------------------------------------------------------------
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_final.to_parquet(output_file, index=False)
    
    logger.info(f"Arquivos unificados e salvos em: {output_file}")
    return True


def proc08_tratamento_SINNAN_01():
    """
    Carrega o arquivo 'sinnan.parquet', realiza limpezas e conversões em suas colunas
    e salva o resultado em 'sinnan_preliminar01.parquet'.

    Etapas Principais:
    ------------------
    1. Tenta carregar o DataFrame a partir de 'data/sinnan.parquet'.
    2. Cria a coluna 'ANO_DIAG' a partir do início de 'DT_DIAG'.
    3. Normaliza espaços e converte marcadores textuais de ausência ('', 'nan',
       'None', '-') em <NA>.
    4. Converte 'ANO_NASC' para inteiro anulável.
    5. Ignora colunas que começam com 'DT', 'ANO' ou 'NU_IDADE_N' para não
       sobrescrever datas/anos/idades.
    6. Converte para inteiro anulável (Int64) as demais colunas cujos valores
       presentes sejam TODOS numéricos, preservando <NA> onde o dado não existe.
    7. Salva o resultado final em 'data/sinnan_preliminar01.parquet'.

    Preservação de ausência
    -----------------------
    Esta etapa não imputa. A versão anterior aplicava `.fillna(0).astype(int)`
    em toda coluna numérica, o que tornava ausência indistinguível do código 0
    do dicionário SINAN (em SITUA_ENCE, fundia 66.420 campos vazios com 13.610
    encerramentos codificados como '0'). A decisão de imputar pertence ao código
    de análise, que a documenta e audita.

    Retorno:
    --------
    bool
        True se o processo ocorrer sem erro, False se o arquivo 'sinnan.parquet' não for encontrado.
    """
  
    # 1) Tenta carregar o DataFrame; se falhar, encerra o processo
    try:
        df = pd.read_parquet('data/sinnan.parquet')
    except FileNotFoundError:
        logger.error('Arquivo "data/sinnan.parquet" não encontrado!')
        return False
    except Exception as e:
        logger.error(f"Falha ao ler o arquivo 'data/sinnan.parquet': {e}")
        return False
    
    logger.info('Iniciando tratamento do dataframe...')

    # -------------------------------------------------------------------------
    # Função auxiliar para identificar se a coluna é "numericamente mista"
    # (ou seja, combina números e valores ausentes, mas sem dados não-numéricos).
    # -------------------------------------------------------------------------
    # Marcadores textuais que representam ausência, não um valor observado.
    # Precisam virar <NA> antes de qualquer conversão: o SINAN grava campo vazio
    # como string vazia, e o Parquet/pandas trazem NaN como 'nan' após astype(str).
    AUSENTES_TEXTUAIS = {"", "nan", "NaN", "None", "<NA>", "NA", "-"}

    def _normaliza_ausentes(series):
        """Aplica strip e converte marcadores textuais de ausência em <NA>."""
        limpa = series.astype("string").str.strip()
        return limpa.mask(limpa.isin(AUSENTES_TEXTUAIS))

    def is_mixed_numeric(series):
        """
        Verifica se TODOS os valores presentes (não-ausentes) da série são
        numéricos — isto é, se a coluna pode virar inteiro sem destruir
        informação.

        A versão anterior comparava 'coerced.isna() + coerced.notna() == len',
        soma que é trivialmente verdadeira para qualquer série. Na prática a
        função respondia apenas "existe ao menos um número aqui?", de modo que
        uma coluna majoritariamente textual com um único número era convertida,
        e todo o texto virava ausente silenciosamente.
        """
        presentes = series.dropna()
        if presentes.empty:
            return False
        return pd.to_numeric(presentes, errors='coerce').notna().all()

    # 2) Cria coluna 'ANO_DIAG' extraindo os 4 primeiros caracteres de 'DT_DIAG' (ano)
    df['ANO_DIAG'] = df['DT_DIAG'].astype('string').str[:4]

    # 3) Normaliza espaços e marcadores de ausência em todas as colunas.
    #    (A versão anterior fazia astype(str) direto, o que convertia NaN na
    #    string literal 'nan' e a carregava adiante como se fosse um valor.)
    for col in df.columns:
        df[col] = _normaliza_ausentes(df[col])

    # 4) Loop por cada coluna para conversões específicas
    for coluna in df.columns:
        # ANO_NASC é ano de nascimento: numérico, mas ausência continua ausência
        # (a versão anterior usava fillna(0), criando nascimentos no ano 0).
        if coluna.startswith("ANO_NASC"):
            df[coluna] = pd.to_numeric(df[coluna], errors='coerce').astype('Int64')
            continue

        # Ignora demais colunas de data/ano/idade (tratadas em etapa própria)
        if coluna.startswith(('DT', 'ANO', 'NU_IDADE_N')):
            continue

        # Converte para inteiro anulável (Int64) apenas quando todos os valores
        # presentes são numéricos.
        #
        # IMPORTANTE: nada de fillna(0). A versão anterior fazia
        # `.fillna(0).astype(int)` em todas as colunas numéricas, o que fundia
        # ausência com o código 0 real do dicionário SINAN. Em SITUA_ENCE, por
        # exemplo, isso colapsava 66.420 campos vazios e 13.610 códigos '0'
        # legítimos num único valor 0, tornando impossível distinguir "sem
        # encerramento registrado" de um encerramento codificado como 0.
        # Ausência é informação: preservamos <NA> e deixamos a decisão de
        # imputação para o código de análise, que a documenta e audita.
        if is_mixed_numeric(df[coluna]):
            numerica = pd.to_numeric(df[coluna], errors='coerce')
            # Int64 só quando todos os valores são inteiros; caso contrário
            # astype('Int64') levantaria TypeError ao truncar. Colunas com
            # decimal legítimo permanecem float.
            presentes = numerica.dropna()
            if (presentes % 1 == 0).all():
                df[coluna] = numerica.astype('Int64')
            else:
                df[coluna] = numerica.astype('float64')


    # 7) Salva o DataFrame tratado em Parquet
    output_path = 'data/sinnan_preliminar01.parquet'
    df.to_parquet(output_path, index=False)
    
    logger.info(f'Dataframe tratado e salvo em {output_path}')
    return True

def proc09_converter_IBGE():
    """
    Lê o arquivo JSON 'indicadores_todos_municipios.json' (gerado a partir da API de indicadores do IBGE),
    organiza os dados e salva em um CSV 'indicadores_IBGE.csv', onde cada linha representa um município/ano
    com as colunas correspondentes aos diferentes indicadores (ex.: PIB per capita, IDHM etc.).
    
    Padrões de nomenclatura de colunas:
    -----------------------------------
    1) Todas as colunas em maiúsculo, sem acentos.
    2) Usar '_' para separar palavras, e remover espaços/pontuação.
    3) Não usar mais que 4 "segmentos" (palavras) após o tratamento.
    4) Todas as colunas começam com "IBGE_", exceto 'ID_MUNICIP' e 'ANO_DIAG'.

    Fluxo Geral:
    ------------
    1) Informa o início da conversão e os caminhos de arquivos de entrada e saída.
    2) Carrega o JSON com dados de todos os municípios (códigos, indicadores, valores por ano).
    3) Extrai os IDs de indicadores presentes para definir as colunas de saída.
    4) Gera uma estrutura de dados onde cada ano do município recebe o conjunto de indicadores correspondentes.
    5) Monta o DataFrame unificado, padroniza o nome das colunas e salva em CSV.

    Retorno:
    --------
    None
        Esta função não retorna valores. Se ocorrer erro de leitura/escrita, 
        imprime mensagens explicativas no console.
    """
    
   
    # -------------------------------------------------------------------------
    # 1) Caminhos de entrada e saída + mensagens ao usuário
    # -------------------------------------------------------------------------
    json_file_path = 'data/indicadores_todos_municipios.json'
    csv_file_path = 'data/indicadores_IBGE.csv'

    logger.info("Iniciando processo de conversão do IBGE...")
    logger.info(f"Arquivo de entrada JSON: {json_file_path}")
    logger.info(f"Arquivo de saída CSV : {csv_file_path}")

    # Dicionário de indicadores (ID -> Descrição legível)
    indicadores = {
        "96386": "Densidade demográfica",
        "143558": "Salário médio mensal",
        "143514": "Pessoal ocupado",
        "60036": "População ocupada",
        "60037": "População c/ rendimento <= 1/2 salário mínimo (%)",
        "60045": "Taxa de escolarização (6 a 14 anos)",
        "78187": "IDEB – Anos iniciais (Rede pública)",
        "78192": "IDEB – Anos finais (Rede pública)",
        "5950": "Número de estab. de ensino fundamental",
        "5955": "Número de estab. de ensino médio",
        "47001": "PIB per capita",
        "30255": "IDHM",
        "30279": "Mortalidade Infantil",
        "60032": "Internações por diarreia (SUS)",
        "28242": "Estab. de Saúde (SUS)", 
        "95335": "Área urbanizada",
        "60030": "Esgotamento sanitário adequado (%)",
        "60029": "Arborização de vias públicas (%)",
        "60031": "Urbanização de vias públicas (%)",
        "77861": "Bioma",
        "82270": "Sistema Costeiro-Marinho"
    }
    
    # -------------------------------------------------------------------------
    # 2) Carrega o arquivo JSON de entrada, tratando possíveis erros
    # -------------------------------------------------------------------------
    try:
        logger.info("Carregando dados do arquivo JSON...")
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Dados carregados com sucesso! Registros de municípios encontrados: {len(data)}")
    except FileNotFoundError:
        logger.error(f"Arquivo '{json_file_path}' não encontrado. Verifique o caminho.")
        return
    except json.JSONDecodeError as e:
        logger.error(f"Não foi possível decodificar o JSON: {e}")
        return

    # -------------------------------------------------------------------------
    # 3) Extrai IDs de indicadores para montar as colunas (fieldnames)
    # -------------------------------------------------------------------------
    logger.info("Extraindo IDs de indicadores presentes no arquivo...")
    indicadores_ids = set()
    for municipio in data:
        for indicador in municipio['indicadores']:
            indicadores_ids.add(str(indicador['id']))
    
    logger.info(f"Total de IDs de indicadores diferentes encontrados: {len(indicadores_ids)}")

    fieldnames = ['ID_MUNICIP', 'ANO_DIAG'] + [
        indicadores[id_str] for id_str in sorted(indicadores_ids) if id_str in indicadores
    ]
    logger.info(f"Colunas definidas (descrições originais): {fieldnames}")

    # -------------------------------------------------------------------------
    # 4) Montagem de dados: agrupa os valores por ano de cada município
    # -------------------------------------------------------------------------
    dados_list = []
    
    logger.info("Processando dados de cada município e seus indicadores...")
    for municipio in data:
        codigo_municipio = municipio['localidade']['codigo']
        
        # anos_dados[ano] -> dicionário com colunas e valores
        anos_dados = {}
        for indicador in municipio['indicadores']:
            id_indicador = str(indicador['id'])
            descricao_indicador = indicadores.get(id_indicador, 'Descrição não encontrada')
            
            # 'indicador["res"]' é um dict do tipo {ano: valor, ano: valor, ...}
            for ano, valor in indicador['res'].items():
                if ano not in anos_dados:
                    anos_dados[ano] = {
                        'ID_MUNICIP': codigo_municipio,
                        'ANO_DIAG': ano
                    }
                # Preenche o valor deste indicador na linha correspondente ao ano
                anos_dados[ano][descricao_indicador] = valor

        # Adiciona as linhas deste município à lista geral
        dados_list.append(list(anos_dados.values()))
    
    logger.info("Estruturando DataFrame final...")

    # Achata a lista de listas (uma por município) e monta o DataFrame de uma só vez,
    # em vez de um pd.concat por município (O(n^2) para ~5.570 municípios).
    linhas_flat = [linha for bloco_municipio in dados_list for linha in bloco_municipio]
    new_df = pd.DataFrame(linhas_flat, columns=fieldnames) if linhas_flat else pd.DataFrame(columns=fieldnames)

    logger.info(f"Tamanho final do DataFrame: {new_df.shape[0]} linhas, {new_df.shape[1]} colunas.")

    # -------------------------------------------------------------------------
    # 5) Padroniza os nomes das colunas conforme regras definidas
    # -------------------------------------------------------------------------
    logger.info("Padronizando nomes das colunas...")

    def _palavras_normalizadas(colname):
        """Remove acentos, força maiúsculas e devolve a lista de palavras."""
        sem_acentos = unidecode(colname).upper()
        return re.sub(r'[^A-Z0-9]+', ' ', sem_acentos).split()

    def padronizar_nome_coluna(colname, n_palavras=4):
        """
        Aplica as seguintes regras:
          1) Se for 'ID_MUNICIP' ou 'ANO_DIAG', mantém como está.
          2) Caso contrário, remove acentos, transforma em maiúsculo, substitui
             caracteres não alfanuméricos por espaço, usa as 'n_palavras'
             primeiras palavras unidas por '_' e prefixa 'IBGE_'.
        """
        if colname in ('ID_MUNICIP', 'ANO_DIAG'):
            return colname
        return f'IBGE_{"_".join(_palavras_normalizadas(colname)[:n_palavras])}'

    # Gera o dicionário de renome garantindo nomes ÚNICOS e DISTINGUÍVEIS.
    #
    # Truncar em 4 palavras faz indicadores distintos colidirem: "Número de
    # estab. de ensino fundamental" e "Número de estab. de ensino médio"
    # produziam ambos IBGE_NUMERO_DE_ESTAB_DE. Nomes repetidos geravam duas
    # colunas homônimas no CSV, que o pandas depois desambiguava sozinho como
    # 'IBGE_NUMERO_DE_ESTAB_DE' e 'IBGE_NUMERO_DE_ESTAB_DE.1' — dois indicadores
    # diferentes com nomes que não dizem qual é qual.
    #
    # Quando um grupo de colunas colide, estendemos TODAS as colunas do grupo
    # (não apenas a segunda) com mais palavras do nome original, até que fiquem
    # mutuamente distintas. Estender só a colidente produziria
    # 'IBGE_NUMERO_DE_ESTAB_DE' e 'IBGE_NUMERO_DE_ESTAB_DE_ENSINO': únicos, mas
    # ainda sem dizer qual é fundamental e qual é médio.
    grupos = {}
    for col in new_df.columns:
        grupos.setdefault(padronizar_nome_coluna(col), []).append(col)

    rename_dict = {}
    for nome_base, colunas_do_grupo in grupos.items():
        if len(colunas_do_grupo) == 1:
            rename_dict[colunas_do_grupo[0]] = nome_base
            continue

        max_palavras = max(len(_palavras_normalizadas(c)) for c in colunas_do_grupo)
        for n in range(5, max_palavras + 1):
            candidatos = {c: padronizar_nome_coluna(c, n_palavras=n) for c in colunas_do_grupo}
            if len(set(candidatos.values())) == len(colunas_do_grupo):
                logger.info(
                    f"Colisão em '{nome_base}' resolvida com {n} palavras: "
                    + ", ".join(f"'{c}' -> '{nome}'" for c, nome in candidatos.items())
                )
                rename_dict.update(candidatos)
                break
        else:
            # Nomes de origem indistinguíveis mesmo por extensão: sufixo numérico.
            logger.warning(
                f"Colisão em '{nome_base}' não resolvível por extensão "
                f"({colunas_do_grupo}): usando sufixos numéricos."
            )
            for i, col in enumerate(colunas_do_grupo, start=1):
                rename_dict[col] = nome_base if i == 1 else f"{nome_base}_{i}"

    if len(set(rename_dict.values())) != len(rename_dict):
        raise ValueError(
            "Falha ao gerar nomes únicos para as colunas do IBGE — "
            "isso produziria colunas homônimas no CSV."
        )

    # Aplica o dicionário de renome ao DataFrame
    new_df.rename(columns=rename_dict, inplace=True)

    # Salva o mapa nome original -> nome padronizado. O briefing (§8.3) exige um
    # mapa bidirecional entre nome bruto e harmonizado; sem ele, IBGE_* é opaco.
    try:
        pd.DataFrame(
            sorted(rename_dict.items()), columns=["nome_original", "nome_padronizado"]
        ).to_csv("data/dicionario_colunas_IBGE.csv", index=False)
        logger.info("Mapa de nomes das colunas IBGE salvo em 'data/dicionario_colunas_IBGE.csv'.")
    except OSError as e:
        logger.warning(f"Não foi possível salvar o mapa de nomes das colunas IBGE: {e}")

    logger.info("Colunas após padronização:")
    logger.info(list(new_df.columns))

    # -------------------------------------------------------------------------
    # 6) Salva o DataFrame em CSV
    # -------------------------------------------------------------------------
    os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)

    logger.info("Salvando DataFrame final em CSV...")
    new_df.to_csv(csv_file_path, index=False, encoding='utf-8')
    logger.info(f"Conversão concluída! Arquivo gerado: {csv_file_path}")
    return True


def proc10_tratamento_SINNAN_02():
    """
    1) Lê 'data/sinnan_preliminar01.parquet'.
    2) Faz requisição à API do IBGE para obter informações de municípios.
       - Ajusta códigos incompletos (ID_MUNICIP / ID_MN_RESI).
       - Insere colunas de nome de município, sigla de estado e região.
    3) Se restarem linhas em que ID_MUNICIP e ID_MN_RESI ficaram "0",
       verifica a coluna ID_MUNIC_A para tentar corrigir.
    4) Salva o resultado em 'data/sinnan_tratado.parquet'.
    """
   
    logger.info("Iniciando processo de ajuste de códigos do SINAN...")

    # URL para obter a lista de municípios (API IBGE)
    url_municipios = "https://servicodados.ibge.gov.br/api/v1/localidades/municipios"

    # 1) Carrega o DataFrame inicial
    parquet_path = "data/sinnan_preliminar01.parquet"
    if not os.path.exists(parquet_path):
        logger.error(f"Arquivo não encontrado: {parquet_path}")
        return False

    df = pd.read_parquet(parquet_path)
    logger.info(f"DataFrame carregado: {df.shape[0]} linhas, {df.shape[1]} colunas.")

    # 2) Faz requisição ao IBGE para montar os dicionários de lookup
    logger.info("Fazendo requisição à API do IBGE para obter dados de municípios...")
    try:
        response = requests.get(url_municipios)
        response.raise_for_status()
    except requests.RequestException as e:
        logger.error(f"Falha ao obter lista de municípios do IBGE: {e}")
        return False

    # Monta dicionários
    codigo_municipio_completo = {}  # Mapeia código_incompleto -> código_completo
    nome_municipio_dict = {}
    siglas_estados_dict = {}
    regiao_municipio_dict = {}

    # Preenche dicionários percorrendo cada município do JSON
    lista_municipios = response.json()
    for municipio in lista_municipios:
        cod_municipio_completo_7 = str(municipio.get("id"))
        nome_mun = municipio.get("nome")

        microrregiao = municipio.get("microrregiao")
        mesorregiao = microrregiao.get("mesorregiao") if microrregiao else None
        uf = mesorregiao.get("UF") if mesorregiao else None
        regiao = uf.get("regiao") if uf else None

        if not uf or not uf.get("sigla") or not regiao or not regiao.get("nome"):
            logger.warning(f"Estrutura incompleta para município: {nome_mun} (ID: {cod_municipio_completo_7})")
            continue  # pula para o próximo

        sigla_est = uf["sigla"]
        regiao_nome = regiao["nome"]

        # Truncar para 6 dígitos (caso seja essa a regra de 'incompleto')
        cod_incompleto_6 = cod_municipio_completo_7[:6]

        # Preencher mapeamentos
        codigo_municipio_completo[cod_incompleto_6] = cod_municipio_completo_7
        nome_municipio_dict[cod_municipio_completo_7] = nome_mun
        siglas_estados_dict[cod_municipio_completo_7] = sigla_est
        regiao_municipio_dict[cod_municipio_completo_7] = regiao_nome
    

    logger.info("Dicionários de município/estado/região montados com sucesso.")

    # 3) Ajuste de códigos incompletos nas colunas ID_MUNICIP e ID_MN_RESI
    #    (Series.replace com dict é vetorizado: troca só as chaves presentes no
    #    dicionário e mantém as demais como estavam — mesmo efeito de
    #    'mapa.get(cod, cod)', sem o custo de um apply() por linha.)
    logger.info("Ajustando códigos incompletos nas colunas ID_MUNICIP / ID_MN_RESI...")

    for coluna in ["ID_MUNICIP", "ID_MN_RESI"]:
        df[coluna] = (
            df[coluna]
            .astype(str)
            .str.replace(r"\.0$", "", regex=True)
            .replace(codigo_municipio_completo)
        )

    # 4) Verifica códigos inválidos e corrige (vetorizado com máscara booleana)
    # (A) Marca ID_MUNICIP == "0" ou ausente no dicionário de municípios,
    #     e substitui pelo valor de ID_MN_RESI nessas linhas.
    municipios_validos = set(nome_municipio_dict.keys())

    mask_municip_invalido = (
        (df["ID_MUNICIP"].str.strip() == "0") | (~df["ID_MUNICIP"].isin(municipios_validos))
    )
    df.loc[mask_municip_invalido, "ID_MUNICIP"] = df.loc[mask_municip_invalido, "ID_MN_RESI"]

    # (B) Mesma verificação para ID_MN_RESI, corrigindo com base no ID_MUNICIP
    #     (já eventualmente corrigido no passo anterior).
    mask_resi_invalido = (
        (df["ID_MN_RESI"].str.strip() == "0") | (~df["ID_MN_RESI"].isin(municipios_validos))
    )
    df.loc[mask_resi_invalido, "ID_MN_RESI"] = df.loc[mask_resi_invalido, "ID_MUNICIP"]

    # -------------------------------------------------------------------------
    # 5) Se ainda restarem registros com ID_MUNICIP == "0" e ID_MN_RESI == "0",
    #    usar ID_MUNIC_A para tentar corrigir esses casos derradeiros.
    # -------------------------------------------------------------------------
    mask_zeros = (df["ID_MUNICIP"] == "0") & (df["ID_MN_RESI"] == "0")

    if "ID_MUNIC_A" in df.columns:
        # Para essas linhas, se ID_MUNIC_A não for nulo/vazio/zero, substitui
        # ID_MUNICIP e ID_MN_RESI pelo valor de ID_MUNIC_A.
        # Obs.: Ajuste conforme sua lógica. Aqui usamos a string "0" como valor zero.
        for idx in df[mask_zeros].index:
            munic_a_val = str(df.at[idx, "ID_MUNIC_A"]).strip()
            if munic_a_val not in ("", "0", "nan", "None"):
                # Substitui
                df.at[idx, "ID_MUNICIP"] = munic_a_val
                df.at[idx, "ID_MN_RESI"] = munic_a_val
    else:
        logger.warning("Coluna 'ID_MUNIC_A' não encontrada no DF, não foi possível corrigir zeros remanescentes.")

    # -------------------------------------------------------------------------
    # 6) Resolve, de uma vez por todas, o MUNICÍPIO DE ANÁLISE.
    #
    #    Este é o ponto único onde a geografia da análise é decidida. Todo
    #    enriquecimento geográfico posterior (lat/long em proc12, CNES em
    #    proc13, IBGE/IPEA em proc14) passa a ler ID_MUNIC_ANALISE, em vez de
    #    cada etapa escolher sozinha entre residência e notificação.
    #
    #    Antes, lat/long/nomes vinham de ID_MN_RESI e os indicadores
    #    socioeconômicos de ID_MUNICIP. Como as duas divergem em ~10% dos
    #    registros, cerca de 130 mil notificações recebiam contexto municipal
    #    de um município diferente daquele que definia suas coordenadas — e,
    #    portanto, seu cluster espacial. Uma única coluna resolvida elimina
    #    essa classe de inconsistência na origem.
    # -------------------------------------------------------------------------
    logger.info(
        f"Resolvendo município de análise (chave='{CHAVE_MUNICIPIO_ANALISE}', "
        f"fallback='{CHAVE_MUNICIPIO_FALLBACK}') em '{COLUNA_MUNICIPIO_RESOLVIDO}'..."
    )

    chave = df[CHAVE_MUNICIPIO_ANALISE].where(
        df[CHAVE_MUNICIPIO_ANALISE].isin(municipios_validos)
    )
    fallback = df[CHAVE_MUNICIPIO_FALLBACK].where(
        df[CHAVE_MUNICIPIO_FALLBACK].isin(municipios_validos)
    )
    df[COLUNA_MUNICIPIO_RESOLVIDO] = chave.fillna(fallback)

    n_total = len(df)
    n_fallback = int(chave.isna().sum() - df[COLUNA_MUNICIPIO_RESOLVIDO].isna().sum())
    n_sem_municipio = int(df[COLUNA_MUNICIPIO_RESOLVIDO].isna().sum())
    n_divergentes = int((df["ID_MN_RESI"] != df["ID_MUNICIP"]).sum())

    logger.info(
        f"Município de análise resolvido: {n_total - n_sem_municipio} de {n_total} registros. "
        f"{n_fallback} usaram o fallback '{CHAVE_MUNICIPIO_FALLBACK}'; "
        f"{n_sem_municipio} ficaram sem município válido."
    )
    logger.info(
        f"Divergência residência vs. notificação: {n_divergentes} registros "
        f"({n_divergentes / n_total * 100:.2f}%). Estes são os registros cujo contexto "
        f"municipal mudaria caso CHAVE_MUNICIPIO_ANALISE fosse trocada."
    )

    # Auditoria da divergência, exigida pelo briefing (§8.5 integridade
    # referencial municipal). Fica em data/ para o relatório de qualidade.
    try:
        os.makedirs("data", exist_ok=True)
        auditoria = pd.DataFrame(
            [
                {"metrica": "registros_total", "valor": n_total},
                {"metrica": "chave_municipio_analise", "valor": CHAVE_MUNICIPIO_ANALISE},
                {"metrica": "registros_com_municipio_valido", "valor": n_total - n_sem_municipio},
                {"metrica": "registros_via_fallback", "valor": n_fallback},
                {"metrica": "registros_sem_municipio_valido", "valor": n_sem_municipio},
                {"metrica": "registros_residencia_diferente_notificacao", "valor": n_divergentes},
            ]
        )
        auditoria.to_csv("data/auditoria_municipio_analise.csv", index=False)
        logger.info("Auditoria da chave municipal salva em 'data/auditoria_municipio_analise.csv'.")
    except OSError as e:
        logger.warning(f"Não foi possível salvar a auditoria da chave municipal: {e}")

    # -------------------------------------------------------------------------
    # 7) Cria colunas de NOME_MUNIC, SIGLA_UF, REGIAO a partir da coluna
    #    resolvida — coerentes, por construção, com lat/long e com os
    #    indicadores municipais.
    # -------------------------------------------------------------------------
    logger.info("Criando colunas de nome de município, sigla e região...")

    df["NOME_MUNIC"] = df[COLUNA_MUNICIPIO_RESOLVIDO].map(nome_municipio_dict)
    df["SIGLA_UF"] = df[COLUNA_MUNICIPIO_RESOLVIDO].map(siglas_estados_dict)
    df["REGIAO"] = df[COLUNA_MUNICIPIO_RESOLVIDO].map(regiao_municipio_dict)

    # -------------------------------------------------------------------------
    # 8) Salva o arquivo final
    # -------------------------------------------------------------------------
    output_path = "data/sinnan_tratado.parquet"
    df.to_parquet(output_path, index=False)
    logger.info(f"Ajuste de códigos concluído e salvo em: {output_path}")
    return True


def proc11_estados_ibge_com_latlon():
    """
    Faz requisição ao IBGE para obter a lista de estados,
    adiciona latitude e longitude de cada UF (de um dicionário local)
    e retorna uma lista de dicionários no formato:
    [
      {
        "codigo_uf": ...,
        "uf": ...,
        "nome": ...,
        "latitude": ...,
        "longitude": ...,
        "regiao": ...
      },
      ...
    ]
    """
    # Dicionário local com as coordenadas aproximadas (ou do centroide) de cada estado
    # Ajuste caso tenha valores mais precisos.
    estados_latlon = {
        "AC":  {"latitude": -8.77,   "longitude": -70.55},
        "AL":  {"latitude": -9.62,   "longitude": -36.82},
        "AM":  {"latitude": -3.47,   "longitude": -65.10},
        "AP":  {"latitude": 1.41,    "longitude": -51.77},
        "BA":  {"latitude": -12.96,  "longitude": -38.51},
        "CE":  {"latitude": -5.20,   "longitude": -39.53},
        "DF":  {"latitude": -15.83,  "longitude": -47.86},
        "ES":  {"latitude": -19.19,  "longitude": -40.34},
        "GO":  {"latitude": -15.98,  "longitude": -49.86},
        "MA":  {"latitude": -5.42,   "longitude": -45.44},
        "MG":  {"latitude": -18.10,  "longitude": -44.38},
        "MS":  {"latitude": -20.51,  "longitude": -54.54},
        "MT":  {"latitude": -12.64,  "longitude": -55.42},
        "PA":  {"latitude": -3.79,   "longitude": -52.48},
        "PB":  {"latitude": -7.28,   "longitude": -36.72},
        "PE":  {"latitude": -8.38,   "longitude": -37.86},
        "PI":  {"latitude": -8.28,   "longitude": -43.68},
        "PR":  {"latitude": -24.89,  "longitude": -51.55},
        "RJ":  {"latitude": -22.25,  "longitude": -42.66},
        "RN":  {"latitude": -5.81,   "longitude": -36.59},
        "RO":  {"latitude": -10.83,  "longitude": -63.34},
        "RR":  {"latitude": 1.99,    "longitude": -61.33},
        "RS":  {"latitude": -30.17,  "longitude": -53.50},
        "SC":  {"latitude": -27.45,  "longitude": -50.95},
        "SE":  {"latitude": -10.57,  "longitude": -37.45},
        "SP":  {"latitude": -22.19,  "longitude": -48.79},
        "TO":  {"latitude": -9.46,   "longitude": -48.26}
    }
    
    # Faz a requisição para obter a lista de estados
    url = "https://servicodados.ibge.gov.br/api/v1/localidades/estados"
    resp = requests.get(url)
    resp.raise_for_status()  # Lança exceção se status >= 400
    
    lista_estados_api = resp.json()
    
    # Monta a estrutura final
    lista_estados_final = []
    for estado in lista_estados_api:
        # Ex.: estado = {"id": 11, "sigla": "RO", "nome": "Rondônia", ...}
        codigo_uf = estado["id"]
        sigla = estado["sigla"]        # ex.: "RO"
        nome = estado["nome"]          # ex.: "Rondônia"
        regiao_nome = estado["regiao"]["nome"]  # ex.: "Norte"

        # Tenta obter latitude/longitude do dicionário local
        lat = None
        lon = None
        if sigla in estados_latlon:
            lat = estados_latlon[sigla]["latitude"]
            lon = estados_latlon[sigla]["longitude"]
        
        lista_estados_final.append({
            "codigo_uf": codigo_uf,
            "uf": sigla,
            "nome": nome,
            "latitude": lat,
            "longitude": lon,
            "regiao": regiao_nome
        })
        
        
    estados_completos = lista_estados_final
    
    # Salva em arquivo JSON, se desejar
    with open("data/estados.json", "w", encoding="utf-8") as f:
        json.dump(estados_completos, f, ensure_ascii=False, indent=4)
    
    logger.info("'estados.json' gerado com sucesso!")
    
    return True

def proc12_inserir_lat_long_SINNAN():
    """
    1) Lê 'data/sinnan_tratado.parquet'.
    2) Insere colunas de latitude e longitude de município (LAT_MUNIC, LONG_MUNIC),
       estado (LAT_UF, LONG_UF) e região (LAT_REG, LONG_REG).
    3) Salva novamente em 'data/sinnan_tratado.parquet'.
    """
    
# =============================================================================
# Funções Auxiliares - Inserir Latitude/Longitude
# =============================================================================

    def return_estado_dict_lat_long():
        """
        Lê 'data/estados.json' e retorna um dicionário:
          {codigo_uf: {'latitude': ..., 'longitude': ..., 'regiao': ...}, ...}
        """
        try:
            json_path = 'data/estados.json'
            with open(json_path, 'r', encoding='utf-8-sig') as file:
                json_data = json.load(file)

            uf_dict = {
                item['codigo_uf']: {
                    'latitude': item['latitude'],
                    'longitude': item['longitude'],
                    'regiao': item['regiao'],
                }
                for item in json_data
            }
            return uf_dict

        except Exception as e:
            raise ValueError(f'[ERRO] Falha ao carregar o arquivo {json_path}: {str(e)}')

    def return_municipio_dict_lat_long():
        """
        Lê 'data/municipios_lat_long.csv' e retorna um dicionário:
          {GEOCODIGO_MUNICIPIO: {'LATITUDE': ..., 'LONGITUDE': ...}, ...}
        """
        try:
            df_path = 'data/municipios_lat_long.csv'
            df = pd.read_csv(df_path)

            # Transforma em dicionário indexado por GEOCODIGO_MUNICIPIO
            dicionario_geocodigos = df.set_index('GEOCODIGO_MUNICIPIO')[['LATITUDE', 'LONGITUDE']].to_dict(orient='index')
            return dicionario_geocodigos

        except Exception as e:
            raise ValueError(f'[ERRO] Falha ao carregar dicionário de geocódigos: {e}')


    def _codigo_numerico(series):
        """
        Converte uma coluna de códigos (possivelmente string/float/NaN) para
        inteiro anulável (Int64): valores não conversíveis viram <NA>, o que
        faz o .map() abaixo devolver NaN para eles — equivalente ao antigo
        try/except int(codigo) que retornava None em caso de falha.
        """
        return pd.to_numeric(series, errors='coerce').astype('Int64')

    def insere_lat_long_municipios(df):
        """
        Insere 'LAT_MUNIC' e 'LONG_MUNIC' a partir de ID_MUNIC_ANALISE — a mesma
        chave usada pelos indicadores do CNES (proc13) e do IBGE/IPEA (proc14).

        A versão anterior resolvia residência-com-fallback aqui dentro, enquanto
        proc13/proc14 usavam ID_MUNICIP. Coordenada e contexto municipal podiam
        então apontar para municípios diferentes na mesma linha, quebrando a
        correspondência que o bloqueio espacial exige.
        """
        if COLUNA_MUNICIPIO_RESOLVIDO not in df.columns:
            raise KeyError(
                f"Coluna '{COLUNA_MUNICIPIO_RESOLVIDO}' ausente: proc12 depende de proc10, "
                f"que a constrói. Rode o pipeline a partir da etapa 10."
            )

        municipios_dict = return_municipio_dict_lat_long()
        lat_map = {cod: valores['LATITUDE'] for cod, valores in municipios_dict.items()}
        lon_map = {cod: valores['LONGITUDE'] for cod, valores in municipios_dict.items()}

        cod = _codigo_numerico(df[COLUNA_MUNICIPIO_RESOLVIDO])

        df['LAT_MUNIC'] = cod.map(lat_map)
        df['LONG_MUNIC'] = cod.map(lon_map)

        n_sem_coord = int(df['LAT_MUNIC'].isna().sum())
        if n_sem_coord:
            logger.warning(
                f"{n_sem_coord} registros ficaram sem coordenada municipal "
                f"(município ausente em 'municipios_lat_long.csv')."
            )

        return df

    def insere_lat_long_estados(df):
        """
        Insere colunas 'LAT_UF' e 'LONG_UF' usando SG_UF_NOT (código do UF)
        como chave de busca (via .map, vetorizado) no dicionário 'uf_dict'.
        """
        uf_dict = return_estado_dict_lat_long()
        lat_map = {cod: valores['latitude'] for cod, valores in uf_dict.items()}
        lon_map = {cod: valores['longitude'] for cod, valores in uf_dict.items()}

        cod_uf = _codigo_numerico(df['SG_UF_NOT'])
        df['LAT_UF'] = cod_uf.map(lat_map)
        df['LONG_UF'] = cod_uf.map(lon_map)

        return df

    def insere_lat_long_regioes(df):
        """
        Insere colunas 'LAT_REG' e 'LONG_REG' baseadas no 'centroids_brasil'
        da respectiva REGIÃO do UF (via .map, vetorizado).
        """
        centroids_brasil = {
            'Norte':        {'latitude': -3.4168,  'longitude': -62.2159},
            'Nordeste':     {'latitude': -8.4703,  'longitude': -37.9980},
            'Centro-Oeste': {'latitude': -15.5989, 'longitude': -52.5625},
            'Sudeste':      {'latitude': -20.3175, 'longitude': -44.3968},
            'Sul':          {'latitude': -27.5712, 'longitude': -51.8339},
        }

        uf_dict = return_estado_dict_lat_long()
        regiao_map = {cod: valores['regiao'] for cod, valores in uf_dict.items()}
        regiao_lat_map = {regiao: valores['latitude'] for regiao, valores in centroids_brasil.items()}
        regiao_lon_map = {regiao: valores['longitude'] for regiao, valores in centroids_brasil.items()}

        cod_uf = _codigo_numerico(df['SG_UF_NOT'])
        regiao_series = cod_uf.map(regiao_map)

        df['LAT_REG'] = regiao_series.map(regiao_lat_map)
        df['LONG_REG'] = regiao_series.map(regiao_lon_map)

        return df

    logger.info("Iniciando processo de inserção de latitude/longitude no SINAN...")

    parquet_path = 'data/sinnan_tratado.parquet'
    if not os.path.exists(parquet_path):
        logger.error(f"Arquivo não encontrado: {parquet_path}")
        return False

    df = pd.read_parquet(parquet_path)
    logger.info(f"DataFrame carregado: {df.shape[0]} linhas, {df.shape[1]} colunas.")

    # Insere colunas do tipo LAT_UF, LONG_UF, LAT_MUNIC, LONG_MUNIC, LAT_REG, LONG_REG
    logger.info("Inserindo latitude/longitude dos estados...")
    df = insere_lat_long_estados(df)

    logger.info("Inserindo latitude/longitude dos municípios...")
    df = insere_lat_long_municipios(df)

    logger.info("Inserindo latitude/longitude das regiões...")
    df = insere_lat_long_regioes(df)

    # Salva o DataFrame final
    df.to_parquet(parquet_path, index=False)
    logger.info(f"Inserção de lat/long concluída e salva em: {parquet_path}")
    return True

def proc13_juntar_SINNAN_CNES():
    """
    Faz a junção de dados do SINAN (sinnan_tratado.parquet) com dados
    de profissionais e estabelecimentos do CNES (profissionais_CNES.parquet
    e estabelecimentos_CNES.parquet). 
    
    Fluxo Geral:
    ------------
    1) Carrega os arquivos Parquet:
       - 'sinnan_tratado.parquet' (informações do SINAN)
       - 'profissionais_CNES.parquet'
       - 'estabelecimentos_CNES.parquet'
       
    2) Cria uma coluna 'ID_MUNICIP_6' em cada DataFrame, 
       extraindo os 6 primeiros dígitos de ID_MUNICIP, 
       para permitir o merge com dados do CNES (que também usam 6 dígitos).
       
    3) Cria (ou renomeia) a coluna 'ANOMES_DIAG' com base em 'DT_DIAG' (ex.: '2023-01').
       Faz o mesmo nas tabelas do CNES (se houver colunas 'data').
       
    4) Junta (merge) os DataFrames profissionais e estabelecimentos ao DF principal,
       usando ['ID_MUNICIP_6', 'ANOMES_DIAG'] como chave (LEFT JOIN).
       
    5) Ordena os dados e usa 'groupby' para preencher NaNs 
       em 'QT_ESTABELECIMENTOS_TOTAL' e 'QT_PROFISSIONAIS_TOTAL' 
       de forma a propagar valores para linhas adjacentes (ffill e bfill).
       
    6) Remove colunas temporárias e salva o resultado em 'data/sinnan_tratado.parquet'.
    
    Retorno:
    --------
    bool
        Retorna True se o processo concluir sem falhas.
    """
   
    # Função auxiliar para preencher valores ausentes do CNES ao longo do tempo
    def preencher_nan(group):
        """
        Preenche ausências apenas com o último valor CONHECIDO ANTERIOR (ffill),
        dentro do município e em ordem cronológica.

        A versão anterior aplicava ffill().bfill(). O bfill propaga um valor
        futuro para trás: um município sem registro de CNES em 2015 mas com
        registro em 2016 recebia, em 2015, a capacidade instalada de 2016 —
        informação que não existia no momento da notificação. Isso é vazamento
        temporal e o briefing o proíbe (§8.4: nunca usar um valor futuro).

        Sem o bfill, meses anteriores ao primeiro registro de CNES do município
        permanecem ausentes, que é a descrição honesta do que se sabe.
        """
        return group.ffill()

    # 1) Carrega DataFrames principais (SINAN) e CNES
    sinnan_path = 'data/sinnan_tratado.parquet'
    prof_cnes_path = 'data/profissionais_CNES.parquet'
    estab_cnes_path = 'data/estabelecimentos_CNES.parquet'
    
    if not (os.path.exists(sinnan_path) and os.path.exists(prof_cnes_path) and os.path.exists(estab_cnes_path)):
        logger.error("Um dos arquivos Parquet não foi encontrado. Verifique os caminhos.")
        return False
    
    df = pd.read_parquet(sinnan_path)
    profissionais_cnes = pd.read_parquet(prof_cnes_path)
    estabelecimentos_cnes = pd.read_parquet(estab_cnes_path)
    
    logger.info("Iniciando junção dos DataFrames SINAN e CNES...")
    
    # 2) Padroniza ID_MUNICIP para 6 dígitos em CNES e SINAN
    profissionais_cnes.rename(columns={'ID_MUNICIP': 'ID_MUNICIP_6'}, inplace=True)
    estabelecimentos_cnes.rename(columns={'ID_MUNICIP': 'ID_MUNICIP_6'}, inplace=True)
    
    # Em df do SINAN, cria nova coluna com 6 dígitos a partir do MUNICÍPIO DE
    # ANÁLISE (proc10) — a mesma chave de lat/long (proc12) e IBGE/IPEA (proc14).
    # Antes usava ID_MUNICIP, divergindo da residência usada nas coordenadas.
    if COLUNA_MUNICIPIO_RESOLVIDO not in df.columns:
        logger.error(
            f"Coluna '{COLUNA_MUNICIPIO_RESOLVIDO}' ausente: proc13 depende de proc10. "
            f"Rode o pipeline a partir da etapa 10."
        )
        return False

    df['ID_MUNICIP_6'] = df[COLUNA_MUNICIPIO_RESOLVIDO].astype(str).str[:6]

    # 3) Cria ou renomeia 'ANOMES_DIAG' para permitir o merge
    df['ANOMES_DIAG'] = df['DT_DIAG'].astype(str).str[:7]  # Ex.: "2023-02"
    
    if 'data' in profissionais_cnes.columns:
        profissionais_cnes.rename(columns={'data': 'ANOMES_DIAG'}, inplace=True)
    if 'data' in estabelecimentos_cnes.columns:
        estabelecimentos_cnes.rename(columns={'data': 'ANOMES_DIAG'}, inplace=True)
    
    # 4) Faz o merge (LEFT JOIN) para trazer dados de profissionais e estab. do CNES
    df = df.merge(profissionais_cnes, on=['ID_MUNICIP_6', 'ANOMES_DIAG'], how='left')
    df = df.merge(estabelecimentos_cnes, on=['ID_MUNICIP_6', 'ANOMES_DIAG'], how='left')
    
    logger.info("DataFrames unificados. Aplicando ordenação e preenchimento de NaNs...")
    
    # 5) Ordena cronologicamente e propaga o último valor conhecido, por município
    #    de análise. A ordenação por ANOMES_DIAG (e não por ANO_DIAG) garante que
    #    o ffill respeite a ordem dos meses dentro de cada ano.
    df = df.sort_values(by=[COLUNA_MUNICIPIO_RESOLVIDO, 'ANOMES_DIAG'])

    for coluna_cnes in ['QT_ESTABELECIMENTOS_TOTAL', 'QT_PROFISSIONAIS_TOTAL']:
        antes = int(df[coluna_cnes].isna().sum())
        df[coluna_cnes] = (
            df.groupby(COLUNA_MUNICIPIO_RESOLVIDO)[coluna_cnes]
              .transform(preencher_nan)
        )
        depois = int(df[coluna_cnes].isna().sum())
        logger.info(
            f"{coluna_cnes}: {antes} ausentes após o merge, {depois} após ffill "
            f"(restantes são meses anteriores ao primeiro registro do município)."
        )


    # 6) Remove colunas temporárias e salva
    df.drop(columns=['ID_MUNICIP_6', 'ANOMES_DIAG'], inplace=True)
    
    output_path = 'data/sinnan_tratado.parquet'
    df.to_parquet(output_path, index=False)
    
    logger.info(f"DataFrames unidos e salvos em '{output_path}'")
    return True

def proc14_juntar_SINNAN_IBGE_IPEA():
    """
    1) Carrega 'data/sinnan_tratado.parquet' (dados SINAN), 'data/indicadores_IBGE.csv'
       (indicadores do IBGE) e 'data/indicadores_IPEA.csv' (Atlas do Desenvolvimento
       Humano, via Ipeadata — ver proc03_download_IndicadoresIPEA).
    2) Converte colunas ID_MUNICIP e ANO_DIAG para int em todos os DataFrames,
       garantindo compatibilidade no merge.
    3) Faz merge (LEFT JOIN) de cada fonte por ['ID_MUNICIP', 'ANO_DIAG'].
    4) Converte colunas específicas do IBGE para numérico (float), forçando coerção de tipos.
    5) Salva o resultado em 'data/sinnan_tratado.parquet'.
    """

    sinnan_path = 'data/sinnan_tratado.parquet'
    ibge_path   = 'data/indicadores_IBGE.csv'
    ipea_path   = 'data/indicadores_IPEA.csv'
    output_path = 'data/sinnan_tratado.parquet'

    # 1) Verifica se os arquivos existem
    if not os.path.exists(sinnan_path):
        logger.error(f"Arquivo não encontrado: {sinnan_path}")
        return False
    if not os.path.exists(ibge_path):
        logger.error(f"Arquivo não encontrado: {ibge_path}")
        return False
    if not os.path.exists(ipea_path):
        logger.error(f"Arquivo não encontrado: {ipea_path}")
        return False

    def _para_int_seguro(series, nome_coluna, origem):
        """
        Converte 'series' para inteiro anulável (Int64) via pd.to_numeric(errors='coerce').

        Diferente de `.astype(int, errors='ignore')` (usado na versão anterior deste
        código), que ao falhar mantinha a coluna no dtype original e podia deixar os
        dois lados do merge com tipos incompatíveis (ex.: int de um lado, string do
        outro) — causando perda silenciosa de linhas no LEFT JOIN — esta função sempre
        devolve Int64 dos dois lados, e avisa no log quantos valores não puderam ser
        convertidos (e portanto não vão casar no merge).
        """
        convertido = pd.to_numeric(series, errors='coerce').astype('Int64')
        n_falhas = int(convertido.isna().sum() - series.isna().sum())
        if n_falhas > 0:
            logger.warning(
                f"{origem}: {n_falhas} valores de '{nome_coluna}' não puderam ser "
                f"convertidos para inteiro (viraram NaN e não vão casar no merge)."
            )
        return convertido

    try:
        logger.info("Lendo arquivo SINAN (Parquet)...")
        sinnan = pd.read_parquet(sinnan_path)

        logger.info("Lendo arquivo IBGE (CSV) com low_memory=False...")
        # low_memory=False => Pandas faz apenas uma varredura do arquivo,
        # minimizando problemas de tipos mistos.
        ibge = pd.read_csv(ibge_path, low_memory=False)

        logger.info("Lendo arquivo IPEA (CSV)...")
        ipea = pd.read_csv(ipea_path, low_memory=False)
    except Exception as e:
        logger.error(f"Falha ao ler arquivos: {e}")
        return False

    # 2) Converte as chaves de merge para Int64 nos três lados.
    #
    #    O SINAN entra pelo MUNICÍPIO DE ANÁLISE (proc10) — a mesma chave de
    #    lat/long (proc12) e CNES (proc13). Antes o merge era por ID_MUNICIP
    #    (notificação), enquanto as coordenadas vinham da residência: as duas
    #    geografias se misturavam na mesma linha.
    if COLUNA_MUNICIPIO_RESOLVIDO not in sinnan.columns:
        logger.error(
            f"Coluna '{COLUNA_MUNICIPIO_RESOLVIDO}' ausente: proc14 depende de proc10. "
            f"Rode o pipeline a partir da etapa 10."
        )
        return False

    sinnan[COLUNA_MUNICIPIO_RESOLVIDO] = _para_int_seguro(
        sinnan[COLUNA_MUNICIPIO_RESOLVIDO], COLUNA_MUNICIPIO_RESOLVIDO, 'SINAN'
    )
    if 'ANO_DIAG' in sinnan.columns:
        sinnan['ANO_DIAG'] = _para_int_seguro(sinnan['ANO_DIAG'], 'ANO_DIAG', 'SINAN')

    # IBGE e IPEA trazem o município em 'ID_MUNICIP'. Renomeamos para a chave de
    # análise para que o merge case com o lado do SINAN sem colidir com a coluna
    # ID_MUNICIP original (município de notificação), que é preservada.
    for nome_fonte, fonte in (('IBGE', ibge), ('IPEA', ipea)):
        if 'ID_MUNICIP' in fonte.columns:
            fonte['ID_MUNICIP'] = _para_int_seguro(fonte['ID_MUNICIP'], 'ID_MUNICIP', nome_fonte)
            fonte.rename(columns={'ID_MUNICIP': COLUNA_MUNICIPIO_RESOLVIDO}, inplace=True)
        if 'ANO_DIAG' in fonte.columns:
            fonte['ANO_DIAG'] = _para_int_seguro(fonte['ANO_DIAG'], 'ANO_DIAG', nome_fonte)

    logger.info("Iniciando a junção dos DataFrames SINAN, IBGE e IPEA...")

    # 3) Merge (LEFT JOIN) por [município de análise, ano].
    #    validate='m:1' faz o merge falhar caso IBGE/IPEA tenham mais de uma
    #    linha por (município, ano) — duplicação que inflaria silenciosamente a
    #    coorte, exatamente o tipo de falha que passa despercebida num LEFT JOIN.
    n_antes = len(sinnan)
    df_merged = pd.merge(
        sinnan, ibge, on=[COLUNA_MUNICIPIO_RESOLVIDO, 'ANO_DIAG'], how='left', validate='m:1'
    )
    df_merged = pd.merge(
        df_merged, ipea, on=[COLUNA_MUNICIPIO_RESOLVIDO, 'ANO_DIAG'], how='left', validate='m:1'
    )
    if len(df_merged) != n_antes:
        logger.error(
            f"O merge alterou o número de linhas ({n_antes} -> {len(df_merged)}). "
            f"Isso indica duplicação nas fontes contextuais."
        )
        return False

    # 4) Converte para numérico as colunas contextuais.
    #
    #    A versão anterior tentava converter três colunas pelos nomes em
    #    português ("PIB per capita" etc.). Mas proc09 já as renomeou para
    #    IBGE_* antes deste ponto, então nenhuma das três existia e o laço era
    #    código morto — motivo pelo qual IBGE_PIB_PER_CAPITA e outras chegavam
    #    ao parquet final como texto.
    #
    #    Agora varremos todas as colunas IBGE_/IPEA_ e convertemos aquelas cujos
    #    valores presentes são todos numéricos. Colunas genuinamente
    #    categóricas (IBGE_BIOMA, IBGE_SISTEMA_COSTEIRO_MARINHO) não passam no
    #    teste e permanecem texto, como devem.
    SENTINELAS_IBGE = {"", "-", "..", "...", "nan", "NaN", "None", "X"}

    convertidas = []
    for col in df_merged.columns:
        if not col.startswith(('IBGE_', 'IPEA_')):
            continue
        if pd.api.types.is_numeric_dtype(df_merged[col]):
            continue

        limpa = df_merged[col].astype('string').str.strip()
        limpa = limpa.mask(limpa.isin(SENTINELAS_IBGE))
        presentes = limpa.dropna()
        if presentes.empty:
            continue
        if pd.to_numeric(presentes, errors='coerce').notna().all():
            df_merged[col] = pd.to_numeric(limpa, errors='coerce')
            convertidas.append(col)

    logger.info(
        f"Colunas contextuais convertidas para numérico ({len(convertidas)}): {convertidas}"
    )

    # 5) Salva em Parquet
    try:
        df_merged.to_parquet(output_path, index=False)
    except Exception as e:
        logger.error(f"Falha ao salvar Parquet: {e}")
        return False
    
    logger.info(f"DataFrames SINAN + IBGE + IPEA unificados e salvos em '{output_path}'.")
    return True

def analisar_sinan_parquet(
    path_or_df, 
    top_missing=5, 
    check_truncamento=True, 
    max_len_trunc=50
):
    """
    Faz uma análise exploratória de um arquivo Parquet do SINAN ou de um DataFrame,
    verificando a completude (NaNs, nulos, zeros) e sugerindo possíveis castings de tipos.

    Parâmetros:
    -----------
    path_or_df : str ou pd.DataFrame
        Caminho para o arquivo .parquet OU um DataFrame já carregado.
    top_missing : int
        Quantidade de colunas a exibir no relatório de "maior número de dados faltantes".
    check_truncamento : bool
        Se True, faz uma verificação simples de truncamento em colunas de texto,
        contando quantos valores excedem 'max_len_trunc'.
    max_len_trunc : int
        Tamanho máximo para verificar se há indícios de truncamento em colunas de texto.

    Retorno:
    --------
    dict
        Retorna um dicionário com as seguintes chaves:
          - 'overview': DataFrame com colunas:
               ["dtype_atual", "missing_count", "missing_perc", "zero_count", "zero_perc",
                "unique_count", "suggested_dtype", "possible_truncated_count", "possible_truncated_perc"]
          - 'top_missing_cols': DataFrame com as colunas que têm mais valores faltantes.
          - 'df_original': Retorna o próprio DataFrame usado na análise (após leitura, se path).
    
    Exemplo de uso:
    ---------------
    >>> result = analisar_sinan_parquet("data/sinnan.parquet", top_missing=3)
    >>> result["overview"].head()
    >>> result["top_missing_cols"]
    """
    
    # -------------------------------------------------------------------------
    # 1) Carregar o DataFrame, caso tenha recebido apenas o caminho
    # -------------------------------------------------------------------------
    if isinstance(path_or_df, str):
        if not os.path.isfile(path_or_df):
            raise FileNotFoundError(f"Arquivo não encontrado: {path_or_df}")
        df = pd.read_parquet(path_or_df)
    elif isinstance(path_or_df, pd.DataFrame):
        df = path_or_df
    else:
        raise TypeError("Parâmetro 'path_or_df' deve ser um caminho (str) ou um DataFrame.")
    
    # -------------------------------------------------------------------------
    # 2) Estrutura para armazenar as informações de cada coluna
    # -------------------------------------------------------------------------
    analysis_results = {
        "dtype_atual": [],
        "missing_count": [],
        "missing_perc": [],
        "zero_count": [],
        "zero_perc": [],
        "unique_count": [],
        "suggested_dtype": [],
        "possible_truncated_count": [],
        "possible_truncated_perc": [],
    }
    col_names = df.columns.tolist()
    
    n_rows = len(df)
    
    # -------------------------------------------------------------------------
    # 3) Função auxiliar para sugerir tipo
    # -------------------------------------------------------------------------
    def sugerir_tipo(col_data):
        """
        Tenta inferir se os dados podem ser numéricos, datas ou 
        booleanos, sugerindo conversão a partir dos valores.
        """
        # Se a coluna já for numérica, retorna seu dtype atual
        if pd.api.types.is_numeric_dtype(col_data):
            return str(col_data.dtype)
        
        # Se a coluna já for datetime
        if pd.api.types.is_datetime64_any_dtype(col_data):
            return "datetime64[ns]"
        
        # Tenta converter para numérico "forçando", se a taxa de erro for muito baixa, sugere float/int
        # Convertendo com 'errors=coerce' para ver quantos NaN geram.
        numeric_test = pd.to_numeric(col_data, errors='coerce')
        notnull_count = numeric_test.notnull().sum()
        # Se a quantidade de não-nulos após conversão for maior que 90%, sugere float
        if notnull_count >= 0.9 * len(col_data):
            # Podemos ver se todos são inteiros (float sem casas decimais) -> sugere int
            # Ex.: se 90% dos valores forem numéricos e desses, se +99% forem "inteiros".
            frac_inteiros = (numeric_test.dropna() % 1 == 0).mean()
            if frac_inteiros > 0.99:
                return "int64"
            else:
                return "float64"
        
        # Podemos tentar converter a data: se for parseável em data para >90% -> datetime
        date_test = pd.to_datetime(col_data, errors='coerce')
        date_notnull_count = date_test.notnull().sum()
        if date_notnull_count >= 0.9 * len(col_data):
            return "datetime64[ns]"
        
        # Caso não se encaixe nos cenários acima, considera string/object
        return "object"
    
    # -------------------------------------------------------------------------
    # 4) Percorrer as colunas e coletar estatísticas
    # -------------------------------------------------------------------------
    for col in col_names:
        col_data = df[col]
        
        # dtype atual
        dtype_atual = str(col_data.dtype)
        
        # Missing
        missing_count = col_data.isna().sum()
        missing_perc = (missing_count / n_rows) * 100 if n_rows > 0 else 0
        
        # Zeros (aplica para colunas numéricas ou strings que possam representar zero)
        # Para manter simples, se a coluna for numérica => zeros. Senão => 0
        zero_count = 0
        zero_perc = 0.0
        
        if pd.api.types.is_numeric_dtype(col_data):
            zero_count = (col_data == 0).sum()
            zero_perc = (zero_count / n_rows) * 100 if n_rows > 0 else 0
        
        # Unique
        unique_count = col_data.nunique(dropna=True)
        
        # Sugerir dtype
        suggested_dtype = sugerir_tipo(col_data)
        
        # Verificar truncamento (opcional)
        # Exemplo: se for string e tiver valores maiores que 'max_len_trunc'
        truncated_count = 0
        truncated_perc = 0.0
        if check_truncamento and pd.api.types.is_object_dtype(col_data):
            len_series = col_data.dropna().astype(str).apply(len)
            truncated_count = (len_series > max_len_trunc).sum()
            truncated_perc = (truncated_count / n_rows) * 100 if n_rows > 0 else 0
        
        analysis_results["dtype_atual"].append(dtype_atual)
        analysis_results["missing_count"].append(missing_count)
        analysis_results["missing_perc"].append(missing_perc)
        analysis_results["zero_count"].append(zero_count)
        analysis_results["zero_perc"].append(zero_perc)
        analysis_results["unique_count"].append(unique_count)
        analysis_results["suggested_dtype"].append(suggested_dtype)
        analysis_results["possible_truncated_count"].append(truncated_count)
        analysis_results["possible_truncated_perc"].append(truncated_perc)
    
    # -------------------------------------------------------------------------
    # 5) Montar DataFrame final de overview e ordenar colunas
    # -------------------------------------------------------------------------
    overview_cols = [
        "dtype_atual", 
        "missing_count", 
        "missing_perc", 
        "zero_count", 
        "zero_perc", 
        "unique_count", 
        "suggested_dtype", 
        "possible_truncated_count", 
        "possible_truncated_perc"
    ]
    df_overview = pd.DataFrame(analysis_results, index=col_names)[overview_cols]
    
    # Ordena para melhorar a visualização (opcional)
    df_overview.sort_values(by="missing_count", ascending=False, inplace=True)
    
    # -------------------------------------------------------------------------
    # 6) Colunas com mais dados faltantes
    # -------------------------------------------------------------------------
    top_missing_cols = df_overview.head(top_missing)[["missing_count", "missing_perc"]]
    
    # -------------------------------------------------------------------------
    # 7) Retornar as informações
    # -------------------------------------------------------------------------
    result = {
        "overview": df_overview,
        "top_missing_cols": top_missing_cols,
        "df_original": df
    }
    
    return result

##############################
# Execução Principal (main)
##############################

def main(start_step=1):
    """
    Função principal que chama cada etapa do pipeline em ordem (exceto a análise final),
    tratando possíveis exceções e reportando falhas de forma amigável.

    Parâmetros:
    -----------
    start_step : int
        Número da etapa (1 a 14) a partir da qual a execução deve começar, pulando
        as anteriores. Útil para retomar o pipeline depois de uma falha sem precisar
        refazer etapas cujos artefatos já estão salvos em 'data/' (ex.: se a etapa 9
        falhou, rode novamente com start_step=9 em vez de baixar tudo de novo).
        Também disponível via linha de comando: `python bot_data_SINNAN_IBGE_CNES.py --start-step 9`.
    """
    log_path = configurar_logging()
    logger.info(f"Log desta execução sendo salvo em: {log_path}")

    etapas = [
        (1, "Baixando arquivos DBC do SINAN...", lambda: proc01_download_SINNANDBC(ano_inicial=2012, replace=True)),
        (2, "Baixando indicadores IBGE...", proc02_download_IndicadoresIBGE),
        (3, "Baixando indicadores do Atlas do Desenvolvimento Humano (Ipeadata)...", proc03_download_IndicadoresIPEA),
        (4, "Baixando estabelecimentos CNES...", proc04_download_estabelecimentosCNES),
        (5, "Baixando profissionais CNES...", proc05_download_profissionaisCNES),
        (6, "Convertendo arquivos DBC -> DBF -> Parquet (SINAN)...", proc06_converter_SINNAN),
        (7, "Unindo tabelas TUBEBR*.parquet em 'data/sinnan.parquet'...", proc07_Unir_tabelas),
        (8, "Tratamento inicial do SINAN (sinnan.parquet -> sinnan_preliminar01.parquet)...", proc08_tratamento_SINNAN_01),
        (9, "Converter IBGE JSON -> CSV (indicadores_IBGE.csv)...", proc09_converter_IBGE),
        (10, "Ajuste de códigos incompletos (sinnan_preliminar01.parquet -> sinnan_tratado.parquet)...", proc10_tratamento_SINNAN_02),
        (11, "Geração do 'estados.json' com lat/long (opcional)...", proc11_estados_ibge_com_latlon),
        (12, "Inserindo latitude/longitude em sinnan_tratado.parquet...", proc12_inserir_lat_long_SINNAN),
        (13, "Juntando SINAN + CNES em sinnan_tratado.parquet...", proc13_juntar_SINNAN_CNES),
        (14, "Juntando SINAN + IBGE + IPEA em sinnan_tratado.parquet final...", proc14_juntar_SINNAN_IBGE_IPEA),
    ]

    try:
        for numero, descricao, func in etapas:
            if numero < start_step:
                logger.info(f"[{numero}] (pulada, --start-step={start_step}) {descricao}")
                continue

            logger.info(f"[{numero}] {descricao}")
            if not func():
                logger.error(f"Falha ao executar a etapa {numero} ({descricao}).")
                logger.error(f"Para retomar a partir desta etapa depois de corrigir o problema: --start-step {numero}")
                return

        logger.info("Pipeline concluído com sucesso!")

    except Exception:
        # Captura qualquer exceção inesperada e mostra o traceback completo no log
        logger.error("Ocorreu um erro inesperado durante a execução do pipeline.")
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pipeline de ingestão SINAN/IBGE/CNES.")
    parser.add_argument(
        "--start-step",
        type=int,
        default=1,
        choices=range(1, 15),
        metavar="[1-14]",
        help="Etapa a partir da qual retomar o pipeline (pula as etapas anteriores).",
    )
    args = parser.parse_args()
    main(start_step=args.start_step)