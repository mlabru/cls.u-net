# Centro Logístico de Simulação Meteorológica
# Copyright (c) 1995-2026, Instituto de Controle do Espaço Aéreo
#
# unt_cfg.sh
# TODA a configuração do nowcast em um só lugar. Lido ("source") pelo
# unt_common.sh -- nunca executado diretamente. Não liga "set -euo
# pipefail": quem faz o source já escolheu as próprias opções de shell.
#
# variáveis marcadas [env] podem ser trocadas por variável de ambiente
# (export antes de rodar, ou no unt_env.sh lido pelo unt_cron.sh). As
# demais são fixas.
#
# ATENÇÃO: os nomes UNT_PATH, UNT_CATALOG_DB, UNT_CAPPI_DIR,
# UNT_AGE_MAX_SECS, UNT_TIMEOUT_SECS e UNT_VERIFY espelham os OFL_* do
# 25.oflow, que o ambiente de teste (21.mvol/04.dvl/tst_env/tst_setup.sh)
# exporta; o tst_env ainda não tem o passo do 25.u-net. Não renomear.
#
# 2026.nov  mlabru  CLS_LIB obrigatória (passo 2): sai o symlink cls_common.sh
#                   da raiz do repo; sem cls.env/CLS_LIB, erro explícito
#
# 2026.nov  mlabru  - banco/CAPPIs compartilhados: sem sobreposição própria,
#                     vêm de CLS_DBASE_FILE/CLS_CAPPI_DIR (cls.env, sob
#                     CLS_ROOT) antes do padrão antigo
#
# 2026.nov  mlabru  - cls.env (CLS_ENV_FILE, padrão ~/.config/cls/cls.env)
#                     carregado antes dos padrões; cls_common.sh vem de
#                     CLS_LIB, com o symlink ao lado do script como reserva
#                   - prefixo met -> cls (cls_log, cls_check_test_env,
#                     CLS_ENV, CLS_TST_ROOT)
#
# 2026.out  mlabru  seção de treino (unt_treino.sh, host de GPU): UNT_CONFIG,
#                   UNT_JANELAS_DIR, UNT_AVALIACAO_DIR, UNT_TREINO_*,
#                   UNT_DESTINO_MODELOS (rsync), logs e lock próprios;
#                   UNT_MONITOR_TIMEOUT_SECS (untMonitor no unt_run.sh)
#
# 2026.out  mlabru  portado de 25.oflow (ofl_cfg.sh): backends pytorch,
#                   rainnet e smaat; UNT_MODELOS_DIR e UNT_DEVICE
#                   (exportados para o untMain.py); UNT_TIMEOUT_SECS padrão
#                   300 s (inferência dos três backends em CPU).
#
# 2026.out  mlabru  UNT_GIF_MOSAICO/UNT_GIFS_POR_PAR (GIF montado por par)
#
# 2026.out  mlabru  - normalização com 21.mvol/27.nowcast: cabeçalho comum,
#                     sem shebang (só "sourceado")
#
#                   - reorganizado por seções; GI_AGE_MAX_SECS virou
#                     UNT_AGE_MAX_SECS (um só nome para a mesma
#                     configuração); GI_NUM_WORKERS -> UNT_MAX_PARES_EM_PARALELO;
#                     UNT_ERR_GIF -> UNT_GIF_SEM_PREVISAO; HOST -> UNT_HOST
#
#                   - removidos os restos do daemon (unt_daemon.sh):
#                     UNT_PROC_QUEUE, GS_SHUTDOWN_FILE, GI_WATCH_INTERVAL_SECS
#
#                   - trava via cls_check_test_env (met_common.sh, symlink
#                     para 04.dvl/bash), comum aos projetos cls
#
#                   - UNT_PATH/idade máxima sobreponíveis por env var,
#                     UNT_LIB_DIR (código) separado de UNT_PATH (dados),
#                     UNT_TIMEOUT_SECS, UNT_VERIFY e trava do ambiente de
#                     teste (CLS_ENV=test, 21.mvol/tst_env). UNT_MAPA_CACHE
#                     (cache das feições do mapa) dentro da árvore de dados.
#
# 2026.ago  mlabru  initial version (Bash/Linux)

# depends on: bash, cls_common.sh
# shellcheck shell=bash

# < código >---------------------------------------------------------------------------

# diretório do código (este repositório): os apps/*.py rodam sempre daqui
UNT_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ambiente da instalação (cls.env: CLS_LIB e sobreposições), carregado
# antes dos padrões abaixo; "set -a" exporta para os programas Python.
# Precedência: ambiente do processo > cls.env > padrões deste arquivo (ver
# cls.common/etc/cls.env.example)
CLS_ENV_FILE="${CLS_ENV_FILE:-${HOME:-}/.config/cls/cls.env}"
if [ -r "${CLS_ENV_FILE}" ]; then set -a; source "${CLS_ENV_FILE}"; set +a; fi

# nome desta máquina (só para o log)
UNT_HOST=$(hostname)

# < árvore de dados >------------------------------------------------------------------
#
#   ${UNT_PATH}/
#     logs/               unt_app.log, unt_error.log
#     data/assets/        no_cast.gif (GIF de "sem previsão")
#     data/output/        <R>_<A>_now_<backend>.gif, _prv_, <R>_<A>_verify.jsonl
#     data/state/         locks e marcadores last_target_<R>_<A>
#     data/temp/          saída do untMain.py, apagada depois do GIF
#     data/cache/mapa/    cache das feições do mapa (unt_mapa.py)
#     data/modelos/       checkpoints <backend>/atual.pt (UNT_MODELOS_DIR)
#     data/janelas/       manifestos do treino (unt_treino.sh)
#     data/avaliacao/     comparacao.json/.csv de cada ciclo de treino

# [env] raiz dos dados
UNT_PATH="${UNT_PATH:-/home/${USER:-$(id -un)}/Public/unet}"

# logs
UNT_LOGS="${UNT_PATH}/logs"

UNT_LOG_APP="${UNT_LOGS}/unt_main.log"
UNT_LOG_ERR="${UNT_LOGS}/unt_main.err"

# limite de tamanho de cada log (10 MB) antes de rotacionar. O nome
# GI_LOG_MAX_SIZE é lido pelo check_log_size do cls_common.sh: não renomear.
GI_LOG_MAX_SIZE=$((10 * 1024 * 1024))

# subdiretórios de data/
UNT_BASE="${UNT_PATH}/data"
UNT_ASSETS="${UNT_BASE}/assets"
UNT_OUT="${UNT_BASE}/output"
UNT_STATE="${UNT_BASE}/state"
UNT_TEMP="${UNT_BASE}/temp"

# [env] cache das feições do mapa. Exportado porque quem o lê são os
# processos Python (apps/work/unt_mapa.py).
export UNT_MAPA_CACHE="${UNT_MAPA_CACHE:-${UNT_BASE}/cache/mapa}"

# < entrada: catálogo e CAPPIs do 27.nowcast >-----------------------------------------

# [env] catálogo SQLite (só lido, nunca gravado)
# (ordem: UNT_* > CLS_DBASE_FILE/CLS_CAPPI_DIR do cls.env > padrão)
UNT_CATALOG_DB="${UNT_CATALOG_DB:-${CLS_DBASE_FILE:-${UNT_BASE}/dbase/catalog.db}}"

# [env] raiz dos CAPPIs (.npy)
UNT_CAPPI_DIR="${UNT_CAPPI_DIR:-${CLS_CAPPI_DIR:-${UNT_BASE}/cappi}}"

# [env] espera máxima (ms) por um lock do catálogo nas consultas sqlite3:
# o 27.nowcast grava no catálogo enquanto lemos, e sem isso o sqlite3 falha na
# hora com "database is locked"
UNT_DB_TIMEOUT_MS="${UNT_DB_TIMEOUT_MS:-10000}"

# [env] checkpoints treinados (untTrain.py): <dir>/<backend>/atual.pt.
# Exportado porque quem o lê são os processos Python (untMain.py).
export UNT_MODELOS_DIR="${UNT_MODELOS_DIR:-${UNT_BASE}/modelos}"

# < execução >-------------------------------------------------------------------------

# [env] virtualenv com as bibliotecas dos backends (torch, cartopy, ...) --
# o mesmo do 27.nowcast. Sem ele, usa o python3 do sistema.
UNT_VENV="${UNT_VENV:-${HOME:-/home/$(id -un)}/.venv}"

# [env] idade máxima (s) do "target" de uma sequência: mais velha que isso
# é ignorada (2 horas). 0 = sem limite (replay de dados antigos em teste).
UNT_AGE_MAX_SECS="${UNT_AGE_MAX_SECS:-$((2 * 60 * 60))}"

# [env] device do PyTorch na inferência: auto (cuda se houver, senão cpu),
# cpu, cuda, cuda:N. Exportado para o untMain.py.
export UNT_DEVICE="${UNT_DEVICE:-auto}"

# [env] tempo máximo (s) de cada chamada a untMain.py e a untMakGIF.py (os
# três backends em CPU, numa grade 600x600, levam dezenas de segundos)
UNT_TIMEOUT_SECS="${UNT_TIMEOUT_SECS:-300}"

# [env] true = compara o nowcast com o target observado (untMain.py
# --verify) e acumula o resultado em data/output/<R>_<A>_verify.jsonl
UNT_VERIFY="${UNT_VERIFY:-false}"

# quantos pares radar/alcance o unt_run.sh processa ao mesmo tempo
UNT_MAX_PARES_EM_PARALELO=3

# lock de instância: impede duas execuções do unt_run.sh ao mesmo tempo
# (o cron disparou de novo antes da anterior terminar)
UNT_INSTANCE_LOCK="${UNT_STATE}/.unt.lock"

# < backends e GIFs >------------------------------------------------------------------

# backends de nowcast -- os mesmos nomes de apps/untMain.py
# (MDCT_MODULOS_BACKEND) e dos subdiretórios que ele grava. A lista é fixa
# porque, se o untMain.py falhar antes de criar qualquer subdiretório, ainda
# é preciso saber quais GIFs marcar como "sem previsão". Ao adicionar um
# backend no untMain.py, adicioná-lo aqui também.
UNT_BACKENDS=(pytorch rainnet smaat)

# GIF montado (backends lado a lado, horário de referência/passo carimbado):
# <R>_<A>_now_mosaico.gif. O nome casa com untMakGIF.CS_NOME_MOSAICO.
UNT_GIF_MOSAICO="mosaico"

# todos os GIFs de um par (um por backend + o montado): rotação now -> prv e
# aviso de "sem previsão" valem para todos
UNT_GIFS_POR_PAR=("${UNT_BACKENDS[@]}" "${UNT_GIF_MOSAICO}")

# GIF mostrado no lugar de um backend que não gerou previsão (em UNT_ASSETS)
UNT_GIF_SEM_PREVISAO="no_cast.gif"

# < treino (unt_treino.sh, host de GPU) >----------------------------------------------
#
# o ciclo de treino roda FORA do cron de 5 min (ex.: semanal): manifesto ->
# untTrain de cada backend (candidatos) -> untEval -> untPromove -> rsync dos
# modelos para a máquina operacional.

# [env] config.ini ("como": janelas, treino, promoção, monitor)
UNT_CONFIG="${UNT_CONFIG:-${UNT_LIB_DIR}/config.ini}"

# [env] manifestos de janelas e avaliações de cada ciclo
UNT_JANELAS_DIR="${UNT_JANELAS_DIR:-${UNT_BASE}/janelas}"
UNT_AVALIACAO_DIR="${UNT_AVALIACAO_DIR:-${UNT_BASE}/avaliacao}"

# [env] backends treinados no ciclo (separados por espaço)
UNT_TREINO_BACKENDS="${UNT_TREINO_BACKENDS:-${UNT_BACKENDS[*]}}"

# [env] device do treino e da avaliação (auto = cuda se houver)
UNT_TREINO_DEVICE="${UNT_TREINO_DEVICE:-auto}"

# [env] argumentos extras do untTrain.py, separados por espaço (ex.: smoke test
# em CPU: "--epocas 1 --recorte 128 --largura-base 16 --workers 0")
UNT_TREINO_ARGS="${UNT_TREINO_ARGS:-}"

# [env] tempo máximo (s) de cada untTrain.py; 0 = sem limite
UNT_TREINO_TIMEOUT_SECS="${UNT_TREINO_TIMEOUT_SECS:-0}"

# [env] destino do rsync dos modelos ("host:/caminho/modelos" ou um diretório
# local); vazio = não distribui
UNT_DESTINO_MODELOS="${UNT_DESTINO_MODELOS:-}"

# logs e lock do treino (separados dos do nowcast)
UNT_LOG_TREINO="${UNT_LOGS}/unt_treino.log"
UNT_LOG_TREINO_ERR="${UNT_LOGS}/unt_treino.err"
UNT_TREINO_LOCK="${UNT_STATE}/.unt_treino.lock"

# estado de um ciclo interrompido (manifesto e candidatos já treinados): o
# próximo unt_treino.sh retoma o ciclo em vez de recomeçar
UNT_TREINO_CICLO="${UNT_STATE}/treino_ciclo"

# < monitor (untMonitor.py, no unt_run.sh com UNT_VERIFY=true) >-----------------------

# [env] tempo máximo (s) do untMonitor.py
UNT_MONITOR_TIMEOUT_SECS="${UNT_MONITOR_TIMEOUT_SECS:-120}"

# < trava do ambiente de teste >-------------------------------------------------------

# cls_check_test_env, cls_log e check_log_size (cls_common.sh: de CLS_LIB,
# do cls.env; comum aos projetos cls)
# CLS_LIB (cls.env) é obrigatória: sem ela não há cls_common.sh -- falha
# aqui, com a causa, em vez de num "command not found" mais adiante
if [ ! -r "${CLS_LIB:-}/cls_common.sh" ]; then
    echo "[error]: CLS_LIB='${CLS_LIB:-}' sem cls_common.sh; defina-a no cls.env (${CLS_ENV_FILE:-~/.config/cls/cls.env})" >&2
    exit 1
fi
source "${CLS_LIB}/cls_common.sh"

# com CLS_ENV=test, nenhum destes caminhos pode apontar para fora de
# CLS_TST_ROOT -- um env.sh incompleto gravaria nos dados de produção. Roda
# já no source, antes de qualquer mkdir/log/lock de quem importou.
# o destino remoto dos modelos ("host:/...") é barrado no unt_treino.sh; o
# local entra na trava como os demais caminhos
CS_DESTINO_LOCAL=""
if [[ "${UNT_DESTINO_MODELOS}" != *:* ]]; then
    CS_DESTINO_LOCAL="${UNT_DESTINO_MODELOS}"
fi

cls_check_test_env "${UNT_PATH}" "${UNT_CATALOG_DB}" "${UNT_CAPPI_DIR}" \
                   "${UNT_MAPA_CACHE}" "${UNT_MODELOS_DIR}" "${UNT_JANELAS_DIR}" \
                   "${UNT_AVALIACAO_DIR}" "${CS_DESTINO_LOCAL}" || exit 1

# < the end >--------------------------------------------------------------------------
