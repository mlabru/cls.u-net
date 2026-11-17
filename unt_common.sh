# Centro Logístico de Simulação Meteorológica
# Copyright (c) 1995-2026, Instituto de Controle do Espaço Aéreo
#
# unt_common.sh
# funções usadas pelo unt_run.sh. Faz o source do unt_cfg.sh (achando-o
# pelo próprio diretório, qualquer que seja o cwd de quem chamou).
#
# sempre "source"ado -- nunca executado diretamente. Não liga "set -euo
# pipefail": quem faz o source já escolheu as próprias opções de shell.
#
# fluxo de um par radar/alcance (processa_par):
#
#   lock do par -> target mais recente no catálogo -> é novo? é recente?
#     -> executa_nowcast (untMain.py: PNGs + .npy em data/temp/<R>_<A>/)
#     -> gera_gifs (rotaciona now->prv, untMakGIF.py, "sem previsão" onde faltar)
#     -> grava o marcador last_target_<R>_<A> (só se deu certo)
#
# CUIDADO com "set -e" (ligado pelo unt_run.sh): processa_par roda em
# background e o unt_run.sh espera por ele com "wait -n". Se processa_par
# terminar com código != 0, o "wait -n" falha e o set -e ABORTA a varredura
# inteira. Por isso todo caminho normal ("nada a fazer" incluído) termina
# com "return 0".
#
# 2026.out  mlabru  executa_monitor: untMonitor.py sobre os verify.jsonl ao fim
#                   da passada (UNT_VERIFY=true); alarme vira [warn] no log.
#
# 2026.out  mlabru  portado de 25.oflow (ofl_common.sh): executa_optical_flow
#                   -> executa_nowcast (passa --modelos-dir/--device),
#                   executa_untmakgif -> executa_untmakgif.
#
# 2026.out  mlabru  copia_gif_sem_previsao atômica (temporário + mv), para o
#                   dashboard nunca ler um GIF pela metade
# 2026.out  mlabru  rotação/aviso de "sem previsão" também no GIF montado
#                   (UNT_GIFS_POR_PAR)
#
# 2026.out  mlabru  normalização com 21.mvol/27.nowcast: nível [error] (era
#                   [erro]), cabeçalho comum, sem shebang (só "sourceado")
#
# 2026.out  mlabru  funções longas quebradas em funções de uma
#                   responsabilidade (cláusulas de guarda, um "return 0" por
#                   saída normal); nomes em português, minúsculos e
#                   snake_case; removido o source duplicado do met_common.sh
#
# 2026.out  mlabru  removida a fila do daemon (enqueue_proc_file/dequeue_file),
#                   sem uso desde a remoção do unt_daemon.sh
#
# 2026.out  mlabru  check_log_size e o corpo do log() vêm do met_common.sh
#                   (comum aos projetos cls); o horário do log() passa a ser
#                   UTC, como os logs Python
#
# 2026.out  mlabru  apps/*.py chamados de UNT_LIB_DIR (código) em vez de
#                   relativos a UNT_PATH (dados); timeout em
#                   UNT_TIMEOUT_SECS; idade máxima 0 desliga o filtro de
#                   idade; com UNT_VERIFY=true o nowcast é verificado contra
#                   o target e o resultado acumulado em
#                   ${UNT_OUT}/<radar>_<alcance>_verify.jsonl.
#
# 2026.ago  mlabru  removido o download de .mvol via SFTP -- a conversão
#                   .mvol -> CAPPI já é feita pelo pipeline do
#                   27.nowcast/04.dvl, que cataloga tudo num SQLite
#                   (catalog.db). O processamento passou a ser por par
#                   (radar, alcance), consultando o catálogo.
#
# 2026.ago  mlabru  extraído de unt_daemon.sh. Corrigida recursão infinita
#                   entre log()/check_log_size(): check_log_size() NÃO pode
#                   chamar log().

# depends on: bash, unt_cfg.sh, sqlite3, flock, timeout
# shellcheck shell=bash

# diretório deste arquivo (o UNT_LIB_DIR só existe depois do source abaixo)
CS_DIRETORIO_COMMON="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# configuração (também traz cls_log/check_log_size do cls_common.sh)
# shellcheck source=unt_cfg.sh
source "${CS_DIRETORIO_COMMON}/unt_cfg.sh"

# < utilitários >----------------------------------------------------------------------

# -------------------------------------------------------------------------------------
# grava uma mensagem em UNT_LOG_APP (e na tela), com data/hora UTC; rotaciona
# os logs grandes demais (cls_log, do cls_common.sh)
#
# uso: log "[info]: mensagem"   (níveis usados: [info], [warn], [error])
#
log()
{
    cls_log "${UNT_LOG_APP}" "${UNT_LOG_ERR}" "$@"

} # log

# -------------------------------------------------------------------------------------
# escreve (sem quebra de linha) uma duração em segundos de forma legível:
# 2700 -> "45min", 69900 -> "19h25min", 183900 -> "2d03h05min"
#
# uso: formata_duracao <segundos>
#
formata_duracao()
{
    local li_segundos="$1"

    local li_dias=$(( li_segundos / 86400 ))
    local li_horas=$(( li_segundos % 86400 / 3600 ))
    local li_minutos=$(( li_segundos % 3600 / 60 ))

    if (( li_dias > 0 )); then
        printf '%dd%02dh%02dmin' "${li_dias}" "${li_horas}" "${li_minutos}"
        return 0
    fi

    if (( li_horas > 0 )); then
        printf '%dh%02dmin' "${li_horas}" "${li_minutos}"
        return 0
    fi

    printf '%dmin' "${li_minutos}"

} # formata_duracao

# -------------------------------------------------------------------------------------
# entra no diretório; se não conseguir, loga e ENCERRA o processo (exit 1)
#
# uso: muda_para_diretorio <diretório>
#
muda_para_diretorio()
{
    local ls_diretorio="$1"

    if ! cd "${ls_diretorio}"; then
        log "[error]: falha ao acessar ${ls_diretorio}"
        exit 1
    fi

} # muda_para_diretorio

# -------------------------------------------------------------------------------------
# ativa o virtualenv UNT_VENV (bibliotecas dos backends), se existir; senão
# avisa e segue com o python3 do sistema (o untMain.py descarta, com aviso,
# cada backend cuja biblioteca faltar). Loga o python3 efetivo -- sob o cron
# ele pode ser diferente do da sessão interativa.
#
ativa_virtualenv()
{
    if [ -f "${UNT_VENV}/bin/activate" ]; then
        # shellcheck source=/dev/null
        source "${UNT_VENV}/bin/activate"
    else
        log "[warn]: virtualenv inexistente (${UNT_VENV}); usando python3 do sistema."
    fi

    log "[info]: python3=$(command -v python3 || echo 'ausente')"

} # ativa_virtualenv

# < catálogo (somente leitura) >--------------------------------------------------------
#
# todas as consultas usam "sqlite3 -readonly" (nunca cria um catálogo vazio
# num caminho errado) e mandam os erros para UNT_LOG_ERR -- nunca para
# /dev/null: um catálogo ausente não pode parecer "nenhum par".

# -------------------------------------------------------------------------------------
# escreve os pares com ao menos uma sequência, um por linha: "<radar> <alcance>"
# retorna != 0 se a consulta falhar
#
lista_pares_catalogados()
{
    sqlite3 -readonly -cmd ".timeout ${UNT_DB_TIMEOUT_MS}" -separator ' ' "${UNT_CATALOG_DB}" \
        "SELECT DISTINCT radar, alcance FROM sequences" 2>> "${UNT_LOG_ERR}"

} # lista_pares_catalogados

# -------------------------------------------------------------------------------------
# confere, antes da varredura, que o catálogo existe, não está vazio e tem as
# tabelas sequences e radar_sites. Loga o motivo e retorna 1 se não.
#
valida_catalogo()
{
    # inexistente ou vazio (ex.: 0 bytes, criado por engano por um sqlite3)
    if [ ! -s "${UNT_CATALOG_DB}" ]; then
        log "[error]: catálogo inexistente ou vazio: ${UNT_CATALOG_DB} (UNT_CATALOG_DB)."
        return 1
    fi

    local li_num_tabelas
    li_num_tabelas=$(sqlite3 -readonly -cmd ".timeout ${UNT_DB_TIMEOUT_MS}" "${UNT_CATALOG_DB}" \
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name IN ('sequences', 'radar_sites')" \
        2>> "${UNT_LOG_ERR}") || li_num_tabelas=""

    if [ "2" != "${li_num_tabelas}" ]; then
        log "[error]: catálogo ilegível ou sem as tabelas sequences/radar_sites: ${UNT_CATALOG_DB}."
        return 1
    fi

    return 0

} # valida_catalogo

# -------------------------------------------------------------------------------------
# escreve o maior "target" (epoch UTC) do par; vazio se não houver sequência
#
# uso: busca_target_mais_recente <radar> <alcance>
#
busca_target_mais_recente()
{
    local ls_radar="$1" li_alcance="$2"

    # aspas simples dobradas (escape do SQL). O radar vem do próprio
    # catálogo, mas escapar não custa nada.
    local ls_radar_escapado="${ls_radar//\'/\'\'}"

    sqlite3 -readonly -cmd ".timeout ${UNT_DB_TIMEOUT_MS}" "${UNT_CATALOG_DB}" \
        "SELECT MAX(target) FROM sequences WHERE radar='${ls_radar_escapado}' AND alcance=${li_alcance}" \
        2>> "${UNT_LOG_ERR}"

} # busca_target_mais_recente

# < marcador do último target processado >---------------------------------------------
#
# data/state/last_target_<R>_<A> guarda o último target processado com
# sucesso -- é o que evita refazer o mesmo nowcast a cada passada do cron.

# -------------------------------------------------------------------------------------
# uso: caminho_marcador <prefixo>
#
caminho_marcador()
{
    echo "${UNT_STATE}/last_target_$1"

} # caminho_marcador

# -------------------------------------------------------------------------------------
# retorna 0 (verdadeiro) se o target ainda não foi processado
#
# uso: tem_target_novo <prefixo> <target>
#
tem_target_novo()
{
    local ls_prefixo="$1" ls_target="$2"
    local ls_marcador
    ls_marcador=$(caminho_marcador "${ls_prefixo}")

    # nunca processou este par
    if [ ! -f "${ls_marcador}" ]; then
        return 0
    fi

    [ "${ls_target}" != "$(cat "${ls_marcador}")" ]

} # tem_target_novo

# -------------------------------------------------------------------------------------
# uso: grava_ultimo_target <prefixo> <target>
#
grava_ultimo_target()
{
    echo "$2" > "$(caminho_marcador "$1")"

} # grava_ultimo_target

# -------------------------------------------------------------------------------------
# retorna 0 (verdadeiro) se o target é mais velho que UNT_AGE_MAX_SECS
# (0 = sem limite: nunca é velho demais)
#
# uso: target_muito_antigo <target>
#
target_muito_antigo()
{
    local li_target="$1"

    if (( 0 == UNT_AGE_MAX_SECS )); then
        return 1
    fi

    local li_idade=$(( $(date +%s) - li_target ))

    (( li_idade > UNT_AGE_MAX_SECS ))

} # target_muito_antigo

# -------------------------------------------------------------------------------------
# explica no log por que uma sequência velha foi ignorada: o target em UTC
# e a idade comparada ao limite, ex.:
#   sequência muito antiga para CGU_250 (target=... = 2026-06-15 17:00:00 UTC,
#   idade 19h25min > limite 2h00min).
#
# uso: loga_target_muito_antigo <prefixo> <target>
#
loga_target_muito_antigo()
{
    local ls_prefixo="$1" li_target="$2"

    local li_idade=$(( $(date +%s) - li_target ))
    local ls_data_utc
    ls_data_utc=$(date -u -d "@${li_target}" '+%Y-%m-%d %H:%M:%S')

    log "[warn]: sequência muito antiga para ${ls_prefixo} (target=${li_target}" \
        "= ${ls_data_utc} UTC," \
        "idade $(formata_duracao "${li_idade}") > limite $(formata_duracao "${UNT_AGE_MAX_SECS}"))."

} # loga_target_muito_antigo

# < nowcast (untMain.py) >-------------------------------------------------------------

# -------------------------------------------------------------------------------------
# registra no log da aplicação por que o untMain.py falhou (o detalhe do
# Python, ex. "CAPPI ausente", fica em UNT_LOG_ERR)
#
# uso: loga_falha_do_untmain <prefixo> <código de saída>
#
loga_falha_do_untmain()
{
    local ls_prefixo="$1" li_codigo_saida="$2"

    if (( 0 == li_codigo_saida )); then
        return 0
    fi

    # 124 = o "timeout" matou o processo
    if (( 124 == li_codigo_saida )); then
        log "[error]: untMain.py excedeu ${UNT_TIMEOUT_SECS}s para ${ls_prefixo} (UNT_TIMEOUT_SECS)."
        return 0
    fi

    log "[error]: untMain.py falhou para ${ls_prefixo} (rc=${li_codigo_saida}); ver ${UNT_LOG_ERR}."

} # loga_falha_do_untmain

# -------------------------------------------------------------------------------------
# acrescenta o verify.json desta execução, em UMA linha, ao histórico
# data/output/<R>_<A>_verify.jsonl (o diretório temporário será apagado)
#
# uso: acumula_verificacao <prefixo> <diretório temporário>
#
acumula_verificacao()
{
    local ls_prefixo="$1" ls_diretorio_temporario="$2"
    local ls_verify_json="${ls_diretorio_temporario}/verify.json"
    local ls_historico="${UNT_OUT}/${ls_prefixo}_verify.jsonl"

    if [ ! -f "${ls_verify_json}" ]; then
        return 0
    fi

    mkdir -p "${UNT_OUT}"
    tr -d '\n' < "${ls_verify_json}" >> "${ls_historico}"
    echo >> "${ls_historico}"

} # acumula_verificacao

# -------------------------------------------------------------------------------------
# roda o untMain.py para o par. Saída em data/temp/<prefixo>/<backend>/.
# retorna o código de saída do untMain.py (0 = ao menos um backend concluiu)
#
# uso: executa_nowcast <radar> <alcance> <prefixo>
#
executa_nowcast()
{
    local ls_radar="$1" li_alcance="$2" ls_prefixo="$3"
    local ls_diretorio_temporario="${UNT_TEMP}/${ls_prefixo}"

    log "[info]: calculando o nowcast (Deep Learning) para ${ls_prefixo}..."

    mkdir -p "${ls_diretorio_temporario}"
    muda_para_diretorio "${UNT_PATH}"

    # --verify só se UNT_VERIFY=true (array vazio = nenhum argumento extra)
    local larr_opcao_verify=()
    if [ "${UNT_VERIFY}" = true ]; then
        larr_opcao_verify=(--verify)
    fi

    local li_codigo_saida=0
    PYTHONPATH="${UNT_LIB_DIR}/apps" timeout "${UNT_TIMEOUT_SECS}" \
        python3 "${UNT_LIB_DIR}/apps/untMain.py" \
        -r "${ls_radar}" -a "${li_alcance}" \
        --catalog "${UNT_CATALOG_DB}" --cappi-dir "${UNT_CAPPI_DIR}" \
        --modelos-dir "${UNT_MODELOS_DIR}" --device "${UNT_DEVICE}" \
        -p "${ls_prefixo}" -s "${ls_diretorio_temporario}" "${larr_opcao_verify[@]}" \
        >> "${UNT_LOG_APP}" 2>> "${UNT_LOG_ERR}" || li_codigo_saida=$?

    loga_falha_do_untmain "${ls_prefixo}" "${li_codigo_saida}"
    acumula_verificacao "${ls_prefixo}" "${ls_diretorio_temporario}"

    return "${li_codigo_saida}"

} # executa_nowcast

# < monitor (untMonitor.py) >----------------------------------------------------------

# -------------------------------------------------------------------------------------
# skill em operação: compara cada backend com a persistência nos verify.jsonl
# (janela móvel do [monitor]) e grava <R>_<A>_monitor.json em UNT_OUT. Só com
# UNT_VERIFY=true. Nunca falha a passada: alarme (rc 2) vira [warn] no log.
#
executa_monitor()
{
    if [ "${UNT_VERIFY}" != true ]; then
        return 0
    fi

    muda_para_diretorio "${UNT_PATH}"

    local li_codigo_saida=0
    PYTHONPATH="${UNT_LIB_DIR}/apps" timeout "${UNT_MONITOR_TIMEOUT_SECS}" \
        python3 "${UNT_LIB_DIR}/apps/untMonitor.py" -i "${UNT_OUT}" --config "${UNT_CONFIG}" \
        >> "${UNT_LOG_APP}" 2>> "${UNT_LOG_ERR}" || li_codigo_saida=$?

    case "${li_codigo_saida}" in
        0) ;;
        1) log "[info]: monitor: ainda sem verify.jsonl em ${UNT_OUT}." ;;
        2) log "[warn]: monitor: backend abaixo da persistência (ver ${UNT_OUT}/*_monitor.json e ${UNT_LOG_ERR})." ;;
        *) log "[error]: untMonitor.py falhou (rc=${li_codigo_saida}); ver ${UNT_LOG_ERR}." ;;
    esac

    return 0

} # executa_monitor

# < GIFs (untMakGIF.py) >--------------------------------------------------------------
#
# cada backend tem dois GIFs em data/output/:
#   <R>_<A>_now_<backend>.gif   previsão atual
#   <R>_<A>_prv_<backend>.gif   previsão anterior
# um backend sem previsão recebe o GIF de aviso UNT_GIF_SEM_PREVISAO.

# -------------------------------------------------------------------------------------
# uso: caminho_gif <prefixo> <now|prv> <backend>
#
caminho_gif()
{
    echo "${UNT_OUT}/$1_$2_$3.gif"

} # caminho_gif

# -------------------------------------------------------------------------------------
# copia o GIF de aviso "sem previsão" para o destino, de forma atômica: copia
# para um temporário oculto no mesmo diretório e renomeia (mv no mesmo
# sistema de arquivos é um rename), para o dashboard (19.dash) nunca ler um
# GIF pela metade -- o mesmo cuidado do grava_gif_atomico do untMakGIF.py
#
# uso: copia_gif_sem_previsao <destino>
#
copia_gif_sem_previsao()
{
    local ls_temp
    ls_temp="$(dirname "$1")/.$(basename "$1").tmp"

    cp "${UNT_ASSETS}/${UNT_GIF_SEM_PREVISAO}" "${ls_temp}" && mv -f "${ls_temp}" "$1"

} # copia_gif_sem_previsao

# -------------------------------------------------------------------------------------
# para cada GIF do par (backends + montado), o "now" atual vira "prv". Sem "now", o "prv" recebe o
# aviso de "sem previsão".
#
# uso: rotaciona_gifs <prefixo>
#
rotaciona_gifs()
{
    local ls_prefixo="$1"
    local ls_gif ls_gif_atual ls_gif_anterior

    for ls_gif in "${UNT_GIFS_POR_PAR[@]}"; do
        ls_gif_atual=$(caminho_gif "${ls_prefixo}" now "${ls_gif}")
        ls_gif_anterior=$(caminho_gif "${ls_prefixo}" prv "${ls_gif}")

        # if/else explícito (e não "A && B || C"): se o mv falhasse, o
        # "|| C" sobrescreveria o anterior com o aviso de erro
        if [ -f "${ls_gif_atual}" ]; then
            mv -f "${ls_gif_atual}" "${ls_gif_anterior}"
        else
            copia_gif_sem_previsao "${ls_gif_anterior}"
        fi
    done

} # rotaciona_gifs

# -------------------------------------------------------------------------------------
# o untMain.py falhou: todos os backends ficam com o aviso de "sem previsão"
#
# uso: marca_todos_os_backends_sem_previsao <prefixo>
#
marca_todos_os_backends_sem_previsao()
{
    local ls_prefixo="$1"
    local ls_gif

    for ls_gif in "${UNT_GIFS_POR_PAR[@]}"; do
        copia_gif_sem_previsao "$(caminho_gif "${ls_prefixo}" now "${ls_gif}")"
    done

    log "[error]: sem nowcast para ${ls_prefixo} (untMain.py falhou); GIFs marcados com aviso de erro."

} # marca_todos_os_backends_sem_previsao

# -------------------------------------------------------------------------------------
# roda o untMakGIF.py: um GIF <prefixo>_now_<backend>.gif por subdiretório
# de backend em data/temp/<prefixo>/. Falha ou timeout são ignorados aqui --
# preenche_gifs_ausentes cuida do que faltar.
#
# uso: executa_untmakgif <prefixo>
#
executa_untmakgif()
{
    local ls_prefixo="$1"

    muda_para_diretorio "${UNT_PATH}"

    PYTHONPATH="${UNT_LIB_DIR}/apps" timeout "${UNT_TIMEOUT_SECS}" \
        python3 "${UNT_LIB_DIR}/apps/untMakGIF.py" \
        -i "${UNT_TEMP}/${ls_prefixo}" -s "${UNT_OUT}" -f "${ls_prefixo}_now.gif" \
        --idade-minima 0 \
        >> "${UNT_LOG_APP}" 2>> "${UNT_LOG_ERR}" || true

} # executa_untmakgif

# -------------------------------------------------------------------------------------
# GIF "now" (de backend ou o montado) não gerado nesta execução (aquele
# backend falhou, ou teve poucos frames) recebe o aviso de "sem previsão"
#
# uso: preenche_gifs_ausentes <prefixo>
#
preenche_gifs_ausentes()
{
    local ls_prefixo="$1"
    local ls_gif ls_gif_atual

    for ls_gif in "${UNT_GIFS_POR_PAR[@]}"; do
        ls_gif_atual=$(caminho_gif "${ls_prefixo}" now "${ls_gif}")

        if [ -f "${ls_gif_atual}" ]; then
            continue
        fi

        copia_gif_sem_previsao "${ls_gif_atual}"
        log "[warn]: GIF ${ls_gif} ausente -- usando aviso de erro."
    done

} # preenche_gifs_ausentes

# -------------------------------------------------------------------------------------
# uso: remove_diretorio_temporario <prefixo>
#
remove_diretorio_temporario()
{
    local ls_diretorio_temporario="${UNT_TEMP}/$1"

    if [ -d "${ls_diretorio_temporario}" ]; then
        rm -rf "${ls_diretorio_temporario}"
    fi

} # remove_diretorio_temporario

# -------------------------------------------------------------------------------------
# atualiza os GIFs do par: rotaciona now -> prv e gera os novos "now".
#
# com o untMain.py tendo falhado (<falhou> = 1), todos os "now" recebem o
# aviso de "sem previsão" e o diretório temporário é mantido para inspeção.
#
# uso: gera_gifs <prefixo> <falhou: 0|1>
#
gera_gifs()
{
    local ls_prefixo="$1" li_falhou="${2:-0}"

    log "[info]: gerando GIF ${ls_prefixo}..."

    mkdir -p "${UNT_OUT}"
    rotaciona_gifs "${ls_prefixo}"

    if (( li_falhou )); then
        marca_todos_os_backends_sem_previsao "${ls_prefixo}"
        return 0
    fi

    executa_untmakgif "${ls_prefixo}"
    preenche_gifs_ausentes "${ls_prefixo}"
    remove_diretorio_temporario "${ls_prefixo}"

    log "[info]: geração de GIF concluída."

} # gera_gifs

# < processamento de um par >----------------------------------------------------------

# -------------------------------------------------------------------------------------
# corpo de processa_par, já com o lock do par. Todo caminho normal termina
# com "return 0" (ver CUIDADO com set -e no topo do arquivo).
#
# uso: processa_par_com_lock <radar> <alcance> <prefixo>
#
processa_par_com_lock()
{
    local ls_radar="$1" li_alcance="$2" ls_prefixo="$3"

    local ls_target
    ls_target=$(busca_target_mais_recente "${ls_radar}" "${li_alcance}")

    if [ -z "${ls_target}" ]; then
        log "[info]: nenhuma sequência catalogada para ${ls_prefixo}."
        return 0
    fi

    if ! tem_target_novo "${ls_prefixo}" "${ls_target}"; then
        log "[info]: sem sequência nova para ${ls_prefixo} (target=${ls_target})."
        return 0
    fi

    if target_muito_antigo "${ls_target}"; then
        loga_target_muito_antigo "${ls_prefixo}" "${ls_target}"
        return 0
    fi

    local li_falhou=0
    if ! executa_nowcast "${ls_radar}" "${li_alcance}" "${ls_prefixo}"; then
        li_falhou=1
    fi

    gera_gifs "${ls_prefixo}" "${li_falhou}"

    # o marcador só avança se deu certo: uma falha passageira (ex.: .npy
    # ainda sendo gravado) é tentada de novo na próxima passada
    if (( 0 == li_falhou )); then
        grava_ultimo_target "${ls_prefixo}" "${ls_target}"
    fi

    log "[info]: processamento de ${ls_prefixo} concluído."

    return 0

} # processa_par_com_lock

# -------------------------------------------------------------------------------------
# gera o nowcast e os GIFs de um par radar/alcance, se houver sequência
# nova e recente. Um lock por par (descritor 300) garante que dois
# processos nunca trabalhem no mesmo par ao mesmo tempo.
#
# uso: processa_par <radar> <alcance>
#
processa_par()
{
    local ls_radar="$1" li_alcance="$2"
    local ls_prefixo="${ls_radar}_${li_alcance}"

    log "[info]: processando ${ls_prefixo}..."

    # outro processo já está neste par: nada a fazer
    exec 300>"${UNT_STATE}/proc_${ls_prefixo}.lock"
    if ! flock -n 300; then
        exec 300>&-
        return 0
    fi

    processa_par_com_lock "${ls_radar}" "${li_alcance}" "${ls_prefixo}"

    # libera o lock fechando o descritor
    exec 300>&-

    return 0

} # processa_par

# < the end >--------------------------------------------------------------------------
