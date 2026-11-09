#!/usr/bin/env bash
#
# Centro Logístico de Simulação Meteorológica
# Copyright (c) 1995-2026, Instituto de Controle do Espaço Aéreo
#
# unt_treino.sh
# um ciclo de treino, no host de GPU e FORA do cron de 5 min do nowcast
# (ex.: semanal, via "unt_cron.sh treino"):
#
#   1. manifesto de janelas de todos os pares do catálogo (untJanelas.py);
#   2. untTrain.py de cada backend de UNT_TREINO_BACKENDS -> candidatos
#      (o atual.pt não muda);
#   3. untEval.py no conjunto de teste: candidatos ao lado dos atuais e da
#      persistência;
#   4. untPromove.py de cada candidato (portão [promocao] do config.ini), com
#      [warn] quando o critério falha em algum par radar/alcance (alertas_por_par
#      do promocoes.jsonl), mesmo com o candidato promovido pelo agregado;
#   5. rsync de UNT_MODELOS_DIR para UNT_DESTINO_MODELOS (máquina operacional),
#      checkpoints antes dos links atual.pt/anterior.pt.
#
# ciclo interrompido (queda, timeout): o estado fica em UNT_TREINO_CICLO
# (manifesto do ciclo e candidatos já prontos) e o próximo unt_treino.sh o
# retoma -- mesmo manifesto, backends prontos pulados, os demais com
# untTrain.py --retoma (continua da última época gravada).
#
# só uma execução por vez (lock UNT_TREINO_LOCK); log em unt_treino.log.
#
# rollback: feito AQUI (untPromove.py --reverte) e distribuído; um --reverte
# só na máquina operacional seria desfeito no próximo rsync.
#
# códigos de saída: 0 = ciclo concluído (com ou sem promoção); 1 = falha
# (catálogo, manifesto, nenhum backend treinado, avaliação ou rsync).
#
# 2026.out  mlabru  "main; exit" na mesma linha (edição do script durante o ciclo).
# 2026.out  mlabru  [warn] dos alertas por par radar/alcance do untPromove.py
# 2026.out  mlabru  initial version (Linux/Bash)

# depends on: unt_common.sh, python3, sqlite3, flock, timeout, rsync

set -euo pipefail

# < configuração + funções >-----------------------------------------------------------

CS_DIRETORIO_TREINO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=unt_common.sh
source "${CS_DIRETORIO_TREINO}/unt_common.sh"

# o log() do unt_common.sh grava onde UNT_LOG_APP/UNT_LOG_ERR apontam: o
# treino tem os próprios
UNT_LOG_APP="${UNT_LOG_TREINO}"
UNT_LOG_ERR="${UNT_LOG_TREINO_ERR}"

# -------------------------------------------------------------------------------------
# roda um apps/<script>.py com o PYTHONPATH do projeto; saída nos logs do treino
#
# uso: executa_app <script.py> [argumentos...]
#
executa_app()
{
    local ls_script="$1"
    shift

    PYTHONPATH="${UNT_LIB_DIR}/apps" python3 "${UNT_LIB_DIR}/apps/${ls_script}" "$@" \
        >> "${UNT_LOG_APP}" 2>> "${UNT_LOG_ERR}"

} # executa_app

# < estado do ciclo >------------------------------------------------------------------
#
# UNT_TREINO_CICLO/manifesto      caminho do manifesto do ciclo
# UNT_TREINO_CICLO/<backend>      caminho do candidato já treinado

# -------------------------------------------------------------------------------------
# escreve o manifesto do ciclo: o do ciclo interrompido, se houver; senão gera
# um novo (untJanelas.py, todos os pares). Retorna 1 se não conseguir.
#
manifesto_do_ciclo()
{
    local ls_estado="${UNT_TREINO_CICLO}/manifesto"

    if [ -f "${ls_estado}" ] && [ -f "$(cat "${ls_estado}")" ]; then
        log "[info]: retomando o ciclo interrompido (manifesto $(cat "${ls_estado}"))."
        cat "${ls_estado}"
        return 0
    fi

    local ls_manifesto
    ls_manifesto="${UNT_JANELAS_DIR}/janelas_$(date -u '+%Y%m%d%H%M%S').csv"

    log "[info]: gerando o manifesto ${ls_manifesto}..."

    if ! executa_app untJanelas.py --catalog "${UNT_CATALOG_DB}" \
            --cappi-dir "${UNT_CAPPI_DIR}" -o "${ls_manifesto}" --config "${UNT_CONFIG}"; then
        log "[error]: untJanelas.py falhou (ver ${UNT_LOG_ERR})."
        return 1
    fi

    # ciclo novo: retomadas de um ciclo abandonado são de outro manifesto (o
    # untTrain.py --retoma as recusaria)
    find "${UNT_MODELOS_DIR}" -name '*.retomada.pt' -print -delete 2>> "${UNT_LOG_ERR}" \
        | while read -r ls_retomada; do log "[warn]: retomada abandonada removida: ${ls_retomada}"; done

    mkdir -p "${UNT_TREINO_CICLO}"
    echo "${ls_manifesto}" > "${ls_estado}"
    echo "${ls_manifesto}"

} # manifesto_do_ciclo

# -------------------------------------------------------------------------------------
# uso: candidato_pronto <backend>   (escreve o caminho; vazio se não há)
#
candidato_pronto()
{
    local ls_estado="${UNT_TREINO_CICLO}/$1"

    if [ -f "${ls_estado}" ] && [ -f "$(cat "${ls_estado}")" ]; then
        cat "${ls_estado}"
    fi

} # candidato_pronto

# < treino >---------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
# treina um backend (candidato) e registra o caminho no estado do ciclo.
# Retorna != 0 se o untTrain.py falhar ou não gravar nada.
#
# uso: treina_backend <backend> <manifesto>
#
treina_backend()
{
    local ls_backend="$1" ls_manifesto="$2"

    if [ -n "$(candidato_pronto "${ls_backend}")" ]; then
        log "[info]: ${ls_backend}: candidato já treinado neste ciclo ($(candidato_pronto "${ls_backend}"))."
        return 0
    fi

    log "[info]: treinando ${ls_backend}..."

    # marcador de tempo: o candidato é o .pt mais novo que ele
    local ls_marcador="${UNT_TREINO_CICLO}/.inicio_${ls_backend}"
    touch "${ls_marcador}"

    # timeout 0 = sem limite
    local larr_timeout=()
    if (( UNT_TREINO_TIMEOUT_SECS > 0 )); then
        larr_timeout=(timeout "${UNT_TREINO_TIMEOUT_SECS}")
    fi

    # UNT_TREINO_ARGS: palavras separadas por espaço (sem aspas internas)
    local larr_extras=()
    read -r -a larr_extras <<< "${UNT_TREINO_ARGS}"

    local li_codigo_saida=0
    PYTHONPATH="${UNT_LIB_DIR}/apps" "${larr_timeout[@]}" \
        python3 "${UNT_LIB_DIR}/apps/untTrain.py" -b "${ls_backend}" \
        --janelas "${ls_manifesto}" --cappi-dir "${UNT_CAPPI_DIR}" \
        --modelos-dir "${UNT_MODELOS_DIR}" --config "${UNT_CONFIG}" \
        --device "${UNT_TREINO_DEVICE}" --retoma "${larr_extras[@]}" \
        >> "${UNT_LOG_APP}" 2>> "${UNT_LOG_ERR}" || li_codigo_saida=$?

    if (( 0 != li_codigo_saida )); then
        log "[error]: untTrain.py falhou para ${ls_backend} (rc=${li_codigo_saida}); ver ${UNT_LOG_ERR}."
        return "${li_codigo_saida}"
    fi

    local ls_candidato
    ls_candidato=$(find "${UNT_MODELOS_DIR}/${ls_backend}" -maxdepth 1 -type f \
                       -name "${ls_backend}_*.pt" -newer "${ls_marcador}" -printf '%T@ %p\n' \
                   | sort -n | tail -1 | cut -d' ' -f2-)

    if [ -z "${ls_candidato}" ]; then
        log "[error]: ${ls_backend}: untTrain.py terminou sem gravar candidato."
        return 1
    fi

    echo "${ls_candidato}" > "${UNT_TREINO_CICLO}/${ls_backend}"
    log "[info]: ${ls_backend}: candidato ${ls_candidato}."

} # treina_backend

# < avaliação e promoção >-------------------------------------------------------------

# -------------------------------------------------------------------------------------
# untEval.py no teste: atuais + candidatos (<backend>_candidato) + persistência.
# Escreve o caminho do comparacao.json; retorna 1 se a avaliação falhar.
#
# uso: avalia_candidatos <manifesto> <backend>...
#
avalia_candidatos()
{
    local ls_manifesto="$1"
    shift

    local ls_saida
    ls_saida="${UNT_AVALIACAO_DIR}/$(date -u '+%Y%m%d%H%M%S')"

    local larr_checkpoints=()
    local ls_backend
    for ls_backend in "$@"; do
        larr_checkpoints+=(--checkpoint "${ls_backend}_candidato=$(candidato_pronto "${ls_backend}")")
    done

    log "[info]: avaliando $* no conjunto de teste..."

    if ! executa_app untEval.py --janelas "${ls_manifesto}" --split teste \
            --cappi-dir "${UNT_CAPPI_DIR}" --modelos-dir "${UNT_MODELOS_DIR}" \
            --device "${UNT_TREINO_DEVICE}" --saida "${ls_saida}" "${larr_checkpoints[@]}"; then
        log "[error]: untEval.py falhou (sem janelas de teste?); nada promovido (ver ${UNT_LOG_ERR})."
        return 1
    fi

    echo "${ls_saida}/comparacao.json"

} # avalia_candidatos

# -------------------------------------------------------------------------------------
# número de linhas do promocoes.jsonl de um backend (0 se ainda não existe)
#
# uso: linhas_do_registro <backend>
#
linhas_do_registro()
{
    local ls_registro="${UNT_MODELOS_DIR}/$1/promocoes.jsonl"

    if [ -f "${ls_registro}" ]; then
        wc -l < "${ls_registro}"
    else
        echo 0
    fi

} # linhas_do_registro

# -------------------------------------------------------------------------------------
# [warn] com os pares radar/alcance em que o critério do portão falha
# (alertas_por_par da decisão que o untPromove.py acabou de registrar). Só lê o
# registro se ele ganhou uma linha nesta chamada: um untPromove.py que falhou
# antes de registrar não pode reavisar a decisão do ciclo anterior.
#
# uso: avisa_alertas_por_par <backend> <linhas do registro antes do untPromove.py>
#
avisa_alertas_por_par()
{
    local ls_backend="$1"
    local li_linhas_antes="$2"
    local ls_registro="${UNT_MODELOS_DIR}/${ls_backend}/promocoes.jsonl"
    local ls_pares

    if [ "$(linhas_do_registro "${ls_backend}")" -le "${li_linhas_antes}" ]; then
        return 0
    fi

    ls_pares=$(tail -n 1 "${ls_registro}" | python3 -c \
        'import json, sys; print(",".join(json.load(sys.stdin).get("alertas_por_par") or {}))' \
        2>> "${UNT_LOG_ERR}") || ls_pares=""

    if [ -n "${ls_pares}" ]; then
        log "[warn]: ${ls_backend}: critério do portão falha em ${ls_pares}" \
            "(promoção julgada pelo agregado; ver alertas_por_par em promocoes.jsonl)."
    fi

} # avisa_alertas_por_par

# -------------------------------------------------------------------------------------
# untPromove.py de cada candidato. Escreve quantos foram promovidos.
#
# uso: promove_candidatos <comparacao.json> <backend>...
#
promove_candidatos()
{
    local ls_comparacao="$1"
    shift

    local li_promovidos=0
    local li_linhas_antes
    local ls_backend

    for ls_backend in "$@"; do
        li_linhas_antes=$(linhas_do_registro "${ls_backend}")

        if executa_app untPromove.py -b "${ls_backend}" \
                --candidato "$(candidato_pronto "${ls_backend}")" \
                --avaliacao "${ls_comparacao}" --modelos-dir "${UNT_MODELOS_DIR}" \
                --config "${UNT_CONFIG}"; then
            log "[info]: ${ls_backend}: candidato promovido a atual.pt."
            li_promovidos=$(( li_promovidos + 1 ))
        else
            log "[warn]: ${ls_backend}: candidato recusado pelo portão (ver promocoes.jsonl)."
        fi

        avisa_alertas_por_par "${ls_backend}" "${li_linhas_antes}"
    done

    echo "${li_promovidos}"

} # promove_candidatos

# < distribuição >---------------------------------------------------------------------

# -------------------------------------------------------------------------------------
# rsync dos modelos para a máquina operacional, em duas passadas: primeiro os
# checkpoints/.json/promocoes.jsonl, depois os links atual.pt/anterior.pt --
# um link nunca chega antes do arquivo para o qual aponta. Arquivos de
# retomada e temporários (.*) não vão. Sem UNT_DESTINO_MODELOS, não faz nada.
#
distribui_modelos()
{
    if [ -z "${UNT_DESTINO_MODELOS}" ]; then
        log "[info]: UNT_DESTINO_MODELOS vazio: modelos não distribuídos."
        return 0
    fi

    if [[ "${UNT_DESTINO_MODELOS}" == *:* ]] && [ "${MET_ENV:-}" = test ]; then
        log "[error]: MET_ENV=test: destino remoto ${UNT_DESTINO_MODELOS} recusado."
        return 1
    fi

    local larr_exclusoes=(--exclude='*.retomada.pt' --exclude='.*')

    # o rsync não cria os pais do destino: local, mkdir -p; remoto, --mkpath
    # (rsync >= 3.2.3 nas duas pontas)
    local larr_cria_destino=()
    if [[ "${UNT_DESTINO_MODELOS}" == *:* ]]; then
        larr_cria_destino=(--mkpath)
    else
        mkdir -p "${UNT_DESTINO_MODELOS}"
    fi

    log "[info]: distribuindo ${UNT_MODELOS_DIR} -> ${UNT_DESTINO_MODELOS}..."

    if ! rsync -a "${larr_cria_destino[@]}" "${larr_exclusoes[@]}" \
            --exclude='atual.pt' --exclude='anterior.pt' \
            "${UNT_MODELOS_DIR}/" "${UNT_DESTINO_MODELOS}/" 2>> "${UNT_LOG_ERR}"; then
        log "[error]: rsync dos checkpoints falhou (ver ${UNT_LOG_ERR})."
        return 1
    fi

    if ! rsync -a "${larr_exclusoes[@]}" \
            "${UNT_MODELOS_DIR}/" "${UNT_DESTINO_MODELOS}/" 2>> "${UNT_LOG_ERR}"; then
        log "[error]: rsync dos links atual.pt/anterior.pt falhou (ver ${UNT_LOG_ERR})."
        return 1
    fi

    log "[info]: modelos distribuídos."

} # distribui_modelos

# < main >-----------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
# lock do treino (descritor 201, aberto até o fim): outro ciclo rodando -> sai com 0
#
adquire_lock_do_treino()
{
    exec 201>"${UNT_TREINO_LOCK}"
    if ! flock -n 201; then
        log "[info]: treino já em execução; saindo."
        exit 0
    fi

} # adquire_lock_do_treino

# -------------------------------------------------------------------------------------
# drive app
#
main()
{
    mkdir -p "${UNT_BASE}" "${UNT_LOGS}" "${UNT_STATE}" "${UNT_MODELOS_DIR}" \
             "${UNT_JANELAS_DIR}" "${UNT_AVALIACAO_DIR}"

    log "-----------------------------------------------------------------"
    log "[info]: iniciando o ciclo de treino (host=${UNT_HOST}, device=${UNT_TREINO_DEVICE})..."

    adquire_lock_do_treino

    log "[info]: catálogo=${UNT_CATALOG_DB} cappi=${UNT_CAPPI_DIR} modelos=${UNT_MODELOS_DIR}"

    if ! valida_catalogo; then
        exit 1
    fi

    ativa_virtualenv
    muda_para_diretorio "${UNT_PATH}"

    # 1. manifesto
    local ls_manifesto
    if ! ls_manifesto=$(manifesto_do_ciclo); then
        exit 1
    fi

    # 2. treino (um backend que falha não impede os outros)
    local larr_treinados=()
    local ls_backend
    for ls_backend in ${UNT_TREINO_BACKENDS}; do
        if treina_backend "${ls_backend}" "${ls_manifesto}"; then
            larr_treinados+=("${ls_backend}")
        fi
    done

    if (( 0 == ${#larr_treinados[@]} )); then
        log "[error]: nenhum backend treinado; o ciclo fica para ser retomado."
        exit 1
    fi

    # 3-4. avaliação e promoção
    local ls_comparacao li_promovidos=0
    if ! ls_comparacao=$(avalia_candidatos "${ls_manifesto}" "${larr_treinados[@]}"); then
        # sem teste não há promoção: o ciclo termina (candidatos ficam no disco)
        rm -rf "${UNT_TREINO_CICLO}"
        exit 1
    fi

    li_promovidos=$(promove_candidatos "${ls_comparacao}" "${larr_treinados[@]}")

    # 5. distribuição (também sem promoção: mantém o destino igual à origem)
    if ! distribui_modelos; then
        exit 1
    fi

    rm -rf "${UNT_TREINO_CICLO}"

    log "[info]: ciclo de treino concluído: ${#larr_treinados[@]} treinados," \
        "${li_promovidos} promovidos; avaliação em ${ls_comparacao}."

} # main

# < main >-----------------------------------------------------------------------------

# permite carregar as funções acima sem executar o main (testes)
[ "${UNT_TREINO_NO_MAIN:-false}" = true ] && return 0

# exit na mesma linha: o bash lê o script aos poucos, e o ciclo dura horas/dias; se o
# arquivo for editado nesse meio-tempo, sem o exit ele seguiria lendo o arquivo novo
# a partir da posição antiga (erro de sintaxe espúrio ao fim do ciclo)
main; exit $?

# < the end >--------------------------------------------------------------------------
