#!/usr/bin/env bash
#
# Centro Logístico de Simulação Meteorológica
# Copyright (c) 1995-2026, Instituto de Controle do Espaço Aéreo
#
# unt_run.sh
# uma passada do nowcast: para cada par radar/alcance do catálogo, gera o
# nowcast dos backends de Deep Learning e os GIFs se houver sequência nova (ver processa_par em
# unt_common.sh), e termina. A repetição fica por conta do cron (unt_cron.sh).
#
# só uma execução por vez: se a anterior ainda estiver rodando (lock de
# instância UNT_INSTANCE_LOCK), esta sai em silêncio com código 0.
#
# códigos de saída: 0 = ok (inclusive "nada a fazer"); 1 = catálogo
# inacessível/inválido ou falha ao consultá-lo.
#
# ATENÇÃO: espelho do ofl_run.sh, que o ambiente de teste (21.mvol/04.dvl/
# tst_env/tst_cycle.sh) chama por nome e sem argumentos. Não renomear.
#
# 2026.out  mlabru  "main; exit" na mesma linha (edição do script durante a varredura).
# 2026.out  mlabru  monitor de skill (executa_monitor) ao fim da varredura.
#
# 2026.out  mlabru  portado de 25.oflow (ofl_run.sh), sem mudança de lógica.
#
# 2026.out  mlabru  normalização com 21.mvol/27.nowcast: corpo do topo em
#                   main() + UNT_RUN_NO_MAIN (carregar sem executar, nos
#                   testes); nível [error] (era [erro]); cabeçalho comum
#
# 2026.out  mlabru  nomes em português; laço de pares com cláusulas de guarda.
#
# 2026.set  mlabru  removido o download via SFTP -- não há mais .mvol a
#                   baixar; a varredura passou a ser dos pares (radar,
#                   alcance) do catálogo SQLite.
#
# 2026.ago  mlabru  initial version (Bash/Linux)

# depends on: unt_common.sh, python3, sqlite3, flock, timeout

set -euo pipefail

# < configuração + funções >-----------------------------------------------------------

CS_DIRETORIO_RUN="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=unt_common.sh
source "${CS_DIRETORIO_RUN}/unt_common.sh"

# -------------------------------------------------------------------------------------
# processa todos os pares do catálogo, no máximo UNT_MAX_PARES_EM_PARALELO
# ao mesmo tempo (cada processa_par roda em background). É seguro rodar
# vários juntos: cada par tem o próprio lock.
#
# retorna 1 se a consulta ao catálogo falhar
#
processa_todos_os_pares()
{
    # logger
    log "[info]: varrendo pares radar/alcance catalogados..."

    # a lista é lida ANTES do laço: dentro de "< <(...)" uma falha do sqlite3
    # se perderia e pareceria uma lista vazia
    local ls_pares
    if ! ls_pares=$(lista_pares_catalogados); then
        # logger
        log "[error]: falha ao consultar o catálogo ${UNT_CATALOG_DB} (ver ${UNT_LOG_ERR})."
        # quit
        return 1
    fi

    local li_num_pares=0
    local li_em_execucao=0
    local ls_radar li_alcance

    while read -r ls_radar li_alcance; do
        # linha vazia (catálogo sem pares produz uma)
        if [ -z "${ls_radar}" ]; then
            continue
        fi

        li_num_pares=$(( li_num_pares + 1 ))

        processa_par "${ls_radar}" "${li_alcance}" &
        li_em_execucao=$(( li_em_execucao + 1 ))

        # limite atingido: espera algum terminar antes de disparar o próximo
        if (( li_em_execucao >= UNT_MAX_PARES_EM_PARALELO )); then
            wait -n
            li_em_execucao=$(( li_em_execucao - 1 ))
        fi
    done <<< "${ls_pares}"

    # espera os que ainda estão rodando
    wait

    if (( 0 == li_num_pares )); then
        # logger
        log "[warn]: nenhum par radar/alcance em ${UNT_CATALOG_DB}."
    fi

    # logger
    log "[info]: varredura concluída (${li_num_pares} pares)."

} # processa_todos_os_pares

# -------------------------------------------------------------------------------------
# lock de instância (descritor 200, mantido aberto até o fim do script):
# outra execução ainda rodando -> sai com 0
#
adquire_lock_de_instancia()
{
    exec 200>"${UNT_INSTANCE_LOCK}"
    if ! flock -n 200; then
        # logger
        log "[info]: nowcast já em execução (execução anterior); saindo."
        # quit
        exit 0
    fi

} # adquire_lock_de_instancia

# -------------------------------------------------------------------------------------
# drive app
#
main()
{
    # cria a árvore de dados ANTES do primeiro log (que grava em UNT_LOGS) e
    # do lock (em UNT_STATE) -- numa raiz nova os dois falhariam
    mkdir -p "${UNT_BASE}" "${UNT_LOGS}" "${UNT_OUT}" "${UNT_STATE}" "${UNT_TEMP}"

    # logger
    log "-----------------------------------------------------------------"
    log "[info]: iniciando execução do nowcast (host=${UNT_HOST})..."

    # lock de instância
    adquire_lock_de_instancia

    # caminhos efetivos -- sob o cron podem ser diferentes da sessão interativa
    log "[info]: catálogo=${UNT_CATALOG_DB} cappi=${UNT_CAPPI_DIR}"

    # catálogo inacessível é erro (rc 1), não "nada a fazer"
    if ! valida_catalogo; then
        # quit
        exit 1
    fi

    # virtualenv ativado uma vez aqui; os processa_par em background herdam
    # o ambiente
    ativa_virtualenv

    # varredura dos pares
    if ! processa_todos_os_pares; then
        # quit
        exit 1
    fi

    # skill em operação x persistência (só com UNT_VERIFY=true)
    executa_monitor

    # logger
    log "[info]: execução concluída."

} # main

# < main >-----------------------------------------------------------------------------

# permite carregar as funções acima sem executar o main (testes)
[ "${UNT_RUN_NO_MAIN:-false}" = true ] && return 0

# exit na mesma linha: o bash lê o script aos poucos; se o arquivo for editado durante a
# varredura, sem o exit ele seguiria lendo o arquivo novo a partir da posição antiga
main; exit $?

# < the end >--------------------------------------------------------------------------
