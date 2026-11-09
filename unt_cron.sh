#!/usr/bin/env bash
#
# Centro Logístico de Simulação Meteorológica
# Copyright (c) 1995-2026, Instituto de Controle do Espaço Aéreo
#
# unt_cron.sh
# ponto de entrada do cron. O cron roda os jobs com um ambiente mínimo (sem
# o PATH e às vezes sem o HOME do usuário, e sem ler .bashrc/.profile). Este
# script só normaliza esse ambiente, lê o unt_env.sh local (opcional) e
# entrega a execução ao unt_run.sh.
#
# crontab -e:
#   */5 * * * * /caminho/do/repositorio/unt_cron.sh          (nowcast, unt_run.sh)
#   0 3 * * 0   /caminho/do/repositorio/unt_cron.sh treino   (host de GPU, unt_treino.sh)
#
# 2026.out  mlabru  argumento opcional "treino" (unt_treino.sh); padrão "run".
#
# 2026.out  mlabru  portado de 25.oflow (ofl_cron.sh), sem mudança de lógica.
#
# 2026.out  mlabru  cabeçalho comum (normalização com 21.mvol/27.nowcast)
#
# 2026.set  mlabru  comentários revistos.
#
# 2026.ago  mlabru  initial version (Linux/Bash)

# depends on: bash, unt_env.sh (opcional), unt_run.sh, unt_treino.sh

set -euo pipefail

# USER/HOME/PATH mínimos, para que sqlite3 e python3 sejam achados como numa
# sessão interativa
export USER="${USER:-$(id -un)}"
export HOME="${HOME:-/home/${USER}}"
export PATH="/usr/local/bin:/usr/bin:/bin:${PATH:-}"

# diretório deste script (independe de onde o cron o chama)
CS_DIRETORIO_CRON="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ambiente local opcional (caminhos desta máquina: UNT_CATALOG_DB,
# UNT_CAPPI_DIR, UNT_PATH, ...). "set -a" exporta tudo o que o arquivo
# atribuir. Modelo: unt_env.sh.example
CS_ARQUIVO_AMBIENTE="${CS_DIRETORIO_CRON}/unt_env.sh"

if [ -f "${CS_ARQUIVO_AMBIENTE}" ]; then
    set -a
    # shellcheck source=/dev/null
    source "${CS_ARQUIVO_AMBIENTE}"
    set +a
fi

# o que rodar: run (nowcast, padrão) ou treino
CS_TAREFA="${1:-run}"

case "${CS_TAREFA}" in
    run|treino) ;;
    *) echo "uso: unt_cron.sh [run|treino]" >&2
       exit 2 ;;
esac

# "exec" substitui este processo pelo unt_<tarefa>.sh (o código de saída dele
# vira o deste). Chamado via bash para não depender do bit de execução.
exec bash "${CS_DIRETORIO_CRON}/unt_${CS_TAREFA}.sh"

# < the end >--------------------------------------------------------------------------
