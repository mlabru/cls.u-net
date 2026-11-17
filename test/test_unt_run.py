# -*- coding: utf-8 -*-
"""
Centro Logístico de Simulação Meteorológica
Copyright (c) 1995-2026, Instituto de Controle do Espaço Aéreo

test_unt_run
testes de unt_run.sh/unt_cron.sh: catálogo inacessível é erro (rc 1, nunca
criado nem silenciado), catálogo sem pares gera aviso, e o unt_cron.sh lê o
ambiente local (unt_env.sh) mesmo sob o ambiente mínimo do cron

2026.out  mlabru  portado de 25.oflow; nomes dos backends trocados (pytorch, rainnet, smaat).

2026.out  mlabru  GIF montado (mosaico) na rotação e no aviso de erro

2026.out  mlabru  UNT_RUN_NO_MAIN (unt_run.sh com main())

2026.out  mlabru  initial version (Linux/Python)
"""
# < imports >----------------------------------------------------------------------------

# python library
import os
import pathlib
import shutil
import sqlite3
import subprocess

# pytest
import pytest

# < constants >--------------------------------------------------------------------------

CS_ROOT = pathlib.Path(__file__).resolve().parent.parent

pytestmark = pytest.mark.skipif(shutil.which("bash") is None or shutil.which("sqlite3") is None,
                                reason="bash/sqlite3 ausente")

# < helpers >----------------------------------------------------------------------------

def _catalog(fpth_db: pathlib.Path, flst_seq: list) -> pathlib.Path:
    """
    cria um catálogo mínimo (sequences, radar_sites) com as sequências dadas.
    """
    lcon = sqlite3.connect(fpth_db)
    lcon.execute("CREATE TABLE sequences (radar TEXT, alcance INTEGER, target INTEGER)")
    lcon.execute("CREATE TABLE radar_sites (radar TEXT)")
    lcon.executemany("INSERT INTO sequences VALUES (?, ?, ?)", flst_seq)
    lcon.commit()
    lcon.close()

    return fpth_db

# -------------------------------------------------------------------------------------
def _run(fpth_tmp: pathlib.Path, fpth_db: pathlib.Path, fpth_venv: pathlib.Path = None) -> tuple:
    """
    roda unt_run.sh com dados em fpth_tmp e retorna (rc, log da aplicação).
    sem fpth_venv, aponta UNT_VENV para um caminho inexistente (o ~/.venv da
    máquina não interfere no teste).
    """
    ldct_env = {lk: lv for lk, lv in os.environ.items()
                if not lk.startswith(("UNT_", "MET_"))}
    ldct_env.update({"UNT_PATH": str(fpth_tmp / "ofl"),
                     "UNT_CATALOG_DB": str(fpth_db),
                     "UNT_CAPPI_DIR": str(fpth_tmp / "cappi"),
                     "UNT_VENV": str(fpth_venv or fpth_tmp / "sem_venv")})

    lproc = subprocess.run(["bash", str(CS_ROOT / "unt_run.sh")],
                           env=ldct_env, capture_output=True, text=True)

    return lproc.returncode, (fpth_tmp / "ofl/logs/unt_main.log").read_text()

# < testes >-----------------------------------------------------------------------------

def test_catalogo_inexistente_e_erro_e_nao_e_criado(tmp_path):
    lpth_db = tmp_path / "nao" / "catalog.db"

    li_rc, ls_log = _run(tmp_path, lpth_db)

    assert 1 == li_rc
    assert "catálogo inexistente ou vazio" in ls_log
    assert not lpth_db.exists()

# -------------------------------------------------------------------------------------
def test_catalogo_vazio_de_0_bytes_e_erro(tmp_path):
    lpth_db = tmp_path / "catalog.db"
    lpth_db.touch()

    li_rc, ls_log = _run(tmp_path, lpth_db)

    assert 1 == li_rc
    assert "catálogo inexistente ou vazio" in ls_log

# -------------------------------------------------------------------------------------
def test_catalogo_sem_tabelas_e_erro(tmp_path):
    lpth_db = tmp_path / "catalog.db"
    lcon = sqlite3.connect(lpth_db)
    lcon.execute("CREATE TABLE x (a)")
    lcon.close()

    li_rc, ls_log = _run(tmp_path, lpth_db)

    assert 1 == li_rc
    assert "sem as tabelas sequences/radar_sites" in ls_log

# -------------------------------------------------------------------------------------
def test_catalogo_sem_pares_gera_aviso(tmp_path):
    lpth_db = _catalog(tmp_path / "catalog.db", [])

    li_rc, ls_log = _run(tmp_path, lpth_db)

    assert 0 == li_rc
    assert "[warn]: nenhum par radar/alcance" in ls_log
    assert "varredura concluída (0 pares)" in ls_log

# -------------------------------------------------------------------------------------
def test_conta_pares_e_registra_caminho_do_catalogo(tmp_path):
    # target antigo: processa_par descarta pela idade, sem rodar o python
    lpth_db = _catalog(tmp_path / "catalog.db", [("CGU", 250, 1000), ("PCO", 400, 1000)])

    li_rc, ls_log = _run(tmp_path, lpth_db)

    assert 0 == li_rc
    assert f"catálogo={lpth_db}" in ls_log
    assert "varredura concluída (2 pares)" in ls_log

# -------------------------------------------------------------------------------------
def test_sequencia_antiga_loga_target_em_utc_e_idade(tmp_path):
    # target = 1970-01-01 00:16:40 UTC: idade em dias, limite padrão de 2 h
    lpth_db = _catalog(tmp_path / "catalog.db", [("CGU", 250, 1000)])

    li_rc, ls_log = _run(tmp_path, lpth_db)

    assert 0 == li_rc
    assert ("[warn]: sequência muito antiga para CGU_250 (target=1000"
            " = 1970-01-01 00:16:40 UTC, idade ") in ls_log
    assert " > limite 2h00min)." in ls_log

# -------------------------------------------------------------------------------------
def test_formata_duracao(tmp_path):
    ldct_env = {lk: lv for lk, lv in os.environ.items()
                if not lk.startswith(("UNT_", "MET_"))}
    ldct_env["UNT_PATH"] = str(tmp_path / "ofl")

    ls_cmd = (f'source "{CS_ROOT / "unt_common.sh"}"; '
              'for s in 0 59 2700 7200 69900 183900; do formata_duracao $s; echo; done')
    lproc = subprocess.run(["bash", "-c", ls_cmd], env=ldct_env, capture_output=True, text=True)

    assert ["0min", "0min", "45min", "2h00min", "19h25min", "2d03h05min"] == lproc.stdout.split()

# -------------------------------------------------------------------------------------
def test_gera_gifs_com_falha_marca_backends_e_mosaico_e_rotaciona(tmp_path):
    ldct_env = {lk: lv for lk, lv in os.environ.items()
                if not lk.startswith(("UNT_", "MET_"))}
    ldct_env["UNT_PATH"] = str(tmp_path / "ofl")

    lpth_assets = tmp_path / "ofl/data/assets"
    lpth_assets.mkdir(parents=True)
    (lpth_assets / "no_cast.gif").write_bytes(b"aviso")

    lpth_out = tmp_path / "ofl/data/output"
    lpth_out.mkdir(parents=True)
    (lpth_out / "CGU_250_now_mosaico.gif").write_bytes(b"anterior")

    ls_cmd = (f'source "{CS_ROOT / "unt_common.sh"}"; mkdir -p "$UNT_LOGS"; '
              "gera_gifs CGU_250 1")
    lproc = subprocess.run(["bash", "-c", ls_cmd], env=ldct_env, capture_output=True, text=True)

    assert 0 == lproc.returncode, lproc.stderr
    for ls_gif in ("pytorch", "rainnet", "smaat", "mosaico"):
        assert b"aviso" == (lpth_out / f"CGU_250_now_{ls_gif}.gif").read_bytes()
    # o "now" anterior do montado virou "prv"
    assert b"anterior" == (lpth_out / "CGU_250_prv_mosaico.gif").read_bytes()

# -------------------------------------------------------------------------------------
def test_ativa_virtualenv(tmp_path):
    # venv falso: bin/activate põe bin/ (com um python3 falso) no início do PATH
    lpth_venv = tmp_path / "venv"
    (lpth_venv / "bin").mkdir(parents=True)
    (lpth_venv / "bin/activate").write_text(f'export PATH="{lpth_venv}/bin:$PATH"\n')
    lpth_py = lpth_venv / "bin/python3"
    lpth_py.write_text("#!/bin/sh\nexit 0\n")
    lpth_py.chmod(0o755)

    lpth_db = _catalog(tmp_path / "catalog.db", [])

    li_rc, ls_log = _run(tmp_path, lpth_db, lpth_venv)

    assert 0 == li_rc
    assert f"python3={lpth_py}" in ls_log
    assert "virtualenv inexistente" not in ls_log

# -------------------------------------------------------------------------------------
def test_sem_virtualenv_avisa_e_segue(tmp_path):
    lpth_db = _catalog(tmp_path / "catalog.db", [])

    li_rc, ls_log = _run(tmp_path, lpth_db)

    assert 0 == li_rc
    assert "[warn]: virtualenv inexistente" in ls_log

# -------------------------------------------------------------------------------------
def test_cron_le_unt_env_sob_ambiente_minimo(tmp_path):
    # cópia dos scripts para poder criar um unt_env.sh sem tocar no repo
    lpth_code = tmp_path / "code"
    lpth_code.mkdir()
    for lpth in CS_ROOT.glob("unt_*.sh"):
        shutil.copy(lpth, lpth_code)

    lpth_db = _catalog(tmp_path / "catalog.db", [("CGU", 250, 1000)])
    (lpth_code / "unt_env.sh").write_text(f"UNT_PATH={tmp_path / 'ofl'}\n"
                                          f"UNT_CATALOG_DB={lpth_db}\n"
                                          f"UNT_CAPPI_DIR={tmp_path / 'cappi'}\n"
                                          f"UNT_VENV={tmp_path / 'sem_venv'}\n")

    # env vazio, como o cron (sem USER/HOME/UNT_*)
    lproc = subprocess.run(["/bin/bash", str(lpth_code / "unt_cron.sh")],
                           env={}, capture_output=True, text=True)

    assert 0 == lproc.returncode, lproc.stderr
    ls_log = (tmp_path / "ofl/logs/unt_main.log").read_text()
    assert f"catálogo={lpth_db}" in ls_log
    assert "varredura concluída (1 pares)" in ls_log

# -------------------------------------------------------------------------------------
def test_no_main_carrega_as_funcoes_sem_executar(tmp_path):
    ldct_env = {lk: lv for lk, lv in os.environ.items()
                if not lk.startswith(("UNT_", "MET_"))}
    ldct_env.update({"UNT_PATH": str(tmp_path / "ofl"), "UNT_RUN_NO_MAIN": "true"})

    lproc = subprocess.run(["bash", "-c", f'source "{CS_ROOT / "unt_run.sh"}"; '
                                          "type -t main processa_todos_os_pares"],
                           env=ldct_env, capture_output=True, text=True)

    assert 0 == lproc.returncode, lproc.stderr
    assert ["function", "function"] == lproc.stdout.split()
    # nada criado: nem a árvore de dados, nem log, nem lock
    assert not (tmp_path / "ofl").exists()

# < the end >----------------------------------------------------------------------------
