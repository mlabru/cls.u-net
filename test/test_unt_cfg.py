# -*- coding: utf-8 -*-
"""
Centro Logístico de Simulação Meteorológica
Copyright (c) 1995-2026, Instituto de Controle do Espaço Aéreo

test_unt_cfg
testes de unt_cfg.sh: defaults de produção, sobreposição por env var e trava
do ambiente de teste (CLS_ENV=test exige tudo dentro de CLS_TST_ROOT),
mesmo padrão de 21.mrad/test/test_rsm_cfg.py

2026.out  mlabru  portado de 25.oflow.

2026.out  mlabru  initial version (Linux/Python)
"""
# < imports >----------------------------------------------------------------------------

# python library
import os
import pathlib
import shutil
import subprocess

# pytest
import pytest

# < constants >--------------------------------------------------------------------------

CS_CFG = pathlib.Path(__file__).resolve().parent.parent / "unt_cfg.sh"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="bash ausente")

# variáveis impressas após o source
CLST_VARS = ("UNT_LIB_DIR", "UNT_PATH", "UNT_CATALOG_DB", "UNT_CAPPI_DIR",
             "UNT_AGE_MAX_SECS", "UNT_TIMEOUT_SECS", "UNT_VERIFY", "UNT_MAPA_CACHE",
             "UNT_VENV", "UNT_DB_TIMEOUT_MS")

# < helpers >----------------------------------------------------------------------------

def _source(fdct_env: dict) -> tuple:
    """
    faz source de unt_cfg.sh num ambiente limpo (sem UNT_*/MET_*) mais
    fdct_env e retorna (rc, {var: valor}, stderr).
    """
    ldct_env = {lk: lv for lk, lv in os.environ.items()
                if not lk.startswith(("UNT_", "MET_"))}
    ldct_env.update(fdct_env)

    ls_echo = "; ".join(f'echo "{lv}=${{{lv}}}"' for lv in CLST_VARS)
    lproc = subprocess.run(["bash", "-c", f'source "{CS_CFG}"; {ls_echo}'],
                           env=ldct_env, capture_output=True, text=True)

    ldct = dict(ll.split("=", 1) for ll in lproc.stdout.splitlines() if "=" in ll)

    return lproc.returncode, ldct, lproc.stderr

# -------------------------------------------------------------------------------------
def _tst_env(fpth_root: pathlib.Path) -> dict:
    """
    variáveis que o env.sh do cls.test exporta para o 25.oflow.
    """
    return {"CLS_ENV": "test",
            "CLS_TST_ROOT": str(fpth_root),
            "UNT_PATH": str(fpth_root / "oflow"),
            "UNT_CATALOG_DB": str(fpth_root / "mvol/data/dbase/catalog.db"),
            "UNT_CAPPI_DIR": str(fpth_root / "nowcast/data/cappi"),
            "UNT_AGE_MAX_SECS": "0",
            "UNT_VERIFY": "true"}

# < defaults / overrides >---------------------------------------------------------------

def test_defaults_de_producao():
    li_rc, ldct, _ = _source({"USER": "fulano", "HOME": "/home/fulano"})

    assert 0 == li_rc
    assert "/home/fulano/Public/unet" == ldct["UNT_PATH"]
    assert "/home/fulano/Public/unet/data/dbase/catalog.db" == ldct["UNT_CATALOG_DB"]
    assert "/home/fulano/Public/unet/data/cappi" == ldct["UNT_CAPPI_DIR"]
    assert "7200" == ldct["UNT_AGE_MAX_SECS"]
    assert "300" == ldct["UNT_TIMEOUT_SECS"]
    assert "10000" == ldct["UNT_DB_TIMEOUT_MS"]
    assert "false" == ldct["UNT_VERIFY"]
    assert "/home/fulano/Public/unet/data/cache/mapa" == ldct["UNT_MAPA_CACHE"]
    # mesmo virtualenv do 27.nowcast
    assert "/home/fulano/.venv" == ldct["UNT_VENV"]
    # código = diretório do próprio unt_cfg.sh, não UNT_PATH
    assert str(CS_CFG.parent) == ldct["UNT_LIB_DIR"]

# -------------------------------------------------------------------------------------
def test_overrides_por_env_var(tmp_path):
    li_rc, ldct, _ = _source({"UNT_PATH": str(tmp_path / "x"),
                              "UNT_AGE_MAX_SECS": "0",
                              "UNT_TIMEOUT_SECS": "300",
                              "UNT_DB_TIMEOUT_MS": "500"})

    assert 0 == li_rc
    assert str(tmp_path / "x") == ldct["UNT_PATH"]
    # catálogo/cappi seguem UNT_PATH quando não sobrepostos
    assert str(tmp_path / "x/data/dbase/catalog.db") == ldct["UNT_CATALOG_DB"]
    assert "0" == ldct["UNT_AGE_MAX_SECS"]
    assert "300" == ldct["UNT_TIMEOUT_SECS"]
    assert "500" == ldct["UNT_DB_TIMEOUT_MS"]

# < trava CLS_ENV=test >-----------------------------------------------------------------

def test_trava_aceita_env_do_tst(tmp_path):
    li_rc, ldct, ls_err = _source(_tst_env(tmp_path))

    assert 0 == li_rc, ls_err
    assert str(tmp_path / "oflow") == ldct["UNT_PATH"]

# -------------------------------------------------------------------------------------
@pytest.mark.parametrize("fs_var", ["UNT_PATH", "UNT_CATALOG_DB", "UNT_CAPPI_DIR",
                                    "UNT_MAPA_CACHE"])
def test_trava_recusa_caminho_fora_da_raiz(tmp_path, fs_var):
    ldct_env = _tst_env(tmp_path / "tst")
    ldct_env[fs_var] = str(tmp_path / "producao" / "x")

    li_rc, _, ls_err = _source(ldct_env)

    assert 1 == li_rc
    assert "fora de" in ls_err

# -------------------------------------------------------------------------------------
def test_trava_recusa_escape_por_dot_dot(tmp_path):
    ldct_env = _tst_env(tmp_path / "tst")
    ldct_env["UNT_PATH"] = str(tmp_path / "tst" / ".." / "producao")

    assert 1 == _source(ldct_env)[0]

# -------------------------------------------------------------------------------------
def test_trava_exige_met_tst_root(tmp_path):
    ldct_env = _tst_env(tmp_path)
    del ldct_env["CLS_TST_ROOT"]

    li_rc, _, ls_err = _source(ldct_env)

    assert 1 == li_rc
    assert "CLS_TST_ROOT" in ls_err

# -------------------------------------------------------------------------------------
def test_trava_bloqueia_defaults_de_producao(tmp_path):
    # CLS_ENV=test sem nenhum UNT_*: os defaults (~/Public/unet) estão
    # fora da raiz e precisam ser recusados
    li_rc, _, _ = _source({"CLS_ENV": "test", "CLS_TST_ROOT": str(tmp_path)})

    assert 1 == li_rc

# < the end >----------------------------------------------------------------------------
