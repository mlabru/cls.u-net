# -*- coding: utf-8 -*-
"""
Centro Logístico de Simulação Meteorológica
Copyright (c) 1995-2026, Instituto de Controle do Espaço Aéreo

test_untListSeq
testes unitários de untListSeq (arg_parse, avalia_sequencia, formata_duracao e
main() sobre um catalog.db temporário)

2026.out  mlabru  portado de 25.oflow.

2026.out  mlabru  nomes novos das funções testadas (refatoração de legibilidade)

2026.ago  mlabru  initial version (Linux/Python)
"""
# < imports >----------------------------------------------------------------------------

# python library
import sqlite3
import sys
import time

# numPy
import numpy as np

# model
import untListSeq as ols
import work.unt_catalog as oc

# < arg_parse >--------------------------------------------------------------------------

def test_arg_parse_defaults(monkeypatch):
    monkeypatch.delenv("UNT_CATALOG_DB", raising=False)
    monkeypatch.delenv("UNT_CAPPI_DIR", raising=False)
    monkeypatch.setattr(sys, "argv", ["untListSeq.py"])

    largs = ols.arg_parse()

    assert largs.radar is None
    assert largs.alcance is None
    assert not largs.all
    assert "./data/dbase/catalog.db" == largs.catalog
    assert "./data/cappi" == largs.cappi_dir
    assert ols.CI_IDADE_MAXIMA_PADRAO_SEGUNDOS == largs.max_age

# -------------------------------------------------------------------------------------
def test_arg_parse_env_vars_definem_defaults(monkeypatch):
    monkeypatch.setenv("UNT_CATALOG_DB", "/env/catalog.db")
    monkeypatch.setenv("UNT_CAPPI_DIR", "/env/cappi")
    monkeypatch.setattr(sys, "argv", ["untListSeq.py"])

    largs = ols.arg_parse()

    assert "/env/catalog.db" == largs.catalog
    assert "/env/cappi" == largs.cappi_dir

# < formata_duracao >--------------------------------------------------------------------

def test_formata_duracao_sem_dias():
    assert "01h05m" == ols.formata_duracao(3900)

# -------------------------------------------------------------------------------------
def test_formata_duracao_com_dias():
    assert "2d01h00m" == ols.formata_duracao(2 * 86400 + 3600)

# -------------------------------------------------------------------------------------
def test_formata_duracao_negativa():
    assert "-01h00m" == ols.formata_duracao(-3600)

# < fixtures do catálogo >---------------------------------------------------------------

def _make_catalog(tmp_path, fi_alcance=400, fi_timestamps=(10, 20, 30, 40),
                  fi_target=None, fv_com_geometria=True):
    """
    monta um catalog.db temporário (schema mínimo) com uma única sequência
    para (CGU, fi_alcance).
    """
    lpth_db = tmp_path / "catalog.db"
    lconn = sqlite3.connect(lpth_db)

    lconn.execute("""CREATE TABLE sequences (
                         radar TEXT, alcance INTEGER,
                         input_1 INTEGER, input_2 INTEGER,
                         input_3 INTEGER, input_4 INTEGER,
                         target INTEGER)""")
    lconn.execute("""CREATE TABLE radar_sites (
                         radar TEXT PRIMARY KEY,
                         latitude REAL, longitude REAL, altitude REAL)""")

    li_target = fi_target if fi_target is not None else fi_timestamps[-1] + 10

    lconn.execute("INSERT INTO sequences VALUES (?, ?, ?, ?, ?, ?, ?)",
                  ("CGU", fi_alcance, *fi_timestamps, li_target))

    if fv_com_geometria:
        lconn.execute("INSERT INTO radar_sites VALUES (?, ?, ?, ?)",
                      ("CGU", -23.5, -46.6, 760.0))

    lconn.commit()
    lconn.close()

    return lpth_db

# -------------------------------------------------------------------------------------
def _make_cappi_files(tmp_path, radar, alcance, timestamps, shape=(4, 4)):
    """
    grava um .npy por timestamp seguindo a convenção de nome esperada.
    """
    lpth_cappi_dir = tmp_path / "cappi"
    lpth_cappi_dir.mkdir()

    for li_ts in timestamps:
        lnp = np.full(shape, float(li_ts), dtype=np.float32)
        np.save(lpth_cappi_dir / f"{radar}_{alcance}_{li_ts}.npy", lnp)

    return lpth_cappi_dir

# < avalia_sequencia >-------------------------------------------------------------------

def test_avalia_sequencia_valida(tmp_path):
    ltimestamps = (10, 20, 30, 40)
    li_target = 50
    lpth_db = _make_catalog(tmp_path, fi_timestamps=ltimestamps, fi_target=li_target)
    lpth_cappi_dir = _make_cappi_files(tmp_path, "CGU", 400, ltimestamps)

    lconn = oc.abre_catalogo_somente_leitura(str(lpth_db))
    ldct_seq = oc.busca_sequencia_mais_recente(lconn, "CGU", 400)

    lv_valida, llst_motivos = ols.avalia_sequencia(lconn, ldct_seq, str(lpth_cappi_dir),
                                                 fi_idade_maxima=3600, fi_agora=li_target + 60)

    assert lv_valida
    assert [] == llst_motivos

# -------------------------------------------------------------------------------------
def test_avalia_sequencia_cappi_ausente(tmp_path):
    ltimestamps = (10, 20, 30, 40)
    lpth_db = _make_catalog(tmp_path, fi_timestamps=ltimestamps)
    lpth_cappi_dir = tmp_path / "cappi"
    lpth_cappi_dir.mkdir()

    lconn = oc.abre_catalogo_somente_leitura(str(lpth_db))
    ldct_seq = oc.busca_sequencia_mais_recente(lconn, "CGU", 400)

    lv_valida, llst_motivos = ols.avalia_sequencia(lconn, ldct_seq, str(lpth_cappi_dir),
                                                 fi_idade_maxima=3600, fi_agora=ldct_seq["target"])

    assert not lv_valida
    assert any("CAPPI ausente" in ls_motivo for ls_motivo in llst_motivos)

# -------------------------------------------------------------------------------------
def test_avalia_sequencia_sem_geometria(tmp_path):
    ltimestamps = (10, 20, 30, 40)
    lpth_db = _make_catalog(tmp_path, fi_timestamps=ltimestamps, fv_com_geometria=False)
    lpth_cappi_dir = _make_cappi_files(tmp_path, "CGU", 400, ltimestamps)

    lconn = oc.abre_catalogo_somente_leitura(str(lpth_db))
    ldct_seq = oc.busca_sequencia_mais_recente(lconn, "CGU", 400)

    lv_valida, llst_motivos = ols.avalia_sequencia(lconn, ldct_seq, str(lpth_cappi_dir),
                                                 fi_idade_maxima=3600, fi_agora=ldct_seq["target"])

    assert not lv_valida
    assert any("geometria" in ls_motivo for ls_motivo in llst_motivos)

# -------------------------------------------------------------------------------------
def test_avalia_sequencia_expirada(tmp_path):
    ltimestamps = (10, 20, 30, 40)
    lpth_db = _make_catalog(tmp_path, fi_timestamps=ltimestamps)
    lpth_cappi_dir = _make_cappi_files(tmp_path, "CGU", 400, ltimestamps)

    lconn = oc.abre_catalogo_somente_leitura(str(lpth_db))
    ldct_seq = oc.busca_sequencia_mais_recente(lconn, "CGU", 400)

    lv_valida, llst_motivos = ols.avalia_sequencia(lconn, ldct_seq, str(lpth_cappi_dir),
                                                 fi_idade_maxima=10, fi_agora=ldct_seq["target"] + 3600)

    assert not lv_valida
    assert any("expirada" in ls_motivo for ls_motivo in llst_motivos)

# -------------------------------------------------------------------------------------
def test_avalia_sequencia_max_age_zero_desliga_filtro_de_idade(tmp_path):
    ltimestamps = (10, 20, 30, 40)
    lpth_db = _make_catalog(tmp_path, fi_timestamps=ltimestamps)
    lpth_cappi_dir = _make_cappi_files(tmp_path, "CGU", 400, ltimestamps)

    lconn = oc.abre_catalogo_somente_leitura(str(lpth_db))
    ldct_seq = oc.busca_sequencia_mais_recente(lconn, "CGU", 400)

    # replay de dados antigos (03.dat/tst): target meses atrás, sem limite
    lv_valida, llst_motivos = ols.avalia_sequencia(lconn, ldct_seq, str(lpth_cappi_dir),
                                                 fi_idade_maxima=0,
                                                 fi_agora=ldct_seq["target"] + 90 * 86400)

    assert lv_valida
    assert [] == llst_motivos

# -------------------------------------------------------------------------------------
def test_avalia_sequencia_layout_com_preset(tmp_path):
    ltimestamps = (10, 20, 30, 40)
    lpth_db = tmp_path / "catalog.db"
    lconn = sqlite3.connect(lpth_db)
    lconn.execute("""CREATE TABLE sequences (
                         radar TEXT, alcance INTEGER,
                         input_1 INTEGER, input_2 INTEGER, input_3 INTEGER,
                         input_4 INTEGER, target INTEGER, quality INTEGER,
                         preset TEXT)""")
    lconn.execute("""CREATE TABLE radar_sites (
                         radar TEXT PRIMARY KEY,
                         latitude REAL, longitude REAL, altitude REAL)""")
    lconn.execute("INSERT INTO sequences VALUES ('STI', 250, ?, ?, ?, ?, 50, 0, 'nwc250v2')",
                  ltimestamps)
    lconn.execute("INSERT INTO radar_sites VALUES ('STI', -29.2, -54.9, 430.0)")
    lconn.commit()
    lconn.close()

    # layout atual do 27.nowcast: <cappi>/<preset>/<R>_<A>/<R>_<A>_<ts>.npy
    lpth_dir = tmp_path / "cappi" / "nwc250v2" / "STI_250"
    lpth_dir.mkdir(parents=True)
    for li_ts in ltimestamps:
        np.save(lpth_dir / f"STI_250_{li_ts}.npy", np.zeros((4, 4), dtype=np.float32))

    lconn = oc.abre_catalogo_somente_leitura(str(lpth_db))
    ldct_seq = oc.busca_sequencia_mais_recente(lconn, "STI", 250)

    lv_valida, llst_motivos = ols.avalia_sequencia(lconn, ldct_seq, str(tmp_path / "cappi"),
                                                 fi_idade_maxima=0, fi_agora=60)

    assert lv_valida, llst_motivos

# < main >-------------------------------------------------------------------------------

def test_main_sem_sequencias_retorna_1(tmp_path, monkeypatch):
    lpth_db = tmp_path / "catalog.db"
    lconn = sqlite3.connect(lpth_db)
    lconn.execute("""CREATE TABLE sequences (
                         radar TEXT, alcance INTEGER,
                         input_1 INTEGER, input_2 INTEGER,
                         input_3 INTEGER, input_4 INTEGER,
                         target INTEGER)""")
    lconn.execute("""CREATE TABLE radar_sites (
                         radar TEXT PRIMARY KEY,
                         latitude REAL, longitude REAL, altitude REAL)""")
    lconn.commit()
    lconn.close()

    monkeypatch.setattr(sys, "argv", ["untListSeq.py", "--catalog", str(lpth_db)])

    assert 1 == ols.main()

# -------------------------------------------------------------------------------------
def test_main_sequencia_valida_retorna_0(tmp_path, monkeypatch, capsys):
    ltimestamps = (10, 20, 30, 40)
    li_target = int(time.time())
    lpth_db = _make_catalog(tmp_path, fi_timestamps=ltimestamps, fi_target=li_target)
    lpth_cappi_dir = _make_cappi_files(tmp_path, "CGU", 400, ltimestamps)

    monkeypatch.setattr(sys, "argv",
                        ["untListSeq.py", "--catalog", str(lpth_db),
                         "--cappi-dir", str(lpth_cappi_dir)])

    assert 0 == ols.main()

    ls_saida = capsys.readouterr().out
    assert "CGU" in ls_saida
    assert "1/1 sequência(s) válida(s)" in ls_saida

# -------------------------------------------------------------------------------------
def test_main_filtra_por_radar_e_alcance_sem_correspondencia(tmp_path, monkeypatch):
    ltimestamps = (10, 20, 30, 40)
    lpth_db = _make_catalog(tmp_path, fi_alcance=400, fi_timestamps=ltimestamps)
    lpth_cappi_dir = _make_cappi_files(tmp_path, "CGU", 400, ltimestamps)

    monkeypatch.setattr(sys, "argv",
                        ["untListSeq.py", "--catalog", str(lpth_db),
                         "--cappi-dir", str(lpth_cappi_dir), "-a", "250"])

    assert 1 == ols.main()

# -------------------------------------------------------------------------------------
def test_main_all_lista_todas_as_sequencias_do_par(tmp_path, monkeypatch, capsys):
    lpth_db = tmp_path / "catalog.db"
    lconn = sqlite3.connect(lpth_db)
    lconn.execute("""CREATE TABLE sequences (
                         radar TEXT, alcance INTEGER,
                         input_1 INTEGER, input_2 INTEGER,
                         input_3 INTEGER, input_4 INTEGER,
                         target INTEGER)""")
    lconn.execute("""CREATE TABLE radar_sites (
                         radar TEXT PRIMARY KEY,
                         latitude REAL, longitude REAL, altitude REAL)""")
    lconn.execute("INSERT INTO sequences VALUES ('CGU', 400, 10, 20, 30, 40, 50)")
    lconn.execute("INSERT INTO sequences VALUES ('CGU', 400, 20, 30, 40, 50, 60)")
    lconn.execute("INSERT INTO radar_sites VALUES ('CGU', -23.5, -46.6, 760.0)")
    lconn.commit()
    lconn.close()

    monkeypatch.setattr(sys, "argv", ["untListSeq.py", "--catalog", str(lpth_db), "--all"])

    ols.main()

    ls_saida = capsys.readouterr().out
    assert "2/2 sequência(s)" in ls_saida or "0/2 sequência(s)" in ls_saida

# < the end >----------------------------------------------------------------------------
