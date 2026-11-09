# -*- coding: utf-8 -*-
"""
Centro Logístico de Simulação Meteorológica
Copyright (c) 1995-2026, Instituto de Controle do Espaço Aéreo

test_unt_catalog
testes unitários de unt_catalog (caminho do CAPPI com preset e fallback
plano, filtros preset/quality, conversão do "fora da cobertura" em NaN e
meio-lado da grade)

2026.out  mlabru  portado de 25.oflow.

2026.out  mlabru  nomes novos das funções testadas (refatoração de legibilidade)

2026.out  mlabru  initial version (Linux/Python)
"""
# < imports >----------------------------------------------------------------------------

# python library
import pathlib
import sqlite3

# numPy
import numpy as np

# model
import work.unt_catalog as oc

# < monta_caminho_cappi >----------------------------------------------------------------

def test_monta_caminho_cappi_com_preset():
    lpth = oc.monta_caminho_cappi("/cappi", "STI", 250, 123, "nwc250v2")
    assert pathlib.Path("/cappi/nwc250v2/STI_250/STI_250_123.npy") == lpth

# -------------------------------------------------------------------------------------
def test_monta_caminho_cappi_sem_preset_usa_layout_plano():
    assert pathlib.Path("/cappi/CGU_400_123.npy") == oc.monta_caminho_cappi("/cappi", "CGU", 400, 123)
    assert pathlib.Path("/cappi/CGU_400_123.npy") == oc.monta_caminho_cappi("/cappi", "CGU", 400, 123, None)

# < filtros >----------------------------------------------------------------------------

def _catalog(tmp_path):
    lconn = sqlite3.connect(tmp_path / "catalog.db")
    lconn.row_factory = sqlite3.Row
    lconn.execute("""CREATE TABLE sequences (
                         radar TEXT, alcance INTEGER,
                         input_1 INTEGER, input_2 INTEGER, input_3 INTEGER,
                         input_4 INTEGER, target INTEGER, quality INTEGER,
                         preset TEXT)""")
    # (target, quality, preset)
    for li_target, li_qlt, ls_preset in ((100, 0, "nwc250v2"), (200, 1, "nwc250v2"),
                                         (300, 0, "nwc250v1")):
        lconn.execute("INSERT INTO sequences VALUES ('STI', 250, 1, 2, 3, 4, ?, ?, ?)",
                      (li_target, li_qlt, ls_preset))
    lconn.commit()
    return lconn

# -------------------------------------------------------------------------------------
def test_busca_sequencia_mais_recente_sem_filtro(tmp_path):
    assert 300 == oc.busca_sequencia_mais_recente(_catalog(tmp_path), "STI", 250)["target"]

# -------------------------------------------------------------------------------------
def test_busca_sequencia_mais_recente_filtra_preset_e_quality(tmp_path):
    lconn = _catalog(tmp_path)

    assert 200 == oc.busca_sequencia_mais_recente(lconn, "STI", 250, fs_preset="nwc250v2")["target"]
    assert 100 == oc.busca_sequencia_mais_recente(lconn, "STI", 250, fs_preset="nwc250v2",
                                         fi_quality_maxima=0)["target"]
    assert oc.busca_sequencia_mais_recente(lconn, "STI", 250, fs_preset="xx") is None

# -------------------------------------------------------------------------------------
def test_busca_sequencia_mais_recente_por_target(tmp_path):
    lconn = _catalog(tmp_path)

    assert 200 == oc.busca_sequencia_mais_recente(lconn, "STI", 250, fi_target=200)["target"]
    assert oc.busca_sequencia_mais_recente(lconn, "STI", 250, fi_target=201) is None

# -------------------------------------------------------------------------------------
def test_lista_sequencias_filtra_quality(tmp_path):
    llst = oc.lista_sequencias(_catalog(tmp_path), "STI", 250, fi_quality_maxima=0)
    assert [300, 100] == [ld["target"] for ld in llst]

# < busca_caminho_cappi_registrado >-----------------------------------------------------

def test_busca_caminho_cappi_registrado(tmp_path):
    lconn = _catalog(tmp_path)

    # catálogo sem cappi_products -> None, sem exceção
    assert oc.busca_caminho_cappi_registrado(lconn, "STI", 250, 1) is None

    lconn.execute("""CREATE TABLE cappi_products (radar TEXT, alcance INTEGER,
                         timestamp INTEGER, preset TEXT, cappi_path TEXT)""")
    lconn.execute("INSERT INTO cappi_products VALUES ('STI', 250, 1, 'nwc250v2', '/a/b.npy')")

    assert "/a/b.npy" == oc.busca_caminho_cappi_registrado(lconn, "STI", 250, 1)
    assert "/a/b.npy" == oc.busca_caminho_cappi_registrado(lconn, "STI", 250, 1, "nwc250v2")
    assert oc.busca_caminho_cappi_registrado(lconn, "STI", 250, 1, "nwc400v2") is None
    assert oc.busca_caminho_cappi_registrado(lconn, "STI", 250, 2) is None

# < carrega_cappi / calcula_raio_da_grade_km >-------------------------------------------

def test_carrega_cappi_converte_nodata_em_nan_e_mantem_sem_eco(tmp_path):
    lnp = np.array([[-9999.0, -32.0], [10.0, 45.5]], dtype=np.float32)
    np.save(tmp_path / "c.npy", lnp)

    lnp_lido = oc.carrega_cappi(tmp_path / "c.npy")

    assert np.isnan(lnp_lido[0, 0])
    assert -32.0 == lnp_lido[0, 1]
    assert np.float32 == lnp_lido.dtype
    assert 45.5 == lnp_lido[1, 1]

# -------------------------------------------------------------------------------------
def test_carrega_pilha_cappi_empilha_na_ordem(tmp_path):
    for li in range(3):
        np.save(tmp_path / f"{li}.npy", np.full((2, 2), float(li)))

    lnp = oc.carrega_pilha_cappi([tmp_path / f"{li}.npy" for li in range(3)])

    assert (3, 2, 2) == lnp.shape
    assert [0.0, 1.0, 2.0] == [float(lnp[li, 0, 0]) for li in range(3)]

# -------------------------------------------------------------------------------------
def test_calcula_raio_da_grade_km_das_grades_dos_presets():
    assert 250.0 == oc.calcula_raio_da_grade_km((500, 500))
    assert 300.0 == oc.calcula_raio_da_grade_km((4, 600, 600))

# < produtos prontos (janelas) >---------------------------------------------------------

def _catalog_produtos(tmp_path):
    lconn = sqlite3.connect(tmp_path / "catalog.db")
    lconn.row_factory = sqlite3.Row
    lconn.execute("""CREATE TABLE cappi_products (
                         radar TEXT, alcance INTEGER, timestamp INTEGER, preset TEXT,
                         origin TEXT, status INTEGER, valid_fraction REAL, rain_fraction REAL)""")
    for li_ts, ls_preset, li_status in ((300, "nwc250v2", 2), (100, "nwc250v2", 2),
                                        (200, "nwc250v2", 3), (100, "nwc250v1", 2)):
        lconn.execute("INSERT INTO cappi_products VALUES ('STI', 250, ?, ?, 'observed', ?, 0.79, 0.1)",
                      (li_ts, ls_preset, li_status))
    lconn.commit()
    return lconn

# -------------------------------------------------------------------------------------
def test_lista_produtos_prontos_so_done_do_preset_em_ordem(tmp_path):
    llst = oc.lista_produtos_prontos(_catalog_produtos(tmp_path), "STI", 250, "nwc250v2")
    assert [100, 300] == [ld["timestamp"] for ld in llst]
    assert {"timestamp", "origin", "valid_fraction", "rain_fraction"} == set(llst[0])

# -------------------------------------------------------------------------------------
def test_lista_produtos_prontos_catalogo_antigo_sem_tabela(tmp_path):
    lconn = sqlite3.connect(tmp_path / "vazio.db")
    assert [] == oc.lista_produtos_prontos(lconn, "STI", 250, "nwc250v2")
    assert [] == oc.lista_pares_com_produtos(lconn)

# -------------------------------------------------------------------------------------
def test_lista_pares_com_produtos_e_preset_ativo(tmp_path):
    assert [("STI", 250)] == oc.lista_pares_com_produtos(_catalog_produtos(tmp_path))
    assert "nwc400v2" == oc.preset_ativo_padrao(400)
    assert oc.preset_ativo_padrao(123) is None

# < the end >----------------------------------------------------------------------------
