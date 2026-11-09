# -*- coding: utf-8 -*-
"""
Centro Logístico de Simulação Meteorológica
Copyright (c) 1995-2026, Instituto de Controle do Espaço Aéreo

test_unt_backend_comum
testes unitários das peças compartilhadas pelos backends (conversões Z-R,
preparo/validação da entrada, caixa geográfica, metadados e gravação dos
frames)

2026.out  mlabru  portado de 25.oflow.

2026.out  mlabru  initial version (Linux/Python)
"""
# < imports >----------------------------------------------------------------------------

# python library
import pathlib

# numPy
import numpy as np

# pytest
import pytest

# model
import work.unt_backend_comum as obc

# < fixtures >---------------------------------------------------------------------------

@pytest.fixture
def ft_radar():
    # (latitude, longitude, altitude, raio do mapa em km)
    return obc.GeometriaRadar(-30.0, -52.0, 200.0, 240.0)

# < conversão Z-R >----------------------------------------------------------------------

def test_conversao_dbz_mm_h_ida_e_volta():
    lnp_dbz = np.array([0., 10., 20., 30., 40., 50.])

    lnp_mm_h = obc.converte_dbz_para_mm_h(lnp_dbz)

    assert np.allclose(lnp_dbz, obc.converte_mm_h_para_dbz(lnp_mm_h), atol=1e-6)

# -------------------------------------------------------------------------------------
def test_zero_dbz_nao_e_zero_mm_h():
    # Z = 200 * R^1.6; em 0 dBZ, Z = 1 => R = (1/200)^(1/1.6), pequeno mas > 0
    assert obc.converte_dbz_para_mm_h(np.array([0.]))[0] > 0.

# -------------------------------------------------------------------------------------
def test_zero_mm_h_vira_menos_infinito_sem_aviso():
    with np.errstate(all="raise"):
        assert -np.inf == obc.converte_mm_h_para_dbz(np.array([0.]))[0]

# < preparação e validação >-------------------------------------------------------------

def test_frame_unico_vira_sequencia_de_copias():
    lnp_frame = np.arange(4.).reshape(2, 2)

    lnp_sequencia = obc.converte_frame_unico_em_sequencia(lnp_frame, fi_num_copias=3)

    assert (3, 2, 2) == lnp_sequencia.shape
    assert all(np.array_equal(lnp_frame, lnp) for lnp in lnp_sequencia)

# -------------------------------------------------------------------------------------
def test_sequencia_3d_nao_e_alterada():
    lnp_sequencia = np.zeros((4, 2, 2))

    assert lnp_sequencia is obc.converte_frame_unico_em_sequencia(lnp_sequencia, 2)

# -------------------------------------------------------------------------------------
def test_valida_refletividade_recusa_tudo_nan():
    with pytest.raises(ValueError, match="não finitos"):
        obc.valida_refletividade(np.full((2, 3, 3), np.nan))

# -------------------------------------------------------------------------------------
def test_existe_chuva():
    assert obc.existe_chuva(np.array([[0., 0.1]]))
    assert not obc.existe_chuva(np.zeros((2, 2)))

# -------------------------------------------------------------------------------------
def test_substitui_invalidos_por_zero():
    lnp = obc.substitui_invalidos_por_zero(np.array([np.nan, np.inf, -np.inf, 2.]))

    assert [0., 0., 0., 2.] == lnp.tolist()

# < geografia e metadados >--------------------------------------------------------------

def test_caixa_geografica_no_equador_e_simetrica():
    lt_radar = obc.GeometriaRadar(0.0, 0.0, 0.0, obc.CF_KM_POR_GRAU)

    lt_caixa = obc.calcula_caixa_geografica(lt_radar)

    # 111.32 km = 1 grau nos dois eixos (cos(0) = 1)
    assert np.allclose([-1., 1., -1., 1.], lt_caixa.como_extensao())

# -------------------------------------------------------------------------------------
def test_caixa_geografica_longitude_cresce_com_a_latitude(ft_radar):
    lt_caixa = obc.calcula_caixa_geografica(ft_radar)

    lf_largura_lng = lt_caixa.lng_max - lt_caixa.lng_min
    lf_altura_lat = lt_caixa.lat_max - lt_caixa.lat_min

    # a 30° S o grau de longitude é menor (em km) -> mais graus para o mesmo raio
    assert lf_largura_lng > lf_altura_lat

# -------------------------------------------------------------------------------------
def test_metadados_descrevem_a_grade(ft_radar):
    ldct = obc.monta_metadados(ft_radar, np.zeros((4, 20, 30)))

    assert (30, 20) == (ldct["xsize"], ldct["ysize"])
    assert (obc.CF_ZR_A, obc.CF_ZR_B) == (ldct["zr_a"], ldct["zr_b"])
    assert np.isclose(ldct["xpixelsize"] * 30, ldct["x2"] - ldct["x1"])

# < frames >-----------------------------------------------------------------------------

def test_esconde_valores_baixos_nao_altera_o_original():
    lnp = np.array([[5., 10., 40.]])

    lnp_exibido = obc.esconde_valores_baixos(lnp)

    assert np.isnan(lnp_exibido[0, 0])
    assert [10., 40.] == lnp_exibido[0, 1:].tolist()
    assert 5. == lnp[0, 0]

# -------------------------------------------------------------------------------------
def test_salva_frames_grava_um_png_por_passo_com_funcao_de_desenho(ft_radar, tmp_path):
    lnp_previsao = np.full((2, 10, 10), 30.0)
    ldct_meta = obc.monta_metadados(ft_radar, lnp_previsao)
    ls_prefixo = str(tmp_path / "nowcast")
    llst_titulos = []

    def _desenha(fax, fnp, fdct, ft_caixa, fs_titulo):
        llst_titulos.append(fs_titulo)
        return obc.desenha_campo_com_imshow(fax, fnp, fdct, ft_caixa, fs_titulo)

    obc.salva_frames_da_previsao(lnp_previsao, ft_radar, ldct_meta, "teste", 10, ls_prefixo,
                                 ff_desenha_campo=_desenha)

    assert pathlib.Path(f"{ls_prefixo}_01.png").exists()
    assert pathlib.Path(f"{ls_prefixo}_02.png").exists()
    assert ["Nowcast +10 min (Raio 240km) [teste]",
            "Nowcast +20 min (Raio 240km) [teste]"] == llst_titulos

# < the end >----------------------------------------------------------------------------
