# -*- coding: utf-8 -*-
"""
Centro Logístico de Simulação Meteorológica
Copyright (c) 1995-2026, Instituto de Controle do Espaço Aéreo

test_unt_verify
testes unitários de unt_verify (POD/FAR/CSI, MAE/bias, exclusão dos pixels
fora da cobertura e escolha do passo previsto que corresponde ao target)

2026.out  mlabru  portado de 25.oflow.

2026.out  mlabru  nomes novos das funções testadas (refatoração de legibilidade)

2026.out  mlabru  initial version (Linux/Python)
"""
# < imports >----------------------------------------------------------------------------

# python library
import json

# pytest
import pytest

# numPy
import numpy as np

# model
import work.unt_verify as ov

# < helpers >----------------------------------------------------------------------------

def _campo(fl_eco, fn=4):
    """
    campo fn x fn em -32 dBZ (sem eco), com 40 dBZ nas posições fl_eco.
    """
    lnp = np.full((fn, fn), -32.0)
    for lt in fl_eco:
        lnp[lt] = 40.0
    return lnp

# < calcula_passo_do_target >------------------------------------------------------------

def test_calcula_passo_do_target():
    assert 1 == ov.calcula_passo_do_target(1000, 1600, 10)
    assert 2 == ov.calcula_passo_do_target(1000, 2190, 10)
    # nunca menor que 1
    assert 1 == ov.calcula_passo_do_target(1000, 1000, 10)

# < calcula_metricas >-------------------------------------------------------------------

def test_calcula_metricas_contagem_e_razoes():
    # obs: eco em (0,0) e (0,1); prev: eco em (0,1) e (1,1)
    ldct = ov.calcula_metricas(_campo([(0, 1), (1, 1)]), _campo([(0, 0), (0, 1)]))
    ld = ldct["limiares"]["36"]

    assert (1, 1, 1) == (ld["hits"], ld["misses"], ld["false_alarms"])
    assert 0.5 == ld["pod"]
    assert 0.5 == ld["far"]
    assert 1.0 / 3.0 == ld["csi"]
    assert 16 == ldct["pixels"]

# -------------------------------------------------------------------------------------
def test_calcula_metricas_ignora_pixels_fora_da_cobertura():
    lnp_obs = _campo([(0, 0)])
    lnp_obs[3, :] = np.nan
    # falso alarme só na linha fora da cobertura -> não conta
    lnp_prev = _campo([(0, 0), (3, 0)])

    ld = ov.calcula_metricas(lnp_prev, lnp_obs)["limiares"]["36"]

    assert (1, 0, 0) == (ld["hits"], ld["misses"], ld["false_alarms"])
    assert 1.0 == ld["csi"]

# -------------------------------------------------------------------------------------
def test_calcula_metricas_sem_evento_retorna_none_e_trata_menos_inf():
    # previsão R=0 convertida em dBZ vira -inf; obs sem eco
    lnp_prev = np.full((3, 3), -np.inf)
    ldct = ov.calcula_metricas(lnp_prev, np.full((3, 3), -32.0))

    assert ldct["limiares"]["20"]["pod"] is None
    assert ldct["limiares"]["20"]["csi"] is None
    assert ldct["mae_dbz"] is None
    assert -32.0 == ldct["max_prev"]

# -------------------------------------------------------------------------------------
def test_calcula_metricas_mae_e_bias_so_sobre_eco():
    lnp_obs = _campo([(0, 0)])
    lnp_prev = lnp_obs.copy()
    lnp_prev[0, 0] = 30.0

    ldct = ov.calcula_metricas(lnp_prev, lnp_obs)

    assert 1 == ldct["pixels_eco"]
    assert 10.0 == ldct["mae_dbz"]
    assert -10.0 == ldct["bias_dbz"]

# -------------------------------------------------------------------------------------
def test_calcula_metricas_mae_eco_perdido_custa_ate_o_limiar_de_eco():
    # eco de 40 dBZ observado, previsão sem eco (-32): erro = 40 - 10, não 72
    ldct = ov.calcula_metricas(_campo([]), _campo([(0, 0)]))

    assert 30.0 == ldct["mae_dbz"]
    assert -30.0 == ldct["bias_dbz"]

# < verifica_previsoes / grava_json >----------------------------------------------------

def test_verifica_previsoes_por_backend_mais_persistencia(tmp_path):
    lnp_obs = _campo([(1, 1)])
    lnp_zf = np.stack([_campo([(1, 1)]), _campo([(2, 2)])])

    ldct = ov.verifica_previsoes({"opencv": lnp_zf, "curto": lnp_zf[:1]}, _campo([(0, 0)]), lnp_obs, 2)

    # passo 2 previu (2,2) -> erro; "curto" não alcança o passo 2 -> omitido
    assert 0.0 == ldct["backends"]["opencv"]["limiares"]["36"]["csi"]
    assert "curto" not in ldct["backends"]
    assert 0.0 == ldct["persistencia"]["limiares"]["36"]["csi"]
    assert 2 == len(ov.formata_resumo(ldct))

    ov.grava_json(ldct, tmp_path / "v.json")
    assert 2 == json.loads((tmp_path / "v.json").read_text())["lead"]

# < calcula_fss >------------------------------------------------------------------------

def _campo_com_nucleo(fi_linha, fi_coluna, fi_lado=40, fi_nucleo=6):
    lnp = np.full((fi_lado, fi_lado), -32.0, dtype=np.float32)
    lnp[fi_linha:fi_linha + fi_nucleo, fi_coluna:fi_coluna + fi_nucleo] = 45.0
    return lnp

# -------------------------------------------------------------------------------------
def test_calcula_fss_campo_identico_vale_1():
    lnp = _campo_com_nucleo(10, 10)
    assert ov.calcula_fss(lnp, lnp, 30.0, 5) == pytest.approx(1.0)

# -------------------------------------------------------------------------------------
def test_calcula_fss_disjunto_em_escala_1_vale_0_e_melhora_com_a_escala():
    lnp_prev = _campo_com_nucleo(5, 5)
    lnp_obs = _campo_com_nucleo(5, 13)

    assert ov.calcula_fss(lnp_prev, lnp_obs, 30.0, 1) == pytest.approx(0.0)
    assert ov.calcula_fss(lnp_prev, lnp_obs, 30.0, 15) > ov.calcula_fss(lnp_prev, lnp_obs, 30.0, 5)

# -------------------------------------------------------------------------------------
def test_calcula_metricas_fss_somas_reproduzem_o_fss():
    lnp_obs = np.full((40, 40), -32.0, dtype=np.float32)
    lnp_prev = lnp_obs.copy()
    lnp_obs[10:20, 10:20] = 40.0
    lnp_prev[14:24, 12:22] = 40.0
    ldct = ov.calcula_metricas(lnp_prev, lnp_obs)
    for ls_escala in ("1", "5", "10", "20"):
        ldct_somas = ldct["fss_somas"]["30"][ls_escala]
        assert ldct["fss"]["30"][ls_escala] == pytest.approx(
            1.0 - ldct_somas["erro"] / ldct_somas["referencia"])
        assert ldct["fss"]["30"][ls_escala] == pytest.approx(
            ov.calcula_fss(lnp_prev, lnp_obs, 30.0, int(ls_escala)))

# -------------------------------------------------------------------------------------
def test_calcula_fss_sem_evento_retorna_none_e_ignora_fora_da_cobertura():
    lnp = np.full((20, 20), -32.0, dtype=np.float32)
    assert ov.calcula_fss(lnp, lnp, 30.0, 5) is None

    # eco previsto só fora da cobertura: não conta
    lnp_obs = lnp.copy()
    lnp_obs[:, :10] = np.nan
    lnp_prev = lnp.copy()
    lnp_prev[:, :10] = 50.0
    assert ov.calcula_fss(lnp_prev, lnp_obs, 30.0, 3) is None

# -------------------------------------------------------------------------------------
def test_calcula_metricas_inclui_fss_e_freq_bias():
    lnp = _campo_com_nucleo(10, 10)
    ldct = ov.calcula_metricas(lnp, lnp)

    assert ldct["fss"]["30"]["10"] == pytest.approx(1.0)
    assert set(ldct["fss"]) == {"30", "36"}
    assert ldct["limiares"]["30"]["freq_bias"] == pytest.approx(1.0)

# -------------------------------------------------------------------------------------
def test_verifica_lead_alem_do_horizonte():
    lnp = np.stack([_campo_com_nucleo(10, 10)] * 2)
    assert ov.verifica_lead(lnp, lnp[0], 3) is None
    assert ov.verifica_lead(lnp, lnp[0], 2)["fss"]["30"]["1"] == pytest.approx(1.0)

# < the end >----------------------------------------------------------------------------
