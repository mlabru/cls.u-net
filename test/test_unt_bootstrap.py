# -*- coding: utf-8 -*-
"""
Centro Logístico de Simulação Meteorológica
Copyright (c) 1995-2026, Instituto de Controle do Espaço Aéreo

test_unt_bootstrap
testes do bootstrap por bloco: pesos de reamostragem, reagregação das somas (CSI e FSS),
intervalo, diferença pareada e o resumo gravado no comparacao.json

2026.out  mlabru  initial version (Linux/Python)
"""
# < imports >----------------------------------------------------------------------------

# numPy
import numpy as np

# pytest
import pytest

# model
import work.unt_bootstrap as ub

# < helpers >----------------------------------------------------------------------------

def _passo(fi_hits, fi_misses, fi_falsos, ff_erro, ff_referencia):
    return {"janelas": 1,
            "limiares": {"30": {"hits": fi_hits, "misses": fi_misses,
                                "false_alarms": fi_falsos}},
            "fss_somas": {"30": {"20": {"erro": ff_erro, "referencia": ff_referencia}}}}


def _somas(flst_por_bloco):
    # flst_por_bloco: [{rótulo: _passo(...)}, ...], um passo ("1")
    return {str(li): {ls_rotulo: {"1": ldct} for ls_rotulo, ldct in ldct_bloco.items()}
            for li, ldct_bloco in enumerate(flst_por_bloco)}

# < testes >-----------------------------------------------------------------------------

def test_pesos_reamostrados_somam_os_blocos_e_sao_reprodutiveis():
    lnp = ub.pesos_reamostrados(7, 100, 42)
    assert (100, 7) == lnp.shape
    assert np.all(7 == lnp.sum(axis=1))
    assert np.array_equal(lnp, ub.pesos_reamostrados(7, 100, 42))


def test_distribuicao_soma_antes_da_razao():
    ldct = _somas([{"a": _passo(10, 0, 0, 1.0, 10.0)}, {"a": _passo(0, 10, 10, 2.0, 2.0)}])
    # pesos fixos: cada bloco uma vez -> o agregado do conjunto todo
    lnp_pesos = np.ones((1, 2))
    assert ub.distribuicao(ldct, "a", "1", ("csi", "30", None), lnp_pesos)[0] == \
        pytest.approx(10 / 30)
    assert ub.distribuicao(ldct, "a", "1", ("fss", "30", "20"), lnp_pesos)[0] == \
        pytest.approx(1 - 3 / 12)


def test_intervalo_e_metrica_indefinida():
    assert ub.intervalo(np.array([np.nan, np.nan]), 0.9) is None
    llst = ub.intervalo(np.arange(101, dtype=float), 0.9)
    assert llst == pytest.approx([5.0, 95.0])


def test_compara_pareado_e_resumo():
    # o rótulo "a" ganha da persistência em todos os blocos
    ldct = _somas([{"a": _passo(6, 4, 0, 0.2, 1.0),
                    ub.CS_PERSISTENCIA: _passo(4, 6, 0, 0.4, 1.0)}] * 6)
    lnp_pesos = ub.pesos_reamostrados(6, 200, 1)
    ldct_dif = ub.compara(ldct, "a", ub.CS_PERSISTENCIA, "1", ("csi", "30", None),
                          lnp_pesos, 0.9)
    assert ldct_dif["ic"] == pytest.approx([0.2, 0.2])
    assert 1.0 == ldct_dif["p_melhor"]

    ldct_resumo = ub.resume(ldct, 200, 0.9, 1)
    assert 6 == ldct_resumo["blocos"]
    assert ldct_resumo["resultados"]["a"]["1"]["ic"]["fss"]["30"]["20"] == pytest.approx([0.8, 0.8])
    assert "dif_persistencia" in ldct_resumo["resultados"]["a"]["1"]
    assert "dif_persistencia" not in ldct_resumo["resultados"][ub.CS_PERSISTENCIA]["1"]
    assert {"blocos": 0}.items() <= ub.resume({}, 10, 0.9, 1).items()
