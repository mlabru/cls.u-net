# -*- coding: utf-8 -*-
"""
Centro Logístico de Simulação Meteorológica
Copyright (c) 1995-2026, Instituto de Controle do Espaço Aéreo

test_untMonitor_untLatencia
testes do monitor de skill em operação (verify.jsonl sintéticos gravados com o
próprio unt_verify: janela móvel, contingência somada, alarme só com eventos
suficientes, linha corrompida, <R>_<A>_monitor.json, códigos de saída) e do
orçamento de latência (medição mínima e estimativas)

2026.out  mlabru  initial version (Linux/Python)
"""
# < imports >----------------------------------------------------------------------------

# python library
import json

# pytest
import pytest

# numPy
import numpy as np

pytest.importorskip("torch")

# model
import untLatencia as nl
import untMonitor as nm
import work.unt_verify as ov

# < helpers >----------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _sem_config_do_repo(tmp_path, monkeypatch):
    # os testes não podem depender do config.ini do repositório
    monkeypatch.chdir(tmp_path)


def _campo(fi_coluna, ff_dbz=40.0):
    lnp = np.full((40, 40), -32.0, dtype=np.float32)
    lnp[15:25, fi_coluna:fi_coluna + 10] = ff_dbz
    return lnp


def _linha(fi_target, fb_com_evento=True):
    # observado: núcleo na coluna 12; persistência: deslocado 2 px (razoável);
    # "bom" acerta em cheio; "ruim" não prevê chuva nenhuma
    lnp_obs = _campo(12) if fb_com_evento else _campo(12, -32.0)
    ldct_prev = {"bom": lnp_obs[None], "ruim": _campo(0, -32.0)[None]}

    ldct = ov.verifica_previsoes(ldct_prev, _campo(10), lnp_obs, 1)
    ldct.update({"radar": "STI", "alcance": 250, "target": fi_target})
    return ldct


def _historico(fpth_dir, flst_linhas, fs_par="STI_250"):
    fpth_dir.mkdir(parents=True, exist_ok=True)
    lpth = fpth_dir / f"{fs_par}{nm.CS_SUFIXO_VERIFY}"
    lpth.write_text("".join(json.dumps(ld) + "\n" for ld in flst_linhas))
    return lpth

# < untMonitor >-------------------------------------------------------------------------

def test_monitor_alarma_backend_abaixo_da_persistencia(tmp_path):
    li_agora = 1_781_100_000
    llst = [_linha(li_agora - 600 * li) for li in range(15)]
    _historico(tmp_path / "out", llst)

    li_rc = nm.main(["-i", str(tmp_path / "out"), "--agora", str(li_agora)])

    assert nm.CI_COM_ALARME == li_rc
    ldct = json.loads((tmp_path / "out" / "STI_250_monitor.json").read_text())
    assert 15 == ldct["verificacoes"] == ldct["verificacoes_com_evento"]
    assert ["ruim"] == [la[0] for la in ldct["alarmes"]]
    assert 1.0 == ldct["rotulos"]["bom"]["pontuacao"]
    assert ldct["rotulos"]["persistencia"]["pontuacao"] > ldct["rotulos"]["ruim"]["pontuacao"]


def test_monitor_sem_eventos_suficientes_nao_julga(tmp_path):
    li_agora = 1_781_100_000
    llst = [_linha(li_agora - 600 * li, fb_com_evento=li < 3) for li in range(15)]
    _historico(tmp_path / "out", llst)

    assert nm.CI_SEM_ALARME == nm.main(["-i", str(tmp_path / "out"), "--agora", str(li_agora)])

    ldct = json.loads((tmp_path / "out" / "STI_250_monitor.json").read_text())
    assert 3 == ldct["verificacoes_com_evento"] and [] == ldct["alarmes"]


def test_monitor_janela_movel_e_linha_corrompida(tmp_path):
    li_agora = 1_781_100_000
    # 15 verificações antigas (fora das 24 h) + 2 recentes
    llst = [_linha(li_agora - 2 * 86400 - 600 * li) for li in range(15)]
    llst += [_linha(li_agora - 600), _linha(li_agora + 600)]
    lpth = _historico(tmp_path / "out", llst)
    with lpth.open("a") as larq:
        larq.write('{"target": 17811\n')

    assert nm.CI_SEM_ALARME == nm.main(["-i", str(lpth), "--agora", str(li_agora),
                                        "--saida", str(tmp_path / "status")])

    ldct = json.loads((tmp_path / "status" / "STI_250_monitor.json").read_text())
    # só a recente dentro da janela (a do futuro também fica de fora)
    assert 1 == ldct["verificacoes"]

    # janela de 72 h pega as antigas
    assert nm.CI_COM_ALARME == nm.main(["-i", str(lpth), "--agora", str(li_agora),
                                        "--janela-h", "72", "--saida", str(tmp_path / "s2")])


def test_monitor_sem_historico_e_config_invalido(tmp_path):
    (tmp_path / "vazio").mkdir()
    assert nm.CI_SEM_HISTORICO == nm.main(["-i", str(tmp_path / "vazio")])
    assert nm.CI_ERRO_CONFIG == nm.main(["-i", str(tmp_path / "vazio"),
                                         "--config", str(tmp_path / "nao_existe.ini")])


def test_monitor_margem_do_config(tmp_path):
    li_agora = 1_781_100_000
    _historico(tmp_path / "out", [_linha(li_agora - 600 * li) for li in range(15)])
    (tmp_path / "c.ini").write_text("[monitor]\nmargem = 1.0\n")

    assert nm.CI_SEM_ALARME == nm.main(["-i", str(tmp_path / "out"), "--agora", str(li_agora),
                                        "--config", str(tmp_path / "c.ini")])

# < untLatencia >------------------------------------------------------------------------

def test_latencia_mede_e_estima(tmp_path):
    lpth = tmp_path / "lat.json"

    assert 0 == nl.main(["--larguras", "4", "--grades", "64", "--repeticoes", "1",
                         "--pares", "5", "--pares-em-paralelo", "2", "--timeout-s", "1e6",
                         "--ciclo-s", "0", "--saida", str(lpth)])

    ldct = json.loads(lpth.read_text())
    assert 3 == len(ldct["medidas"])
    assert all(ld["segundos"] > 0 and ld["parametros"] > 0 for ld in ldct["medidas"])

    (ldct_est,) = ldct["estimativas"]
    lf_par = max(ld["segundos"] for ld in ldct["medidas"])
    assert ldct_est["par_s"] == pytest.approx(lf_par)
    assert ldct_est["par_no_cron_s"] == pytest.approx(2 * lf_par)
    assert ldct_est["passada_s"] == pytest.approx(5 * lf_par)
    assert ldct_est["cabe_no_timeout"] and not ldct_est["cabe_no_ciclo"]


def test_latencia_entrada_sintetica_tem_eco_e_cantos_fora_da_cobertura():
    lnp = nl.entrada_sintetica(64)
    assert (4, 64, 64) == lnp.shape
    assert np.isnan(lnp[:, 0, 0]).all()
    assert (lnp[-1] >= 40).any()

# < the end >----------------------------------------------------------------------------
