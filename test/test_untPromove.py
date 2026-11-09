# -*- coding: utf-8 -*-
"""
Centro Logístico de Simulação Meteorológica
Copyright (c) 1995-2026, Instituto de Controle do Espaço Aéreo

test_untPromove
testes do portão de promoção (julga: candidato x atual x persistência,
tolerância, margem, métrica indefinida), das conferências (diretório, backend,
sha256, conjunto, atual desatualizado), dos alertas por par radar/alcance e das
trocas atual/anterior, com checkpoints falsos (bytes + .json) e comparacao.json
sintéticos

2026.out  mlabru  significância (bootstrap de somas_por_bloco)
2026.out  mlabru  alertas por par (resultados_por_par)
2026.out  mlabru  initial version (Linux/Python)
"""
# < imports >----------------------------------------------------------------------------

# python library
import json

# pytest
import pytest

pytest.importorskip("torch")

# model
import untPromove as np_
import work.unt_config as uc
import work.unt_modelos as um

# < helpers >----------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _sem_config_do_repo(tmp_path, monkeypatch):
    # os testes não podem depender do config.ini do repositório
    monkeypatch.chdir(tmp_path)


def _criterio(**fdct):
    return np_.criterio_do_config(uc.carrega())._replace(**fdct)


def _passo(ff_csi, ff_fss):
    return {"limiares": {"30": {"csi": ff_csi}}, "fss": {"30": {"20": ff_fss}}}


def _resultados(fdct_por_rotulo):
    # {"rotulo": (csi, fss)} igual nos passos 3 e 6
    return {ls: {"3": _passo(*lt), "6": _passo(*lt)} for ls, lt in fdct_por_rotulo.items()}


def _checkpoint_falso(fpth_modelos, fs_backend, fs_nome, fs_conteudo="x", fdct_json=None):
    lpth_dir = fpth_modelos / fs_backend
    lpth_dir.mkdir(parents=True, exist_ok=True)
    lpth = lpth_dir / fs_nome
    lpth.write_text(fs_conteudo)
    ldct = {"backend": fs_backend, "escopo": {"pares": ["STI_250"]}}
    ldct.update(fdct_json or {})
    lpth.with_suffix(".json").write_text(json.dumps(ldct))
    return lpth


def _comparacao(fpth_dir, fdct_resultados, fdct_desempenho, fs_split="teste",
                fdct_por_par=None, fdct_janelas_por_par=None, fdct_somas_por_bloco=None):
    lpth = fpth_dir / "comparacao.json"
    ldct = {"split": fs_split, "resultados": fdct_resultados, "desempenho": fdct_desempenho}
    if fdct_por_par is not None:
        ldct.update({"resultados_por_par": fdct_por_par,
                     "janelas_por_par": fdct_janelas_por_par})
    if fdct_somas_por_bloco is not None:
        ldct["somas_por_bloco"] = fdct_somas_por_bloco
    lpth.write_text(json.dumps(ldct))
    return lpth

# < critério e julgamento >--------------------------------------------------------------

def test_criterio_padrao():
    lcrit = np_.criterio_do_config(uc.carrega())
    assert (3, 6) == lcrit.passos
    assert "30" == lcrit.limiar_dbz and "20" == lcrit.escala_fss_km
    assert "teste" == lcrit.split


def test_julga_aprova_melhor_que_persistencia_e_atual():
    ldct = _resultados({"c": (0.5, 0.7), "a": (0.45, 0.65), "persistencia": (0.3, 0.5)})
    lj = np_.julga(ldct, "c", "a", _criterio())
    assert lj.aprovado and [] == lj.motivos
    assert 0.5 == lj.metricas["3"]["candidato"]["csi"]


def test_julga_tolerancia_frente_ao_atual():
    ldct = _resultados({"c": (0.495, 0.7), "a": (0.5, 0.7), "persistencia": (0.3, 0.5)})
    assert np_.julga(ldct, "c", "a", _criterio(tolerancia=0.01)).aprovado
    lj = np_.julga(ldct, "c", "a", _criterio(tolerancia=0.001))
    assert not lj.aprovado and "abaixo do atual" in lj.motivos[0]


def test_julga_precisa_superar_persistencia():
    ldct = _resultados({"c": (0.3, 0.7), "persistencia": (0.3, 0.5)})
    lj = np_.julga(ldct, "c", None, _criterio())
    assert not lj.aprovado
    assert any("persistência" in ls for ls in lj.motivos)

    ldct = _resultados({"c": (0.35, 0.7), "persistencia": (0.3, 0.5)})
    assert not np_.julga(ldct, "c", None, _criterio(margem_persistencia=0.1)).aprovado


def test_julga_metrica_indefinida_recusa():
    ldct = _resultados({"c": (None, 0.7), "persistencia": (0.3, 0.5)})
    lj = np_.julga(ldct, "c", None, _criterio())
    assert not lj.aprovado and "indefinido" in lj.motivos[0]

# < alertas por par >-------------------------------------------------------------------

def test_alertas_por_par_so_no_par_que_piora():
    ldct_bom = _resultados({"c": (0.5, 0.7), "a": (0.45, 0.65), "persistencia": (0.3, 0.5)})
    ldct_pior = _resultados({"c": (0.3, 0.7), "a": (0.45, 0.65), "persistencia": (0.2, 0.5)})
    ldct_comp = {"resultados_por_par": {"STI_250": ldct_bom, "BEL_400": ldct_pior},
                 "janelas_por_par": {"STI_250": 50, "BEL_400": 50}}

    ldct = np_.alertas_por_par(ldct_comp, "c", "a", _criterio())
    assert ["BEL_400"] == list(ldct)
    assert all("abaixo do atual" in ls for ls in ldct["BEL_400"])


def test_alertas_por_par_ignora_poucas_janelas_e_indefinidos():
    ldct_pior = _resultados({"c": (0.3, 0.7), "a": (0.45, 0.65), "persistencia": (0.2, 0.5)})
    ldct_seco = _resultados({"c": (None, None), "a": (None, None),
                             "persistencia": (None, None)})
    ldct_comp = {"resultados_por_par": {"STI_250": ldct_seco, "BEL_400": ldct_pior},
                 "janelas_por_par": {"STI_250": 50, "BEL_400": 5}}

    # BEL_400 piora, mas tem poucas janelas; STI_250 não tem eventos
    assert {} == np_.alertas_por_par(ldct_comp, "c", "a", _criterio(min_janelas_par=20))
    assert ["BEL_400"] == list(np_.alertas_por_par(ldct_comp, "c", "a",
                                                   _criterio(min_janelas_par=5)))


def test_alertas_por_par_um_par_ou_untEval_antigo():
    ldct_pior = _resultados({"c": (0.3, 0.7), "a": (0.45, 0.65), "persistencia": (0.2, 0.5)})
    ldct_comp = {"resultados_por_par": {"STI_250": ldct_pior},
                 "janelas_por_par": {"STI_250": 50}}

    assert {} == np_.alertas_por_par(ldct_comp, "c", "a", _criterio())
    assert {} == np_.alertas_por_par({}, "c", "a", _criterio())

# < conferências >-----------------------------------------------------------------------

def test_confere_candidato(tmp_path):
    lpth = _checkpoint_falso(tmp_path / "m", "rainnet", "rainnet_1.pt")
    lpth_real, ldct = np_.confere_candidato(str(lpth), str(tmp_path / "m"), "rainnet")
    assert lpth_real == lpth.resolve() and "rainnet" == ldct["backend"]

    lpth_trocado = _checkpoint_falso(tmp_path / "m", "rainnet", "rainnet_3.pt",
                                     fdct_json={"backend": "smaat"})
    with pytest.raises(ValueError, match="backend smaat"):
        np_.confere_candidato(str(lpth_trocado), str(tmp_path / "m"), "rainnet")

    lpth_fora = _checkpoint_falso(tmp_path / "outro", "rainnet", "rainnet_2.pt")
    with pytest.raises(ValueError, match="precisa estar"):
        np_.confere_candidato(str(lpth_fora), str(tmp_path / "m"), "rainnet")

    lpth.with_suffix(".json").unlink()
    with pytest.raises(ValueError, match=".json"):
        np_.confere_candidato(str(lpth), str(tmp_path / "m"), "rainnet")


def test_confere_avaliacao(tmp_path):
    lpth_c = _checkpoint_falso(tmp_path, "rainnet", "rainnet_2.pt", "novo")
    lpth_a = _checkpoint_falso(tmp_path, "rainnet", "rainnet_1.pt", "velho")
    ls_sha_c = um.sha256_de_arquivo(lpth_c)
    ls_sha_a = um.sha256_de_arquivo(lpth_a)
    lcrit = _criterio()

    ldct = {"split": "teste", "desempenho": {"rc": {"sha256": ls_sha_c},
                                             "rainnet": {"sha256": ls_sha_a}}}
    assert "rainnet" == np_.confere_avaliacao(ldct, "rc", lpth_c, lpth_a, "rainnet", lcrit)
    assert np_.confere_avaliacao(ldct, "rc", lpth_c, None, "rainnet", lcrit) is None

    with pytest.raises(ValueError, match="conjunto"):
        np_.confere_avaliacao({**ldct, "split": "treino"}, "rc", lpth_c, lpth_a, "rainnet", lcrit)
    with pytest.raises(ValueError, match="ausente"):
        np_.confere_avaliacao(ldct, "xx", lpth_c, lpth_a, "rainnet", lcrit)
    with pytest.raises(ValueError, match="sha256"):
        np_.confere_avaliacao(ldct, "rc", lpth_a, None, "rainnet", lcrit)
    with pytest.raises(ValueError, match="mudou"):
        np_.confere_avaliacao(ldct, "rc", lpth_c, lpth_c, "rainnet", lcrit)
    with pytest.raises(ValueError, match="não inclui"):
        np_.confere_avaliacao({"split": "teste", "desempenho": {"rc": {"sha256": ls_sha_c}}},
                              "rc", lpth_c, lpth_a, "rainnet", lcrit)

# < main: promove, recusa, reverte >-----------------------------------------------------

def test_main_primeira_promocao_sem_atual(tmp_path):
    lpth_m = tmp_path / "m"
    lpth_c = _checkpoint_falso(lpth_m, "smaat", "smaat_1.pt")
    lpth_aval = _comparacao(tmp_path,
                            _resultados({"smaat_candidato": (0.5, 0.7),
                                         "persistencia": (0.3, 0.5)}),
                            {"smaat_candidato": {"sha256": um.sha256_de_arquivo(lpth_c)}})

    assert 0 == np_.main(["-b", "smaat", "--candidato", str(lpth_c), "--avaliacao",
                          str(lpth_aval), "--modelos-dir", str(lpth_m)])

    assert "smaat_1.pt" == um.caminho_checkpoint_atual(str(lpth_m), "smaat").resolve().name
    assert not um.caminho_checkpoint_anterior(str(lpth_m), "smaat").exists()

    # de novo: já é o atual
    assert 0 == np_.main(["-b", "smaat", "--candidato", str(lpth_c), "--modelos-dir", str(lpth_m)])


def test_main_recusa_mantem_atual_e_registra(tmp_path):
    lpth_m = tmp_path / "m"
    lpth_a = _checkpoint_falso(lpth_m, "smaat", "smaat_1.pt", "velho")
    um.aponta_link(lpth_a, um.CS_CHECKPOINT_ATUAL)
    lpth_c = _checkpoint_falso(lpth_m, "smaat", "smaat_2.pt", "novo")

    lpth_aval = _comparacao(tmp_path,
                            _resultados({"smaat_candidato": (0.2, 0.7), "smaat": (0.5, 0.7),
                                         "persistencia": (0.3, 0.5)}),
                            {"smaat_candidato": {"sha256": um.sha256_de_arquivo(lpth_c)},
                             "smaat": {"sha256": um.sha256_de_arquivo(lpth_a)}})

    assert 1 == np_.main(["-b", "smaat", "--candidato", str(lpth_c), "--avaliacao",
                          str(lpth_aval), "--modelos-dir", str(lpth_m)])
    assert "smaat_1.pt" == um.caminho_checkpoint_atual(str(lpth_m), "smaat").resolve().name

    ldct = json.loads((lpth_m / "smaat" / np_.CS_ARQUIVO_REGISTRO).read_text())
    assert "recusa" == ldct["acao"] and ldct["motivos"]


def test_main_sem_avaliacao_so_com_forca(tmp_path):
    lpth_m = tmp_path / "m"
    lpth_c = _checkpoint_falso(lpth_m, "pytorch", "pytorch_1.pt")
    llst = ["-b", "pytorch", "--candidato", str(lpth_c), "--modelos-dir", str(lpth_m)]

    assert 1 == np_.main(llst)
    assert 0 == np_.main(llst + ["--forca"])
    assert um.caminho_checkpoint_atual(str(lpth_m), "pytorch").exists()


def test_main_avisa_pares_perdidos(tmp_path):
    lpth_m = tmp_path / "m"
    lpth_a = _checkpoint_falso(lpth_m, "smaat", "smaat_1.pt", "velho",
                               {"escopo": {"pares": ["STI_250", "SBMT_400"]}})
    um.aponta_link(lpth_a, um.CS_CHECKPOINT_ATUAL)
    lpth_c = _checkpoint_falso(lpth_m, "smaat", "smaat_2.pt", "novo")

    assert 0 == np_.main(["-b", "smaat", "--candidato", str(lpth_c), "--modelos-dir",
                          str(lpth_m), "--forca"])

    ldct = json.loads((lpth_m / "smaat" / np_.CS_ARQUIVO_REGISTRO).read_text())
    assert ["SBMT_400"] == ldct["pares_perdidos"]


def test_main_promove_pelo_agregado_e_registra_alertas_por_par(tmp_path):
    lpth_m = tmp_path / "m"
    lpth_a = _checkpoint_falso(lpth_m, "smaat", "smaat_1.pt", "velho")
    um.aponta_link(lpth_a, um.CS_CHECKPOINT_ATUAL)
    lpth_c = _checkpoint_falso(lpth_m, "smaat", "smaat_2.pt", "novo")

    ldct_bom = _resultados({"smaat_candidato": (0.6, 0.8), "smaat": (0.45, 0.65),
                            "persistencia": (0.3, 0.5)})
    ldct_pior = _resultados({"smaat_candidato": (0.3, 0.6), "smaat": (0.45, 0.65),
                             "persistencia": (0.3, 0.5)})
    lpth_aval = _comparacao(tmp_path, ldct_bom,
                            {"smaat_candidato": {"sha256": um.sha256_de_arquivo(lpth_c)},
                             "smaat": {"sha256": um.sha256_de_arquivo(lpth_a)}},
                            fdct_por_par={"STI_250": ldct_bom, "BEL_400": ldct_pior},
                            fdct_janelas_por_par={"STI_250": 100, "BEL_400": 30})

    assert 0 == np_.main(["-b", "smaat", "--candidato", str(lpth_c), "--avaliacao",
                          str(lpth_aval), "--modelos-dir", str(lpth_m)])
    assert "smaat_2.pt" == um.caminho_checkpoint_atual(str(lpth_m), "smaat").resolve().name

    ldct = json.loads((lpth_m / "smaat" / np_.CS_ARQUIVO_REGISTRO).read_text())
    assert "promove" == ldct["acao"] and [] == ldct["motivos"]
    assert ["BEL_400"] == list(ldct["alertas_por_par"])
    assert any("persistência" in ls for ls in ldct["alertas_por_par"]["BEL_400"])


def _somas_bloco(fi_hits, fi_erros, ff_erro_fss):
    # CSI = hits / (hits + erros); FSS = 1 - erro (referência 1)
    return {"janelas": 10,
            "limiares": {"30": {"hits": fi_hits, "misses": fi_erros, "false_alarms": 0}},
            "fss_somas": {"30": {"20": {"erro": ff_erro_fss, "referencia": 1.0}}}}


def _somas_por_bloco(flst_candidato, flst_persistencia):
    # um bloco por par (candidato, persistência) de (hits, erros, erro_fss), passos 3 e 6
    return {str(li): {ls_rotulo: {ls_passo: _somas_bloco(*lt) for ls_passo in ("3", "6")}
                      for ls_rotulo, lt in (("smaat_candidato", lt_c), ("persistencia", lt_p))}
            for li, (lt_c, lt_p) in enumerate(zip(flst_candidato, flst_persistencia))}


def test_significancia_aprova_ganho_consistente_e_recusa_ganho_de_um_bloco_so():
    lcrit = _criterio(significancia=True, reamostras=500)

    # o candidato ganha em todos os 8 blocos
    ldct = _somas_por_bloco([(6, 4, 0.3)] * 8, [(4, 6, 0.5)] * 8)
    llst_motivos, ldct_ic = np_.julga_significancia({"somas_por_bloco": ldct},
                                                    "smaat_candidato", lcrit)
    assert [] == llst_motivos
    assert ldct_ic["3"]["csi"]["p_melhor"] == pytest.approx(1.0)
    assert ldct_ic["6"]["fss"]["ic"][0] > 0.0

    # um bloco com ganho enorme e sete com perda: o agregado pode ganhar, o IC não
    ldct = _somas_por_bloco([(90, 10, 0.1)] + [(3, 7, 0.6)] * 7,
                            [(10, 90, 0.9)] + [(4, 6, 0.5)] * 7)
    llst_motivos, _ = np_.julga_significancia({"somas_por_bloco": ldct},
                                              "smaat_candidato", lcrit)
    assert any("IC" in ls and "csi" in ls for ls in llst_motivos)


def test_significancia_poucos_blocos_ou_sem_somas_recusa():
    lcrit = _criterio(significancia=True, min_blocos=5)
    ldct = _somas_por_bloco([(6, 4, 0.3)] * 3, [(4, 6, 0.5)] * 3)

    for ldct_comparacao in ({"somas_por_bloco": ldct}, {}):
        llst_motivos, _ = np_.julga_significancia(ldct_comparacao, "smaat_candidato", lcrit)
        assert 1 == len(llst_motivos) and "bloco" in llst_motivos[0]


def test_main_significancia_desligada_por_padrao_e_ligada_no_config(tmp_path):
    lpth_m = tmp_path / "m"
    lpth_c = _checkpoint_falso(lpth_m, "smaat", "smaat_2.pt", "novo")
    # o agregado passa, mas o ganho vem de um bloco só
    ldct_somas = _somas_por_bloco([(90, 10, 0.1)] + [(3, 7, 0.6)] * 7,
                                  [(10, 90, 0.9)] + [(4, 6, 0.5)] * 7)
    lpth_aval = _comparacao(tmp_path, _resultados({"smaat_candidato": (0.6, 0.8),
                                                   "persistencia": (0.3, 0.5)}),
                            {"smaat_candidato": {"sha256": um.sha256_de_arquivo(lpth_c)}},
                            fdct_somas_por_bloco=ldct_somas)
    llst_args = ["-b", "smaat", "--candidato", str(lpth_c), "--avaliacao", str(lpth_aval),
                 "--modelos-dir", str(lpth_m)]

    (tmp_path / "config.ini").write_text("[promocao]\nsignificancia = true\n")
    assert 1 == np_.main(llst_args)
    ldct = json.loads((lpth_m / "smaat" / np_.CS_ARQUIVO_REGISTRO).read_text())
    assert "recusa" == ldct["acao"] and "dif_persistencia" in ldct["metricas"]["3"]

    (tmp_path / "config.ini").unlink()
    assert 0 == np_.main(llst_args)


def test_reverte_sem_anterior_falha(tmp_path):
    lpth_m = tmp_path / "m"
    lpth_a = _checkpoint_falso(lpth_m, "smaat", "smaat_1.pt")
    um.aponta_link(lpth_a, um.CS_CHECKPOINT_ATUAL)

    assert 1 == np_.main(["-b", "smaat", "--reverte", "--modelos-dir", str(lpth_m)])


def test_reverte_sem_links_troca_copias(tmp_path):
    lpth_m = tmp_path / "m"
    lpth_dir = lpth_m / "smaat"
    lpth_dir.mkdir(parents=True)
    (lpth_dir / um.CS_CHECKPOINT_ATUAL).write_text("novo")
    (lpth_dir / um.CS_CHECKPOINT_ANTERIOR).write_text("velho")

    assert 0 == np_.main(["-b", "smaat", "--reverte", "--modelos-dir", str(lpth_m)])

    assert "velho" == (lpth_dir / um.CS_CHECKPOINT_ATUAL).read_text()
    assert "novo" == (lpth_dir / um.CS_CHECKPOINT_ANTERIOR).read_text()

# < the end >----------------------------------------------------------------------------
