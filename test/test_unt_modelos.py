# -*- coding: utf-8 -*-
"""
Centro Logístico de Simulação Meteorológica
Copyright (c) 1995-2026, Instituto de Controle do Espaço Aéreo

test_unt_modelos
testes da parte de Deep Learning, todos em CPU com redes minúsculas
(largura_base 4) e grades 64x64: transformações de dados e padding
(unt_dados), as três redes e o checkpoint (unt_modelos), dataset
(unt_dataset), perdas mascaradas e treino (unt_treino), previsão nos três
modos (unt_inferencia) e o contrato gera_nowcast dos backends.

2026.nov  mlabru  regressao do smaat (linear/log)
2026.out  mlabru  verificação de rede colapsada (confere_colapso, passos_colapso)
2026.out  mlabru  perda ponderada por intensidade (peso_dbz), media_metrica
2026.out  mlabru  dtype_amp (bfloat16/float16), treino interrompido por lotes não
                  finitos (max_nao_finitos)
2026.out  mlabru  passos_rollout, lotes não finitos descartados, métrica de validação
                  só com evento
2026.out  mlabru  seleção pela métrica/perda, val_cada, passos iguais por época,
                  clip_grad, retomada, peso_bins = auto
2026.out  mlabru  checkpoint candidato sem promoção, versão 1 ainda lida, escopo
2026.out  mlabru  initial version (Linux/Python)
"""
# < imports >----------------------------------------------------------------------------

# python library
import pathlib

# pytest
import pytest

# numPy
import numpy as np

torch = pytest.importorskip("torch")

# model
import work.unt_avaliacao as ua
import work.unt_backend_comum as obc
import work.unt_config as uc
import work.unt_dados as ud
import work.unt_dataset as uds
import work.unt_device as ue
import work.unt_inferencia as ui
import work.unt_janelas as uj
import work.unt_modelos as um
import work.unt_treino as ut

# < helpers >----------------------------------------------------------------------------

# sobrescritos que deixam qualquer backend minúsculo e rápido em CPU
MDCT_RAPIDO = {"largura_base": 4, "recorte_px": 32, "batch": 4, "epocas": 2, "workers": 0,
               "device": "cpu", "paciencia": 5, "amp": False}


def _hparams(fs_backend, **fdct_extra):
    ldct = dict(MDCT_RAPIDO)
    ldct.update(fdct_extra)
    return um.hiperparametros(uc.carrega(), fs_backend, 4, 6, ldct)


def _janelas(fdct_cat):
    llst_produtos = []
    for li_ts in fdct_cat["timestamps"]:
        llst_produtos.append(uj.SProduto(li_ts, "observed", 0.99, 0.05))

    lt_param = uj.SParametrosJanela(frac_treino=1.0, frac_validacao=0.0)
    llst, _ = uj.monta_janelas(llst_produtos, "STI", 250, "nwc250v2", lt_param)
    return llst

@pytest.fixture(autouse=True)
def _sem_config_do_repo(tmp_path, monkeypatch):
    # os testes não podem depender do config.ini do repositório
    monkeypatch.chdir(tmp_path)

# < unt_dados >--------------------------------------------------------------------------

def test_transformacoes_ida_e_volta_em_dbz():
    lnp_dbz = np.array([[-32.0, 20.0], [35.0, np.nan]], dtype=np.float32)

    for ltransf in (ud.TransformacaoRainNet(), ud.TransformacaoSmaat()):
        lnp_volta = ltransf.saida_para_dbz(ltransf.alvo(lnp_dbz))
        assert lnp_volta[0, 1] == pytest.approx(20.0, abs=0.01)
        assert lnp_volta[1, 0] == pytest.approx(35.0, abs=0.01)
        # sem eco / inválido viram o piso, nunca -inf
        assert -32.0 <= lnp_volta.min()

# -------------------------------------------------------------------------------------
def test_transformacao_pytorch_entrada_alvo_e_mediana():
    ltransf = ud.TransformacaoPytorch([10, 20, 30])
    lnp_dbz = np.array([[[-32.0, 15.0], [35.0, np.nan]]], dtype=np.float32)

    lnp_x = ltransf.entrada(lnp_dbz)
    assert (2, 2, 2) == lnp_x.shape
    assert [[1.0, 1.0], [1.0, 0.0]] == lnp_x[-1].tolist()

    lnp_y = ltransf.alvo(lnp_dbz[0])
    assert [[0, 1], [3, ud.CI_CLASSE_IGNORADA]] == lnp_y.tolist()

    # probabilidade concentrada na classe 2 -> centro 25 dBZ; classe 0 -> sem eco
    lnp_p = np.zeros((4, 1, 2), dtype=np.float32)
    lnp_p[2, 0, 0] = 1.0
    lnp_p[0, 0, 1] = 1.0
    assert [[25.0, -32.0]] == ltransf.probabilidades_para_dbz(lnp_p).tolist()

# -------------------------------------------------------------------------------------
def test_transformacao_de_dict_e_desconhecida():
    ltransf = ud.TransformacaoPytorch([10, 20], "esperanca")
    assert ltransf.como_dict() == ud.transformacao_de_dict(ltransf.como_dict()).como_dict()
    with pytest.raises(ValueError):
        ud.transformacao_de_dict({"nome": "xx"})

# -------------------------------------------------------------------------------------
@pytest.mark.parametrize("ft_shape", [(500, 500), (600, 600), (64, 64), (3, 61, 70)])
def test_padding_ida_e_volta(ft_shape):
    lnp = np.random.default_rng(0).random(ft_shape).astype(np.float32)
    lt_pad = ud.calcula_padding(lnp.shape)
    lnp_pad = ud.aplica_padding(lnp, lt_pad)

    assert 0 == lnp_pad.shape[-1] % 32 and 0 == lnp_pad.shape[-2] % 32
    assert np.array_equal(lnp, ud.remove_padding(lnp_pad, lt_pad))

# < unt_modelos >------------------------------------------------------------------------

@pytest.mark.parametrize("fs_backend", um.CT_BACKENDS)
def test_redes_shapes(fs_backend):
    ldct = _hparams(fs_backend)
    lrede = um.constroi_rede(ldct).eval()

    if "pytorch" == fs_backend:
        lsaida = lrede(torch.zeros(2, 5, 64, 64), torch.tensor([0, 5]))
        assert (2, 10, 64, 64) == tuple(lsaida.shape)
    else:
        assert (2, 1, 64, 64) == tuple(lrede(torch.zeros(2, 4, 64, 64)).shape)

# -------------------------------------------------------------------------------------
def test_smaat_modo_direto_tem_6_saidas():
    lrede = um.constroi_rede(_hparams("smaat", modo="direto")).eval()
    assert 6 == lrede(torch.zeros(1, 4, 32, 32)).shape[1]

# -------------------------------------------------------------------------------------
def test_largura_dos_artigos():
    # RainNet ~31.4 M, SmaAt-UNet ~4 M parâmetros (artigos)
    assert 31_000_000 < um.conta_parametros(um.constroi_rede(_hparams("rainnet", largura_base=64))) < 32_000_000
    assert 3_900_000 < um.conta_parametros(um.constroi_rede(_hparams("smaat", largura_base=64))) < 4_200_000

# -------------------------------------------------------------------------------------
def test_hiperparametros_precedencia_e_validacao():
    ldct = um.hiperparametros(uc.carrega(), "rainnet", 4, 6, {"lr": None, "batch": 7})
    assert 3e-4 == ldct["lr"] and 7 == ldct["batch"] and "recursivo" == ldct["modo"]
    assert 1e-3 == um.hiperparametros(uc.carrega(), "smaat", 4, 6)["lr"]

    with pytest.raises(ValueError, match="backend"):
        um.hiperparametros(uc.carrega(), "xx", 4, 6)
    with pytest.raises(ValueError, match="múltiplo"):
        _hparams("rainnet", recorte_px=100)
    with pytest.raises(ValueError, match="modo"):
        _hparams("smaat", modo="xx")
    with pytest.raises(ValueError, match="passos_rollout"):
        _hparams("rainnet", passos_rollout=7)
    with pytest.raises(ValueError, match="recursivo"):
        _hparams("smaat", modo="direto", passos_rollout=3)
    with pytest.raises(ValueError, match="recursivo"):
        _hparams("pytorch", passos_rollout=2)

# -------------------------------------------------------------------------------------
def test_checkpoint_ida_e_volta_e_atual(tmp_path):
    ldct = _hparams("smaat")
    lrede = um.constroi_rede(ldct)
    ltransf = um.transformacao_padrao(ldct)

    lpth = um.salva_checkpoint(str(tmp_path), ldct, lrede, ltransf, {"epoca": 3},
                               fb_promove=True)
    lpth_atual = um.caminho_checkpoint_atual(str(tmp_path), "smaat")

    assert lpth_atual.exists() and lpth_atual.resolve() == lpth.resolve()

    lmodelo = um.carrega_checkpoint(lpth_atual, "cpu")
    assert "smaat" == lmodelo.backend and 3 == lmodelo.checkpoint["epoca"]
    assert not lmodelo.rede.training

    for lchave, ltensor in lrede.state_dict().items():
        assert torch.equal(ltensor, lmodelo.rede.state_dict()[lchave])

# -------------------------------------------------------------------------------------
def test_checkpoint_sem_promove_nao_toca_atual(tmp_path):
    ldct = _hparams("rainnet")
    lrede = um.constroi_rede(ldct)
    ltransf = um.transformacao_padrao(ldct)

    # um atual.pt em produção, apontando para um checkpoint mais antigo
    lpth_dir = tmp_path / "rainnet"
    lpth_dir.mkdir()
    (lpth_dir / "rainnet_20000101000000.pt").write_bytes(b"antigo")
    lpth_atual = um.caminho_checkpoint_atual(str(tmp_path), "rainnet")
    lpth_atual.symlink_to("rainnet_20000101000000.pt")

    lpth_novo = um.salva_checkpoint(str(tmp_path), ldct, lrede, ltransf, {"epoca": 9})

    assert lpth_novo.exists()
    assert "rainnet_20000101000000.pt" == lpth_atual.resolve().name

# -------------------------------------------------------------------------------------
def test_checkpoint_mesmo_segundo_nao_sobrescreve(tmp_path):
    ldct = _hparams("smaat")
    lrede = um.constroi_rede(ldct)
    ltransf = um.transformacao_padrao(ldct)

    lset = {um.salva_checkpoint(str(tmp_path), ldct, lrede, ltransf, {}) for _ in range(3)}
    assert 3 == len(lset)

# -------------------------------------------------------------------------------------
def test_checkpoint_versao_1_ainda_e_lido(tmp_path):
    ldct = _hparams("smaat")
    lpth = um.salva_checkpoint(str(tmp_path), ldct, um.constroi_rede(ldct),
                               um.transformacao_padrao(ldct), {})
    ldct_ck = torch.load(lpth, weights_only=False)
    assert um.CI_VERSAO_CHECKPOINT == ldct_ck["versao"]

    ldct_ck["versao"] = 1
    torch.save(ldct_ck, lpth)
    assert "smaat" == um.carrega_checkpoint(lpth).backend

    ldct_ck["versao"] = 99
    torch.save(ldct_ck, lpth)
    with pytest.raises(ValueError, match="versão"):
        um.carrega_checkpoint(lpth)

# -------------------------------------------------------------------------------------
def test_escopo_e_cobre_par(tmp_path):
    llst = [uj.SJanela("STI", 250, "p", (1,), (2,), 0, "treino"),
            uj.SJanela("SBMT", 400, "p", (1,), (2,), 0, "teste"),
            uj.SJanela("STI", 250, "p", (3,), (4,), 0, "teste")]
    ldct_escopo = um.escopo_das_janelas(llst)
    assert {"pares": ["SBMT_400", "STI_250"]} == ldct_escopo

    assert um.cobre_par({"escopo": ldct_escopo}, "STI", 250) is True
    assert um.cobre_par({"escopo": ldct_escopo}, "STI", 400) is False
    assert um.cobre_par({"hparams": {}}, "STI", 250) is None
    assert um.cobre_par(None, "STI", 250) is None

    # o resumo é lido pelo link, sem carregar os pesos
    (tmp_path / "rainnet_1.pt").write_bytes(b"x")
    (tmp_path / "rainnet_1.json").write_text('{"escopo": {"pares": ["STI_250"]}}')
    (tmp_path / "atual.pt").symlink_to("rainnet_1.pt")
    assert um.cobre_par(um.le_resumo_checkpoint(tmp_path / "atual.pt"), "STI", 250)
    assert um.le_resumo_checkpoint(tmp_path / "nao_existe.pt") is None

    assert len(um.sha256_de_arquivo(tmp_path / "rainnet_1.pt")) == 64
    assert um.sha256_de_arquivo(tmp_path / "nao_existe.pt") is None

# -------------------------------------------------------------------------------------
def test_checkpoint_ausente():
    with pytest.raises(FileNotFoundError):
        um.carrega_checkpoint("/nao/existe.pt")

# < unt_device >-------------------------------------------------------------------------

def test_device_auto_sem_gpu_e_cuda_explicito(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(ue, "_mps_disponivel", lambda: False)

    assert "cpu" == ue.escolhe_device("auto").type
    assert not ue.usa_amp(torch.device("cpu"), True)
    with pytest.raises(RuntimeError, match="GPU"):
        ue.escolhe_device("cuda")

# -------------------------------------------------------------------------------------
@pytest.mark.parametrize("fb_bf16, fdtype", [(True, torch.bfloat16), (False, torch.float16)])
def test_dtype_amp_bfloat16_so_com_suporte_nativo(monkeypatch, fb_bf16, fdtype):
    llst_args = []

    def _suporta(**fdct_kw):
        llst_args.append(fdct_kw)
        return fb_bf16

    monkeypatch.setattr(torch.cuda, "is_bf16_supported", _suporta)

    assert fdtype == ue.dtype_amp(torch.device("cuda"))
    assert [{"including_emulation": False}] == llst_args
    assert torch.float16 == ue.dtype_amp(torch.device("cpu"))

# < unt_dataset >------------------------------------------------------------------------

@pytest.mark.parametrize("fs_backend,ft_y", [("rainnet", (1, 32, 32)), ("pytorch", (32, 32))])
def test_dataset_itens(catalogo_sintetico, fs_backend, ft_y):
    ldct = _hparams(fs_backend)
    lds = uds.JanelasDataset(_janelas(catalogo_sintetico), catalogo_sintetico["cappi_dir"],
                             ldct, um.transformacao_padrao(ldct), fb_treino=True)

    ldct_item = lds[0]
    assert ft_y == tuple(ldct_item["y"].shape)

    if "pytorch" == fs_backend:
        assert len(lds) == 6 * len(lds.janelas)
        assert (5, 32, 32) == tuple(ldct_item["x"].shape)
    else:
        assert (4, 32, 32) == tuple(ldct_item["x"].shape)
        assert ldct_item["m"].shape == ldct_item["y"].shape

# -------------------------------------------------------------------------------------
def test_dataset_recursivo_com_rollout_tem_k_alvos(catalogo_sintetico):
    ldct = _hparams("smaat", passos_rollout=3)
    lds = uds.JanelasDataset(_janelas(catalogo_sintetico), catalogo_sintetico["cappi_dir"],
                             ldct, um.transformacao_padrao(ldct), fb_treino=True)
    ldct_item = lds[0]
    assert (3, 32, 32) == tuple(ldct_item["y"].shape) == tuple(ldct_item["m"].shape)

# -------------------------------------------------------------------------------------
def test_dataset_recorte_procura_eco_e_validacao_deterministica():
    lnp = np.full((5, 64, 64), -32.0, dtype=np.float32)
    lnp[3, 40:60, 40:60] = 40.0
    lrng = np.random.default_rng(1)

    li_lin, li_col = uds.escolhe_recorte(lnp, 16, 4, lrng, 0.2, 200)
    assert np.mean(lnp[3, li_lin:li_lin + 16, li_col:li_col + 16] >= 10) >= 0.2
    assert (24, 24) == uds.escolhe_recorte(lnp, 16, 4, None, 0.2, 10)

# < unt_treino >-------------------------------------------------------------------------

def test_perdas_ignoram_pixels_invalidos():
    lprev = torch.zeros(1, 1, 2, 2)
    lalvo = torch.tensor([[[[0.0, 0.0], [0.0, 100.0]]]])
    lmasc = torch.tensor([[[[1.0, 1.0], [1.0, 0.0]]]])

    assert 0.0 == float(ut.perda_mse(lprev, lalvo, lmasc))
    assert 0.0 == pytest.approx(float(ut.perda_log_cosh(lprev, lalvo, lmasc)), abs=1e-6)
    # log-cosh estável para erros grandes (cosh estouraria)
    assert torch.isfinite(ut.perda_log_cosh(lprev, lalvo * 1000, torch.ones_like(lmasc)))

    lperda = ut.PerdaEntropiaBins([1.0, 1.0])
    lclasses = torch.full((1, 2, 2), ud.CI_CLASSE_IGNORADA, dtype=torch.int64)
    assert 0.0 == float(lperda(torch.zeros(1, 2, 2, 2), lclasses))

# -------------------------------------------------------------------------------------
def test_perda_ponderada_por_intensidade():
    ltransf = ud.TransformacaoRainNet()
    lpeso = ut.PesoIntensidade([20.0, 30.0], [1.0, 2.0, 5.0], ltransf)

    # o limite cai no espaço do alvo (log(R + 0.01)); o próprio limite já é a faixa de cima
    lalvo = torch.as_tensor(ltransf.alvo(np.array([[-32.0, 25.0], [30.0, 45.0]],
                                                  dtype=np.float32)))[None, None]
    assert [[1.0, 2.0], [5.0, 5.0]] == lpeso(lalvo)[0, 0].tolist()

    lprev = lalvo.clone()
    lprev[0, 0, 1, 1] -= 2.0                       # erro só no eco forte
    lmasc = torch.ones_like(lalvo)

    # pesos uniformes = perda sem peso
    lpeso_1 = ut.PesoIntensidade([20.0, 30.0], [3.0, 3.0, 3.0], ltransf)
    for lperda in (ut.perda_log_cosh, ut.perda_mse):
        assert float(lperda(lprev, lalvo, lmasc)) == pytest.approx(
            float(lperda(lprev, lalvo, lmasc, lpeso_1)))
        # o peso maior no eco forte aumenta a perda do erro lá
        assert float(lperda(lprev, lalvo, lmasc, lpeso)) > float(lperda(lprev, lalvo, lmasc))

    # pixel inválido continua fora, com qualquer peso
    lmasc[0, 0, 1, 1] = 0.0
    assert 0.0 == pytest.approx(float(ut.perda_log_cosh(lprev, lalvo, lmasc, lpeso)), abs=1e-6)

# -------------------------------------------------------------------------------------
def test_hparams_peso_dbz_e_media_metrica():
    # padrão: sem peso (artigos) e sem média; o pytorch não tem as chaves de peso
    ldct = um.hiperparametros(uc.carrega(), "rainnet", 4, 6)
    assert ([], [], 1) == (ldct["limites_peso_dbz"], ldct["peso_dbz"], ldct["media_metrica"])
    assert ut.peso_intensidade(ldct, ud.TransformacaoRainNet()) is None
    assert "peso_dbz" not in um.hiperparametros(uc.carrega(), "pytorch", 4, 6)

    pathlib.Path("config.ini").write_text("[rainnet]\nlimites_peso_dbz = 20, 30\n"
                                          "peso_dbz = 1, 2, 4\nmedia_metrica = 3\n")
    ldct = um.hiperparametros(uc.carrega(), "rainnet", 4, 6)
    assert ([20.0, 30.0], [1.0, 2.0, 4.0], 3) == (
        ldct["limites_peso_dbz"], ldct["peso_dbz"], ldct["media_metrica"])
    assert isinstance(ut.peso_intensidade(ldct, ud.TransformacaoRainNet()), ut.PesoIntensidade)
    # o smaat não herda a seção do rainnet
    assert [] == um.hiperparametros(uc.carrega(), "smaat", 4, 6)["peso_dbz"]

    # os pesos mudam a função objetivo: outra identidade de retomada
    ldct_sem = dict(ldct, peso_dbz=[], limites_peso_dbz=[])
    assert ut.identidade_retomada(ldct, "x") != ut.identidade_retomada(ldct_sem, "x")

    for ldct_ruim, ls_erro in (({"peso_dbz": [1.0, 2.0]}, "3 valores"),
                               ({"limites_peso_dbz": [30.0, 20.0]}, "crescente"),
                               ({"peso_dbz": [1.0, 0.0, 2.0]}, "positivo"),
                               ({"media_metrica": 0}, "media_metrica"),
                               ({"passos_colapso": -1}, "passos_colapso"),
                               ({"tentativas_colapso": 0}, "tentativas_colapso")):
        with pytest.raises(ValueError, match=ls_erro):
            um.hiperparametros(uc.carrega(), "rainnet", 4, 6, ldct_ruim)

# -------------------------------------------------------------------------------------
def test_regressao_do_smaat():
    # padrão: linear, a do artigo (R / escala com MSE); rainnet sempre log
    ldct = um.hiperparametros(uc.carrega(), "smaat", 4, 6)
    assert um.CS_REGRESSAO_LINEAR == um.regressao(ldct)
    assert isinstance(um.transformacao_padrao(ldct), ud.TransformacaoSmaat)
    assert um.CS_REGRESSAO_LOG == um.regressao(um.hiperparametros(uc.carrega(), "rainnet", 4, 6))
    # checkpoint antigo sem a chave: linear
    assert um.CS_REGRESSAO_LINEAR == um.regressao({"backend": "smaat"})

    pathlib.Path("config.ini").write_text("[smaat]\nregressao = log\n")
    ldct_log = um.hiperparametros(uc.carrega(), "smaat", 4, 6)
    assert um.CS_REGRESSAO_LOG == um.regressao(ldct_log)
    assert isinstance(um.transformacao_padrao(ldct_log), ud.TransformacaoRainNet)
    assert ut.identidade_retomada(ldct, "x") != ut.identidade_retomada(ldct_log, "x")

    # perda do espaço log: log-cosh (|erro| 1 -> log(cosh(1)), não 1)
    lrede = _RedeSoma()
    lx = torch.zeros(1, 4, 2, 2)
    ly = torch.zeros(1, 1, 2, 2)
    ldct_lote = {"x": lx, "y": ly, "m": torch.ones_like(ly)}
    lf_log = float(ut.calcula_perda(lrede, ldct_lote, ldct_log, None))
    assert lf_log == pytest.approx(np.log(np.cosh(1.0)), abs=1.0e-5)

    with pytest.raises(ValueError, match="regressao"):
        um.hiperparametros(uc.carrega(), "smaat", 4, 6, {"regressao": "raiz"})

# -------------------------------------------------------------------------------------
def test_metrica_media():
    llst_hist = [{"metrica_validacao": 0.4}, {"metrica_validacao": None},
                 {"metrica_validacao": 0.1}, {"metrica_validacao": 0.9}]

    assert 0.3 == ut.metrica_media(llst_hist, 0.3, 1)
    # as N - 1 métricas definidas anteriores mais a da época
    assert (0.1 + 0.9 + 0.2) / 3 == pytest.approx(ut.metrica_media(llst_hist, 0.2, 3))
    # início do treino: só as que existem
    assert 0.3 == pytest.approx(ut.metrica_media([{"metrica_validacao": 0.4}], 0.2, 3))
    assert ut.metrica_media(llst_hist, None, 3) is None

# -------------------------------------------------------------------------------------
@pytest.mark.parametrize("fi_media,fi_epoca", [(1, 3), (3, 7)])
def test_treina_media_metrica_ignora_pico(catalogo_sintetico, monkeypatch, fi_media, fi_epoca):
    ldct = _hparams("rainnet", epocas=7, paciencia=10, media_metrica=fi_media)
    lds_treino, lds_val = _datasets(catalogo_sintetico, ldct)
    llst_metricas = [0.4, 0.1, 0.9, 0.1, 0.5, 0.6, 0.6]
    monkeypatch.setattr(ut, "avalia_metrica", lambda *_: llst_metricas.pop(0))

    lres = ut.treina(um.constroi_rede(ldct), lds_treino, lds_val, ldct, torch.device("cpu"))

    # sem média, o pico isolado da época 3; com média de 3, a época 7 (média 0,567)
    assert fi_epoca == lres.epoca
    assert lres.historico[6]["metrica_media"] == pytest.approx(
        0.6 if 1 == fi_media else (0.5 + 0.6 + 0.6) / 3)

# -------------------------------------------------------------------------------------
def test_confere_colapso():
    # saída constante (pesos zerados: só o viés) = colapsada; rede comum, não
    lrede = torch.nn.Conv2d(4, 1, 3, padding=1)
    ldct_lote = {"x": torch.randn(2, 4, 8, 8)}
    ut.confere_colapso(lrede, ldct_lote, 10, None, torch.device("cpu"))

    torch.nn.init.zeros_(lrede.weight)
    with pytest.raises(ut.TreinoColapsou, match="10 passos"):
        ut.confere_colapso(lrede, ldct_lote, 10, None, torch.device("cpu"))

# -------------------------------------------------------------------------------------
def test_treina_verifica_colapso_so_no_inicio(catalogo_sintetico, monkeypatch):
    ldct = _hparams("rainnet", epocas=2, passos_colapso=1)
    lds_treino, lds_val = _datasets(catalogo_sintetico, ldct)
    llst_chamadas = []
    monkeypatch.setattr(ut, "confere_colapso", lambda *fargs: llst_chamadas.append(fargs[2]))

    ut.treina(um.constroi_rede(ldct), lds_treino, lds_val, ldct, torch.device("cpu"))

    # uma vez, no 1º passo da 1ª época; nunca no pytorch
    assert [1] == llst_chamadas

    ldct = _hparams("pytorch", epocas=1, passos_colapso=1)
    lds_treino, lds_val = _datasets(catalogo_sintetico, ldct)
    ut.treina(um.constroi_rede(ldct), lds_treino, lds_val, ldct, torch.device("cpu"))
    assert [1] == llst_chamadas

# -------------------------------------------------------------------------------------
class _RedeSoma(torch.nn.Module):
    """
    "rede" de teste: a soma das entradas + 1, guardando o que viu a cada chamada.
    """
    def __init__(self):
        super().__init__()
        self.vistas = []

    def forward(self, fx):
        self.vistas.append(fx.detach().clone())
        return fx.sum(dim=1, keepdim=True) + 1.0


def test_perda_rollout_realimenta_a_saida_e_media_dos_passos():
    lrede = _RedeSoma()
    lx = torch.zeros(1, 4, 2, 2)
    ly = torch.stack([torch.full((2, 2), lf) for lf in (1.0, 2.0, 4.0)])[None]
    ldct_lote = {"x": lx, "y": ly, "m": torch.ones_like(ly)}

    # saídas 1, 2, 4 (cada uma realimentada): perda zero nos 3 passos
    lperda = ut.perda_rollout(lrede, ldct_lote, um.CS_REGRESSAO_LINEAR, ut.perda_mse)
    assert 0.0 == float(lperda)
    assert 3 == len(lrede.vistas)
    assert torch.equal(lrede.vistas[2][0, 2:], torch.stack([torch.full((2, 2), 1.0),
                                                           torch.full((2, 2), 2.0)]))

    # linear (smaat) realimenta sem chuva negativa
    assert 0.0 == float(ut.realimenta(torch.tensor([-1.0]), um.CS_REGRESSAO_LINEAR))
    assert -1.0 == float(ut.realimenta(torch.tensor([-1.0]), um.CS_REGRESSAO_LOG))

    # rainnet realimenta dentro da faixa física, igual no treino e na inferência
    lf_min, lf_max = ud.faixa_realimentacao_rainnet()
    assert lf_min == pytest.approx(np.log(0.01))
    assert 6.0 < lf_max < 7.5
    ltensor = ut.realimenta(torch.tensor([-204.0, 0.5, 1.0e4]), um.CS_REGRESSAO_LOG)
    assert [lf_min, 0.5, lf_max] == pytest.approx(ltensor.tolist(), abs=1.0e-5)
    lnp = ud.TransformacaoRainNet().realimenta(np.array([-204.0, 0.5, 1.0e4]))
    assert [lf_min, 0.5, lf_max] == pytest.approx(lnp.tolist(), abs=1.0e-5)

# -------------------------------------------------------------------------------------
def test_executa_epoca_descarta_lote_de_perda_nao_finita(catalogo_sintetico, monkeypatch):
    ldct = _hparams("rainnet")
    lds_treino, _ = _datasets(catalogo_sintetico, ldct)
    lrede = um.constroi_rede(ldct)
    lcarregador = ut.monta_carregador(lds_treino, ldct, torch.device("cpu"), True)
    lotim = torch.optim.Adam(lrede.parameters())
    lescala = torch.amp.GradScaler("cpu", enabled=False)

    lperda_original = ut.calcula_perda
    llst_chamadas = []

    def _primeira_nan(*flst_args):
        llst_chamadas.append(1)
        lperda = lperda_original(*flst_args)
        return lperda * float("nan") if 1 == len(llst_chamadas) else lperda

    monkeypatch.setattr(ut, "calcula_perda", _primeira_nan)
    lf_perda, li_lotes, li_descartados = ut.executa_epoca(
        lrede, lcarregador, ldct, torch.device("cpu"), None, lotim, lescala)

    assert 1 == li_descartados and li_lotes == len(llst_chamadas) - 1
    assert np.isfinite(lf_perda)
    assert all(torch.isfinite(lp).all() for lp in lrede.parameters())

# -------------------------------------------------------------------------------------
def test_grad_scaler_so_com_float16():
    lrede = torch.nn.Linear(2, 2)
    ldct = {"lr": 1.0e-3, "paciencia": 8}

    for ldtype, lb_ativo in ((None, False), (torch.bfloat16, False), (torch.float16, True)):
        ldct_otim = ut._monta_otimizacao(lrede, ldct, ldtype, torch.device("cpu"))
        assert lb_ativo == ldct_otim["escalonador_amp"].is_enabled()

# -------------------------------------------------------------------------------------
def test_confere_nao_finitos_no_limite():
    ut._confere_nao_finitos(3, "treino", 99, 1, 0.01)
    ut._confere_nao_finitos(3, "treino", 0, 0, 0.0)
    ut._confere_nao_finitos(3, "treino", 0, 5, 1.0)

    with pytest.raises(ut.TreinoDivergiu, match="época 3: perda não finita em 2 de 100"):
        ut._confere_nao_finitos(3, "treino", 98, 2, 0.01)

# -------------------------------------------------------------------------------------
def test_treina_interrompe_com_lotes_nao_finitos(tmp_path, catalogo_sintetico, monkeypatch):
    ldct = _hparams("rainnet", epocas=2, max_nao_finitos=0.5)
    lds_treino, lds_val = _datasets(catalogo_sintetico, ldct)
    lpth_retomada = tmp_path / "rainnet.retomada.pt"

    lperda_original = ut.calcula_perda
    ldct_epoca = {"nan": False}

    def _nan_na_segunda_epoca(*flst_args):
        lperda = lperda_original(*flst_args)
        return lperda * float("nan") if ldct_epoca["nan"] else lperda

    def _ao_fim(_fdct_registro):
        ldct_epoca["nan"] = True

    monkeypatch.setattr(ut, "calcula_perda", _nan_na_segunda_epoca)

    with pytest.raises(ut.TreinoDivergiu, match="época 2"):
        ut.treina(um.constroi_rede(ldct), lds_treino, lds_val, ldct, torch.device("cpu"),
                  ff_ao_fim_da_epoca=_ao_fim, fpth_retomada=lpth_retomada)

    # a retomada da época 1 (a última boa) ficou
    assert 1 == torch.load(lpth_retomada, weights_only=False)["loop"]["epoca"]

# -------------------------------------------------------------------------------------
def test_janelas_da_metrica_so_com_evento_e_fixas(catalogo_sintetico, monkeypatch):
    ldct = _hparams("rainnet", max_janelas_val=3, min_pixels_evento_val=1)
    _, lds_val = _datasets(catalogo_sintetico, ldct)

    # nenhum alvo com 1e6 pixels >= limiar: nenhuma janela
    assert [] == ua.janelas_com_evento(lds_val.janelas, lds_val.cappi_dir, 30.0, 10 ** 6, 4)

    llst = ut.janelas_da_metrica(lds_val, ldct)
    assert 1 <= len(llst) <= 3
    # escolhidas uma vez: a segunda chamada não lê os CAPPIs de novo
    monkeypatch.setattr(ua, "janelas_com_evento", None)
    assert llst is ut.janelas_da_metrica(lds_val, ldct)

# -------------------------------------------------------------------------------------
def _datasets(fdct_cat, fdct_hparams):
    ltransf = um.transformacao_padrao(fdct_hparams)
    llst = _janelas(fdct_cat)
    return (uds.JanelasDataset(llst, fdct_cat["cappi_dir"], fdct_hparams, ltransf, True),
            uds.JanelasDataset(llst[:4], fdct_cat["cappi_dir"], fdct_hparams, ltransf, False))

# -------------------------------------------------------------------------------------
@pytest.mark.parametrize("fs_backend", um.CT_BACKENDS)
def test_treina_selecao_pela_perda(catalogo_sintetico, fs_backend):
    ldct = _hparams(fs_backend, epocas=3, lr=3e-3, selecao="perda")
    lds_treino, lds_val = _datasets(catalogo_sintetico, ldct)

    lres = ut.treina(um.constroi_rede(ldct), lds_treino, lds_val, ldct, torch.device("cpu"))

    assert 3 == len(lres.historico)
    assert 1 <= lres.epoca <= 3
    assert "perda" == lres.criterio
    assert np.isfinite(lres.perda_validacao)
    assert lres.perda_validacao == min(lr["perda_validacao"] for lr in lres.historico)
    # sem cálculo da métrica quando a seleção é pela perda
    assert all(lr["metrica_validacao"] is None for lr in lres.historico)

# -------------------------------------------------------------------------------------
@pytest.mark.parametrize("fs_backend", um.CT_BACKENDS)
def test_treina_selecao_pela_metrica_do_rollout(catalogo_sintetico, fs_backend):
    ldct = _hparams(fs_backend, epocas=3, lr=3e-3, selecao="metrica")
    lds_treino, lds_val = _datasets(catalogo_sintetico, ldct)

    lres = ut.treina(um.constroi_rede(ldct), lds_treino, lds_val, ldct, torch.device("cpu"))

    # o catálogo sintético sempre tem eco >= 30 dBZ nos alvos: a métrica é definida
    llst_metricas = [lr["metrica_validacao"] for lr in lres.historico]
    assert all(lf is not None for lf in llst_metricas)

    assert "metrica" == lres.criterio
    assert lres.metrica_validacao == max(lf for lf in llst_metricas if lf is not None)
    assert lres.historico[lres.epoca - 1]["metrica_validacao"] == lres.metrica_validacao

# -------------------------------------------------------------------------------------
def test_treina_recursivo_com_rollout(catalogo_sintetico):
    ldct = _hparams("rainnet", epocas=1, selecao="perda", passos_rollout=3)
    lds_treino, lds_val = _datasets(catalogo_sintetico, ldct)

    lres = ut.treina(um.constroi_rede(ldct), lds_treino, lds_val, ldct, torch.device("cpu"))

    assert np.isfinite(lres.perda_validacao)
    assert {"treino": 0, "validacao": 0} == lres.historico[0]["lotes_nao_finitos"]

# -------------------------------------------------------------------------------------
def test_estado_loop_metrica_com_fallback_para_perda():
    lrede = torch.nn.Linear(1, 1)
    lestado = ut.EstadoLoop(lrede)

    # sem métrica definida: o critério ativo é a perda
    lestado.epoca = 1
    assert lestado.registra(lrede, "metrica", 1.0, None)
    lestado.epoca = 2
    assert not lestado.registra(lrede, "metrica", 2.0, None)
    assert (1, "perda") == lestado.escolhida("metrica")[::2]

    # a partir da primeira métrica, só ela conta (mesmo com a perda piorando)
    lestado.epoca = 3
    assert lestado.registra(lrede, "metrica", 5.0, 0.2)
    lestado.epoca = 4
    assert not lestado.registra(lrede, "metrica", 0.5, 0.1)
    assert (3, "metrica") == lestado.escolhida("metrica")[::2]
    assert (4, "perda") == lestado.escolhida("perda")[::2]

# -------------------------------------------------------------------------------------
def test_pontuacao_e_amostra_de_janelas():
    import work.unt_avaliacao as ua

    ldct = {"1": {"limiares": {"30": {"csi": 0.4}}, "fss": {"30": {"20": 0.8}}},
            "2": {"limiares": {"30": {"csi": None}}, "fss": {"30": {"20": 0.6}}}}
    # média(CSI definidos) = 0.4, média(FSS) = 0.7 -> 0.55
    assert ua.pontuacao(ldct, "30", "20") == pytest.approx(0.55)
    assert ua.pontuacao({"1": {"limiares": {}, "fss": {}}}, "30", "20") is None

    assert list(range(10)) == ua.amostra_janelas(list(range(10)), 0)
    assert [0, 3, 6, 9] == ua.amostra_janelas(list(range(10)), 4)

# -------------------------------------------------------------------------------------
def test_treina_val_cada_pula_metrica(catalogo_sintetico):
    ldct = _hparams("smaat", epocas=3, val_cada=2)
    lds_treino, lds_val = _datasets(catalogo_sintetico, ldct)

    lres = ut.treina(um.constroi_rede(ldct), lds_treino, lds_val, ldct, torch.device("cpu"))

    assert lres.historico[0]["metrica_validacao"] is None
    assert lres.historico[2]["metrica_validacao"] is None

# -------------------------------------------------------------------------------------
def test_mesmo_numero_de_passos_por_epoca_em_todos_os_backends(catalogo_sintetico):
    llst_passos = []

    for ls_backend in um.CT_BACKENDS:
        ldct = _hparams(ls_backend, epocas=1, selecao="perda")
        lds_treino, lds_val = _datasets(catalogo_sintetico, ldct)
        lres = ut.treina(um.constroi_rede(ldct), lds_treino, lds_val, ldct, torch.device("cpu"))
        llst_passos.append(lres.passos)

    # pytorch: 6 itens por janela no dataset, mas uma janela por item na época
    assert 1 == len(set(llst_passos))
    assert llst_passos[0] == -(-len(_janelas(catalogo_sintetico)) // MDCT_RAPIDO["batch"])

# -------------------------------------------------------------------------------------
def test_clip_grad_limita_a_norma():
    lrede = torch.nn.Linear(4, 1)
    lotim = torch.optim.SGD(lrede.parameters(), lr=0.0)
    lescala = torch.amp.GradScaler("cpu", enabled=False)
    lperda = (lrede(torch.full((8, 4), 100.0)) ** 2).mean()

    ut._passo_otimizador(lrede, lperda, lotim, lescala, 1.0)

    lf_norma = torch.sqrt(sum((lp.grad ** 2).sum() for lp in lrede.parameters()))
    assert float(lf_norma) == pytest.approx(1.0, rel=1e-4)

# -------------------------------------------------------------------------------------
def test_treina_retoma_do_estado_gravado(tmp_path, catalogo_sintetico):
    lpth = tmp_path / "rainnet.retomada.pt"
    ldct = _hparams("rainnet", epocas=2, selecao="perda")
    lds_treino, lds_val = _datasets(catalogo_sintetico, ldct)
    ldct_id = ut.identidade_retomada(ldct, "abc")

    lres_2 = ut.treina(um.constroi_rede(ldct), lds_treino, lds_val, ldct, torch.device("cpu"),
                       fpth_retomada=lpth, fdct_identidade=ldct_id)
    assert lpth.exists() and 2 == len(lres_2.historico)

    # retoma com mais épocas: continua da 3ª, com o histórico e os passos acumulados
    ldct_4 = dict(ldct, epocas=4)
    lres_4 = ut.treina(um.constroi_rede(ldct_4), lds_treino, lds_val, ldct_4,
                       torch.device("cpu"), fpth_retomada=lpth, fb_retoma=True,
                       fdct_identidade=ldct_id)
    assert [1, 2, 3, 4] == [lr["epoca"] for lr in lres_4.historico]
    assert lres_4.historico[:2] == lres_2.historico
    assert lres_4.passos == 2 * lres_2.passos

    # outro manifesto: recusa
    with pytest.raises(ValueError, match="manifesto_sha256"):
        ut.treina(um.constroi_rede(ldct_4), lds_treino, lds_val, ldct_4, torch.device("cpu"),
                  fpth_retomada=lpth, fb_retoma=True,
                  fdct_identidade=ut.identidade_retomada(ldct_4, "outro"))

# -------------------------------------------------------------------------------------
def test_peso_bins_auto(catalogo_sintetico):
    llst = _janelas(catalogo_sintetico)
    llst_pesos = uds.calcula_pesos_bins(llst, catalogo_sintetico["cappi_dir"], [10, 20, 30, 50])

    assert 5 == len(llst_pesos)
    # classe 0 (sem eco) é a mais frequente; a classe acima de 50 dBZ não existe: teto
    assert 1.0 == llst_pesos[0]
    assert uds.CF_PESO_BIN_MAXIMO == llst_pesos[-1]
    assert all(lf >= 1.0 for lf in llst_pesos)

    lcfg_dir = pathlib.Path(".")
    (lcfg_dir / "config.ini").write_text("[pytorch]\npeso_bins = auto\n")
    assert "auto" == um.hiperparametros(uc.carrega(), "pytorch", 4, 6)["peso_bins"]

# -------------------------------------------------------------------------------------
def test_treina_sem_treino_recusa(catalogo_sintetico):
    ldct = _hparams("rainnet")
    lds = uds.JanelasDataset([], catalogo_sintetico["cappi_dir"], ldct, ud.TransformacaoRainNet(), True)
    with pytest.raises(ValueError, match="vazio"):
        ut.treina(um.constroi_rede(ldct), lds, lds, ldct, torch.device("cpu"))

# < unt_inferencia >---------------------------------------------------------------------

def _modelo_salvo(fpth_dir, fs_backend, **fdct_extra):
    ldct = _hparams(fs_backend, **fdct_extra)
    um.salva_checkpoint(str(fpth_dir), ldct, um.constroi_rede(ldct),
                        um.transformacao_padrao(ldct), {}, fb_promove=True)
    ui.carrega_modelo_atual.cache_clear()
    return ui.carrega_modelo_atual(fs_backend, str(fpth_dir), "cpu")

# -------------------------------------------------------------------------------------
@pytest.mark.parametrize("fs_backend,fdct_extra", [("rainnet", {}), ("smaat", {}),
                                                   ("smaat", {"modo": "direto"}),
                                                   ("pytorch", {})])
def test_preve_dbz_shape_e_nan_fora_da_cobertura(tmp_path, fs_backend, fdct_extra):
    lmodelo = _modelo_salvo(tmp_path, fs_backend, **fdct_extra)

    lnp = np.full((4, 50, 70), 20.0, dtype=np.float32)
    lnp[-1, :5, :5] = np.nan

    lnp_prev = ui.preve_dbz(lmodelo, lnp, 6)

    assert (6, 50, 70) == lnp_prev.shape
    assert np.isnan(lnp_prev[:, :5, :5]).all()
    assert np.isfinite(lnp_prev[:, 10:, 10:]).all()

# -------------------------------------------------------------------------------------
def test_preve_dbz_numero_de_entradas_errado(tmp_path):
    lmodelo = _modelo_salvo(tmp_path, "rainnet")
    with pytest.raises(ValueError, match="entradas"):
        ui.preve_dbz(lmodelo, np.zeros((3, 32, 32), dtype=np.float32), 6)

# -------------------------------------------------------------------------------------
@pytest.mark.parametrize("fs_backend", um.CT_BACKENDS)
def test_gera_nowcast_contrato(tmp_path, monkeypatch, fs_backend):
    import importlib
    _modelo_salvo(tmp_path / "modelos", fs_backend)
    monkeypatch.setenv("UNT_MODELOS_DIR", str(tmp_path / "modelos"))
    monkeypatch.setenv("UNT_DEVICE", "cpu")

    lmod = importlib.import_module(f"work.unt_{fs_backend}")
    lt_radar = obc.GeometriaRadar(-29.2, -54.9, 400.0, 32.0)

    # sem chuva -> None
    lres = lmod.gera_nowcast(np.full((4, 64, 64), -4000.0), lt_radar, 6, 10, str(tmp_path / "a"))
    assert lres.previsao_dbz is None and lres.previsao_mm_h is None

    # tudo NaN -> ValueError
    with pytest.raises(ValueError, match="não finitos"):
        lmod.gera_nowcast(np.full((4, 64, 64), np.nan), lt_radar, 6, 10, str(tmp_path / "b"))

    # chuva -> 6 passos, 6 PNGs (o diretório é criado pelo untMain.executa_backend)
    (tmp_path / "c").mkdir()
    lnp = np.random.default_rng(0).uniform(20, 45, (4, 64, 64)).astype(np.float32)
    lres = lmod.gera_nowcast(lnp, lt_radar, 6, 10, str(tmp_path / "c" / "nowcast"))
    assert (6, 64, 64) == lres.previsao_dbz.shape
    for li in range(1, 7):
        assert (tmp_path / "c" / f"nowcast_{li:02d}.png").exists()

# -------------------------------------------------------------------------------------
def test_gera_nowcast_sem_checkpoint(tmp_path, monkeypatch):
    import work.unt_rainnet as lmod
    ui.carrega_modelo_atual.cache_clear()
    monkeypatch.setenv("UNT_MODELOS_DIR", str(tmp_path / "vazio"))
    lt_radar = obc.GeometriaRadar(-29.2, -54.9, 400.0, 32.0)
    with pytest.raises(FileNotFoundError):
        lmod.gera_nowcast(np.full((4, 32, 32), 30.0, dtype=np.float32), lt_radar, 6, 10,
                          str(tmp_path / "x"))

# < the end >----------------------------------------------------------------------------
