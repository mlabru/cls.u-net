# -*- coding: utf-8 -*-
"""
Centro Logístico de Simulação Meteorológica
Copyright (c) 1995-2026, Instituto de Controle do Espaço Aéreo

test_untTrain_untEval
testes de ponta a ponta em CPU sobre o catálogo sintético: untJanelas ->
untTrain (os três backends) -> untEval (comparacao.json/.csv com os três +
persistência x 6 passos) e untMain com checkpoints reais (backends_treinados,
verify.json com FSS); candidato sem promoção -> untEval --checkpoint ->
untPromove (recusa, --forca, --reverte); escopo no untMain

2026.out  mlabru  semente fixa também na inicialização da rede; nova semente se colapsa
2026.out  mlabru  untEval com somas_por_bloco e bootstrap
2026.out  mlabru  untEval por par radar/alcance (resultados_por_par, CSV por par)
2026.out  mlabru  fluxo candidato/promoção/rollback e escopo
2026.out  mlabru  initial version (Linux/Python)
"""
# < imports >----------------------------------------------------------------------------

# python library
import csv
import json
import shutil
import sys

# pytest
import pytest

torch = pytest.importorskip("torch")

# model
import untEval as ne
import untJanelas as nj
import untMain as no
import untPromove as np_
import untTrain as nt
import work.unt_avaliacao as ua
import work.unt_inferencia as ui
import work.unt_modelos as um

# < fixture: manifesto + três modelos treinados >----------------------------------------

@pytest.fixture(scope="module")
def ambiente_treinado(tmp_path_factory):
    import conftest

    lpth = tmp_path_factory.mktemp("e2e")
    ldct_cat = conftest.cria_catalogo_sintetico(lpth / "sintetico", fi_num_volumes=60)

    # blocos de 1 hora: o catálogo sintético cobre ~10 h, então os três conjuntos aparecem
    (lpth / "c.ini").write_text("[janelas]\nbloco_split_s = 3600\n")

    lpth_manifesto = lpth / "janelas.csv"
    assert 0 == nj.main(["--catalog", ldct_cat["catalogo"], "--cappi-dir", ldct_cat["cappi_dir"],
                         "-o", str(lpth_manifesto), "--config", str(lpth / "c.ini")])

    lpth_modelos = lpth / "modelos"
    for ls_backend in um.CT_BACKENDS:
        assert 0 == nt.main(["-b", ls_backend, "--janelas", str(lpth_manifesto),
                             "--cappi-dir", ldct_cat["cappi_dir"],
                             "--modelos-dir", str(lpth_modelos), "--device", "cpu",
                             "--epocas", "1", "--recorte", "32", "--largura-base", "4",
                             "--workers", "0", "--batch", "4", "--threads", "2",
                             "--promove"])

    ui.carrega_modelo_atual.cache_clear()

    return {"raiz": lpth, "manifesto": lpth_manifesto, "modelos": lpth_modelos, **ldct_cat}

# < untTrain >---------------------------------------------------------------------------

def test_untTrain_grava_checkpoint_e_historico(ambiente_treinado):
    for ls_backend in um.CT_BACKENDS:
        lpth_atual = um.caminho_checkpoint_atual(str(ambiente_treinado["modelos"]), ls_backend)
        assert lpth_atual.exists()

        ldct_hist = json.loads(lpth_atual.resolve().with_suffix(".json").read_text())
        assert ls_backend == ldct_hist["hparams"]["backend"]
        assert 1 == len(ldct_hist["historico"])

        lmodelo = um.carrega_checkpoint(lpth_atual, "cpu")
        assert "cpu" == lmodelo.checkpoint["device"]

        # proveniência: escopo do manifesto (só STI_250 no catálogo sintético) e hash
        assert {"pares": ["STI_250"]} == ldct_hist["escopo"]
        assert {"pares": ["STI_250"]} == lmodelo.checkpoint["escopo"]
        assert um.sha256_de_arquivo(ambiente_treinado["manifesto"]) == ldct_hist["manifesto_sha256"]

        # seleção e orçamento registrados; o arquivo de retomada some no fim
        assert ldct_hist["criterio_selecao"] in ("metrica", "perda")
        assert ldct_hist["passos"] == ldct_hist["historico"][-1]["passos"] > 0
        assert not list(lpth_atual.parent.glob("*.retomada.pt"))

# -------------------------------------------------------------------------------------
def test_untTrain_manifesto_sem_treino_retorna_1(tmp_path, ambiente_treinado):
    import work.unt_janelas as uj
    llst, ldct = uj.le_manifesto(ambiente_treinado["manifesto"])
    llst_teste = [lj._replace(split="teste") for lj in llst]
    uj.grava_manifesto(llst_teste, tmp_path / "so_teste.csv", ldct)

    assert 1 == nt.main(["-b", "rainnet", "--janelas", str(tmp_path / "so_teste.csv"),
                         "--cappi-dir", ambiente_treinado["cappi_dir"],
                         "--modelos-dir", str(tmp_path / "m"), "--device", "cpu"])

# -------------------------------------------------------------------------------------
def test_untTrain_semente_fixa_inicializacao(tmp_path, monkeypatch, ambiente_treinado):
    """
    a semente vale também para os pesos iniciais (a rede é construída depois dela):
    mesma semente -> mesma inicialização; outra semente -> outra.
    """
    import work.unt_treino as ut
    llst_iniciais = []

    def treina_falso(frede, *fargs, **fkwargs):
        llst_iniciais.append({lk: lv.clone() for lk, lv in frede.state_dict().items()})
        raise ValueError("só a inicialização")

    monkeypatch.setattr(ut, "treina", treina_falso)

    for ls_semente in ("7", "7", "8"):
        assert 1 == nt.main(["-b", "rainnet", "--janelas", str(ambiente_treinado["manifesto"]),
                             "--cappi-dir", ambiente_treinado["cappi_dir"],
                             "--modelos-dir", str(tmp_path / "m"), "--device", "cpu",
                             "--recorte", "32", "--largura-base", "4", "--workers", "0",
                             "--semente", ls_semente])

    lpar_igual = zip(llst_iniciais[0].values(), llst_iniciais[1].values())
    lpar_outra = zip(llst_iniciais[0].values(), llst_iniciais[2].values())
    assert all(torch.equal(la, lb) for la, lb in lpar_igual)
    assert not all(torch.equal(la, lb) for la, lb in lpar_outra)

# -------------------------------------------------------------------------------------
def test_untTrain_recomeca_com_outra_semente_se_colapsa(tmp_path, monkeypatch,
                                                        ambiente_treinado):
    import work.unt_treino as ut
    llst_sementes = []

    def treina_falso(frede, fds_t, fds_v, fdct_hparams, *fargs, **fkwargs):
        llst_sementes.append(fdct_hparams["semente"])
        raise ut.TreinoColapsou("saída constante")

    monkeypatch.setattr(ut, "treina", treina_falso)
    llst_args = ["-b", "rainnet", "--janelas", str(ambiente_treinado["manifesto"]),
                 "--cappi-dir", ambiente_treinado["cappi_dir"],
                 "--modelos-dir", str(tmp_path / "m"), "--device", "cpu",
                 "--recorte", "32", "--largura-base", "4", "--workers", "0",
                 "--semente", "7"]

    # colapsa em todas as tentativas (5, padrão): sementes 7..11 e rc 1
    assert 1 == nt.main(llst_args)
    assert [7, 8, 9, 10, 11] == llst_sementes

    (tmp_path / "c.ini").write_text("[treino]\ntentativas_colapso = 1\n")
    llst_sementes.clear()
    assert 1 == nt.main(llst_args + ["--config", str(tmp_path / "c.ini")])
    assert [7] == llst_sementes

# < untEval >----------------------------------------------------------------------------

def test_untEval_compara_tres_backends_e_persistencia(tmp_path, ambiente_treinado):
    lpth_saida = tmp_path / "aval"
    li_rc = ne.main(["--janelas", str(ambiente_treinado["manifesto"]), "--split", "treino",
                     "--cappi-dir", ambiente_treinado["cappi_dir"],
                     "--modelos-dir", str(ambiente_treinado["modelos"]),
                     "--device", "cpu", "--saida", str(lpth_saida), "--max-janelas", "3"])
    assert 0 == li_rc

    ldct = json.loads((lpth_saida / ne.CS_ARQUIVO_JSON).read_text())
    assert set(ldct["resultados"]) == {"persistencia", *um.CT_BACKENDS}
    assert [str(li) for li in range(1, 7)] == list(ldct["resultados"]["rainnet"])
    assert set(ldct["desempenho"]) == set(um.CT_BACKENDS)
    assert ldct["desempenho"]["smaat"]["inferencia_ms_media"] > 0

    # persistência: o núcleo anda 2 px por passo, então o acerto cai com o lead e o
    # FSS numa escala maior perdoa mais que o CSI ponto a ponto
    ldct_p1 = ldct["resultados"]["persistencia"]["1"]
    ldct_p6 = ldct["resultados"]["persistencia"]["6"]
    assert ldct_p1["limiares"]["20"]["csi"] > 0.3
    assert ldct_p1["limiares"]["20"]["csi"] > ldct_p6["limiares"]["20"]["csi"]
    assert ldct_p1["fss"]["30"]["20"] >= ldct_p1["fss"]["30"]["1"]

    with (lpth_saida / ne.CS_ARQUIVO_CSV).open() as larq:
        llst_linhas = list(csv.DictReader(larq))
    assert 4 * 6 == len(llst_linhas)
    assert "fss30_10km" in llst_linhas[0]
    assert "par" not in llst_linhas[0]

    # um só par no catálogo sintético: o resultado do par é o agregado
    assert {"STI_250": 3} == ldct["janelas_por_par"]
    assert {"STI_250": ldct["resultados"]} == ldct["resultados_por_par"]

    # bootstrap por bloco: as somas dos blocos reagregadas dão o mesmo CSI do conjunto
    assert ldct["somas_por_bloco"] and ldct["bootstrap"]["blocos"] == len(ldct["somas_por_bloco"])
    li_hits = sum(ldct_b["persistencia"]["1"]["limiares"]["20"]["hits"]
                  for ldct_b in ldct["somas_por_bloco"].values())
    assert li_hits == ldct["resultados"]["persistencia"]["1"]["limiares"]["20"]["hits"]
    assert "dif_persistencia" in ldct["bootstrap"]["resultados"]["rainnet"]["1"]
    assert "dif_persistencia" not in ldct["bootstrap"]["resultados"]["persistencia"]["1"]

    with (lpth_saida / ne.CS_ARQUIVO_CSV_POR_PAR).open() as larq:
        llst_linhas_par = list(csv.DictReader(larq))
    assert 4 * 6 == len(llst_linhas_par)
    assert {"STI_250"} == {ldct_linha["par"] for ldct_linha in llst_linhas_par}

# -------------------------------------------------------------------------------------
def test_untEval_separa_por_par(tmp_path, ambiente_treinado):
    import work.unt_janelas as uj

    # segundo par: cópia dos CAPPI do STI_250 renomeada como XYZ_250, e metade das
    # janelas do manifesto atribuídas a ele
    lpth_cappi = tmp_path / "cappi"
    shutil.copytree(ambiente_treinado["cappi_dir"], lpth_cappi)

    for lpth_dir in lpth_cappi.glob("*/STI_250"):
        lpth_novo = lpth_dir.with_name("XYZ_250")
        lpth_novo.mkdir()
        for lpth_npy in lpth_dir.glob("STI_250_*.npy"):
            shutil.copy(lpth_npy, lpth_novo / lpth_npy.name.replace("STI_250", "XYZ_250"))

    llst, ldct_man = uj.le_manifesto(ambiente_treinado["manifesto"])
    llst = [lj._replace(split="treino", radar="XYZ" if li % 2 else lj.radar)
            for li, lj in enumerate(llst[:4])]
    uj.grava_manifesto(llst, tmp_path / "dois_pares.csv", ldct_man)

    lpth_saida = tmp_path / "aval"
    assert 0 == ne.main(["--janelas", str(tmp_path / "dois_pares.csv"), "--split", "treino",
                         "--cappi-dir", str(lpth_cappi),
                         "--modelos-dir", str(ambiente_treinado["modelos"]),
                         "--backends", "rainnet", "--device", "cpu",
                         "--saida", str(lpth_saida)])

    ldct = json.loads((lpth_saida / ne.CS_ARQUIVO_JSON).read_text())
    assert {"STI_250": 2, "XYZ_250": 2} == ldct["janelas_por_par"]
    assert ["STI_250", "XYZ_250"] == list(ldct["resultados_por_par"])

    # contingência somada por par fecha com a do agregado
    for ls_rotulo in ("persistencia", "rainnet"):
        for ls_passo, ldct_agregado in ldct["resultados"][ls_rotulo].items():
            llst_por_par = [ldct["resultados_por_par"][ls_par][ls_rotulo][ls_passo]
                            for ls_par in ("STI_250", "XYZ_250")]
            assert ldct_agregado["janelas"] == sum(ld["janelas"] for ld in llst_por_par)

            for ls_limiar, ldct_lim in ldct_agregado["limiares"].items():
                for ls_chave in ("hits", "misses", "false_alarms"):
                    assert ldct_lim[ls_chave] == sum(ld["limiares"][ls_limiar][ls_chave]
                                                     for ld in llst_por_par)

    with (lpth_saida / ne.CS_ARQUIVO_CSV_POR_PAR).open() as larq:
        llst_linhas = list(csv.DictReader(larq))
    assert 2 * 2 * 6 == len(llst_linhas)
    assert "par" == list(llst_linhas[0])[0]

# -------------------------------------------------------------------------------------
def test_untEval_split_vazio_retorna_1(tmp_path, ambiente_treinado):
    import work.unt_janelas as uj
    llst, ldct = uj.le_manifesto(ambiente_treinado["manifesto"])
    uj.grava_manifesto([lj._replace(split="treino") for lj in llst], tmp_path / "t.csv", ldct)

    assert 1 == ne.main(["--janelas", str(tmp_path / "t.csv"), "--split", "teste",
                         "--cappi-dir", ambiente_treinado["cappi_dir"]])

# -------------------------------------------------------------------------------------
def test_acumulador_soma_contingencia_antes_das_razoes():
    lacum = ua.AcumuladorLead()
    for li_h, li_m, li_f in ((10, 0, 0), (0, 10, 10)):
        lacum.adiciona({"limiares": {"30": {"hits": li_h, "misses": li_m, "false_alarms": li_f}},
                        "fss": {"30": {"10": 0.5}}, "pixels_eco": 0, "mae_dbz": None,
                        "bias_dbz": None})
    ldct = lacum.resumo()
    # 10 / (10 + 10 + 10), não a média de 1.0 e 0.0
    assert ldct["limiares"]["30"]["csi"] == pytest.approx(1 / 3)
    assert ldct["fss"]["30"]["10"] == pytest.approx(0.5)
    assert ldct["mae_dbz"] is None

# -------------------------------------------------------------------------------------
def test_acumulador_fss_soma_os_termos_antes_da_razao():
    lacum = ua.AcumuladorLead()
    # janela grande quase perfeita (FSS 0.9) e janela pequena sem acerto (FSS 0.0)
    for lf_erro, lf_ref in ((10.0, 100.0), (2.0, 2.0)):
        lacum.adiciona({"limiares": {}, "pixels_eco": 0, "mae_dbz": None, "bias_dbz": None,
                        "fss": {"30": {"10": 1.0 - lf_erro / lf_ref}},
                        "fss_somas": {"30": {"10": {"erro": lf_erro, "referencia": lf_ref}}}})
    ldct = lacum.resumo()
    # 1 - 12 / 102, não a média de 0.9 e 0.0
    assert ldct["fss"]["30"]["10"] == pytest.approx(1.0 - 12.0 / 102.0)
    assert ldct["fss_media_janelas"]["30"]["10"] == pytest.approx(0.45)

# < untMain com checkpoints reais >------------------------------------------------------

class _PoolSerial:
    def __init__(self, processes=None):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def map(self, ff, flst):
        return [ff(lx) for lx in flst]

# -------------------------------------------------------------------------------------
def test_backends_treinados_pula_sem_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(no, "MLST_BACKENDS_DISPONIVEIS", list(um.CT_BACKENDS))
    (tmp_path / "smaat").mkdir()
    (tmp_path / "smaat" / "atual.pt").write_bytes(b"x")

    assert ["smaat"] == no.backends_treinados(str(tmp_path))

# -------------------------------------------------------------------------------------
def test_backends_treinados_respeita_escopo(tmp_path, monkeypatch):
    monkeypatch.setattr(no, "MLST_BACKENDS_DISPONIVEIS", list(um.CT_BACKENDS))

    for ls_backend, ls_json in (("pytorch", '{"escopo": {"pares": ["STI_250"]}}'),
                                ("rainnet", '{"escopo": {"pares": ["SBMT_400"]}}'),
                                ("smaat", None)):
        (tmp_path / ls_backend).mkdir()
        (tmp_path / ls_backend / "x_1.pt").write_bytes(b"x")
        (tmp_path / ls_backend / "atual.pt").symlink_to("x_1.pt")
        if ls_json:
            (tmp_path / ls_backend / "x_1.json").write_text(ls_json)

    # smaat: escopo desconhecido (sem .json) roda, com aviso
    assert ["pytorch", "smaat"] == no.backends_treinados(str(tmp_path), "STI", 250)
    assert ["rainnet", "smaat"] == no.backends_treinados(str(tmp_path), "SBMT", 400)
    assert list(um.CT_BACKENDS) == no.backends_treinados(str(tmp_path))

# -------------------------------------------------------------------------------------
def test_candidato_avaliado_promovido_e_revertido(tmp_path, ambiente_treinado):
    # cópia dos modelos: não mexe no atual.pt do fixture compartilhado
    lpth_modelos = tmp_path / "modelos"
    shutil.copytree(ambiente_treinado["modelos"], lpth_modelos, symlinks=True)
    lpth_atual = um.caminho_checkpoint_atual(str(lpth_modelos), "rainnet")
    ls_era_atual = lpth_atual.resolve().name

    # 1. treino sem --promove: candidato gravado, atual.pt intocado
    llst_antes = set((lpth_modelos / "rainnet").glob("rainnet_*.pt"))
    assert 0 == nt.main(["-b", "rainnet", "--janelas", str(ambiente_treinado["manifesto"]),
                         "--cappi-dir", ambiente_treinado["cappi_dir"],
                         "--modelos-dir", str(lpth_modelos), "--device", "cpu",
                         "--epocas", "1", "--recorte", "32", "--largura-base", "4",
                         "--workers", "0", "--batch", "4", "--threads", "2"])
    assert ls_era_atual == lpth_atual.resolve().name

    (lpth_candidato,) = set((lpth_modelos / "rainnet").glob("rainnet_*.pt")) - llst_antes

    # 2. avaliação do candidato ao lado do atual
    lpth_aval = tmp_path / "aval"
    assert 0 == ne.main(["--janelas", str(ambiente_treinado["manifesto"]), "--split", "treino",
                         "--cappi-dir", ambiente_treinado["cappi_dir"],
                         "--modelos-dir", str(lpth_modelos), "--backends", "rainnet",
                         "--device", "cpu", "--saida", str(lpth_aval), "--max-janelas", "3",
                         "--checkpoint", f"rainnet_candidato={lpth_candidato}"])
    ldct_aval = json.loads((lpth_aval / ne.CS_ARQUIVO_JSON).read_text())
    assert {"persistencia", "rainnet", "rainnet_candidato"} == set(ldct_aval["resultados"])
    assert um.sha256_de_arquivo(lpth_candidato) == \
        ldct_aval["desempenho"]["rainnet_candidato"]["sha256"]

    # 3. portão: o conjunto avaliado (treino) não é o exigido (teste) -> recusa
    llst_base = ["-b", "rainnet", "--candidato", str(lpth_candidato),
                 "--avaliacao", str(lpth_aval / ne.CS_ARQUIVO_JSON),
                 "--modelos-dir", str(lpth_modelos)]
    assert 1 == np_.main(llst_base)
    assert ls_era_atual == lpth_atual.resolve().name

    # 4. --forca promove (registrado), --reverte volta
    assert 0 == np_.main(llst_base + ["--forca"])
    assert lpth_candidato.name == lpth_atual.resolve().name
    lpth_anterior = um.caminho_checkpoint_anterior(str(lpth_modelos), "rainnet")
    assert ls_era_atual == lpth_anterior.resolve().name

    assert 0 == np_.main(["-b", "rainnet", "--reverte", "--modelos-dir", str(lpth_modelos)])
    assert ls_era_atual == lpth_atual.resolve().name

    llst_registro = [json.loads(ls) for ls in
                     (lpth_modelos / "rainnet" / np_.CS_ARQUIVO_REGISTRO).read_text().splitlines()]
    assert ["recusa", "promove", "reverte"] == [ld["acao"] for ld in llst_registro]
    assert llst_registro[1]["forcado"] is True

# -------------------------------------------------------------------------------------
def test_untMain_com_modelos_reais_e_verify(tmp_path, monkeypatch, ambiente_treinado):
    monkeypatch.setattr(no.mp, "Pool", _PoolSerial)
    for ls_var in ("UNT_MODELOS_DIR", "UNT_DEVICE", "UNT_THREADS"):
        monkeypatch.delenv(ls_var, raising=False)
    ui.carrega_modelo_atual.cache_clear()

    lpth_saida = tmp_path / "saida"
    monkeypatch.setattr(sys, "argv", ["untMain.py", "-r", "STI", "-a", "250",
                                      "--catalog", ambiente_treinado["catalogo"],
                                      "--cappi-dir", ambiente_treinado["cappi_dir"],
                                      "--modelos-dir", str(ambiente_treinado["modelos"]),
                                      "--device", "cpu", "-s", str(lpth_saida),
                                      "-p", "STI_250", "--verify"])
    assert 0 == no.main()

    ldct_v = json.loads((lpth_saida / "verify.json").read_text())
    assert set(ldct_v["backends"]) == set(um.CT_BACKENDS)
    assert "fss" in ldct_v["persistencia"]

    for ls_backend in um.CT_BACKENDS:
        assert (lpth_saida / ls_backend / "STI_250_06.png").exists()
        assert (lpth_saida / ls_backend / "STI_250.npy").exists()

# < the end >----------------------------------------------------------------------------
