# -*- coding: utf-8 -*-
"""
Centro Logístico de Simulação Meteorológica
Copyright (c) 1995-2026, Instituto de Controle do Espaço Aéreo

test_untMakGIF
testes unitários de untMakGIF (decisão de gerar/regenerar GIF por backend e
orquestração de gera_gifs), com util.utl_gifs stubado via monkeypatch para
não depender de imagens PNG reais

2026.out  mlabru  portado de 25.oflow; nomes dos backends trocados (pytorch, rainnet, smaat).

2026.out  mlabru  GIF montado (backends lado a lado, carimbo)

2026.out  mlabru  nomes novos das funções testadas (refatoração de legibilidade)

2026.ago  mlabru  initial version (Linux/Python)
"""
# < imports >----------------------------------------------------------------------------

# python library
import json
import os
import pathlib
import time
import types

# PIL
from PIL import Image

# model
import untMakGIF as ng

# < gera_gif_do_diretorio >--------------------------------------------------------------

def test_gera_gif_do_diretorio_menos_de_3_frames_nao_gera_gif(monkeypatch):
    llst_cria_calls = []

    monkeypatch.setattr(ng.utg, "arquivos_estaveis", lambda *a, **k: ["a.png", "b.png"])
    monkeypatch.setattr(ng.utg, "precisa_regerar", lambda *a, **k: True)
    monkeypatch.setattr(ng.utg, "cria_gif", lambda *a, **k: llst_cria_calls.append(a))

    ng.gera_gif_do_diretorio(pathlib.Path("/fake/in"), pathlib.Path("/fake/out.gif"))

    assert 0 == len(llst_cria_calls)

# -------------------------------------------------------------------------------------
def test_gera_gif_do_diretorio_gif_ja_atualizado_nao_regenera(monkeypatch):
    llst_cria_calls = []

    monkeypatch.setattr(ng.utg, "arquivos_estaveis",
                        lambda *a, **k: ["a.png", "b.png", "c.png"])
    monkeypatch.setattr(ng.utg, "precisa_regerar", lambda *a, **k: False)
    monkeypatch.setattr(ng.utg, "cria_gif", lambda *a, **k: llst_cria_calls.append(a))

    ng.gera_gif_do_diretorio(pathlib.Path("/fake/in"), pathlib.Path("/fake/out.gif"))

    assert 0 == len(llst_cria_calls)

# -------------------------------------------------------------------------------------
def test_gera_gif_do_diretorio_gera_gif_quando_necessario(monkeypatch, tmp_path):
    llst_cria_calls = []

    monkeypatch.setattr(ng.utg, "arquivos_estaveis",
                        lambda *a, **k: ["a.png", "b.png", "c.png"])
    monkeypatch.setattr(ng.utg, "precisa_regerar", lambda *a, **k: True)
    def _falso_cria_gif(flst, fpth):
        llst_cria_calls.append(fpth)
        pathlib.Path(fpth).write_bytes(b"GIF89a")

    monkeypatch.setattr(ng.utg, "cria_gif", _falso_cria_gif)

    lpth_out = tmp_path / "sub" / "out.gif"
    ng.gera_gif_do_diretorio(tmp_path / "in", lpth_out)

    assert 1 == len(llst_cria_calls)
    # diretório de saída deve ter sido criado antes de gerar o GIF
    assert lpth_out.parent.is_dir()
    # gravado num temporário e renomeado para o destino (grava_gif_atomico)
    assert llst_cria_calls[0] != lpth_out
    assert b"GIF89a" == lpth_out.read_bytes()
    assert [lpth_out] == list(lpth_out.parent.iterdir())

# < gera_gifs — uso antigo (pasta plana, sem subdiretórios de backend) >-----------------

def test_gera_gifs_pasta_plana_gera_um_unico_gif(monkeypatch, tmp_path):
    lpth_image = tmp_path / "flat"
    lpth_image.mkdir()

    llst_calls = []
    monkeypatch.setattr(ng, "gera_gif_do_diretorio",
                        lambda fpth_in, fpth_out: llst_calls.append((fpth_in, fpth_out)))

    largs = types.SimpleNamespace(image=str(lpth_image), saida=str(tmp_path / "saida"),
                                  fname="nowcast.gif")
    ng.gera_gifs(largs)

    assert 1 == len(llst_calls)
    lpth_in, lpth_out = llst_calls[0]
    assert lpth_image == lpth_in
    assert (tmp_path / "saida" / "nowcast.gif") == lpth_out

# -------------------------------------------------------------------------------------
def test_gera_gifs_pasta_inexistente_nao_levanta(monkeypatch, tmp_path):
    llst_calls = []
    monkeypatch.setattr(ng, "gera_gif_do_diretorio",
                        lambda fpth_in, fpth_out: llst_calls.append((fpth_in, fpth_out)))

    largs = types.SimpleNamespace(image=str(tmp_path / "nao-existe"),
                                  saida=str(tmp_path / "saida"), fname="nowcast.gif")
    ng.gera_gifs(largs)

    # uso antigo: mesmo sem a pasta existir, tenta gerar um único GIF
    assert 1 == len(llst_calls)

# < gera_gifs — uso novo (um subdiretório por backend) >---------------------------------

def test_gera_gifs_gera_um_gif_por_backend(monkeypatch, tmp_path):
    lpth_base = tmp_path / "base"

    for ls_backend in ("pytorch", "rainnet"):
        lpth_backend = lpth_base / ls_backend
        lpth_backend.mkdir(parents=True)
        (lpth_backend / "nowcast_01.png").write_bytes(b"\x89PNG")

    llst_calls = []
    monkeypatch.setattr(ng, "gera_gif_do_diretorio",
                        lambda fpth_in, fpth_out: llst_calls.append((fpth_in, fpth_out)))

    largs = types.SimpleNamespace(image=str(lpth_base), saida=str(tmp_path / "saida"),
                                  fname="nowcast.gif")
    ng.gera_gifs(largs)

    assert 2 == len(llst_calls)

    ldct_out_by_backend = {fpth_in.name: fpth_out for fpth_in, fpth_out in llst_calls}
    assert (tmp_path / "saida" / "nowcast_rainnet.gif") == ldct_out_by_backend["rainnet"]
    assert (tmp_path / "saida" / "nowcast_pytorch.gif") == ldct_out_by_backend["pytorch"]

# -------------------------------------------------------------------------------------
def test_gera_gifs_falha_em_um_backend_nao_impede_os_demais(monkeypatch, tmp_path, caplog):
    lpth_base = tmp_path / "base"

    for ls_backend in ("pytorch", "rainnet"):
        lpth_backend = lpth_base / ls_backend
        lpth_backend.mkdir(parents=True)
        (lpth_backend / "nowcast_01.png").write_bytes(b"\x89PNG")

    llst_ok = []

    def _falso_gera_gif_do_diretorio(fpth_in, fpth_out):
        if "rainnet" == fpth_in.name:
            raise RuntimeError("falha simulada")
        llst_ok.append(fpth_in.name)

    monkeypatch.setattr(ng, "gera_gif_do_diretorio", _falso_gera_gif_do_diretorio)

    largs = types.SimpleNamespace(image=str(lpth_base), saida=str(tmp_path / "saida"),
                                  fname="nowcast.gif")
    ng.gera_gifs(largs)

    # pytorch concluiu apesar da falha em rainnet
    assert ["pytorch"] == llst_ok

# -------------------------------------------------------------------------------------
def test_gera_gifs_ignora_subdiretorios_sem_png(monkeypatch, tmp_path):
    lpth_base = tmp_path / "base"
    (lpth_base / "pytorch").mkdir(parents=True)
    (lpth_base / "pytorch" / "nowcast_01.png").write_bytes(b"\x89PNG")
    # subdiretório sem nenhum PNG -- não deve ser tratado como backend
    (lpth_base / "vazio").mkdir()

    llst_calls = []
    monkeypatch.setattr(ng, "gera_gif_do_diretorio",
                        lambda fpth_in, fpth_out: llst_calls.append(fpth_in.name))

    largs = types.SimpleNamespace(image=str(lpth_base), saida=str(tmp_path / "saida"),
                                  fname="nowcast.gif")
    ng.gera_gifs(largs)

    assert ["pytorch"] == llst_calls

# < GIF montado (backends lado a lado) >-------------------------------------------------

def _png(fpth: pathlib.Path, ft_cor: tuple) -> None:
    """
    PNG pequeno real, com mtime no passado (passa pela idade mínima).
    """
    fpth.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (100, 80), ft_cor).save(fpth)

    lf_antigo = time.time() - 60
    os.utime(fpth, (lf_antigo, lf_antigo))

# -------------------------------------------------------------------------------------
def _referencia(fpth_base: pathlib.Path, flst_backends: list) -> None:
    fpth_base.mkdir(parents=True, exist_ok=True)
    (fpth_base / ng.CS_ARQUIVO_REFERENCIA).write_text(json.dumps(
        {"radar": "CGU", "alcance": 250, "referencia": 1_781_000_000,
         "target": 1_781_000_600, "timestep_min": 10, "num_passos": 3,
         "backends": flst_backends}))

# -------------------------------------------------------------------------------------
def test_numero_do_passo():
    assert 3 == ng.numero_do_passo("/x/CGU_250_03.png")
    assert 12 == ng.numero_do_passo("nowcast_12.png")
    assert ng.numero_do_passo("/x/sem_numero.png") is None

# -------------------------------------------------------------------------------------
def test_texto_do_carimbo_com_referencia_em_utc():
    ldct_ref = {"radar": "CGU", "alcance": 250, "referencia": 1_781_000_000,
                "timestep_min": 10}

    # 1_781_000_000 = 2026-06-09 10:13:20 UTC; +20 min -> 10:33
    assert ("CGU 250 km | ref 2026-06-09 10:13 UTC | +20 min (10:33 UTC)"
            == ng.texto_do_carimbo(ldct_ref, 2))

# -------------------------------------------------------------------------------------
def test_texto_do_carimbo_sem_referencia_so_passo():
    assert "passo 04" == ng.texto_do_carimbo(None, 4)

# -------------------------------------------------------------------------------------
def test_ordena_backends_usa_referencia_e_senao_os_diretorios(tmp_path):
    llst_dirs = [tmp_path / "rainnet", tmp_path / "pytorch"]

    assert ["rainnet", "pytorch"] == ng.ordena_backends(llst_dirs, None)
    assert ["pytorch", "smaat", "rainnet"] == ng.ordena_backends(
        llst_dirs, {"backends": ["pytorch", "smaat", "rainnet"]})

# -------------------------------------------------------------------------------------
def test_gera_gif_montado_paineis_lado_a_lado_e_backend_ausente(tmp_path):
    lpth_base = tmp_path / "base"
    # smaat falhou: está na referência, mas sem frames
    _referencia(lpth_base, ["pytorch", "smaat", "rainnet"])

    for ls_backend, lt_cor in (("pytorch", (255, 0, 0)), ("rainnet", (0, 0, 255))):
        for li_passo in (1, 2, 3):
            _png(lpth_base / ls_backend / f"CGU_250_{li_passo:02d}.png", lt_cor)

    llst_dirs = ng.lista_diretorios_de_backend(lpth_base)
    lpth_gif = tmp_path / "saida" / "CGU_250_now_mosaico.gif"

    ng.gera_gif_montado(lpth_base, llst_dirs, lpth_gif)

    with Image.open(lpth_gif) as limg:
        assert 3 == limg.n_frames
        assert (3 * ng.CT_TAMANHO_PAINEL[0],
                ng.CI_ALTURA_CARIMBO + ng.CT_TAMANHO_PAINEL[1]) == limg.size

        # centro de cada painel: vermelho (pytorch), cinza (vazio), azul (rainnet)
        # (o frame pequeno não é ampliado: fica centrado no fundo preto)
        limg_rgb = limg.convert("RGB")
        li_y = ng.CI_ALTURA_CARIMBO + ng.CT_TAMANHO_PAINEL[1] // 2
        li_largura = ng.CT_TAMANHO_PAINEL[0]
        assert limg_rgb.getpixel((li_largura // 2, li_y))[0] > 200
        assert ng.CT_COR_PAINEL_VAZIO == limg_rgb.getpixel((li_largura + 5, li_y + 100))
        assert limg_rgb.getpixel((2 * li_largura + li_largura // 2, li_y))[2] > 200

# -------------------------------------------------------------------------------------
def test_gera_gif_montado_poucos_passos_nao_gera(tmp_path):
    lpth_base = tmp_path / "base"
    for li_passo in (1, 2):
        _png(lpth_base / "rainnet" / f"nowcast_{li_passo:02d}.png", (0, 0, 255))

    lpth_gif = tmp_path / "saida" / "nowcast_mosaico.gif"
    ng.gera_gif_montado(lpth_base, ng.lista_diretorios_de_backend(lpth_base), lpth_gif)

    assert not lpth_gif.exists()

# -------------------------------------------------------------------------------------
def test_gera_gifs_por_backend_tambem_gera_o_montado(monkeypatch, tmp_path):
    lpth_base = tmp_path / "base"
    for li_passo in (1, 2, 3):
        _png(lpth_base / "rainnet" / f"nowcast_{li_passo:02d}.png", (0, 0, 255))

    monkeypatch.setattr(ng, "gera_gif_do_diretorio", lambda fpth_in, fpth_out: None)

    largs = types.SimpleNamespace(image=str(lpth_base), saida=str(tmp_path / "saida"),
                                  fname="CGU_250_now.gif")
    ng.gera_gifs(largs)

    # sem referencia.json: sai assim mesmo, só com o passo carimbado
    assert (tmp_path / "saida" / "CGU_250_now_mosaico.gif").exists()

# < grava_gif_atomico >------------------------------------------------------------------

# -------------------------------------------------------------------------------------
def test_grava_gif_atomico_substitui_o_destino_sem_deixar_temporario(tmp_path):
    lpth_gif = tmp_path / "CGU_400_now_mosaico.gif"
    lpth_gif.write_bytes(b"antigo")

    llst_caminhos = []

    def _grava(fpth):
        llst_caminhos.append(fpth)
        fpth.write_bytes(b"novo")

    ng.grava_gif_atomico(lpth_gif, _grava)

    assert b"novo" == lpth_gif.read_bytes()
    assert llst_caminhos[0].parent == tmp_path
    assert llst_caminhos[0].name.startswith(".")
    assert llst_caminhos[0].name.endswith(".gif")
    assert [lpth_gif] == list(tmp_path.iterdir())

# -------------------------------------------------------------------------------------
def test_grava_gif_atomico_falha_mantem_destino_e_apaga_temporario(tmp_path):
    lpth_gif = tmp_path / "CGU_400_now_mosaico.gif"
    lpth_gif.write_bytes(b"antigo")

    def _grava_e_falha(fpth):
        fpth.write_bytes(b"pela metade")
        raise OSError("disco cheio")

    try:
        ng.grava_gif_atomico(lpth_gif, _grava_e_falha)
        assert False, "deveria ter propagado o erro"

    except OSError:
        pass

    assert b"antigo" == lpth_gif.read_bytes()
    assert [lpth_gif] == list(tmp_path.iterdir())

# -------------------------------------------------------------------------------------
def test_gera_gif_montado_nao_deixa_temporario(tmp_path):
    lpth_base = tmp_path / "base"
    for li_passo in (1, 2, 3):
        _png(lpth_base / "rainnet" / f"nowcast_{li_passo:02d}.png", (0, 0, 255))

    lpth_saida = tmp_path / "saida"
    ng.gera_gif_montado(lpth_base, ng.lista_diretorios_de_backend(lpth_base),
                        lpth_saida / "nowcast_mosaico.gif")

    assert ["nowcast_mosaico.gif"] == [p.name for p in lpth_saida.iterdir()]

# < the end >----------------------------------------------------------------------------
