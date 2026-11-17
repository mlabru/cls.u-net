# -*- coding: utf-8 -*-
"""
Centro Logístico de Simulação Meteorológica
Copyright (c) 1995-2026, Instituto de Controle do Espaço Aéreo

untMakGIF

monta os GIFs animados a partir dos frames PNG gravados pelo untMain.py.

  - modo por backend (o normal): --image é o diretório de saída do
    untMain.py, com um subdiretório por backend (pytorch/, rainnet/, smaat/).
    Cada subdiretório com PNGs vira um GIF <nome>_<backend>.gif; falha num
    backend não impede os outros. Além deles, um GIF montado
    <nome>_mosaico.gif com os backends lado a lado e, em cada frame, o
    horário de referência e o passo carimbados (lidos de referencia.json,
    gravado pelo untMain.py; sem ele, só o passo).
  - modo plano (uso antigo): --image não tem subdiretórios com PNG; gera um
    único GIF com o nome de --fname.

um GIF só é (re)gerado se houver frames suficientes, já completamente
gravados, e mais novos que o GIF existente (util/utl_gifs).

todo GIF é gravado de forma atômica (grava_gif_atomico): primeiro num
arquivo temporário oculto no mesmo diretório, depois renomeado por cima do
destino. Quem lê os GIFs (o dashboard 19.dash) nunca pega um arquivo pela
metade.

$ python untMakGIF.py -i ./saida -s ./out -f nowcast.gif

2026.out  mlabru  portado de 25.oflow (oflMakGIF); backends pytorch, rainnet
                  e smaat.

2026.out  mlabru  gravação atômica dos GIFs (grava_gif_atomico: temporário
                  + os.replace); feita aqui e não em util/utl_gifs, que é
                  compartilhado com o 23.goes.

2026.out  mlabru  GIF montado (backends lado a lado) com horário de
                  referência/passo carimbado: <nome>_mosaico.gif.

2026.out  mlabru  modos plano e por backend em funções próprias;
                  CI_MINIMO_FRAMES; main() devolve o código de saída;
                  removidos imports sem uso.

2026.ago  mlabru  um GIF independente por subdiretório de backend
                  (pysteps, rainymotion, opencv); mantém o uso antigo
                  (pasta plana de frames, um único GIF).

2025.nov  mlabru  initial version (Linux/Python)
"""
# < imports >----------------------------------------------------------------------------

# python library
import argparse
import datetime
import json
import logging
import os
import pathlib
import re
import sys
import typing

# PIL
from PIL import Image, ImageDraw, ImageFont

# utils (módulo compartilhado com outros projetos cls: apps/util é symlink)
import util.utl_gifs as utg

# < constants >--------------------------------------------------------------------------

# idade mínima (s) de um PNG para ser usado: arquivo mais novo que isso pode
# ainda estar sendo gravado
CI_IDADE_MINIMA_FRAME_SEGUNDOS = 3

# idade mínima em uso (--idade-minima). O unt_common.sh passa 0: ele só chama o
# untMakGIF depois que o untMain.py terminou, quando todos os PNGs já estão
# completos. Com 3 s, os PNGs do último backend a terminar (gravados no último
# segundo) eram descartados e o diretório temporário apagado em seguida -- o GIF
# daquele backend virava "sem previsão" (2026-11-04: 7 de 31 passadas)
gi_idade_minima_frame = CI_IDADE_MINIMA_FRAME_SEGUNDOS

# menos frames que isso não vale um GIF animado
CI_MINIMO_FRAMES = 3

# GIF montado: "backend" usado no nome (<nome>_mosaico.gif) e arquivo com o
# horário de referência gravado pelo untMain.py (CS_ARQUIVO_REFERENCIA)
CS_NOME_MOSAICO = "mosaico"
CS_ARQUIVO_REFERENCIA = "referencia.json"

# tamanho de cada painel (um por backend), faixa do carimbo no topo e
# duração de cada frame do GIF montado
CT_TAMANHO_PAINEL = (600, 480)
CI_ALTURA_CARIMBO = 40
CI_TAMANHO_FONTE = 22
CI_DURACAO_FRAME_MS = 500

# cores (RGB): fundo do carimbo / texto / painel de backend sem frame
CT_COR_CARIMBO = (20, 20, 20)
CT_COR_TEXTO = (255, 255, 255)
CT_COR_PAINEL_VAZIO = (60, 60, 60)

# fonte TrueType do carimbo (se ausente, usa a fonte padrão do PIL)
CS_FONTE = "DejaVuSans.ttf"

# número do passo no nome do frame: <prefixo>_NN.png
MRE_NUMERO_DO_PASSO = re.compile(r"_(\d+)$")

# < logging >----------------------------------------------------------------------------

# logger
M_LOG = logging.getLogger(__name__)
M_LOG.setLevel(logging.INFO)

# -------------------------------------------------------------------------------------
def gera_gif_do_diretorio(fpth_diretorio_frames: pathlib.Path,
                          fpth_gif_saida: pathlib.Path) -> None:
    """
    gera (ou regenera) um GIF com os PNGs de um diretório.

    não faz nada se houver menos de CI_MINIMO_FRAMES frames prontos ou se o
    GIF já estiver mais novo que todos os frames.

    :param fpth_diretorio_frames: diretório com os <prefixo>_NN.png
    :param fpth_gif_saida: GIF a gravar
    """
    llst_frames = utg.arquivos_estaveis(fpth_diretorio_frames, "png",
                                        gi_idade_minima_frame)

    if len(llst_frames) < CI_MINIMO_FRAMES:
        return

    if not utg.precisa_regerar(llst_frames, fpth_gif_saida):
        return

    fpth_gif_saida.parent.mkdir(parents=True, exist_ok=True)

    grava_gif_atomico(fpth_gif_saida, lambda fpth_temp: utg.cria_gif(llst_frames, fpth_temp))

    M_LOG.info(f"[✓] GIF gerado: {fpth_gif_saida}")

# -------------------------------------------------------------------------------------
def grava_gif_atomico(fpth_gif_saida: pathlib.Path,
                      fcb_grava: typing.Callable[[pathlib.Path], None]) -> None:
    """
    grava um GIF sem que um leitor veja o arquivo pela metade.

    fluxo:
      1. fcb_grava grava em ".<nome>.tmp.gif", no mesmo diretório do destino
         (mesmo sistema de arquivos: o passo 2 é um rename, atômico). O ponto
         inicial esconde o temporário de quem lista o diretório; o ".gif" no
         fim é porque o PIL escolhe o formato pela extensão;
      2. os.replace troca o destino pelo temporário de uma vez.
    Se a gravação falhar, o temporário é apagado, o destino anterior fica
    intacto e o erro é propagado.

    :param fpth_gif_saida: GIF final
    :param fcb_grava: função que grava o GIF no caminho recebido
    :raises Exception: o erro da gravação ou do rename
    """
    lpth_temp = fpth_gif_saida.with_name(f".{fpth_gif_saida.name}.tmp.gif")

    try:
        fcb_grava(lpth_temp)
        os.replace(lpth_temp, fpth_gif_saida)

    except Exception:
        lpth_temp.unlink(missing_ok=True)
        raise

# -------------------------------------------------------------------------------------
def lista_diretorios_de_backend(fpth_base: pathlib.Path) -> list:
    """
    subdiretórios de fpth_base que contêm ao menos um PNG (cada um é a
    saída de um backend), em ordem alfabética.

    :param fpth_base: diretório de saída do untMain.py

    :returns: lista de diretórios (vazia se fpth_base não existe)
    """
    if not fpth_base.is_dir():
        return []

    llst_diretorios = []

    for lpth_item in sorted(fpth_base.iterdir()):
        if not lpth_item.is_dir():
            continue

        if not any(lpth_item.glob("*.png")):
            continue

        llst_diretorios.append(lpth_item)

    return llst_diretorios

# -------------------------------------------------------------------------------------
def nome_do_gif_do_backend(fs_nome_gif: str, fs_nome_backend: str) -> str:
    """
    nome do GIF de um backend: "nowcast.gif" + "opencv" -> "nowcast_opencv.gif".
    """
    lpth_nome = pathlib.Path(fs_nome_gif)

    return f"{lpth_nome.stem}_{fs_nome_backend}{lpth_nome.suffix}"

# -------------------------------------------------------------------------------------
def gera_gif_unico(fpth_base: pathlib.Path, fpth_saida: pathlib.Path, fs_nome_gif: str) -> None:
    """
    modo plano (uso antigo): um único GIF com os PNGs de fpth_base. Erros
    são logados, nunca propagados.
    """
    try:
        gera_gif_do_diretorio(fpth_base, fpth_saida / fs_nome_gif)

    except Exception as le_erro:
        M_LOG.error(f"{le_erro}")

# -------------------------------------------------------------------------------------
def gera_gifs_por_backend(flst_diretorios_backend: list, fpth_saida: pathlib.Path,
                          fs_nome_gif: str) -> None:
    """
    modo por backend: um GIF por subdiretório. Erro num backend é logado e
    não impede os demais.
    """
    for lpth_diretorio in flst_diretorios_backend:
        ls_nome_backend = lpth_diretorio.name

        lpth_gif = fpth_saida / nome_do_gif_do_backend(fs_nome_gif, ls_nome_backend)

        try:
            gera_gif_do_diretorio(lpth_diretorio, lpth_gif)

        except Exception as le_erro:
            M_LOG.error(f"[{ls_nome_backend}] {le_erro}")

# < GIF montado (backends lado a lado) >-------------------------------------------------

def le_referencia(fpth_base: pathlib.Path) -> typing.Optional[dict]:
    """
    lê o referencia.json gravado pelo untMain.py.

    :param fpth_base: diretório de saída do untMain.py

    :returns: dicionário, ou None se ausente/ilegível
    """
    lpth_json = fpth_base / CS_ARQUIVO_REFERENCIA

    if not lpth_json.is_file():
        return None

    try:
        return json.loads(lpth_json.read_text())

    except (OSError, ValueError) as le_erro:
        M_LOG.warning(f"[!] {lpth_json} ilegível: {le_erro}")
        return None

# -------------------------------------------------------------------------------------
def numero_do_passo(fs_arquivo_png: str) -> typing.Optional[int]:
    """
    "…/CGU_250_03.png" -> 3; None se o nome não termina em _NN.
    """
    lmatch = MRE_NUMERO_DO_PASSO.search(pathlib.Path(fs_arquivo_png).stem)

    if lmatch is None:
        return None

    return int(lmatch.group(1))

# -------------------------------------------------------------------------------------
def frames_por_passo(fpth_diretorio: pathlib.Path) -> dict:
    """
    PNGs prontos de um backend, indexados pelo número do passo.

    :param fpth_diretorio: subdiretório do backend

    :returns: {passo: caminho do PNG}
    """
    ldct_frames = {}

    for ls_arquivo in utg.arquivos_estaveis(fpth_diretorio, "png",
                                            gi_idade_minima_frame):
        li_passo = numero_do_passo(ls_arquivo)

        if li_passo is not None:
            ldct_frames[li_passo] = ls_arquivo

    return ldct_frames

# -------------------------------------------------------------------------------------
def ordena_backends(flst_diretorios_backend: list,
                    fdct_referencia: typing.Optional[dict]) -> list:
    """
    ordem dos painéis: a dos backends executados (referencia.json), que
    inclui os que falharam (viram painel vazio); sem referência, os
    subdiretórios com PNG em ordem alfabética.

    :returns: nomes dos backends
    """
    if fdct_referencia and fdct_referencia.get("backends"):
        return list(fdct_referencia["backends"])

    return [lpth_diretorio.name for lpth_diretorio in flst_diretorios_backend]

# -------------------------------------------------------------------------------------
def formata_hora_utc(fi_epoch: int, fs_formato: str) -> str:
    """
    epoch (s) -> texto em UTC no formato dado.
    """
    ldt_hora = datetime.datetime.fromtimestamp(fi_epoch, tz=datetime.timezone.utc)

    return ldt_hora.strftime(fs_formato)

# -------------------------------------------------------------------------------------
def texto_do_carimbo(fdct_referencia: typing.Optional[dict], fi_passo: int) -> str:
    """
    texto carimbado no frame do passo fi_passo, ex.:
    "CGU 250 km | ref 2026-10-09 12:00 UTC | +10 min (12:10 UTC)".
    Sem referência, só "passo NN".
    """
    if not fdct_referencia:
        return f"passo {fi_passo:02d}"

    li_referencia = int(fdct_referencia["referencia"])
    li_minutos = fi_passo * int(fdct_referencia["timestep_min"])
    li_valido = li_referencia + li_minutos * 60

    ls_referencia = formata_hora_utc(li_referencia, "%Y-%m-%d %H:%M UTC")
    ls_valido = formata_hora_utc(li_valido, "%H:%M UTC")

    return (f"{fdct_referencia['radar']} {fdct_referencia['alcance']} km | "
            f"ref {ls_referencia} | +{li_minutos} min ({ls_valido})")

# -------------------------------------------------------------------------------------
def carrega_fonte() -> ImageFont.ImageFont:
    """
    fonte do carimbo: CS_FONTE se instalada, senão a padrão do PIL.
    """
    try:
        return ImageFont.truetype(CS_FONTE, CI_TAMANHO_FONTE)

    except OSError:
        return ImageFont.load_default()

# -------------------------------------------------------------------------------------
def monta_painel(fs_arquivo_png: typing.Optional[str], fs_nome_backend: str,
                 ffonte: ImageFont.ImageFont) -> Image.Image:
    """
    painel de um backend: o frame redimensionado (proporção mantida) ou,
    sem frame, um painel cinza "<backend>: sem previsão".
    """
    if fs_arquivo_png is not None:
        return utg.redimensionar_com_proporcao(fs_arquivo_png, CT_TAMANHO_PAINEL)

    limg_painel = Image.new("RGB", CT_TAMANHO_PAINEL, CT_COR_PAINEL_VAZIO)

    ldraw = ImageDraw.Draw(limg_painel)
    ldraw.text((CT_TAMANHO_PAINEL[0] // 2, CT_TAMANHO_PAINEL[1] // 2),
               f"{fs_nome_backend}: sem previsão", fill=CT_COR_TEXTO, font=ffonte,
               anchor="mm")

    return limg_painel

# -------------------------------------------------------------------------------------
def monta_frame_montado(flst_paineis: list, fs_carimbo: str,
                        ffonte: ImageFont.ImageFont) -> Image.Image:
    """
    painéis lado a lado, com a faixa do carimbo no topo.
    """
    li_largura = CT_TAMANHO_PAINEL[0] * len(flst_paineis)
    li_altura = CI_ALTURA_CARIMBO + CT_TAMANHO_PAINEL[1]

    limg_frame = Image.new("RGB", (li_largura, li_altura), CT_COR_CARIMBO)

    for li_indice, limg_painel in enumerate(flst_paineis):
        limg_frame.paste(limg_painel, (li_indice * CT_TAMANHO_PAINEL[0], CI_ALTURA_CARIMBO))

    ldraw = ImageDraw.Draw(limg_frame)
    ldraw.text((li_largura // 2, CI_ALTURA_CARIMBO // 2), fs_carimbo,
               fill=CT_COR_TEXTO, font=ffonte, anchor="mm")

    return limg_frame

# -------------------------------------------------------------------------------------
def gera_gif_montado(fpth_base: pathlib.Path, flst_diretorios_backend: list,
                     fpth_gif_saida: pathlib.Path) -> None:
    """
    gera (ou regenera) o GIF montado: um frame por passo, com o frame de
    cada backend lado a lado e o horário de referência/passo carimbado.

    não faz nada se houver menos de CI_MINIMO_FRAMES passos com frame ou se
    o GIF já estiver mais novo que todos os frames.

    :param fpth_base: diretório de saída do untMain.py
    :param flst_diretorios_backend: subdiretórios de backend com PNG
    :param fpth_gif_saida: GIF a gravar
    """
    ldct_referencia = le_referencia(fpth_base)
    llst_backends = ordena_backends(flst_diretorios_backend, ldct_referencia)

    # {backend: {passo: png}}; backend sem subdiretório fica sem frames
    ldct_frames = {}
    for ls_nome_backend in llst_backends:
        ldct_frames[ls_nome_backend] = frames_por_passo(fpth_base / ls_nome_backend)

    llst_todos_os_frames = []
    for ldct_do_backend in ldct_frames.values():
        llst_todos_os_frames.extend(ldct_do_backend.values())

    llst_passos = sorted({li_passo for ldct_do_backend in ldct_frames.values()
                          for li_passo in ldct_do_backend})

    if len(llst_passos) < CI_MINIMO_FRAMES:
        return

    if not utg.precisa_regerar(llst_todos_os_frames, fpth_gif_saida):
        return

    lfonte = carrega_fonte()

    llst_imagens = []
    for li_passo in llst_passos:
        llst_paineis = []

        for ls_nome_backend in llst_backends:
            ls_arquivo_png = ldct_frames[ls_nome_backend].get(li_passo)
            llst_paineis.append(monta_painel(ls_arquivo_png, ls_nome_backend, lfonte))

        ls_carimbo = texto_do_carimbo(ldct_referencia, li_passo)
        llst_imagens.append(monta_frame_montado(llst_paineis, ls_carimbo, lfonte))

    fpth_gif_saida.parent.mkdir(parents=True, exist_ok=True)

    grava_gif_atomico(fpth_gif_saida,
                      lambda fpth_temp: llst_imagens[0].save(fpth_temp, save_all=True,
                                                             append_images=llst_imagens[1:],
                                                             duration=CI_DURACAO_FRAME_MS,
                                                             loop=0))

    M_LOG.info(f"[✓] GIF montado gerado: {fpth_gif_saida}")

# -------------------------------------------------------------------------------------
def gera_gif_montado_sem_propagar(fpth_base: pathlib.Path, flst_diretorios_backend: list,
                                  fpth_saida: pathlib.Path, fs_nome_gif: str) -> None:
    """
    gera <nome>_mosaico.gif. Erro é logado, nunca propagado: os GIFs por
    backend já foram gerados.
    """
    lpth_gif = fpth_saida / nome_do_gif_do_backend(fs_nome_gif, CS_NOME_MOSAICO)

    try:
        gera_gif_montado(fpth_base, flst_diretorios_backend, lpth_gif)

    except Exception as le_erro:
        M_LOG.error(f"[{CS_NOME_MOSAICO}] {le_erro}")

# -------------------------------------------------------------------------------------
def gera_gifs(fargs: argparse.Namespace) -> None:
    """
    escolhe o modo (por backend ou plano) e gera os GIFs.

    :param fargs: argumentos (image, saida, fname, idade_minima)
    """
    global gi_idade_minima_frame
    gi_idade_minima_frame = fargs.idade_minima

    lpth_base = pathlib.Path(fargs.image)
    lpth_saida = pathlib.Path(fargs.saida)

    llst_diretorios_backend = lista_diretorios_de_backend(lpth_base)

    if not llst_diretorios_backend:
        gera_gif_unico(lpth_base, lpth_saida, fargs.fname)
        return

    gera_gifs_por_backend(llst_diretorios_backend, lpth_saida, fargs.fname)
    gera_gif_montado_sem_propagar(lpth_base, llst_diretorios_backend, lpth_saida,
                                  fargs.fname)

# -------------------------------------------------------------------------------------
def arg_parse() -> argparse.Namespace:
    """
    lê os argumentos da linha de comando.

    :returns: argumentos lidos
    """
    lparser = argparse.ArgumentParser(description="Gera um GIF de imagens do nowcast")

    lparser.add_argument("-i", "--image", type=str, default="./",
                         help="diretório de imagens (pai, com um subdiretório por backend "
                              "de nowcast -- ou uma pasta plana de frames, no uso antigo)")
    lparser.add_argument("-s", "--saida", type=str, default="./",
                         help="diretório de saída")
    lparser.add_argument("-f", "--fname", type=str, default="nowcast.gif",
                         help="nome-base do arquivo GIF (cada backend gera "
                              "<nome>_<backend>.gif)")
    lparser.add_argument("--idade-minima", type=float,
                         default=CI_IDADE_MINIMA_FRAME_SEGUNDOS,
                         help="idade mínima (s) de um PNG para ser usado; 0 quando "
                              "quem gravou os frames já terminou")

    return lparser.parse_args()

# -------------------------------------------------------------------------------------
def main() -> int:
    """
    gera os GIFs. Falhas são logadas, não interrompem: o shell
    (unt_common.sh) confere quais GIFs foram gerados e põe o aviso de erro
    nos que faltarem.

    :returns: sempre 0
    """
    gera_gifs(arg_parse())

    return 0

# -------------------------------------------------------------------------------------
# this is the bootstrap process

if "__main__" == __name__:
    # logger
    logging.basicConfig(level=logging.INFO)

    # run
    sys.exit(main())

# < the end >----------------------------------------------------------------------------
