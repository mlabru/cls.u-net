# -*- coding: utf-8 -*-
"""
Centro Logístico de Simulação Meteorológica
Copyright (c) 1995-2026, Instituto de Controle do Espaço Aéreo

untJanelas

monta o manifesto de janelas de treino/avaliação (4 entradas + 6 alvos,
+10..+60 min) a partir dos CAPPIs prontos de cappi_products. O catálogo é aberto
SOMENTE para leitura. Os três backends (pytorch, rainnet, smaat) treinam e são
avaliados sobre o mesmo manifesto (princípio de comparabilidade do relatório).

fluxo de main():
  1. lista os pares (radar, alcance): os de -r/-a, ou todos com CAPPI pronto;
  2. por par: produtos prontos do preset (--preset ou o ativo do alcance) ->
     unt_janelas.monta_janelas() (slots encadeados, filtros, split por dia UTC);
  3. descarta janelas com algum .npy ausente (salvo --sem-verificar-arquivos);
  4. grava <saida>.csv + <saida>.json e loga a contagem por conjunto e as
     rejeições por motivo.

código de saída: 0 se gravou ao menos uma janela; 1 caso contrário.

$ python untJanelas.py [-r STI] [-a 250] [--preset nwc250v2] [--catalog catalog.db]
                       [--cappi-dir ./data/cappi] [-o data/janelas/janelas.csv]
                       [--config config.ini] [--sem-verificar-arquivos]

2026.out  mlabru  initial version (Linux/Python)
"""
# < imports >----------------------------------------------------------------------------

# python library
import argparse
import logging
import sqlite3
import sys
import typing

# model
import work.unt_catalog as oc
import work.unt_config as uc
import work.unt_janelas as uj

# < constants >--------------------------------------------------------------------------

# manifesto padrão
CS_MANIFESTO_PADRAO = "./data/janelas/janelas.csv"

# < logging >----------------------------------------------------------------------------

# logger
M_LOG = logging.getLogger(__name__)
M_LOG.setLevel(logging.INFO)

# < argumentos >-------------------------------------------------------------------------

def arg_parse(flst_argv: typing.Optional[list]=None) -> argparse.Namespace:
    """
    lê os argumentos da linha de comando.

    :param flst_argv: argumentos (None = sys.argv)

    :returns: argumentos lidos
    """
    lparser = argparse.ArgumentParser(
        description="Monta o manifesto de janelas 4 entradas + 6 alvos (treino, "
                    "validação e teste) a partir dos CAPPIs prontos do catálogo")

    lparser.add_argument("-r", "--radar", type=str, default=None,
                         help="código do radar (ex: STI); se omitido, todos")
    lparser.add_argument("-a", "--alcance", type=int, default=None,
                         help="alcance em km (ex: 250); se omitido, todos")
    lparser.add_argument("--preset", type=str, default=None,
                         help="preset do CAPPI; se omitido, o ativo de cada alcance")
    lparser.add_argument("--catalog", type=str, default=oc.caminho_catalogo_padrao(),
                         help="caminho do catálogo SQLite")
    lparser.add_argument("--cappi-dir", type=str, default=oc.diretorio_cappi_padrao(),
                         help="diretório dos CAPPI (.npy)")
    lparser.add_argument("-o", "--saida", type=str, default=CS_MANIFESTO_PADRAO,
                         help="manifesto (.csv; os parâmetros vão no .json ao lado)")
    lparser.add_argument("--config", type=str, default=None,
                         help="config.ini (seção [janelas]); padrão ./config.ini se existir")
    lparser.add_argument("--sem-verificar-arquivos", action="store_true",
                         help="não confere a existência dos .npy")

    return lparser.parse_args(flst_argv)

# < passos de main() >-------------------------------------------------------------------

def le_parametros(fconfig: uc.SConfig) -> uj.SParametrosJanela:
    """
    critério das janelas a partir da seção [janelas].
    """
    return uj.SParametrosJanela(intervalo_s=fconfig.inteiro("janelas", "intervalo_s"),
                                tolerancia_s=fconfig.inteiro("janelas", "tolerancia_s"),
                                num_entradas=fconfig.inteiro("janelas", "num_entradas"),
                                num_alvos=fconfig.inteiro("janelas", "num_alvos"),
                                min_valid_ratio=fconfig.real("janelas", "min_valid_ratio"),
                                min_rain_fraction=fconfig.real("janelas", "min_rain_fraction"),
                                frac_treino=fconfig.real("janelas", "frac_treino"),
                                frac_validacao=fconfig.real("janelas", "frac_validacao"),
                                bloco_split_s=fconfig.inteiro("janelas", "bloco_split_s"))

# -------------------------------------------------------------------------------------
def lista_pares(fconexao: sqlite3.Connection, fargs: argparse.Namespace) -> list:
    """
    passo 1: pares (radar, alcance) a processar.

    :returns: os pares com CAPPI pronto que casam com -r/-a (se dados)
    """
    llst_pares = []

    for ls_radar, li_alcance in oc.lista_pares_com_produtos(fconexao):
        if fargs.radar is not None and ls_radar != fargs.radar:
            continue

        if fargs.alcance is not None and li_alcance != fargs.alcance:
            continue

        llst_pares.append((ls_radar, li_alcance))

    return llst_pares

# -------------------------------------------------------------------------------------
def _arquivos_presentes(fs_cappi_dir: str, fjanela: uj.SJanela) -> bool:
    """
    todos os .npy (entradas e alvos) da janela existem.
    """
    for li_timestamp in fjanela.entradas + fjanela.alvos:
        lpth_cappi = oc.monta_caminho_cappi(fs_cappi_dir, fjanela.radar, fjanela.alcance,
                                            li_timestamp, fjanela.preset)
        if not lpth_cappi.exists():
            return False

    return True

# -------------------------------------------------------------------------------------
def janelas_do_par(fconexao: sqlite3.Connection, fargs: argparse.Namespace, fs_radar: str,
                   fi_alcance: int, ft_parametros: uj.SParametrosJanela) -> tuple:
    """
    passos 2 e 3: janelas aceitas de um par.

    :returns: (lista de SJanela, dicionário motivo -> rejeições)
    """
    ls_preset = fargs.preset or oc.preset_ativo_padrao(fi_alcance)

    if ls_preset is None:
        M_LOG.warning(f"[!] {fs_radar}-{fi_alcance}: alcance sem preset ativo conhecido; "
                      f"use --preset.")
        return [], {}

    llst_linhas = oc.lista_produtos_prontos(fconexao, fs_radar, fi_alcance, ls_preset)
    llst_produtos = uj.produtos_de_linhas(llst_linhas)

    llst_janelas, ldct_rejeicoes = uj.monta_janelas(llst_produtos, fs_radar, fi_alcance,
                                                    ls_preset, ft_parametros)

    if fargs.sem_verificar_arquivos:
        return llst_janelas, ldct_rejeicoes

    llst_com_arquivos = []

    for ljanela in llst_janelas:
        if _arquivos_presentes(fargs.cappi_dir, ljanela):
            llst_com_arquivos.append(ljanela)
        else:
            ldct_rejeicoes["arquivo_ausente"] = ldct_rejeicoes.get("arquivo_ausente", 0) + 1

    M_LOG.info(f"[=] {fs_radar}-{fi_alcance} ({ls_preset}): {len(llst_produtos)} produtos, "
               f"{len(llst_com_arquivos)} janelas, rejeições {ldct_rejeicoes}")

    return llst_com_arquivos, ldct_rejeicoes

# -------------------------------------------------------------------------------------
def monta_parametros_gravados(fargs: argparse.Namespace,
                              ft_parametros: uj.SParametrosJanela) -> dict:
    """
    parâmetros gravados no .json do manifesto (reprodutibilidade).
    """
    ldct_parametros = dict(ft_parametros._asdict())
    ldct_parametros.update({"catalog": fargs.catalog,
                            "cappi_dir": fargs.cappi_dir,
                            "preset": fargs.preset,
                            "radar": fargs.radar,
                            "alcance": fargs.alcance})

    return ldct_parametros

# < main >-------------------------------------------------------------------------------

def main(flst_argv: typing.Optional[list]=None) -> int:
    """
    monta e grava o manifesto (ver o fluxo no topo do módulo).

    :returns: 0 se gravou ao menos uma janela, 1 caso contrário
    """
    largs = arg_parse(flst_argv)
    lt_parametros = le_parametros(uc.carrega(largs.config))

    lconexao = oc.abre_catalogo_somente_leitura(largs.catalog)

    # 1. pares
    llst_pares = lista_pares(lconexao, largs)

    if not llst_pares:
        M_LOG.error(f"[x] nenhum CAPPI pronto em {largs.catalog} "
                    f"(radar={largs.radar} alcance={largs.alcance}).")
        return 1

    # 2-3. janelas
    llst_janelas = []

    for ls_radar, li_alcance in llst_pares:
        llst_do_par, _ = janelas_do_par(lconexao, largs, ls_radar, li_alcance, lt_parametros)
        llst_janelas.extend(llst_do_par)

    if not llst_janelas:
        M_LOG.error("[x] nenhuma janela aceita; manifesto não gravado.")
        return 1

    # 4. manifesto
    uj.grava_manifesto(llst_janelas, largs.saida,
                       monta_parametros_gravados(largs, lt_parametros))

    M_LOG.info(f"[✓] {len(llst_janelas)} janelas em {largs.saida}: "
               f"{uj.conta_por_split(llst_janelas)}")

    return 0

# -------------------------------------------------------------------------------------
# this is the bootstrap process

if "__main__" == __name__:
    # logger
    logging.basicConfig(level=logging.INFO)

    # run
    sys.exit(main())

# < the end >----------------------------------------------------------------------------
