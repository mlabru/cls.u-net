# -*- coding: utf-8 -*-
"""
Centro Logístico de Simulação Meteorológica
Copyright (c) 1995-2026, Instituto de Controle do Espaço Aéreo

untTrain

treina UM backend (pytorch, rainnet ou smaat) sobre um manifesto de janelas
(untJanelas) e grava o checkpoint CANDIDATO em <modelos>/<backend>/. O atual.pt (o que
a produção usa) não muda: a promoção é do untPromove, depois do untEval. --promove
aponta atual.pt direto para o candidato (só para dev/testes).

roda com ou sem GPU: --device auto escolhe cuda se houver, senão cpu. Em cpu, use
recortes menores e redes estreitas (ex.: --recorte 128 --largura-base 16); no host de
treino com GPU, os valores do config.ini (recorte 256, largura 64 = artigos).

fluxo de main():
  1. lê o manifesto e os hiperparâmetros (CLI > config.ini > defaults);
  2. monta os datasets de treino e validação (unt_dataset); peso_bins = auto (pytorch)
     é calculado aqui, das frequências dos alvos de treino;
  3. treina (unt_treino): seleção da melhor época pela métrica do rollout na validação
     (ou pela perda), estado gravado a cada época em
     <modelos>/<backend>/<backend>.retomada.pt (--retoma continua dali);
  4. grava o checkpoint (pesos da época escolhida, hiperparâmetros, transformação,
     histórico, tempo de treino, passos, device, escopo radar/alcance, sha256 do
     manifesto, commit git) e o resumo .json ao lado; apaga o arquivo de retomada.

código de saída: 0 se gravou o checkpoint; 1 caso contrário.

$ python untTrain.py -b rainnet --janelas data/janelas/janelas.csv
                     [--cappi-dir ./data/cappi] [--modelos-dir ./data/modelos]
                     [--config config.ini] [--device auto] [--epocas N] [--batch N]
                     [--lr X] [--recorte PX] [--largura-base N] [--workers N]
                     [--amostras-por-epoca N] [--semente N] [--selecao metrica|perda]
                     [--clip-grad X] [--retoma] [--promove]

2026.out  mlabru  semente fixada antes de construir a rede (inicialização reprodutível);
                  rede colapsada no início -> recomeça com a semente seguinte
2026.out  mlabru  treino interrompido por lotes não finitos (TreinoDivergiu) -> rc 1
2026.out  mlabru  --retoma (estado por época), --selecao, --clip-grad, peso_bins = auto;
                  métrica e passos no log, no checkpoint e no .json
2026.out  mlabru  grava candidato sem promover (--promove só em dev); escopo e
                  proveniência no checkpoint e no .json
2026.out  mlabru  initial version (Linux/Python)
"""
# < imports >----------------------------------------------------------------------------

# python library
import argparse
import json
import logging
import pathlib
import sys
import typing

# model
import work.unt_catalog as oc
import work.unt_config as uc
import work.unt_dataset as uds
import work.unt_device as ue
import work.unt_janelas as uj
import work.unt_modelos as um
import work.unt_treino as ut

# < logging >----------------------------------------------------------------------------

# logger
M_LOG = logging.getLogger(__name__)
M_LOG.setLevel(logging.INFO)

# < argumentos >-------------------------------------------------------------------------

def arg_parse(flst_argv: typing.Optional[list]=None) -> argparse.Namespace:
    """
    lê os argumentos da linha de comando. Os opcionais de treino ficam None quando
    omitidos, para que valha o config.ini.

    :param flst_argv: argumentos (None = sys.argv)

    :returns: argumentos lidos
    """
    lparser = argparse.ArgumentParser(
        description="Treina um backend de nowcasting por Deep Learning "
                    "(pytorch, rainnet ou smaat) sobre um manifesto de janelas")

    lparser.add_argument("-b", "--backend", type=str, required=True, choices=um.CT_BACKENDS,
                         help="backend a treinar")
    lparser.add_argument("--janelas", type=str, required=True,
                         help="manifesto de janelas (.csv do untJanelas)")
    lparser.add_argument("--cappi-dir", type=str, default=oc.diretorio_cappi_padrao(),
                         help="diretório dos CAPPI (.npy)")
    lparser.add_argument("--modelos-dir", type=str, default=um.diretorio_modelos_padrao(),
                         help="raiz dos checkpoints (<dir>/<backend>/)")
    lparser.add_argument("--config", type=str, default=None,
                         help="config.ini; padrão ./config.ini se existir")
    lparser.add_argument("--device", type=str, default=None,
                         help="auto, cpu, cuda, cuda:N ou mps")
    lparser.add_argument("--epocas", type=int, default=None)
    lparser.add_argument("--batch", type=int, default=None)
    lparser.add_argument("--lr", type=float, default=None)
    lparser.add_argument("--paciencia", type=int, default=None)
    lparser.add_argument("--recorte", dest="recorte_px", type=int, default=None,
                         help="lado do recorte (px, múltiplo de 32; 0 = grade inteira)")
    lparser.add_argument("--largura-base", type=int, default=None,
                         help="largura do 1º nível da rede (64 = artigos)")
    lparser.add_argument("--workers", type=int, default=None,
                         help="processos do DataLoader")
    lparser.add_argument("--amostras-por-epoca", type=int, default=None,
                         help="itens de treino por época (0 = todos)")
    lparser.add_argument("--semente", type=int, default=None)
    lparser.add_argument("--threads", type=int, default=0,
                         help="threads do torch em cpu (0 = todas as CPUs)")
    lparser.add_argument("--selecao", type=str, default=None,
                         choices=(um.CS_SELECAO_METRICA, um.CS_SELECAO_PERDA),
                         help="critério da melhor época / early stopping")
    lparser.add_argument("--clip-grad", type=float, default=None,
                         help="norma máxima do gradiente (0 = sem corte)")
    lparser.add_argument("--retoma", action="store_true",
                         help="continua o treino interrompido "
                              "(<modelos>/<backend>/<backend>.retomada.pt), se existir")
    lparser.add_argument("--promove", action="store_true",
                         help="aponta atual.pt para o checkpoint gravado, sem passar "
                              "pelo untEval/untPromove (só dev/testes)")

    return lparser.parse_args(flst_argv)

# < passos de main() >-------------------------------------------------------------------

def sobrescritos_da_cli(fargs: argparse.Namespace) -> dict:
    """
    hiperparâmetros dados na linha de comando (None = não dado).
    """
    return {"device": fargs.device,
            "epocas": fargs.epocas,
            "batch": fargs.batch,
            "lr": fargs.lr,
            "paciencia": fargs.paciencia,
            "recorte_px": fargs.recorte_px,
            "largura_base": fargs.largura_base,
            "workers": fargs.workers,
            "amostras_por_epoca": fargs.amostras_por_epoca,
            "semente": fargs.semente,
            "selecao": fargs.selecao,
            "clip_grad": fargs.clip_grad}

# -------------------------------------------------------------------------------------
def caminho_retomada(fs_modelos_dir: str, fs_backend: str) -> pathlib.Path:
    """
    <modelos>/<backend>/<backend>.retomada.pt (um por backend; apagado no fim).
    """
    return pathlib.Path(fs_modelos_dir) / fs_backend / f"{fs_backend}.retomada.pt"

# -------------------------------------------------------------------------------------
def resolve_peso_bins(fargs: argparse.Namespace, fds_treino, fdct_hparams: dict) -> None:
    """
    passo 2b: peso_bins = auto -> pesos das frequências dos alvos de treino (gravados
    nos hparams, logo no checkpoint).

    :raises ValueError: alvos sem nenhum pixel válido
    """
    if um.CS_PESO_BINS_AUTO != fdct_hparams.get("peso_bins"):
        return

    fdct_hparams["peso_bins"] = uds.calcula_pesos_bins(fds_treino.janelas, fargs.cappi_dir,
                                                       fdct_hparams["bins_dbz"])

# -------------------------------------------------------------------------------------
def monta_datasets(fargs: argparse.Namespace, flst_janelas: list, fdct_hparams: dict,
                   ftransformacao) -> tuple:
    """
    passo 2: datasets de treino (aleatório) e de validação (determinístico).
    """
    lds_treino = uds.JanelasDataset(uj.filtra_split(flst_janelas, uj.CS_SPLIT_TREINO),
                                    fargs.cappi_dir, fdct_hparams, ftransformacao,
                                    fb_treino=True)
    lds_validacao = uds.JanelasDataset(uj.filtra_split(flst_janelas, uj.CS_SPLIT_VALIDACAO),
                                       fargs.cappi_dir, fdct_hparams, ftransformacao,
                                       fb_treino=False)

    M_LOG.info(f"[+] {fdct_hparams['backend']}: {len(lds_treino)} itens de treino, "
               f"{len(lds_validacao)} de validação (modo {fdct_hparams['modo']}).")

    return lds_treino, lds_validacao

# -------------------------------------------------------------------------------------
def loga_epoca(fdct_registro: dict) -> None:
    """
    uma linha de log por época.
    """
    lf_val = fdct_registro["perda_validacao"]
    ls_val = "-" if lf_val is None else f"{lf_val:.5f}"

    lf_metrica = fdct_registro["metrica_validacao"]
    ls_metrica = "-" if lf_metrica is None else f"{lf_metrica:.4f}"

    M_LOG.info(f"[=] época {fdct_registro['epoca']:3d}  treino={fdct_registro['perda_treino']:.5f}  "
               f"val={ls_val}  métrica={ls_metrica}  lr={fdct_registro['lr']:.2e}  "
               f"passos={fdct_registro['passos']}  {fdct_registro['duracao_s']:.1f}s")

# -------------------------------------------------------------------------------------
def proveniencia(fargs: argparse.Namespace, flst_janelas: list) -> dict:
    """
    de onde veio o modelo: escopo radar/alcance do manifesto, sha256 do manifesto e
    commit do código.
    """
    return {"manifesto": str(fargs.janelas),
            "manifesto_sha256": um.sha256_de_arquivo(fargs.janelas),
            "escopo": um.escopo_das_janelas(flst_janelas),
            "git_commit": um.commit_git()}

# -------------------------------------------------------------------------------------
def grava_historico(fpth_checkpoint: pathlib.Path, fresultado: ut.SResultadoTreino,
                    fdct_hparams: dict, fdct_proveniencia: dict) -> None:
    """
    passo 4b: <checkpoint>.json com hiperparâmetros, proveniência e histórico (leitura
    humana; o untMain lê dele o escopo, sem carregar os pesos).
    """
    ldct = {"checkpoint": fpth_checkpoint.name,
            "backend": fdct_hparams["backend"],
            "hparams": fdct_hparams,
            **fdct_proveniencia,
            "melhor_epoca": fresultado.epoca,
            "criterio_selecao": fresultado.criterio,
            "perda_validacao": fresultado.perda_validacao,
            "metrica_validacao": fresultado.metrica_validacao,
            "passos": fresultado.passos,
            "tempo_treino_s": fresultado.tempo_treino_s,
            "historico": fresultado.historico}

    fpth_checkpoint.with_suffix(".json").write_text(json.dumps(ldct, indent=2,
                                                               ensure_ascii=False))

# -------------------------------------------------------------------------------------
def treina_com_tentativas(fargs: argparse.Namespace, fdct_hparams: dict, fds_treino,
                          fds_validacao, fdevice, fpth_retomada: pathlib.Path,
                          fs_manifesto_sha256: str) -> ut.SResultadoTreino:
    """
    passo 3: constrói a rede (semente antes: a inicialização também é reprodutível) e
    treina. Rede colapsada no início (TreinoColapsou) -> nova rede com a semente
    seguinte, até tentativas_colapso vezes; a semente usada fica nos hparams.

    :raises ValueError, TreinoDivergiu: do treino
    :raises TreinoColapsou: colapsou em todas as tentativas
    """
    li_tentativas = fdct_hparams.get("tentativas_colapso", 1)

    for li_tentativa in range(1, li_tentativas + 1):
        ut.fixa_semente(fdct_hparams["semente"])
        lrede = um.constroi_rede(fdct_hparams)

        if 1 == li_tentativa:
            M_LOG.info(f"[+] {fargs.backend}: {um.conta_parametros(lrede):,} parâmetros.")

        try:
            return ut.treina(lrede, fds_treino, fds_validacao, fdct_hparams, fdevice,
                             ff_ao_fim_da_epoca=loga_epoca, fpth_retomada=fpth_retomada,
                             fb_retoma=fargs.retoma,
                             fdct_identidade=ut.identidade_retomada(fdct_hparams,
                                                                    fs_manifesto_sha256))
        except ut.TreinoColapsou as le_erro:
            if li_tentativa == li_tentativas:
                raise ut.TreinoColapsou(f"{le_erro} em {li_tentativas} tentativa(s)") \
                    from le_erro

            M_LOG.warning(f"[!] semente {fdct_hparams['semente']}: {le_erro}; "
                          f"recomeçando com a semente {fdct_hparams['semente'] + 1}.")
            fdct_hparams["semente"] += 1

    raise AssertionError("inalcançável")

# < main >-------------------------------------------------------------------------------

def main(flst_argv: typing.Optional[list]=None) -> int:
    """
    treina e grava o checkpoint (ver o fluxo no topo do módulo).

    :returns: 0 se gravou o checkpoint, 1 caso contrário
    """
    largs = arg_parse(flst_argv)

    # 1. manifesto e hiperparâmetros
    llst_janelas, ldct_manifesto = uj.le_manifesto(largs.janelas)

    ldct_hparams = um.hiperparametros(uc.carrega(largs.config), largs.backend,
                                      ldct_manifesto["num_entradas"],
                                      ldct_manifesto["num_alvos"],
                                      sobrescritos_da_cli(largs))

    ldevice = ue.escolhe_device(ldct_hparams["device"])
    ue.configura_threads_cpu(ldevice, largs.threads)
    M_LOG.info(f"[+] device: {ue.descreve(ldevice)}")

    # 2. datasets
    ltransformacao = um.transformacao_padrao(ldct_hparams)
    lds_treino, lds_validacao = monta_datasets(largs, llst_janelas, ldct_hparams,
                                               ltransformacao)

    if 0 == len(lds_treino):
        M_LOG.error(f"[x] nenhuma janela de treino em {largs.janelas}.")
        return 1

    try:
        resolve_peso_bins(largs, lds_treino, ldct_hparams)

    except ValueError as le_erro:
        M_LOG.error(f"[x] {le_erro}")
        return 1

    # 3. treino
    ldct_proveniencia = proveniencia(largs, llst_janelas)
    lpth_retomada = caminho_retomada(largs.modelos_dir, largs.backend)

    if lpth_retomada.exists() and not largs.retoma:
        M_LOG.warning(f"[!] {lpth_retomada.name} existe e será sobrescrito (use --retoma "
                      f"para continuar aquele treino).")

    try:
        lresultado = treina_com_tentativas(largs, ldct_hparams, lds_treino, lds_validacao,
                                           ldevice, lpth_retomada,
                                           ldct_proveniencia["manifesto_sha256"])
    except (ValueError, ut.TreinoDivergiu, ut.TreinoColapsou) as le_erro:
        M_LOG.error(f"[x] {le_erro}")
        return 1

    # 4. checkpoint (candidato; atual.pt só muda com --promove)
    lpth_checkpoint = um.salva_checkpoint(largs.modelos_dir, ldct_hparams, lresultado.rede,
                                          ltransformacao,
                                          {"epoca": lresultado.epoca,
                                           "criterio_selecao": lresultado.criterio,
                                           "perda_validacao": lresultado.perda_validacao,
                                           "metrica_validacao": lresultado.metrica_validacao,
                                           "passos": lresultado.passos,
                                           "historico": lresultado.historico,
                                           "tempo_treino_s": lresultado.tempo_treino_s,
                                           "device": str(ldevice),
                                           **ldct_proveniencia},
                                          fb_promove=largs.promove)
    grava_historico(lpth_checkpoint, lresultado, ldct_hparams, ldct_proveniencia)
    lpth_retomada.unlink(missing_ok=True)

    M_LOG.info(f"[✓] checkpoint {lpth_checkpoint} (época {lresultado.epoca} pela "
               f"{lresultado.criterio}, perda {lresultado.perda_validacao:.5f}, "
               f"{lresultado.tempo_treino_s:.0f}s, escopo "
               f"{','.join(ldct_proveniencia['escopo']['pares'])}).")

    if largs.promove:
        M_LOG.warning(f"[!] --promove: atual.pt aponta para {lpth_checkpoint.name} sem "
                      f"avaliação.")
    else:
        M_LOG.info(f"[+] candidato (não promovido): avalie com untEval --checkpoint "
                   f"{largs.backend}_candidato={lpth_checkpoint} e promova com "
                   f"untPromove.")

    return 0

# -------------------------------------------------------------------------------------
# this is the bootstrap process

if "__main__" == __name__:
    # logger
    logging.basicConfig(level=logging.INFO)

    # run
    sys.exit(main())

# < the end >----------------------------------------------------------------------------
