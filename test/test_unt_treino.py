# -*- coding: utf-8 -*-
"""
Centro Logístico de Simulação Meteorológica
Copyright (c) 1995-2026, Instituto de Controle do Espaço Aéreo

test_unt_treino
testes de unt_treino.sh (ciclo completo no catálogo sintético: manifesto ->
untTrain -> untEval -> untPromove -> rsync; retomada do ciclo; descarte de
retomadas abandonadas; rsync sem retomadas/temporários e com os links;
destino remoto barrado em MET_ENV=test; aviso dos alertas por par do
untPromove), do unt_cron.sh treino e do executa_monitor do unt_run.sh

2026.out  mlabru  avisa_alertas_por_par
2026.out  mlabru  initial version (Linux/Python)
"""
# < imports >----------------------------------------------------------------------------

# python library
import json
import os
import pathlib
import shutil
import subprocess
import sys

# pytest
import pytest

pytest.importorskip("torch")

# < constants >--------------------------------------------------------------------------

CS_ROOT = pathlib.Path(__file__).resolve().parent.parent

pytestmark = pytest.mark.skipif(
    any(shutil.which(ls) is None for ls in ("bash", "sqlite3", "rsync", "flock")),
    reason="bash/sqlite3/rsync/flock ausente")

# backends e argumentos mínimos (CPU, segundos)
CS_ARGS_RAPIDOS = "--epocas 1 --recorte 32 --largura-base 4 --workers 0 --batch 4 --threads 2"

# < helpers >----------------------------------------------------------------------------

def _ambiente(fpth_tmp: pathlib.Path, fdct_cat: dict = None, **fdct_extra) -> dict:
    """
    ambiente limpo (sem UNT_*/MET_*), com o python3 desta sessão (torch) no PATH e
    os dados em fpth_tmp.
    """
    ldct_env = {lk: lv for lk, lv in os.environ.items()
                if not lk.startswith(("UNT_", "MET_"))}

    lpth_ini = fpth_tmp / "c.ini"
    lpth_ini.write_text("[janelas]\nbloco_split_s = 3600\n\n[treino]\nselecao = perda\n")

    ldct_env.update({"PATH": f"{pathlib.Path(sys.executable).parent}:{ldct_env['PATH']}",
                     "UNT_PATH": str(fpth_tmp / "untw"),
                     "UNT_VENV": sys.prefix,
                     "UNT_CONFIG": str(lpth_ini),
                     "UNT_TREINO_BACKENDS": "rainnet smaat",
                     "UNT_TREINO_DEVICE": "cpu",
                     "UNT_TREINO_ARGS": CS_ARGS_RAPIDOS})

    if fdct_cat is not None:
        ldct_env.update({"UNT_CATALOG_DB": fdct_cat["catalogo"],
                         "UNT_CAPPI_DIR": fdct_cat["cappi_dir"]})

    ldct_env.update(fdct_extra)

    return ldct_env


def _bash(fs_cmd: str, fdct_env: dict) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", "-c", fs_cmd], env=fdct_env, capture_output=True, text=True)


def _sem_main(fs_corpo: str) -> str:
    return (f'UNT_TREINO_NO_MAIN=true source "{CS_ROOT / "unt_treino.sh"}"; '
            f'mkdir -p "$UNT_LOGS" "$UNT_STATE"; {fs_corpo}')

# < ciclo completo >---------------------------------------------------------------------

def test_ciclo_completo_treina_avalia_promove_e_distribui(tmp_path):
    import conftest
    ldct_cat = conftest.cria_catalogo_sintetico(tmp_path / "sintetico", fi_num_volumes=60)
    lpth_ops = tmp_path / "ops" / "modelos"

    ldct_env = _ambiente(tmp_path, ldct_cat, UNT_DESTINO_MODELOS=str(lpth_ops))
    lproc = subprocess.run(["bash", str(CS_ROOT / "unt_treino.sh")], env=ldct_env,
                           capture_output=True, text=True, timeout=900)

    lpth_base = tmp_path / "untw" / "data"
    ls_log = (tmp_path / "untw/logs/unt_treino.log").read_text()

    assert 0 == lproc.returncode, ls_log + (tmp_path / "untw/logs/unt_treino.err").read_text()
    assert "ciclo de treino concluído: 2 treinados" in ls_log

    # avaliação do ciclo, com os candidatos
    (lpth_comparacao,) = lpth_base.glob("avaliacao/*/comparacao.json")
    ldct_aval = json.loads(lpth_comparacao.read_text())
    assert {"rainnet_candidato", "smaat_candidato"} <= set(ldct_aval["desempenho"])

    for ls_backend in ("rainnet", "smaat"):
        lpth_dir = lpth_base / "modelos" / ls_backend

        # decisão registrada (o modelo de 1 época pode ou não bater a persistência)
        ldct_decisao = json.loads((lpth_dir / "promocoes.jsonl").read_text().splitlines()[-1])
        assert ldct_decisao["acao"] in ("promove", "recusa")

        # sem retomada; o destino tem os checkpoints, .json e o registro
        assert not list(lpth_dir.glob("*.retomada.pt"))
        for lpth in lpth_dir.glob(f"{ls_backend}_*"):
            assert (lpth_ops / ls_backend / lpth.name).is_file()
        assert (lpth_ops / ls_backend / "promocoes.jsonl").is_file()

        if (lpth_dir / "atual.pt").exists():
            assert (lpth_ops / ls_backend / "atual.pt").is_symlink()

    # ciclo encerrado: o próximo começa outro
    assert not (lpth_base / "state" / "treino_ciclo").exists()

# < retomada do ciclo >------------------------------------------------------------------

def test_manifesto_do_ciclo_reaproveita_o_interrompido(tmp_path):
    ldct_env = _ambiente(tmp_path)
    lpth_ciclo = tmp_path / "untw/data/state/treino_ciclo"
    lpth_ciclo.mkdir(parents=True)
    (tmp_path / "m.csv").write_text("x")
    (lpth_ciclo / "manifesto").write_text(f"{tmp_path / 'm.csv'}\n")
    (tmp_path / "c.pt").write_text("x")
    (lpth_ciclo / "rainnet").write_text(f"{tmp_path / 'c.pt'}\n")

    lproc = _bash(_sem_main("manifesto_do_ciclo; echo \"[$(candidato_pronto rainnet)]\"; "
                            "echo \"[$(candidato_pronto smaat)]\""), ldct_env)

    assert 0 == lproc.returncode, lproc.stderr
    assert [str(tmp_path / "m.csv"), f"[{tmp_path / 'c.pt'}]", "[]"] == lproc.stdout.split()
    assert "retomando o ciclo interrompido" in lproc.stderr


def test_ciclo_novo_descarta_retomadas_abandonadas(tmp_path):
    import conftest
    ldct_cat = conftest.cria_catalogo_sintetico(tmp_path / "sintetico", fi_num_volumes=20)
    ldct_env = _ambiente(tmp_path, ldct_cat)

    lpth_modelos = tmp_path / "untw/data/modelos/rainnet"
    lpth_modelos.mkdir(parents=True)
    (lpth_modelos / "rainnet.retomada.pt").write_text("velho")

    lproc = _bash(_sem_main('mkdir -p "$UNT_MODELOS_DIR"; manifesto_do_ciclo'), ldct_env)

    assert 0 == lproc.returncode, lproc.stderr
    assert pathlib.Path(lproc.stdout.strip()).is_file()
    assert not (lpth_modelos / "rainnet.retomada.pt").exists()
    assert "retomada abandonada removida" in lproc.stderr
    assert (tmp_path / "untw/data/state/treino_ciclo/manifesto").is_file()

# < distribuição >-----------------------------------------------------------------------

def test_distribui_sem_retomadas_e_com_links(tmp_path):
    lpth_ops = tmp_path / "ops"
    ldct_env = _ambiente(tmp_path, UNT_DESTINO_MODELOS=str(lpth_ops))

    lpth_dir = tmp_path / "untw/data/modelos/smaat"
    lpth_dir.mkdir(parents=True)
    (lpth_dir / "smaat_1.pt").write_text("pesos")
    (lpth_dir / "smaat_1.json").write_text("{}")
    (lpth_dir / "atual.pt").symlink_to("smaat_1.pt")
    (lpth_dir / "smaat.retomada.pt").write_text("estado")
    (lpth_dir / ".smaat_2.pt.tmp").write_text("pela metade")

    lproc = _bash(_sem_main("distribui_modelos"), ldct_env)

    assert 0 == lproc.returncode, lproc.stderr
    assert "pesos" == (lpth_ops / "smaat/atual.pt").read_text()
    assert (lpth_ops / "smaat/atual.pt").is_symlink()
    assert (lpth_ops / "smaat/smaat_1.json").exists()
    assert not (lpth_ops / "smaat/smaat.retomada.pt").exists()
    assert not (lpth_ops / "smaat/.smaat_2.pt.tmp").exists()


def test_destino_remoto_barrado_no_ambiente_de_teste(tmp_path):
    ldct_env = _ambiente(tmp_path, MET_ENV="test", MET_TST_ROOT=str(tmp_path),
                         UNT_CATALOG_DB=str(tmp_path / "c.db"),
                         UNT_CAPPI_DIR=str(tmp_path / "cappi"),
                         UNT_DESTINO_MODELOS="ops:/srv/modelos")

    lproc = _bash(_sem_main("distribui_modelos"), ldct_env)

    assert 0 != lproc.returncode
    assert "destino remoto ops:/srv/modelos recusado" in lproc.stderr


def test_destino_local_fora_da_raiz_barrado_pela_trava(tmp_path):
    ldct_env = _ambiente(tmp_path, MET_ENV="test", MET_TST_ROOT=str(tmp_path / "raiz"),
                         UNT_PATH=str(tmp_path / "raiz/untw"),
                         UNT_CATALOG_DB=str(tmp_path / "raiz/c.db"),
                         UNT_CAPPI_DIR=str(tmp_path / "raiz/cappi"),
                         UNT_DESTINO_MODELOS=str(tmp_path / "fora"))

    lproc = _bash(f'source "{CS_ROOT / "unt_cfg.sh"}"', ldct_env)

    assert 1 == lproc.returncode
    assert "fora de" in lproc.stderr

# < promoção >---------------------------------------------------------------------------

def test_avisa_alertas_por_par_so_da_decisao_nova(tmp_path):
    ldct_env = _ambiente(tmp_path)
    lpth_registro = tmp_path / "untw" / "data" / "modelos" / "smaat" / "promocoes.jsonl"
    lpth_registro.parent.mkdir(parents=True)
    lpth_registro.write_text(
        json.dumps({"acao": "promove", "alertas_por_par": {}}) + "\n" +
        json.dumps({"acao": "promove", "alertas_por_par": {"BEL_400": ["x"],
                                                           "MCA_250": ["y"]}}) + "\n")

    # a última decisão é nova (o registro tinha 1 linha antes): avisa os pares
    lproc = _bash(_sem_main("avisa_alertas_por_par smaat 1"), ldct_env)
    assert 0 == lproc.returncode, lproc.stderr
    assert "[warn]: smaat: critério do portão falha em BEL_400,MCA_250" in lproc.stderr

    # o registro não cresceu (untPromove.py falhou antes de registrar): silêncio
    lproc = _bash(_sem_main("avisa_alertas_por_par smaat 2"), ldct_env)
    assert 0 == lproc.returncode and "critério do portão" not in lproc.stderr

    # decisão nova sem alertas: silêncio
    lpth_registro.write_text(json.dumps({"acao": "recusa", "alertas_por_par": {}}) + "\n")
    lproc = _bash(_sem_main("avisa_alertas_por_par smaat 0"), ldct_env)
    assert 0 == lproc.returncode and "critério do portão" not in lproc.stderr

    # backend sem registro
    lproc = _bash(_sem_main("[ 0 -eq $(linhas_do_registro rainnet) ]"), ldct_env)
    assert 0 == lproc.returncode, lproc.stderr

# < unt_cron.sh e monitor >--------------------------------------------------------------

def test_cron_tarefa_invalida(tmp_path):
    lproc = subprocess.run(["bash", str(CS_ROOT / "unt_cron.sh"), "xx"],
                           env=_ambiente(tmp_path), capture_output=True, text=True)

    assert 2 == lproc.returncode
    assert "uso: unt_cron.sh [run|treino]" in lproc.stderr


def test_executa_monitor_sem_historico_e_desligado(tmp_path):
    ldct_env = _ambiente(tmp_path, UNT_VERIFY="true")
    ls_cmd = (f'source "{CS_ROOT / "unt_common.sh"}"; mkdir -p "$UNT_LOGS" "$UNT_OUT"; '
              "executa_monitor")

    lproc = _bash(ls_cmd, ldct_env)
    assert 0 == lproc.returncode, lproc.stderr
    assert "monitor: ainda sem verify.jsonl" in lproc.stderr

    # UNT_VERIFY=false: nem roda
    lproc = _bash(ls_cmd, dict(ldct_env, UNT_VERIFY="false"))
    assert 0 == lproc.returncode and "monitor" not in lproc.stderr

# < the end >----------------------------------------------------------------------------
