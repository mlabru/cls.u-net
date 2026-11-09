# -*- coding: utf-8 -*-
"""
Centro Logístico de Simulação Meteorológica
Copyright (c) 1995-2026, Instituto de Controle do Espaço Aéreo

test_untMain
testes unitários de untMain (arg_parse, main() lendo sequência do catálogo
SQLite, e o contrato de contenção de exceções de executa_backend -- nunca
propaga, sempre retorna (nome, sucesso, erro))

2026.out  mlabru  portado de 25.oflow; nomes dos backends trocados (pytorch, rainnet, smaat).

2026.out  mlabru  backends injetados via MLST_BACKENDS_DISPONIVEIS e
                  gera_nowcast(); tarefas como no.TarefaBackend.

2026.out  mlabru  caminho com preset, passo de tempo e raio derivados da
                  sequência/grade, --verify (verify.json) e gravação da
                  previsão (<prefix>.npy) por _run_backend.

2026.ago  mlabru  atualizado para o novo contrato de CLI (--radar/--alcance/
                  --catalog/--cappi-dir em vez de 3 arquivos .mvol
                  posicionais) e para o caminho de leitura de sequência via
                  apps/unt_catalog.py.

2026.ago  mlabru  initial version (Linux/Python)
"""
# < imports >----------------------------------------------------------------------------

# python library
import os
import pathlib
import sqlite3
import subprocess
import sys

# numPy
import numpy as np

# model
import untMain as no
import work.unt_backend_comum as obc

# < helpers >----------------------------------------------------------------------------

def _injeta_backend(monkeypatch, fs_nome: str, fmod) -> None:
    """
    substitui o backend work.unt_<fs_nome> por fmod. "import work.unt_x as m"
    resolve pelo atributo do pacote work (já preenchido pelo import real no
    topo do untMain), então trocar só sys.modules não basta.
    """
    import work

    monkeypatch.setitem(sys.modules, f"work.unt_{fs_nome}", fmod)
    monkeypatch.setattr(work, f"unt_{fs_nome}", fmod, raising=False)

# < arg_parse >--------------------------------------------------------------------------

def test_arg_parse_defaults(monkeypatch):
    monkeypatch.delenv("UNT_CATALOG_DB", raising=False)
    monkeypatch.delenv("UNT_CAPPI_DIR", raising=False)
    monkeypatch.setattr(sys, "argv", ["untMain.py", "-r", "CGU", "-a", "400"])

    largs = no.arg_parse()

    assert "CGU" == largs.radar
    assert 400 == largs.alcance
    assert "nowcast" == largs.prefix
    # opções sem uso removidas
    assert not hasattr(largs, "mask")
    assert not hasattr(largs, "shape")
    assert "./data/dbase/catalog.db" == largs.catalog
    assert "./data/cappi" == largs.cappi_dir
    assert "./data/cache/nowcast" == largs.saida

# -------------------------------------------------------------------------------------
def test_arg_parse_exige_radar_e_alcance(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["untMain.py"])

    try:
        no.arg_parse()
        assert False, "deveria ter chamado sys.exit por --radar/--alcance ausentes"
    except SystemExit as le_err:
        assert 0 != le_err.code

# -------------------------------------------------------------------------------------
def test_arg_parse_env_vars_definem_defaults(monkeypatch):
    monkeypatch.setenv("UNT_CATALOG_DB", "/env/catalog.db")
    monkeypatch.setenv("UNT_CAPPI_DIR", "/env/cappi")
    monkeypatch.setattr(sys, "argv", ["untMain.py", "-r", "CGU", "-a", "400"])

    largs = no.arg_parse()

    assert "/env/catalog.db" == largs.catalog
    assert "/env/cappi" == largs.cappi_dir

# < unt_catalog (via oc.*) >-------------------------------------------------------------

def test_monta_caminho_cappi_segue_convencao():
    lpth = no.oc.monta_caminho_cappi("/cappi", "CGU", 400, 123)
    assert pathlib.Path("/cappi/CGU_400_123.npy") == lpth

# < main() -- leitura de sequência do catálogo >-----------------------------------------

def _make_catalog(tmp_path, fi_alcance=400, fi_timestamps=(10, 20, 30, 40),
                  fv_com_geometria=True):
    """
    monta um catalog.db temporário com o schema mínimo (sequences,
    radar_sites) e retorna seu caminho.
    """
    lpth_db = tmp_path / "catalog.db"
    lconn = sqlite3.connect(lpth_db)

    lconn.execute("""CREATE TABLE sequences (
                         radar TEXT, alcance INTEGER,
                         input_1 INTEGER, input_2 INTEGER,
                         input_3 INTEGER, input_4 INTEGER,
                         target INTEGER)""")
    lconn.execute("""CREATE TABLE radar_sites (
                         radar TEXT PRIMARY KEY,
                         latitude REAL, longitude REAL, altitude REAL)""")

    lconn.execute("INSERT INTO sequences VALUES (?, ?, ?, ?, ?, ?, ?)",
                  ("CGU", fi_alcance, *fi_timestamps, fi_timestamps[-1] + 10))

    if fv_com_geometria:
        lconn.execute("INSERT INTO radar_sites VALUES (?, ?, ?, ?)",
                      ("CGU", -23.5, -46.6, 760.0))

    lconn.commit()
    lconn.close()

    return lpth_db

# -------------------------------------------------------------------------------------
def _make_cappi_files(tmp_path, radar, alcance, timestamps, shape=(4, 4)):
    """
    grava um .npy por timestamp seguindo a convenção de nome esperada.
    """
    lpth_cappi_dir = tmp_path / "cappi"
    lpth_cappi_dir.mkdir()

    for li_ts in timestamps:
        lnp = np.full(shape, float(li_ts), dtype=np.float32)
        np.save(lpth_cappi_dir / f"{radar}_{alcance}_{li_ts}.npy", lnp)

    return lpth_cappi_dir

# -------------------------------------------------------------------------------------
def test_main_monta_stack_e_lt_radar_a_partir_do_catalogo(tmp_path, monkeypatch):
    ltimestamps = (10, 20, 30, 40)
    lpth_db = _make_catalog(tmp_path, fi_alcance=400, fi_timestamps=ltimestamps)
    lpth_cappi_dir = _make_cappi_files(tmp_path, "CGU", 400, ltimestamps)

    monkeypatch.setattr(sys, "argv",
                        ["untMain.py", "-r", "CGU", "-a", "400",
                         "--catalog", str(lpth_db), "--cappi-dir", str(lpth_cappi_dir),
                         "-s", str(tmp_path / "saida")])

    # sem nenhum backend disponível -> main() retorna 1 após montar o stack,
    # mas antes disso já exercitou toda a leitura do catálogo
    monkeypatch.setattr(no, "MLST_BACKENDS_DISPONIVEIS", [])

    lret = no.main()

    assert 1 == lret

# -------------------------------------------------------------------------------------
def test_main_sem_sequencia_retorna_1(tmp_path, monkeypatch):
    lpth_db = tmp_path / "catalog.db"
    lconn = sqlite3.connect(lpth_db)
    lconn.execute("""CREATE TABLE sequences (
                         radar TEXT, alcance INTEGER,
                         input_1 INTEGER, input_2 INTEGER,
                         input_3 INTEGER, input_4 INTEGER,
                         target INTEGER)""")
    lconn.execute("""CREATE TABLE radar_sites (
                         radar TEXT PRIMARY KEY,
                         latitude REAL, longitude REAL, altitude REAL)""")
    lconn.commit()
    lconn.close()

    monkeypatch.setattr(sys, "argv",
                        ["untMain.py", "-r", "CGU", "-a", "400", "--catalog", str(lpth_db)])

    assert 1 == no.main()

# -------------------------------------------------------------------------------------
def test_main_sem_geometria_retorna_1(tmp_path, monkeypatch):
    ltimestamps = (10, 20, 30, 40)
    lpth_db = _make_catalog(tmp_path, fi_alcance=400, fi_timestamps=ltimestamps,
                            fv_com_geometria=False)
    lpth_cappi_dir = _make_cappi_files(tmp_path, "CGU", 400, ltimestamps)

    monkeypatch.setattr(sys, "argv",
                        ["untMain.py", "-r", "CGU", "-a", "400",
                         "--catalog", str(lpth_db), "--cappi-dir", str(lpth_cappi_dir)])

    assert 1 == no.main()

# -------------------------------------------------------------------------------------
def test_main_cappi_ausente_retorna_1(tmp_path, monkeypatch):
    ltimestamps = (10, 20, 30, 40)
    lpth_db = _make_catalog(tmp_path, fi_alcance=400, fi_timestamps=ltimestamps)
    # não grava nenhum .npy -- diretório vazio
    lpth_cappi_dir = tmp_path / "cappi"
    lpth_cappi_dir.mkdir()

    monkeypatch.setattr(sys, "argv",
                        ["untMain.py", "-r", "CGU", "-a", "400",
                         "--catalog", str(lpth_db), "--cappi-dir", str(lpth_cappi_dir)])

    assert 1 == no.main()

# -------------------------------------------------------------------------------------
def test_main_cappi_ausente_loga_caminho_procurado_e_o_do_catalogo(tmp_path, monkeypatch,
                                                                    caplog):
    ltimestamps = (10, 20, 30, 40)
    lpth_db = _make_catalog(tmp_path, fi_alcance=400, fi_timestamps=ltimestamps)
    # o 27.nowcast gravou os .npy em outra raiz (ex: UNT_CAPPI_DIR errado)
    (tmp_path / "outra").mkdir()
    lpth_outra = _make_cappi_files(tmp_path / "outra", "CGU", 400, ltimestamps)
    lconn = sqlite3.connect(lpth_db)
    lconn.execute("""CREATE TABLE cappi_products (radar TEXT, alcance INTEGER,
                         timestamp INTEGER, preset TEXT, cappi_path TEXT)""")
    lconn.executemany("INSERT INTO cappi_products VALUES ('CGU', 400, ?, 'p', ?)",
                      [(li, str(lpth_outra / f"CGU_400_{li}.npy")) for li in ltimestamps])
    lconn.commit()
    lconn.close()

    lpth_cappi_dir = tmp_path / "cappi"
    lpth_cappi_dir.mkdir()

    monkeypatch.setattr(sys, "argv",
                        ["untMain.py", "-r", "CGU", "-a", "400",
                         "--catalog", str(lpth_db), "--cappi-dir", str(lpth_cappi_dir)])

    assert 1 == no.main()
    assert f"{lpth_cappi_dir / 'CGU_400_10.npy'} (--cappi-dir={lpth_cappi_dir})" in caplog.text
    assert f"o catálogo registra este CAPPI em {lpth_outra / 'CGU_400_10.npy'} (existe" \
        in caplog.text

# -------------------------------------------------------------------------------------
def test_import_forca_matplotlib_agg_mesmo_com_display():
    # com DISPLAY o matplotlib escolheria QtAgg, e o Qt aborta nos processos
    # do Pool (worker morto -> Pool trava até o timeout do shell)
    ldct_env = {lk: lv for lk, lv in os.environ.items() if "MPLBACKEND" != lk}
    ldct_env.update({"DISPLAY": ":0", "PYTHONPATH": str(pathlib.Path(no.__file__).parent)})

    lproc = subprocess.run([sys.executable, "-c",
                            "import untMain, matplotlib; print(matplotlib.get_backend())"],
                           env=ldct_env, capture_output=True, text=True)

    # última linha -- o pytorch imprime a própria configuração ao ser importado
    assert "agg" == lproc.stdout.split()[-1].lower()

# < main() -- dados no layout do 27.nowcast (preset, 10 min, ±300 km) >------------------

class _PoolSerial:
    """
    substituto de multiprocessing.Pool que roda as tarefas no próprio
    processo (os backends falsos ficam visíveis via sys.modules).
    """
    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def map(self, ff, flst):
        return [ff(lt) for lt in flst]

# -------------------------------------------------------------------------------------
def _make_catalog_nwc(tmp_path, fi_alcance=400, fs_preset="nwc400v2", fi_quality=0,
                      fi_t0=1_781_000_000, fi_dt=600):
    """
    catálogo com o schema atual do 27.nowcast (sequences.preset/quality).
    """
    lpth_db = tmp_path / "catalog.db"
    lconn = sqlite3.connect(lpth_db)

    lconn.execute("""CREATE TABLE sequences (
                         id INTEGER PRIMARY KEY AUTOINCREMENT,
                         radar TEXT, alcance INTEGER,
                         input_1 INTEGER, input_2 INTEGER,
                         input_3 INTEGER, input_4 INTEGER,
                         target INTEGER, quality INTEGER, created_at TEXT,
                         preset TEXT)""")
    lconn.execute("""CREATE TABLE radar_sites (
                         radar TEXT PRIMARY KEY,
                         latitude REAL, longitude REAL, altitude REAL)""")

    ltimestamps = tuple(fi_t0 + li * fi_dt for li in range(5))
    lconn.execute("""INSERT INTO sequences (radar, alcance, input_1, input_2, input_3,
                                            input_4, target, quality, preset)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                  ("STI", fi_alcance, *ltimestamps, fi_quality, fs_preset))
    lconn.execute("INSERT INTO radar_sites VALUES (?, ?, ?, ?)",
                  ("STI", -29.225, -54.930, 430.0))
    lconn.commit()
    lconn.close()

    return lpth_db, ltimestamps

# -------------------------------------------------------------------------------------
def _make_cappi_nwc(tmp_path, fs_preset, fs_radar, fi_alcance, ftimestamps, fi_n=6):
    """
    grava os CAPPIs no layout <cappi>/<preset>/<R>_<A>/<R>_<A>_<ts>.npy, com
    borda fora da cobertura (-9999), fundo sem eco (-32) e um eco de 40 dBZ.
    """
    lpth_cappi = tmp_path / "cappi"
    lpth_dir = lpth_cappi / fs_preset / f"{fs_radar}_{fi_alcance}"
    lpth_dir.mkdir(parents=True)

    for li_ts in ftimestamps:
        lnp = np.full((fi_n, fi_n), -32.0, dtype=np.float32)
        lnp[0, :] = -9999.0
        lnp[2:4, 2:4] = 40.0
        np.save(lpth_dir / f"{fs_radar}_{fi_alcance}_{li_ts}.npy", lnp)

    return lpth_cappi

# -------------------------------------------------------------------------------------
def test_main_layout_nwc_passo_raio_e_verify(tmp_path, monkeypatch):
    lpth_db, ltimestamps = _make_catalog_nwc(tmp_path)
    lpth_cappi = _make_cappi_nwc(tmp_path, "nwc400v2", "STI", 400, ltimestamps)
    lpth_saida = tmp_path / "saida"

    llst_chamadas = []

    def _apply(f_refl, ft_radar, **k):
        llst_chamadas.append((f_refl, ft_radar, k))
        # previsão = persistência do último frame (NaN -> sem eco)
        lnp_last = np.nan_to_num(f_refl[-1], nan=-32.0)
        return obc.ResultadoNowcast(np.stack([lnp_last] * k["fi_num_passos"]), None, {})

    lmod_fake = type(sys)("unt_rainnet_fake_nwc")
    lmod_fake.gera_nowcast = _apply
    _injeta_backend(monkeypatch, "rainnet", lmod_fake)

    monkeypatch.setattr(no, "MLST_BACKENDS_DISPONIVEIS", ["rainnet"])
    monkeypatch.setattr(no.mp, "Pool", _PoolSerial)
    # sem checkpoints reais: todo backend injetado conta como treinado
    monkeypatch.setattr(no, "backends_treinados",
                        lambda fs_dir, *largs: list(no.MLST_BACKENDS_DISPONIVEIS))

    monkeypatch.setattr(sys, "argv",
                        ["untMain.py", "-r", "STI", "-a", "400",
                         "--catalog", str(lpth_db), "--cappi-dir", str(lpth_cappi),
                         "-s", str(lpth_saida), "--preset", "nwc400v2",
                         "--quality-max", "0", "--verify"])

    assert 0 == no.main()

    lnp_refl, lt_radar, ldct_k = llst_chamadas[0]

    # -9999 (fora da cobertura) virou NaN antes de chegar ao backend
    assert np.isnan(lnp_refl[:, 0, :]).all()
    assert (-32.0 == lnp_refl[:, 1, 0]).all()
    # passo de tempo vem da sequência (600 s -> 10 min), não mais 5 fixo
    assert 10 == ldct_k["fi_passo_minutos"]
    # raio do mapa vem da grade (6x6 a 1 km/px -> 3 km), não do alcance (400)
    assert 3.0 == lt_radar.raio_mapa_km

    ldct_v = __import__("json").loads((lpth_saida / "verify.json").read_text())
    assert 1 == ldct_v["lead"]
    assert "nwc400v2" == ldct_v["preset"]
    assert 1.0 == ldct_v["backends"]["rainnet"]["limiares"]["36"]["csi"]

    # referencia.json, para o GIF montado do untMakGIF
    ldct_r = __import__("json").loads((lpth_saida / "referencia.json").read_text())
    assert ltimestamps[3] == ldct_r["referencia"]
    assert ltimestamps[4] == ldct_r["target"]
    assert 10 == ldct_r["timestep_min"]
    assert ["rainnet"] == ldct_r["backends"]
    assert ("STI", 400) == (ldct_r["radar"], ldct_r["alcance"])

# -------------------------------------------------------------------------------------
def test_main_prefix_nomeia_frames_e_previsao(tmp_path, monkeypatch):
    lpth_db, ltimestamps = _make_catalog_nwc(tmp_path)
    lpth_cappi = _make_cappi_nwc(tmp_path, "nwc400v2", "STI", 400, ltimestamps)
    lpth_saida = tmp_path / "saida"

    llst_prefix = []

    def _apply(f_refl, ft_radar, **k):
        llst_prefix.append(k["fs_prefixo_saida"])
        lnp_last = np.nan_to_num(f_refl[-1], nan=-32.0)
        return obc.ResultadoNowcast(np.stack([lnp_last] * k["fi_num_passos"]), None, {})

    lmod_fake = type(sys)("unt_rainnet_fake_prefix")
    lmod_fake.gera_nowcast = _apply
    _injeta_backend(monkeypatch, "rainnet", lmod_fake)
    monkeypatch.setattr(no, "MLST_BACKENDS_DISPONIVEIS", ["rainnet"])
    monkeypatch.setattr(no.mp, "Pool", _PoolSerial)
    # sem checkpoints reais: todo backend injetado conta como treinado
    monkeypatch.setattr(no, "backends_treinados",
                        lambda fs_dir, *largs: list(no.MLST_BACKENDS_DISPONIVEIS))

    monkeypatch.setattr(sys, "argv",
                        ["untMain.py", "-r", "STI", "-a", "400", "-p", "STI_400",
                         "--catalog", str(lpth_db), "--cappi-dir", str(lpth_cappi),
                         "-s", str(lpth_saida), "--verify"])

    assert 0 == no.main()

    # o backend recebe <saida>/<backend>/<prefix> e a verificação lê <prefix>.npy
    assert [str(lpth_saida / "rainnet" / "STI_400")] == llst_prefix
    assert (lpth_saida / "rainnet" / "STI_400.npy").exists()
    assert (lpth_saida / "verify.json").exists()

# -------------------------------------------------------------------------------------
def test_main_filtro_de_quality_sem_correspondencia_retorna_1(tmp_path, monkeypatch):
    lpth_db, ltimestamps = _make_catalog_nwc(tmp_path, fi_quality=1)
    lpth_cappi = _make_cappi_nwc(tmp_path, "nwc400v2", "STI", 400, ltimestamps)

    monkeypatch.setattr(sys, "argv",
                        ["untMain.py", "-r", "STI", "-a", "400",
                         "--catalog", str(lpth_db), "--cappi-dir", str(lpth_cappi),
                         "--quality-max", "0"])

    assert 1 == no.main()

# < executa_backend >--------------------------------------------------------------------

def test_executa_backend_sucesso_retorna_tupla_ok(monkeypatch, tmp_path):
    llst_chamadas = []

    lmod_fake = type(sys)("unt_rainnet_fake")

    def _apply(*a, **k):
        llst_chamadas.append((a, k))
        return obc.ResultadoNowcast(np.ones((6, 4, 4)), None, {})

    lmod_fake.gera_nowcast = _apply

    _injeta_backend(monkeypatch, "rainnet", lmod_fake)

    lnp_refl = np.zeros((4, 4, 4))
    ls_prefix = str(tmp_path / "out" / "nowcast")

    lresult = no.executa_backend(no.TarefaBackend("rainnet", lnp_refl,
                                                  obc.GeometriaRadar(-30., -52., 200., 240.),
                                                  6, 5, ls_prefix))

    assert ("rainnet", True, None) == lresult
    assert 1 == len(llst_chamadas)
    # diretório de saída deve ter sido criado antes de chamar o backend
    assert (tmp_path / "out").is_dir()
    # previsão (dBZ) gravada para a verificação no processo pai
    assert (6, 4, 4) == np.load(ls_prefix + ".npy").shape

# -------------------------------------------------------------------------------------
def test_executa_backend_sem_previsao_nao_grava_npy(monkeypatch, tmp_path):
    lmod_fake = type(sys)("unt_rainnet_fake_sem_chuva")
    lmod_fake.gera_nowcast = lambda *a, **k: obc.ResultadoNowcast(None, None, {})
    _injeta_backend(monkeypatch, "rainnet", lmod_fake)

    ls_prefix = str(tmp_path / "out" / "nowcast")

    lresult = no.executa_backend(no.TarefaBackend("rainnet", np.zeros((4, 4, 4)),
                                                  obc.GeometriaRadar(0., 0., 0., 1.), 6, 10,
                                                  ls_prefix))

    assert ("rainnet", True, None) == lresult
    assert not pathlib.Path(ls_prefix + ".npy").exists()

# -------------------------------------------------------------------------------------
def test_executa_backend_backend_desconhecido_retorna_tupla_falha():
    lresult = no.executa_backend(no.TarefaBackend("xyz", np.zeros((4, 2, 2)),
                                                  obc.GeometriaRadar(0., 0., 0., 1.), 6, 5,
                                                  "prefix"))

    ls_nome, lv_ok, ls_erro = lresult
    assert "xyz" == ls_nome
    assert lv_ok is False
    assert "backend desconhecido" in ls_erro

# -------------------------------------------------------------------------------------
def test_executa_backend_excecao_do_backend_e_contida(monkeypatch, tmp_path):
    lmod_fake = type(sys)("unt_rainnet_fake_falha")

    def _falha(*a, **k):
        raise RuntimeError("falha simulada no backend")

    lmod_fake.gera_nowcast = _falha
    _injeta_backend(monkeypatch, "rainnet", lmod_fake)

    ls_nome, lv_ok, ls_erro = no.executa_backend(
        no.TarefaBackend("rainnet", np.zeros((4, 2, 2)), obc.GeometriaRadar(0., 0., 0., 1.),
                         6, 5, str(tmp_path / "prefix")))

    assert "rainnet" == ls_nome
    assert lv_ok is False
    assert "falha simulada no backend" == ls_erro

# < the end >----------------------------------------------------------------------------
