# -*- coding: utf-8 -*-
"""
Centro Logístico de Simulação Meteorológica
Copyright (c) 1995-2026, Instituto de Controle do Espaço Aéreo

conftest
configuração compartilhada pela suíte de testes unitários (backend do
matplotlib, cache do mapa em diretório temporário, catálogo sintético com
CAPPIs 64x64 para os testes de janelas/treino/inferência)

2026.out  mlabru  portado de 25.oflow; + fixture catalogo_sintetico.

2026.ago  mlabru  initial version (Linux/Python)
"""
# < matplotlib backend >-----------------------------------------------------------------
# força um backend não-interativo antes que qualquer módulo sob teste importe
# matplotlib.pyplot, evitando falha em ambiente sem display (headless/CI)

import os
import matplotlib
matplotlib.use("Agg")

# < cache do mapa >-----------------------------------------------------------------------
# os testes que plotam não podem gravar o cache de feições (apps/unt_mapa.py)
# no HOME do usuário: aponta para um diretório temporário da sessão

import pathlib
import sys

import pytest

@pytest.fixture(autouse=True, scope="session")
def _cache_mapa_temporario(tmp_path_factory):
    lmp = pytest.MonkeyPatch()
    lmp.setenv("UNT_MAPA_CACHE", str(tmp_path_factory.mktemp("unt_mapa")))
    yield
    lmp.undo()

# < catálogo sintético >-----------------------------------------------------------------
# catálogo SQLite (cappi_products, sequences, radar_sites) + CAPPIs .npy num layout
# de preset, com um núcleo de chuva que anda 2 px por volume -- pequeno o bastante
# para treinar uma rede minúscula em CPU em poucos segundos

import sqlite3

import numpy as np

# t0 e cadência (com a deriva real de alguns segundos)
CI_T0_SINTETICO = 1_781_000_000
CT_DELTAS_SINTETICOS = (600, 598, 603, 600, 597, 602)


def cria_cappi_sintetico(fi_passo: int, fi_lado: int=64) -> np.ndarray:
    """
    campo dBZ: -32 (sem eco), canto fora da cobertura (-9999) e um núcleo de 45 dBZ
    com borda de 25 dBZ deslocado 2 px por passo.
    """
    lnp = np.full((fi_lado, fi_lado), -32.0, dtype=np.float32)
    lnp[:4, :4] = -9999.0

    li_col = 8 + 2 * fi_passo
    lnp[20:36, li_col:li_col + 16] = 25.0
    lnp[24:32, li_col + 4:li_col + 12] = 45.0

    return lnp


def cria_catalogo_sintetico(fpth_raiz, fi_num_volumes: int=40, fs_radar: str="STI",
                            fi_alcance: int=250, fs_preset: str="nwc250v2",
                            fset_interpolados: tuple=()) -> dict:
    """
    grava catalog.db + CAPPIs e devolve {"catalogo", "cappi_dir", "timestamps"}.
    """
    lpth_raiz = pathlib.Path(fpth_raiz)
    lpth_cappi = lpth_raiz / "cappi"
    lpth_dir = lpth_cappi / fs_preset / f"{fs_radar}_{fi_alcance}"
    lpth_dir.mkdir(parents=True, exist_ok=True)

    llst_ts = [CI_T0_SINTETICO]
    for li in range(1, fi_num_volumes):
        llst_ts.append(llst_ts[-1] + CT_DELTAS_SINTETICOS[li % len(CT_DELTAS_SINTETICOS)])

    lpth_db = lpth_raiz / "catalog.db"
    lconn = sqlite3.connect(lpth_db)
    lconn.execute("""CREATE TABLE cappi_products (
                         id INTEGER PRIMARY KEY AUTOINCREMENT, radar TEXT, alcance INTEGER,
                         timestamp INTEGER, preset TEXT, origin TEXT, status INTEGER,
                         attempts INTEGER, cappi_path TEXT, rain_fraction REAL,
                         valid_fraction REAL, created_at TEXT, updated_at TEXT)""")
    lconn.execute("""CREATE TABLE sequences (
                         id INTEGER PRIMARY KEY AUTOINCREMENT, radar TEXT, alcance INTEGER,
                         input_1 INTEGER, input_2 INTEGER, input_3 INTEGER, input_4 INTEGER,
                         target INTEGER, quality INTEGER, created_at TEXT, preset TEXT)""")
    lconn.execute("""CREATE TABLE radar_sites (radar TEXT PRIMARY KEY, latitude REAL,
                         longitude REAL, altitude REAL, updated_at TEXT)""")
    lconn.execute("INSERT INTO radar_sites VALUES (?, -29.2, -54.9, 400.0, NULL)", (fs_radar,))

    for li_passo, li_ts in enumerate(llst_ts):
        lnp = cria_cappi_sintetico(li_passo % 20)
        np.save(lpth_dir / f"{fs_radar}_{fi_alcance}_{li_ts}.npy", lnp)

        ls_origem = "interpolated" if li_passo in fset_interpolados else "observed"
        lconn.execute("""INSERT INTO cappi_products (radar, alcance, timestamp, preset, origin,
                             status, rain_fraction, valid_fraction)
                         VALUES (?, ?, ?, ?, ?, 2, 0.05, 0.99)""",
                      (fs_radar, fi_alcance, li_ts, fs_preset, ls_origem))

    for li in range(4, len(llst_ts)):
        lconn.execute("""INSERT INTO sequences (radar, alcance, input_1, input_2, input_3,
                             input_4, target, quality, preset) VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?)""",
                      (fs_radar, fi_alcance, *llst_ts[li - 4:li + 1], fs_preset))

    lconn.commit()
    lconn.close()

    return {"catalogo": str(lpth_db), "cappi_dir": str(lpth_cappi), "timestamps": llst_ts}


@pytest.fixture
def catalogo_sintetico(tmp_path):
    return cria_catalogo_sintetico(tmp_path / "sintetico")

# < ambiente cls >---------------------------------------------------------------------

# a suíte não depende do cls.env de quem a roda (~/.config/cls/cls.env): os
# scripts shell testados herdam este ambiente e não carregam nada de lá
os.environ["CLS_ENV_FILE"] = os.devnull

# cls_common.sh: CLS_LIB do ambiente ou, no checkout de desenvolvimento, o
# repo cls.common irmão (<raiz>/cls.common, com o projeto em
# <raiz>/<nn>.<proj>/cls.<proj>)
os.environ.setdefault("CLS_LIB", str(pathlib.Path(__file__).resolve().parents[3]
                                     / "cls.common" / "bash"))

# pacote Python comum (cls_common): CLS_PYLIB do ambiente ou o cls.common
# irmão -- no sys.path desta suíte e no PYTHONPATH dos subprocessos (os
# scripts shell o recebem de novo por cls_setup_pythonpath, sem repetir)
os.environ.setdefault("CLS_PYLIB", str(pathlib.Path(__file__).resolve().parents[3]
                                       / "cls.common" / "python"))
if os.environ["CLS_PYLIB"] not in sys.path:
    sys.path.insert(0, os.environ["CLS_PYLIB"])
if os.environ["CLS_PYLIB"] not in os.environ.get("PYTHONPATH", "").split(":"):
    os.environ["PYTHONPATH"] = ":".join(filter(None, (os.environ["CLS_PYLIB"],
                                                      os.environ.get("PYTHONPATH"))))

# < the end >--------------------------------------------------------------------------
