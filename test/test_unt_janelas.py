# -*- coding: utf-8 -*-
"""
Centro Logístico de Simulação Meteorológica
Copyright (c) 1995-2026, Instituto de Controle do Espaço Aéreo

test_unt_janelas
testes de unt_janelas (associação encadeada com deriva, lacuna, duplicata no
mesmo slot, janelas 4 + 6, alvo interpolado rejeitado, filtros, split estável,
manifesto ida e volta, purge na fronteira do split), de untJanelas.main
sobre o catálogo sintético e da leitura do config.ini (unt_config)

2026.out  mlabru  purge na fronteira do split
2026.out  mlabru  initial version (Linux/Python)
"""
# < imports >----------------------------------------------------------------------------

# pytest
import pytest

# model
import untJanelas as nj
import work.unt_config as uc
import work.unt_janelas as uj

# < helpers >----------------------------------------------------------------------------

def _produtos(flst_deltas, fs_origem="observed", ff_valid=0.79, ff_chuva=0.1, fi_t0=1000):
    llst = [uj.SProduto(fi_t0, fs_origem, ff_valid, ff_chuva)]
    for li_delta in flst_deltas:
        llst.append(uj.SProduto(llst[-1].timestamp + li_delta, fs_origem, ff_valid, ff_chuva))
    return llst

# < associa_slots >----------------------------------------------------------------------

def test_associa_slots_deriva_nao_acumula():
    # BVA real: 610-622 s por volume; uma grade fixa perderia a série
    llst = _produtos([615] * 30)
    llst_slots = uj.associa_slots(llst)
    assert len(llst) == len(llst_slots)
    assert all(lprod is not None for lprod in llst_slots)

# -------------------------------------------------------------------------------------
def test_associa_slots_lacuna_vira_slots_vazios():
    llst_slots = uj.associa_slots(_produtos([600, 1800, 600]))
    assert [lprod is None for lprod in llst_slots] == [False, False, True, True, False, False]

# -------------------------------------------------------------------------------------
def test_associa_slots_mesmo_slot_observado_vence():
    llst = [uj.SProduto(0, "interpolated", 0.8, 0.1), uj.SProduto(100, "observed", 0.8, 0.1)]
    llst_slots = uj.associa_slots(llst)
    assert 1 == len(llst_slots)
    assert "observed" == llst_slots[0].origin

# -------------------------------------------------------------------------------------
def test_associa_slots_tolerancia_maior_que_meio_intervalo_recusada():
    with pytest.raises(ValueError, match="tolerancia_s"):
        uj.associa_slots([], 600, 301)

# < monta_janelas >----------------------------------------------------------------------

def test_monta_janelas_conta_e_ordem():
    llst_janelas, ldct_rej = uj.monta_janelas(_produtos([600] * 11), "STI", 250, "nwc250v2")

    # 12 produtos, janelas de 10: 3
    assert 3 == len(llst_janelas)
    assert {} == ldct_rej
    ljanela = llst_janelas[0]
    assert 4 == len(ljanela.entradas) and 6 == len(ljanela.alvos)
    assert ljanela.entradas[-1] < ljanela.alvos[0]
    assert 0 == ljanela.quality

# -------------------------------------------------------------------------------------
def test_monta_janelas_alvo_interpolado_rejeitado_entrada_interpolada_aceita():
    llst = _produtos([600] * 9)
    # entrada 2 interpolada: quality 1; nenhum alvo interpolado
    llst[1] = llst[1]._replace(origin="interpolated")
    llst_janelas, _ = uj.monta_janelas(llst, "STI", 250, "p")
    assert 1 == len(llst_janelas) and 1 == llst_janelas[0].quality

    # alvo interpolado: rejeitado
    llst[1] = llst[1]._replace(origin="observed")
    llst[7] = llst[7]._replace(origin="interpolated")
    llst_janelas, ldct_rej = uj.monta_janelas(llst, "STI", 250, "p")
    assert [] == llst_janelas
    assert {"alvo_interpolado": 1} == ldct_rej

# -------------------------------------------------------------------------------------
def test_monta_janelas_filtros_cobertura_e_seca():
    llst = _produtos([600] * 9)
    llst[3] = llst[3]._replace(valid_fraction=0.5)
    _, ldct_rej = uj.monta_janelas(llst, "STI", 250, "p")
    assert {"cobertura": 1} == ldct_rej

    lt_param = uj.SParametrosJanela(min_rain_fraction=0.5)
    _, ldct_rej = uj.monta_janelas(_produtos([600] * 9), "STI", 250, "p", lt_param)
    assert {"seca": 1} == ldct_rej

# -------------------------------------------------------------------------------------
def test_monta_janelas_lacuna_quebra_janela():
    llst_janelas, _ = uj.monta_janelas(_produtos([600] * 5 + [1200] + [600] * 5), "STI", 250, "p")
    assert [] == llst_janelas

# < split >------------------------------------------------------------------------------

def test_atribui_split_estavel_e_por_bloco():
    li_ts = 1_781_000_000
    ls_split = uj.atribui_split(li_ts, 0.7, 0.15)
    # mesmo dia, mesmo conjunto
    assert ls_split == uj.atribui_split(li_ts + 3600, 0.7, 0.15)
    assert ls_split in uj.CT_SPLITS
    assert "treino" == uj.atribui_split(li_ts, 1.0, 0.0)
    assert "teste" == uj.atribui_split(li_ts, 0.0, 0.0)

# -------------------------------------------------------------------------------------
def test_atribui_split_distribui_blocos():
    llst = [uj.atribui_split(li * 86400, 0.7, 0.15) for li in range(2000)]
    assert 0.6 < llst.count("treino") / 2000 < 0.8

# -------------------------------------------------------------------------------------
def _dia_com_vizinho(fb_mesmo_split):
    # primeiro dia d em que o split de d e o de d+1 são (ou não) o mesmo
    for li_dia in range(20000, 21000):
        lb_mesmo = (uj.atribui_split(li_dia * 86400, 0.7, 0.15) ==
                    uj.atribui_split((li_dia + 1) * 86400, 0.7, 0.15))
        if lb_mesmo == fb_mesmo_split:
            return li_dia
    raise AssertionError("sem dia adequado")


def test_monta_janelas_purge_na_fronteira_do_split():
    # 21 volumes cruzando a meia-noite (5 antes): as janelas que começam nos 5
    # primeiros têm quadros nos dois dias
    li_dia = _dia_com_vizinho(fb_mesmo_split=False)
    li_t0 = (li_dia + 1) * 86400 - 5 * 600

    llst_janelas, ldct_rej = uj.monta_janelas(_produtos([600] * 20, fi_t0=li_t0), "STI", 250, "p")

    assert 5 == ldct_rej["fronteira_split"]
    assert 7 == len(llst_janelas)
    ls_split_dia_seguinte = uj.atribui_split((li_dia + 1) * 86400, 0.7, 0.15)
    assert {ls_split_dia_seguinte} == {lj.split for lj in llst_janelas}


def test_monta_janelas_sem_purge_quando_dias_vizinhos_tem_o_mesmo_split():
    li_dia = _dia_com_vizinho(fb_mesmo_split=True)
    li_t0 = (li_dia + 1) * 86400 - 5 * 600

    llst_janelas, ldct_rej = uj.monta_janelas(_produtos([600] * 20, fi_t0=li_t0), "STI", 250, "p")

    assert "fronteira_split" not in ldct_rej
    assert 12 == len(llst_janelas)


def test_split_da_janela():
    assert "treino" == uj.split_da_janela([0, 600, 86400], 1.0, 0.0)
    li_dia = _dia_com_vizinho(fb_mesmo_split=False)
    assert uj.split_da_janela([li_dia * 86400, (li_dia + 1) * 86400], 0.7, 0.15) is None

# < manifesto >--------------------------------------------------------------------------

def test_manifesto_ida_e_volta(tmp_path):
    llst_janelas, _ = uj.monta_janelas(_produtos([600] * 11), "STI", 250, "nwc250v2")
    lpth = tmp_path / "j" / "m.csv"

    uj.grava_manifesto(llst_janelas, lpth, {"num_entradas": 4, "num_alvos": 6, "x": 1})
    llst_lidas, ldct_param = uj.le_manifesto(lpth)

    assert llst_janelas == llst_lidas
    assert 1 == ldct_param["x"]
    assert sum(uj.conta_por_split(llst_lidas).values()) == 3

# < untJanelas.main >--------------------------------------------------------------------

def test_untjanelas_main_catalogo_sintetico(tmp_path, catalogo_sintetico):
    lpth = tmp_path / "janelas.csv"
    li_rc = nj.main(["--catalog", catalogo_sintetico["catalogo"],
                     "--cappi-dir", catalogo_sintetico["cappi_dir"], "-o", str(lpth)])
    assert 0 == li_rc

    llst, ldct = uj.le_manifesto(lpth)
    # 40 volumes -> 31 janelas de 10
    assert 31 == len(llst)
    assert "nwc250v2" == llst[0].preset
    assert catalogo_sintetico["catalogo"] == ldct["catalog"]

# -------------------------------------------------------------------------------------
def test_untjanelas_main_arquivo_ausente_descarta(tmp_path, catalogo_sintetico):
    import pathlib
    lpth_npy = next(pathlib.Path(catalogo_sintetico["cappi_dir"]).rglob("*.npy"))
    lpth_npy.unlink()

    lpth = tmp_path / "janelas.csv"
    assert 0 == nj.main(["--catalog", catalogo_sintetico["catalogo"],
                         "--cappi-dir", catalogo_sintetico["cappi_dir"], "-o", str(lpth)])
    llst, _ = uj.le_manifesto(lpth)
    assert len(llst) < 31

# -------------------------------------------------------------------------------------
def test_untjanelas_main_sem_produtos_retorna_1(tmp_path, catalogo_sintetico):
    assert 1 == nj.main(["--catalog", catalogo_sintetico["catalogo"], "-r", "XXX",
                         "-o", str(tmp_path / "j.csv")])

# < unt_config >-------------------------------------------------------------------------

def test_config_defaults_sem_arquivo_e_backend_sobrescreve_treino(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    lcfg = uc.carrega()
    assert 600 == lcfg.inteiro("janelas", "intervalo_s")
    assert "rainnet" == lcfg.secao_do_backend("rainnet", "lr")
    assert "treino" == lcfg.secao_do_backend("rainnet", "batch")

    (tmp_path / "c.ini").write_text("[treino]\nbatch = 3\n[rainnet]\nbatch = 5\n")
    lcfg = uc.carrega(str(tmp_path / "c.ini"))
    assert 5 == lcfg.inteiro(lcfg.secao_do_backend("rainnet", "batch"), "batch")
    assert 3 == lcfg.inteiro(lcfg.secao_do_backend("smaat", "batch"), "batch")

# -------------------------------------------------------------------------------------
def test_config_valor_invalido_cita_secao_e_arquivo_ausente(tmp_path):
    (tmp_path / "c.ini").write_text("[treino]\nepocas = muitas\namp = talvez\n")
    lcfg = uc.carrega(str(tmp_path / "c.ini"))
    with pytest.raises(ValueError, match=r"\[treino\] epocas"):
        lcfg.inteiro("treino", "epocas")
    with pytest.raises(ValueError, match="booleano"):
        lcfg.booleano("treino", "amp")
    with pytest.raises(FileNotFoundError):
        uc.carrega(str(tmp_path / "nao_existe.ini"))

# < the end >----------------------------------------------------------------------------
