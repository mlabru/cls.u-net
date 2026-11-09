# -*- coding: utf-8 -*-
"""
Centro Logístico de Simulação Meteorológica
Copyright (c) 1995-2026, Instituto de Controle do Espaço Aéreo

test_unt_mapa
testes do cache de feições do mapa (unt_mapa): grava em disco na primeira
chamada, reaproveita nas seguintes e nunca falha por causa do cache

2026.out  mlabru  portado de 25.oflow.

2026.out  mlabru  nomes novos das funções testadas (refatoração de legibilidade)

2026.out  mlabru  initial version (Linux/Python)
"""
# < imports >----------------------------------------------------------------------------

# python library
import pickle

# pytest
import pytest

# model
import work.unt_mapa as om

# < fixtures >---------------------------------------------------------------------------

@pytest.fixture
def feicoes_falsas(monkeypatch, tmp_path):
    """
    troca as feições do Cartopy por objetos que contam as chamadas (sem ler
    shapefile) e aponta o cache para tmp_path.
    """
    class _Feicao:
        def __init__(self, fs_nome):
            self.nome, self.chamadas = fs_nome, 0

        def intersecting_geometries(self, fl_extent):
            self.chamadas += 1
            return [f"{self.nome}:{fl_extent}"]

    llst = [_Feicao("borders"), _Feicao("coast"), _Feicao("states")]
    monkeypatch.setattr(om, "MT_FEICOES", tuple((lf, {}) for lf in llst))
    monkeypatch.setenv("UNT_MAPA_CACHE", str(tmp_path / "cache"))
    om.geometrias_da_area.cache_clear()
    yield llst
    om.geometrias_da_area.cache_clear()

# < testes >-----------------------------------------------------------------------------

def test_geometrias_gravam_e_reusam_cache_em_disco(feicoes_falsas, tmp_path):
    lt_ext = (-58.0, -52.0, -32.0, -26.5)

    lt_1 = om.geometrias_da_area(lt_ext)

    assert [1, 1, 1] == [lf.chamadas for lf in feicoes_falsas]
    assert 1 == len(list((tmp_path / "cache").glob("*.pkl")))

    # novo "processo": sem cache em memória, lê do disco sem recalcular
    om.geometrias_da_area.cache_clear()
    assert lt_1 == om.geometrias_da_area(lt_ext)
    assert [1, 1, 1] == [lf.chamadas for lf in feicoes_falsas]

# -------------------------------------------------------------------------------------
def test_cache_corrompido_recalcula(feicoes_falsas, tmp_path):
    lt_ext = (0.0, 1.0, 0.0, 1.0)
    lpth = om.caminho_do_cache(lt_ext)
    lpth.parent.mkdir(parents=True)
    lpth.write_bytes(b"lixo")

    lt = om.geometrias_da_area(lt_ext)

    assert ("borders:[0.0, 1.0, 0.0, 1.0]", ) == lt[0]
    # regravado com conteúdo válido
    assert lt == pickle.loads(lpth.read_bytes())

# -------------------------------------------------------------------------------------
def test_cache_sem_permissao_nao_impede_o_mapa(feicoes_falsas, monkeypatch, tmp_path):
    lpth_ro = tmp_path / "ro"
    lpth_ro.mkdir()
    lpth_ro.chmod(0o500)
    monkeypatch.setenv("UNT_MAPA_CACHE", str(lpth_ro / "sub"))

    try:
        assert 3 == len(om.geometrias_da_area((0.0, 1.0, 0.0, 1.0)))
    finally:
        lpth_ro.chmod(0o700)

# -------------------------------------------------------------------------------------
def test_extensoes_diferentes_tem_arquivos_diferentes():
    assert om.caminho_do_cache((0., 1., 0., 1.)) != om.caminho_do_cache((0., 1., 0., 2.))

# < the end >----------------------------------------------------------------------------
