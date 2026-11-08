"""
teste_logica.py - Testes da lógica de contagem, área restrita, alertas e série temporal,
SEM usar YOLO nem Streamlit (os alertas usam um modelo falso).

Execute:  python teste_logica.py
Serve para mostrar ao professor como funciona o cruzamento da linha.
"""

import csv
import io

import numpy as np

import vision

LINHA = 100  # linha horizontal em y = 100


def testar_sentidos():
    assert vision.sentido_de_passagem(90, 110, LINHA) == "descendo"
    assert vision.sentido_de_passagem(110, 90, LINHA) == "subindo"
    assert vision.sentido_de_passagem(80, 95, LINHA) is None      # ainda acima
    assert vision.sentido_de_passagem(120, 140, LINHA) is None    # já estava abaixo
    assert vision.sentido_de_passagem(99.9, 100, LINHA) == "descendo"  # tocar a linha conta
    print("OK  sentidos de passagem")


def testar_contagem_unica_por_id():
    c = vision.ContadorDeLinha(LINHA)
    c.atualizar(1, "carro", 90, 0.0, 0)
    assert c.atualizar(1, "carro", 110, 0.1, 1) is not None   # cruzou: conta
    assert c.atualizar(1, "carro", 90, 0.2, 2) is None        # voltou: NÃO conta de novo
    assert c.atualizar(1, "carro", 120, 0.3, 3) is None       # cruzou outra vez: NÃO conta de novo
    assert c.total == 1 and c.contagem["carro"] == 1
    print("OK  cada ID é contado uma única vez")


def testar_categorias_e_sentidos():
    c = vision.ContadorDeLinha(LINHA)
    c.atualizar(1, "carro", 90, 0.0, 0)
    c.atualizar(1, "carro", 110, 0.1, 1)       # descendo
    c.atualizar(2, "moto", 130, 0.0, 0)
    c.atualizar(2, "moto", 70, 0.1, 1)         # subindo
    c.atualizar(3, "caminhão", 50, 0.0, 0)
    c.atualizar(3, "caminhão", 60, 0.1, 1)     # não cruzou
    assert {k: c.contagem[k] for k in vision.CATEGORIAS} == {"carro": 1, "moto": 1, "ônibus": 0, "caminhão": 0}
    assert c.por_sentido["carro"]["descendo"] == 1
    assert c.por_sentido["moto"]["subindo"] == 1
    assert c.total == 2 and len(c.eventos) == 2
    print("OK  categorias e sentidos")


def testar_id_que_aparece_depois_da_linha():
    c = vision.ContadorDeLinha(LINHA)
    assert c.atualizar(7, "carro", 150, 0.0, 0) is None  # primeira aparição já abaixo: sem posição anterior
    assert c.total == 0
    print("OK  ID que surge depois da linha não é contado")


def testar_novo_contador_comeca_zerado():
    a = vision.ContadorDeLinha(LINHA)
    a.atualizar(1, "carro", 90, 0.0, 0)
    a.atualizar(1, "carro", 110, 0.1, 1)
    b = vision.ContadorDeLinha(LINHA)  # nova análise
    assert b.total == 0 and b.eventos == []
    b.atualizar(1, "carro", 90, 0.0, 0)
    assert b.atualizar(1, "carro", 110, 0.1, 1) is not None  # o mesmo número de ID pode contar de novo
    print("OK  nova análise começa do zero")


def testar_mapa_de_classes():
    coco = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
    assert vision.mapa_de_classes(coco) == {0: "pessoa", 2: "carro", 3: "moto", 5: "ônibus", 7: "caminhão"}
    assert vision.mapa_de_classes({0: "Caminhão", 1: "Ônibus"}) == {0: "caminhão", 1: "ônibus"}
    assert vision.mapa_de_classes({0: "gato", 1: "cachorro"}) == {}
    assert vision.ids_epi(coco) == {}                      # o modelo COCO não tem classes de EPI
    assert vision.ids_epi({0: "helmet", 1: "No-Helmet", 2: "sem colete"}) == {1: "No-Helmet", 2: "sem colete"}
    print("OK  mapa de classes e classes de EPI")


def testar_area_restrita():
    area = vision.area_em_pixels((0.5, 0.5, 1.0, 1.0), 200, 100)   # x 100..200, y 50..100
    assert area == (100, 50, 200, 100)
    assert vision.ponto_na_area(150, 80, area)
    assert not vision.ponto_na_area(50, 80, area)
    assert not vision.ponto_na_area(150, 20, area)
    assert vision.area_em_pixels((1.0, 1.0, 0.5, 0.5), 200, 100) == area   # aceita cantos invertidos
    print("OK  área restrita")


def testar_serie_temporal():
    eventos = [
        {"tempo_s": 0.5, "categoria": "carro"}, {"tempo_s": 1.5, "categoria": "moto"},
        {"tempo_s": 5.2, "categoria": "carro"}, {"tempo_s": 99.0, "categoria": "carro"},  # além do fim: vai p/ o último
    ]
    serie = vision.serie_temporal(eventos, 10, 5)   # intervalos 0-5 e 5-10 (+1 de folga)
    assert [l["início (s)"] for l in serie] == [0, 5, 10]
    assert serie[0]["total"] == 2 and serie[0]["carro"] == 1 and serie[0]["moto"] == 1
    assert serie[1]["total"] == 1
    assert serie[2]["total"] == 1
    assert vision.intervalo_sugerido(10) == 1 and vision.intervalo_sugerido(40) == 2 and vision.intervalo_sugerido(120) == 5
    print("OK  série temporal")


class _T:  # imita um tensor do PyTorch (só o que o analisador usa)
    def __init__(self, a): self.a = np.array(a)
    def cpu(self): return self
    def numpy(self): return self.a
    def int(self): return _T(self.a.astype(int))
    def tolist(self): return self.a.tolist()


class _Caixas:
    def __init__(self, xyxy, ids, cls, conf):
        self.xyxy, self.cls, self.conf = _T(xyxy), _T(cls), _T(conf)
        self.id = None if ids is None else _T(ids)
    def __len__(self): return len(self.xyxy.a)


class _Res:
    def __init__(self, caixas): self.boxes = caixas


class _ModeloFalso:
    """Uma pessoa (id 1) caminha para dentro da área; um 'sem capacete' (id 2) aparece uma vez."""
    predictor = "estado_antigo"
    def __init__(self): self.n = 0
    def track(self, quadro, **kwargs):
        self.n += 1
        x = 10 + self.n * 20
        caixas = [[x, 40, x + 20, 90]]; ids = [1]; cls = [0]; conf = [0.9]
        if self.n in (3, 4):  # mesmo ID 2 em dois quadros: deve gerar UM alerta
            caixas.append([5, 5, 25, 25]); ids.append(2); cls.append(9); conf.append(0.8)
        return [_Res(_Caixas(caixas, ids, cls, conf))]


def testar_alertas_com_modelo_falso():
    modelo = _ModeloFalso()
    analisador = vision.AnalisadorDeQuadros(
        modelo, {0: "pessoa"}, {9: "no_helmet"}, 0.3, linha_rel=0.5,
        area=(0.5, 0.0, 1.0, 1.0), alvos_area=("pessoa",))
    assert modelo.predictor is None  # estado do rastreador reiniciado
    novos_total = []
    for i in range(8):
        quadro = np.zeros((100, 200, 3), np.uint8)
        novos_total += analisador.processar(quadro, i, i / 10)
    tipos = [a["tipo"] for a in analisador.alertas]
    assert tipos.count("falta de EPI") == 1, tipos          # 1 alerta por ID, mesmo aparecendo 2 vezes
    assert tipos.count("invasão de área") == 1, tipos       # a pessoa entrou na área e só alertou uma vez
    assert len(novos_total) == 2
    assert analisador.conf_media is not None and analisador.ms_medio >= 0
    csv_texto = vision.csv_alertas(analisador.alertas).decode("utf-8-sig")
    assert csv_texto.splitlines()[0] == "tempo_s,quadro,tipo,alvo,id"
    analisador.encerrar()
    assert modelo.predictor is None
    print("OK  alertas de invasão de área e de EPI (modelo falso)")


def testar_webcam_inexistente():
    try:
        vision.abrir_webcam(97)
    except vision.ErroDeArquivo:
        print("OK  webcam inexistente gera erro claro")
        return
    raise AssertionError("era esperado ErroDeArquivo")


def testar_csv():
    dados = vision.csv_video([{"id": 4, "categoria": "ônibus", "tempo_s": 1.5, "quadro": 45, "sentido": "descendo"}])
    linhas = list(csv.reader(io.StringIO(dados.decode("utf-8-sig"))))
    assert linhas[0] == ["id", "categoria", "tempo_s", "quadro", "sentido"]
    assert linhas[1] == ["4", "ônibus", "1.5", "45", "descendo"]
    dados = vision.csv_imagem([{"categoria": "carro", "confianca": 0.9, "x1": 1, "y1": 2, "x2": 3, "y2": 4}])
    linhas = list(csv.reader(io.StringIO(dados.decode("utf-8-sig"))))
    assert linhas[0] == ["categoria", "confianca", "x1", "y1", "x2", "y2"]
    print("OK  CSV")


def testar_arquivo_invalido():
    try:
        vision.decodificar_imagem(b"isto nao e uma imagem")
    except vision.ErroDeArquivo:
        print("OK  imagem inválida gera erro claro")
        return
    raise AssertionError("era esperado ErroDeArquivo")


if __name__ == "__main__":
    testar_sentidos()
    testar_contagem_unica_por_id()
    testar_categorias_e_sentidos()
    testar_id_que_aparece_depois_da_linha()
    testar_novo_contador_comeca_zerado()
    testar_mapa_de_classes()
    testar_area_restrita()
    testar_serie_temporal()
    testar_alertas_com_modelo_falso()
    testar_webcam_inexistente()
    testar_csv()
    testar_arquivo_invalido()
    print("\nTodos os testes passaram.")
