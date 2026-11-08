"""
vision.py - Lógica de visão computacional do ViaVision.

Este arquivo NÃO usa Streamlit. Ele concentra:
  - carregamento do modelo YOLO (Ultralytics);
  - detecção em imagens;
  - rastreamento e contagem em vídeo/webcam (linha horizontal);
  - alertas: invasão de área restrita e falta de EPI;
  - série temporal do fluxo, CSV e utilitários de arquivos temporários.

O app.py apenas chama as funções daqui e mostra os resultados.
"""

from __future__ import annotations

import csv
import io
import math
import os
import struct
import tempfile
import threading
import time
import unicodedata
import wave
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

CATEGORIAS = ["carro", "moto", "ônibus", "caminhão"]   # veículos
PESSOA = "pessoa"
CATEGORIAS_TODAS = CATEGORIAS + [PESSOA]

# Nomes de classe aceitos (inglês, como no COCO, ou português) -> categoria.
_ALIAS_CLASSES = {
    "car": "carro", "carro": "carro",
    "motorcycle": "moto", "motorbike": "moto", "moto": "moto", "motocicleta": "moto",
    "bus": "ônibus", "onibus": "ônibus",
    "truck": "caminhão", "caminhao": "caminhão",
    "person": PESSOA, "pessoa": PESSOA, "pedestre": PESSOA,
}

# Classes de um modelo CUSTOMIZADO que significam "falta de EPI" (violação).
# O modelo pré-treinado (COCO) não tem essas classes: o alerta de EPI só funciona
# com um modelo treinado pelo grupo que as contenha.
_ALIAS_EPI = {
    "no_helmet", "nohelmet", "without_helmet", "sem_capacete",
    "no_hardhat", "nohardhat", "without_hardhat",
    "no_vest", "novest", "without_vest", "sem_colete", "no_safety_vest",
    "no_ppe", "sem_epi",
}

# Cores em BGR (padrão do OpenCV).
_CORES = {
    "carro": (60, 180, 60), "moto": (0, 140, 255),
    "ônibus": (200, 120, 0), "caminhão": (60, 60, 220),
    PESSOA: (200, 200, 0),
}
_COR_LINHA = (0, 255, 255)
_COR_ALERTA = (0, 0, 255)
_COR_EPI = (255, 0, 255)

SENTIDOS = ("descendo", "subindo")  # em relação à imagem: de cima para baixo / de baixo para cima


class ErroDeModelo(Exception):
    """Problema ao carregar ou usar o modelo (mensagem já pronta para o usuário)."""


class ErroDeArquivo(Exception):
    """Imagem, vídeo ou webcam inválidos (mensagem já pronta para o usuário)."""


# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------

def _normalizar(texto) -> str:
    """Minúsculas, sem acentos, com espaços/hífens trocados por '_'."""
    sem_acento = unicodedata.normalize("NFKD", str(texto)).encode("ascii", "ignore").decode("ascii")
    return sem_acento.strip().lower().replace("-", "_").replace(" ", "_")


def mapa_de_classes(nomes: dict) -> dict:
    """{id da classe: nome} do modelo -> {id: categoria} (veículos e pessoa)."""
    mapa = {}
    for id_classe, nome in nomes.items():
        categoria = _ALIAS_CLASSES.get(_normalizar(nome))
        if categoria is not None:
            mapa[int(id_classe)] = categoria
    return mapa


def ids_epi(nomes: dict) -> dict:
    """{id da classe: nome original} das classes que indicam falta de EPI."""
    return {int(i): str(n) for i, n in nomes.items() if _normalizar(n) in _ALIAS_EPI}


def carregar_modelo(origem: str, customizado: bool = False):
    """
    Carrega um modelo YOLO e devolve (modelo, mapa_de_classes, classes_epi).

    - origem: nome dos pesos pré-treinados (ex.: "yolo11n.pt") ou caminho de um .pt local.
    - customizado: True quando 'origem' é um arquivo local do usuário.

    Na primeira execução, os pesos pré-treinados são baixados da internet pela
    própria Ultralytics e ficam salvos na pasta onde o app é executado.
    """
    try:
        from ultralytics import YOLO
    except ImportError as erro:
        raise ErroDeModelo(
            "A biblioteca 'ultralytics' não está instalada neste ambiente. "
            "Rode: python -m pip install -r requirements.txt"
        ) from erro

    if customizado:
        caminho = Path(origem.strip().strip('"')).expanduser()
        if not caminho.is_file():
            raise ErroDeModelo(f"Modelo customizado não encontrado: {caminho}")
        if caminho.suffix.lower() != ".pt":
            raise ErroDeModelo("O modelo customizado deve ser um arquivo .pt (ex.: best.pt).")
        try:
            modelo = YOLO(str(caminho))
        except Exception as erro:  # arquivo corrompido, versão incompatível etc.
            raise ErroDeModelo(
                f"Não foi possível abrir o modelo customizado ({type(erro).__name__}: {erro})."
            ) from erro
    else:
        try:
            modelo = YOLO(origem)
        except Exception as erro:
            if not Path(origem).exists():
                raise ErroDeModelo(
                    f"Não foi possível carregar os pesos '{origem}'. Na primeira execução eles "
                    "precisam ser baixados da internet: verifique sua conexão e tente de novo. "
                    f"({type(erro).__name__}: {erro})"
                ) from erro
            raise ErroDeModelo(
                f"Falha ao carregar o modelo '{origem}' ({type(erro).__name__}: {erro}). "
                "Se o arquivo foi baixado pela metade, apague-o e tente novamente."
            ) from erro

    if getattr(modelo, "task", "detect") != "detect":
        raise ErroDeModelo(
            "O modelo carregado não é de detecção de objetos. Use um modelo YOLO do tipo 'detect'."
        )

    nomes = getattr(modelo, "names", None) or {}
    mapa = mapa_de_classes(nomes)
    epi = ids_epi(nomes)
    if not mapa and not epi:
        raise ErroDeModelo(
            "O modelo não possui classes reconhecidas (car, motorcycle, bus, truck, person ou "
            f"classes de falta de EPI como no_helmet). Classes encontradas: {list(nomes.values())}"
        )
    return modelo, mapa, epi


# ---------------------------------------------------------------------------
# Arquivos temporários (compatível com Windows: nada fica aberto ao usar o caminho)
# ---------------------------------------------------------------------------

def criar_caminho_temporario(sufixo: str) -> str:
    descritor, caminho = tempfile.mkstemp(suffix=sufixo, prefix="viavision_")
    os.close(descritor)
    return caminho


def salvar_temporario(dados: bytes, sufixo: str) -> str:
    caminho = criar_caminho_temporario(sufixo)
    with open(caminho, "wb") as arquivo:
        arquivo.write(dados)
    return caminho


def remover_arquivo(caminho) -> None:
    """Apaga um arquivo sem lançar erro se ele não existir ou estiver em uso."""
    if not caminho:
        return
    try:
        os.remove(caminho)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Desenho
# ---------------------------------------------------------------------------

def _ascii(texto: str) -> str:
    """O OpenCV não desenha acentos; removemos só no texto desenhado na imagem."""
    return unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode("ascii")


def _desenhar_caixa(quadro, x1, y1, x2, y2, texto, cor) -> None:
    p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
    cv2.rectangle(quadro, p1, p2, cor, 2)
    texto = _ascii(texto)
    (largura, altura), base = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    topo = max(p1[1] - altura - base - 4, 0)
    cv2.rectangle(quadro, (p1[0], topo), (p1[0] + largura + 4, topo + altura + base + 4), cor, -1)
    cv2.putText(quadro, texto, (p1[0] + 2, topo + altura + 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def desenhar_linha(quadro, linha_rel: float) -> None:
    """Desenha a linha horizontal de contagem (linha_rel entre 0 e 1 da altura)."""
    altura, largura = quadro.shape[:2]
    y = int(round(altura * linha_rel))
    cv2.line(quadro, (0, y), (largura, y), _COR_LINHA, 2)


def area_em_pixels(area_rel, largura: int, altura: int):
    """(x1, y1, x2, y2) relativos (0 a 1) -> pixels, ordenados e dentro da imagem."""
    x1, y1, x2, y2 = area_rel
    xa, xb = sorted((int(round(x1 * largura)), int(round(x2 * largura))))
    ya, yb = sorted((int(round(y1 * altura)), int(round(y2 * altura))))
    return max(xa, 0), max(ya, 0), min(xb, largura), min(yb, altura)


def ponto_na_area(x: float, y: float, area_px) -> bool:
    """True se o ponto (x, y) está dentro do retângulo (x1, y1, x2, y2)."""
    x1, y1, x2, y2 = area_px
    return x1 <= x <= x2 and y1 <= y <= y2


def desenhar_area(quadro, area_rel, alerta: bool = False) -> None:
    """Desenha a área restrita (retângulo semitransparente)."""
    altura, largura = quadro.shape[:2]
    x1, y1, x2, y2 = area_em_pixels(area_rel, largura, altura)
    if x2 <= x1 or y2 <= y1:
        return
    cor = _COR_ALERTA if alerta else (0, 165, 255)
    regiao = quadro[y1:y2, x1:x2]
    camada = np.full_like(regiao, cor)
    cv2.addWeighted(camada, 0.25, regiao, 0.75, 0, regiao)
    cv2.rectangle(quadro, (x1, y1), (x2, y2), cor, 2)
    cv2.putText(quadro, "AREA RESTRITA", (x1 + 4, max(y1 + 16, 16)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor, 2, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Imagem
# ---------------------------------------------------------------------------

def decodificar_imagem(dados: bytes):
    """Converte bytes de JPG/PNG em imagem BGR. Lança ErroDeArquivo se for inválida."""
    matriz = np.frombuffer(dados, dtype=np.uint8)
    imagem = cv2.imdecode(matriz, cv2.IMREAD_COLOR) if matriz.size else None
    if imagem is None:
        raise ErroDeArquivo("Não foi possível abrir a imagem. O arquivo pode estar corrompido.")
    return imagem


def codificar_png(imagem_bgr) -> bytes:
    ok, buffer = cv2.imencode(".png", imagem_bgr)
    if not ok:
        raise ErroDeArquivo("Não foi possível gerar o PNG da imagem processada.")
    return buffer.tobytes()


@dataclass
class ResultadoImagem:
    imagem: object
    deteccoes: list           # dict(categoria, confianca, x1, y1, x2, y2)
    contagem: dict
    ms_inferencia: float
    violacoes_epi: int = 0
    conf_media: float = None

    @property
    def total(self) -> int:
        return sum(self.contagem.values())


def analisar_imagem(modelo, mapa: dict, epi: dict, imagem_bgr, confianca: float) -> ResultadoImagem:
    """Detecta veículos/pessoas (e falta de EPI, se o modelo tiver essas classes) em uma imagem."""
    inicio = time.perf_counter()
    try:
        resultado = modelo.predict(
            imagem_bgr, conf=confianca, classes=list(mapa.keys()) + list(epi.keys()),
            device="cpu", verbose=False
        )[0]
    except Exception as erro:
        raise ErroDeModelo(f"Falha durante a detecção ({type(erro).__name__}: {erro}).") from erro
    ms = (time.perf_counter() - inicio) * 1000.0

    anotada = imagem_bgr.copy()
    deteccoes = []
    contagem = {c: 0 for c in CATEGORIAS_TODAS}
    violacoes = 0
    confs = []

    caixas = resultado.boxes
    if caixas is not None and len(caixas) > 0:
        xyxy = caixas.xyxy.cpu().numpy()
        classes = caixas.cls.int().cpu().tolist()
        valores = caixas.conf.cpu().tolist()
        for (x1, y1, x2, y2), id_classe, conf in zip(xyxy, classes, valores):
            if id_classe in epi:
                violacoes += 1
                rotulo = f"SEM EPI: {epi[id_classe]}"
                cor = _COR_EPI
            else:
                rotulo = mapa.get(id_classe)
                if rotulo is None:
                    continue
                contagem[rotulo] += 1
                cor = _CORES[rotulo]
            confs.append(float(conf))
            deteccoes.append({
                "categoria": rotulo, "confianca": round(float(conf), 4),
                "x1": round(float(x1), 1), "y1": round(float(y1), 1),
                "x2": round(float(x2), 1), "y2": round(float(y2), 1),
            })
            _desenhar_caixa(anotada, x1, y1, x2, y2, f"{rotulo} {conf:.2f}", cor)

    conf_media = sum(confs) / len(confs) if confs else None
    return ResultadoImagem(anotada, deteccoes, contagem, ms, violacoes, conf_media)


# ---------------------------------------------------------------------------
# Cruzamento de linha e contagem
# ---------------------------------------------------------------------------

def _lado_da_linha(y: float, y_linha: float) -> int:
    """-1 se o ponto está acima da linha (y menor), +1 se está sobre ela ou abaixo."""
    return -1 if y < y_linha else 1


def sentido_de_passagem(y_anterior: float, y_atual: float, y_linha: float):
    """Devolve "descendo", "subindo" ou None (não cruzou)."""
    antes = _lado_da_linha(y_anterior, y_linha)
    depois = _lado_da_linha(y_atual, y_linha)
    if antes == depois:
        return None
    return "descendo" if antes < depois else "subindo"


class ContadorDeLinha:
    """
    Conta cada ID rastreado no máximo uma vez, quando o centro da caixa cruza a linha.
    Um objeto novo é criado a cada análise, então o estado sempre começa do zero.
    """

    def __init__(self, y_linha: float):
        self.y_linha = y_linha
        self._ultimo_y = {}
        self._ja_contados = set()
        self.contagem = {c: 0 for c in CATEGORIAS_TODAS}
        self.por_sentido = {c: {s: 0 for s in SENTIDOS} for c in CATEGORIAS_TODAS}
        self.eventos = []

    @property
    def total(self) -> int:
        return sum(self.contagem.values())

    def atualizar(self, id_rastreio: int, categoria: str, y_centro: float, tempo_s: float, quadro: int):
        """Registra a posição do ID. Devolve o evento (dict) se houve uma nova passagem."""
        anterior = self._ultimo_y.get(id_rastreio)
        self._ultimo_y[id_rastreio] = y_centro
        if anterior is None or id_rastreio in self._ja_contados:
            return None
        sentido = sentido_de_passagem(anterior, y_centro, self.y_linha)
        if sentido is None:
            return None
        self._ja_contados.add(id_rastreio)
        self.contagem[categoria] += 1
        self.por_sentido[categoria][sentido] += 1
        evento = {
            "id": int(id_rastreio), "categoria": categoria,
            "tempo_s": round(float(tempo_s), 2), "quadro": int(quadro), "sentido": sentido,
        }
        self.eventos.append(evento)
        return evento


# ---------------------------------------------------------------------------
# Série temporal (gráfico de tendência do fluxo)
# ---------------------------------------------------------------------------

def intervalo_sugerido(duracao_s: float) -> int:
    """Tamanho (em segundos) de cada barra/ponto do gráfico de tendência."""
    if duracao_s < 20:
        return 1
    if duracao_s < 60:
        return 2
    if duracao_s < 300:
        return 5
    return 10


def serie_temporal(eventos: list, duracao_s: float, intervalo_s: int) -> list:
    """Passagens por intervalo de tempo: [{'início (s)': 0, 'carro': 2, ..., 'total': 3}, ...]."""
    intervalo_s = max(1, int(intervalo_s))
    n = int(max(duracao_s, 0) // intervalo_s) + 1
    linhas = []
    for i in range(n):
        linha = {"início (s)": i * intervalo_s}
        linha.update({c: 0 for c in CATEGORIAS_TODAS})
        linhas.append(linha)
    for evento in eventos:
        i = min(int(evento["tempo_s"] // intervalo_s), n - 1)
        linhas[i][evento["categoria"]] += 1
    for linha in linhas:
        linha["total"] = sum(linha[c] for c in CATEGORIAS_TODAS)
    return linhas


# ---------------------------------------------------------------------------
# Analisador de quadros (usado para vídeo e webcam)
# ---------------------------------------------------------------------------

class AnalisadorDeQuadros:
    """
    Recebe quadros um a um e faz: detecção + rastreamento, contagem na linha,
    alerta de invasão de área restrita e alerta de falta de EPI.

    Cada análise cria um objeto novo, e o estado do rastreador do modelo é
    reiniciado aqui, então nada "sobra" de análises anteriores.
    """

    def __init__(self, modelo, mapa: dict, epi: dict, confianca: float, linha_rel: float = 0.5,
                 area=None, alvos_area=(PESSOA,)):
        self.modelo = modelo
        self.mapa = dict(mapa)
        self.epi = dict(epi)
        self.confianca = confianca
        self.linha_rel = linha_rel
        self.area = area                  # (x1, y1, x2, y2) relativos, ou None
        self.alvos_area = set(alvos_area)
        self.contador = None              # criado no 1º quadro (precisa da altura)
        self.alertas = []
        self.tempos_ms = []
        self.em_area_agora = 0
        self.epi_agora = 0
        self._alertados = set()
        self._soma_conf = 0.0
        self._n_conf = 0
        self._classes = list(self.mapa.keys()) + list(self.epi.keys())
        modelo.predictor = None           # reinicia o rastreamento

    @property
    def ms_medio(self) -> float:
        return sum(self.tempos_ms) / len(self.tempos_ms) if self.tempos_ms else 0.0

    @property
    def conf_media(self):
        return self._soma_conf / self._n_conf if self._n_conf else None

    @property
    def total(self) -> int:
        return self.contador.total if self.contador else 0

    @property
    def contagem(self) -> dict:
        return self.contador.contagem if self.contador else {c: 0 for c in CATEGORIAS_TODAS}

    def encerrar(self) -> None:
        self.modelo.predictor = None

    def _alerta(self, tipo, alvo, id_rastreio, tempo_s, quadro, janela_s=3):
        """Registra um alerta, sem repetir: 1 por ID (ou 1 a cada 'janela_s' s se não houver ID)."""
        chave = (tipo, id_rastreio if id_rastreio is not None else (alvo, int(tempo_s // janela_s)))
        if chave in self._alertados:
            return None
        self._alertados.add(chave)
        alerta = {
            "tempo_s": round(float(tempo_s), 2), "quadro": int(quadro), "tipo": tipo,
            "alvo": alvo, "id": "" if id_rastreio is None else int(id_rastreio),
        }
        self.alertas.append(alerta)
        return alerta

    def processar(self, quadro, indice: int, tempo_s: float) -> list:
        """Analisa e desenha em cima do quadro. Devolve a lista de NOVOS alertas."""
        altura, largura = quadro.shape[:2]
        if self.contador is None:
            self.contador = ContadorDeLinha(int(round(altura * self.linha_rel)))

        inicio = time.perf_counter()
        try:
            resultado = self.modelo.track(
                quadro, persist=True, tracker="bytetrack.yaml", conf=self.confianca,
                classes=self._classes, device="cpu", verbose=False
            )[0]
        except Exception as erro:
            texto = str(erro)
            if "lap" in texto.lower():
                raise ErroDeModelo(
                    "Faltou a dependência de rastreamento 'lap'. Rode: "
                    "python -m pip install -r requirements.txt "
                    "(ou, se falhar no seu Python: python -m pip install lapx)."
                ) from erro
            raise ErroDeModelo(f"Falha durante o rastreamento ({type(erro).__name__}: {texto}).") from erro
        self.tempos_ms.append((time.perf_counter() - inicio) * 1000.0)

        area_px = area_em_pixels(self.area, largura, altura) if self.area else None
        if area_px:
            desenhar_area(quadro, self.area)
        desenhar_linha(quadro, self.linha_rel)

        novos = []
        em_area = 0
        epi_agora = 0
        caixas = resultado.boxes
        if caixas is not None and len(caixas) > 0:
            xyxy = caixas.xyxy.cpu().numpy()
            classes = caixas.cls.int().cpu().tolist()
            confs = caixas.conf.cpu().tolist()
            ids = caixas.id.int().cpu().tolist() if caixas.id is not None else [None] * len(classes)
            for (x1, y1, x2, y2), id_classe, conf, id_rastreio in zip(xyxy, classes, confs, ids):
                self._soma_conf += float(conf)
                self._n_conf += 1

                if id_classe in self.epi:  # violação de EPI (só com modelo customizado)
                    epi_agora += 1
                    nome = self.epi[id_classe]
                    _desenhar_caixa(quadro, x1, y1, x2, y2, f"SEM EPI: {nome}", _COR_EPI)
                    alerta = self._alerta("falta de EPI", nome, id_rastreio, tempo_s, indice)
                    if alerta:
                        novos.append(alerta)
                    continue

                categoria = self.mapa.get(id_classe)
                if categoria is None:
                    continue
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                if id_rastreio is not None:
                    self.contador.atualizar(id_rastreio, categoria, cy, tempo_s, indice)

                dentro = (area_px is not None and categoria in self.alvos_area
                          and ponto_na_area(cx, y2, area_px))  # usa o "pé" (base da caixa)
                if dentro:
                    em_area += 1
                    alerta = self._alerta("invasão de área", categoria, id_rastreio, tempo_s, indice)
                    if alerta:
                        novos.append(alerta)
                cor = _COR_ALERTA if dentro else _CORES[categoria]
                texto = f"#{id_rastreio} {categoria}" if id_rastreio is not None else categoria
                _desenhar_caixa(quadro, x1, y1, x2, y2, texto, cor)
                cv2.circle(quadro, (int(cx), int(cy)), 4, cor, -1)

        self.em_area_agora = em_area
        self.epi_agora = epi_agora

        cv2.putText(quadro, f"Passagens: {self.contador.total}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(quadro, f"Passagens: {self.contador.total}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
        if em_area or epi_agora:
            cv2.rectangle(quadro, (0, 0), (largura - 1, altura - 1), _COR_ALERTA, 8)
            cv2.putText(quadro, "ALERTA", (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 1.0, _COR_ALERTA, 3, cv2.LINE_AA)
        return novos


# ---------------------------------------------------------------------------
# Vídeo
# ---------------------------------------------------------------------------

@dataclass
class ResultadoVideo:
    contagem: dict
    por_sentido: dict
    eventos: list
    alertas: list
    ms_inferencia_medio: float
    conf_media: float
    quadros_analisados: int
    duracao_s: float
    video_gerado: bool = False
    total: int = field(default=0)


def _abrir_escritor(caminho: str, fps: float, largura: int, altura: int):
    """Abre o gravador de vídeo (codec mp4v). Devolve None se não for possível."""
    escritor = cv2.VideoWriter(caminho, cv2.VideoWriter_fourcc(*"mp4v"), fps, (largura, altura))
    if not escritor.isOpened():
        escritor.release()
        return None
    return escritor


def processar_video(modelo, mapa: dict, epi: dict, caminho_video: str, confianca: float, linha_rel: float,
                    passo: int = 1, area=None, alvos_area=(PESSOA,), caminho_saida=None,
                    ao_progresso=None) -> ResultadoVideo:
    """
    Analisa um arquivo de vídeo quadro a quadro.

    - passo: analisa 1 a cada 'passo' quadros (1 = todos).
    - caminho_saida: se informado, grava o vídeo anotado nesse caminho.
    - ao_progresso(indice, total_quadros, quadro_anotado, analisador, novos_alertas): chamado a
      cada quadro analisado; quem chama decide com que frequência atualiza a interface.
    """
    passo = max(1, int(passo))
    captura = cv2.VideoCapture(str(caminho_video))
    escritor = None
    analisador = None
    analisados = 0
    indice = 0
    fps = 30.0

    try:
        if not captura.isOpened():
            raise ErroDeArquivo("Não foi possível abrir o vídeo. Use um MP4 válido (codec H.264 é o mais comum).")

        fps_lido = captura.get(cv2.CAP_PROP_FPS)
        if fps_lido and fps_lido == fps_lido and 0 < fps_lido <= 240:  # descarta 0, NaN e absurdos
            fps = float(fps_lido)
        total_quadros = int(captura.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        analisador = AnalisadorDeQuadros(modelo, mapa, epi, confianca, linha_rel, area, alvos_area)

        while True:
            if indice % passo == 0:
                ok, quadro = captura.read()
            else:
                ok, quadro = captura.grab(), None
            if not ok:
                if indice == 0:
                    raise ErroDeArquivo("O vídeo abriu, mas não foi possível ler nenhum quadro.")
                break

            if quadro is not None:
                if escritor is None and caminho_saida and analisados == 0:
                    altura, largura = quadro.shape[:2]
                    escritor = _abrir_escritor(caminho_saida, fps / passo, largura, altura)
                novos = analisador.processar(quadro, indice, indice / fps)
                if escritor is not None:
                    escritor.write(quadro)
                analisados += 1
                if ao_progresso is not None:
                    ao_progresso(indice, total_quadros, quadro, analisador, novos)
            indice += 1
    finally:
        captura.release()
        if escritor is not None:
            escritor.release()
        if analisador is not None:
            analisador.encerrar()

    if analisador is None or analisador.contador is None:
        raise ErroDeArquivo("Nenhum quadro foi analisado.")

    return ResultadoVideo(
        contagem=analisador.contador.contagem,
        por_sentido=analisador.contador.por_sentido,
        eventos=analisador.contador.eventos,
        alertas=analisador.alertas,
        ms_inferencia_medio=analisador.ms_medio,
        conf_media=analisador.conf_media,
        quadros_analisados=analisados,
        duracao_s=indice / fps,
        video_gerado=escritor is not None,
        total=analisador.contador.total,
    )


def ler_primeiro_quadro(caminho_video: str):
    """Lê o primeiro quadro de um vídeo (para pré-visualizar linha e área). None se falhar."""
    captura = cv2.VideoCapture(str(caminho_video))
    try:
        if not captura.isOpened():
            return None
        ok, quadro = captura.read()
        return quadro if ok else None
    finally:
        captura.release()


# ---------------------------------------------------------------------------
# Webcam e alerta sonoro
# ---------------------------------------------------------------------------

def abrir_webcam(indice: int = 0):
    """Abre a webcam local. Lança ErroDeArquivo se não for possível."""
    captura = None
    if os.name == "nt":  # no Windows, o backend DirectShow costuma abrir mais rápido
        captura = cv2.VideoCapture(int(indice), cv2.CAP_DSHOW)
    if captura is None or not captura.isOpened():
        if captura is not None:
            captura.release()
        captura = cv2.VideoCapture(int(indice))
    if not captura.isOpened():
        captura.release()
        raise ErroDeArquivo(
            f"Não foi possível abrir a webcam de índice {indice}. Feche outros programas que usam a "
            "câmera, confira as permissões do Windows e tente outro índice (0, 1, 2...)."
        )
    return captura


def tocar_beep() -> bool:
    """Beep do próprio computador (Windows). Devolve False se não for possível."""
    try:
        import winsound
    except ImportError:
        return False

    def _beep():
        try:
            winsound.Beep(1000, 350)
        except Exception:
            pass

    threading.Thread(target=_beep, daemon=True).start()
    return True


def wav_do_alerta() -> bytes:
    """Gera um 'bip' em WAV (biblioteca padrão), para tocar no navegador quando não há winsound."""
    taxa = 22050
    quadros = b"".join(
        struct.pack("<h", int(12000 * math.sin(2 * math.pi * 880 * i / taxa)))
        for i in range(int(taxa * 0.4))
    )
    saida = io.BytesIO()
    with wave.open(saida, "wb") as arquivo:
        arquivo.setnchannels(1)
        arquivo.setsampwidth(2)
        arquivo.setframerate(taxa)
        arquivo.writeframes(quadros)
    return saida.getvalue()


# ---------------------------------------------------------------------------
# CSV (biblioteca padrão). utf-8-sig para o Excel abrir acentos corretamente.
# ---------------------------------------------------------------------------

def _gerar_csv(cabecalho: list, linhas: list) -> bytes:
    saida = io.StringIO(newline="")
    escritor = csv.writer(saida)
    escritor.writerow(cabecalho)
    escritor.writerows(linhas)
    return saida.getvalue().encode("utf-8-sig")


def csv_imagem(deteccoes: list) -> bytes:
    cabecalho = ["categoria", "confianca", "x1", "y1", "x2", "y2"]
    linhas = [[d["categoria"], d["confianca"], d["x1"], d["y1"], d["x2"], d["y2"]] for d in deteccoes]
    return _gerar_csv(cabecalho, linhas)


def csv_video(eventos: list) -> bytes:
    cabecalho = ["id", "categoria", "tempo_s", "quadro", "sentido"]
    linhas = [[e["id"], e["categoria"], e["tempo_s"], e["quadro"], e["sentido"]] for e in eventos]
    return _gerar_csv(cabecalho, linhas)


def csv_alertas(alertas: list) -> bytes:
    cabecalho = ["tempo_s", "quadro", "tipo", "alvo", "id"]
    linhas = [[a["tempo_s"], a["quadro"], a["tipo"], a["alvo"], a["id"]] for a in alertas]
    return _gerar_csv(cabecalho, linhas)
