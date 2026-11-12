"""
extrair_quadros.py - Prepara um conjunto de TESTE para avaliar o modelo (evaluate.py).

O que ele faz (nada aqui é inventado: os rótulos você cria à mão, em uma ferramenta de anotação):

  1) "extrair": tira quadros de um vídeo e separa em train/val/test POR TRECHO DO VÍDEO
     (70% do início = train, 15% = val, 15% do fim = test), para não misturar quadros quase
     iguais entre os conjuntos. Cria também o dataset.yaml.

  2) "remapear": se você anotou com as classes 0=carro, 1=moto, 2=ônibus, 3=caminhão,
     converte os rótulos para os números do COCO (carro=2, moto=3, ônibus=5, caminhão=7),
     que são os que o modelo pré-treinado (yolo11n.pt) usa. Sem isso, as métricas saem erradas.

Exemplos (PowerShell):
    python extrair_quadros.py extrair --video exemplos\\transito_demo.mp4 --saida dataset_avaliacao
    (anote as imagens; salve os .txt na pasta labels correspondente, formato YOLO)
    python extrair_quadros.py remapear --pasta dataset_avaliacao
    python evaluate.py --model yolo11n.pt --data dataset_avaliacao\\dataset.yaml
"""

import argparse
import sys
from pathlib import Path

import cv2

# Nomes do COCO (80 classes), na ordem em que o yolo11n.pt as usa.
NOMES_COCO = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush",
]

# Conversão das classes anotadas (0 a 3) para os números do COCO.
MAPA_CLASSES = {0: 2, 1: 3, 2: 5, 3: 7}


def extrair(video: Path, saida: Path, intervalo_s: float, maximo: int) -> None:
    if not video.is_file():
        sys.exit(f"ERRO: vídeo não encontrado: {video}")
    captura = cv2.VideoCapture(str(video))
    if not captura.isOpened():
        sys.exit("ERRO: não foi possível abrir o vídeo.")
    fps = captura.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(captura.get(cv2.CAP_PROP_FRAME_COUNT))
    passo = max(1, int(round(fps * intervalo_s)))

    indices = list(range(0, total, passo))
    if maximo and len(indices) > maximo:
        # Reduz a quantidade mantendo os quadros espalhados pelo vídeo todo.
        salto = len(indices) / maximo
        indices = [indices[int(i * salto)] for i in range(maximo)]
    if len(indices) < 10:
        sys.exit("ERRO: poucos quadros. Diminua --intervalo (ex.: 0.5) ou use um vídeo maior.")

    corte_train = int(len(indices) * 0.70)
    corte_val = int(len(indices) * 0.85)

    for pasta in ("train", "val", "test"):
        (saida / "images" / pasta).mkdir(parents=True, exist_ok=True)
        (saida / "labels" / pasta).mkdir(parents=True, exist_ok=True)

    contagem = {"train": 0, "val": 0, "test": 0}
    nome_base = video.stem
    for posicao, indice in enumerate(indices):
        captura.set(cv2.CAP_PROP_POS_FRAMES, indice)
        ok, quadro = captura.read()
        if not ok:
            continue
        conjunto = "train" if posicao < corte_train else ("val" if posicao < corte_val else "test")
        cv2.imwrite(str(saida / "images" / conjunto / f"{nome_base}_{indice:06d}.jpg"), quadro)
        contagem[conjunto] += 1
    captura.release()

    linhas = [
        "# Dataset de avaliação gerado por extrair_quadros.py (formato YOLO).",
        f"path: {saida.resolve().as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
    ]
    linhas += [f"  {i}: {nome}" for i, nome in enumerate(NOMES_COCO)]
    (saida / "dataset.yaml").write_text("\n".join(linhas) + "\n", encoding="utf-8")

    print(f"Quadros salvos em {saida}: train={contagem['train']}, val={contagem['val']}, test={contagem['test']}")
    print("Próximo passo: anote os quadros (principalmente os de test) e salve os .txt em labels/<conjunto>/.")
    print("Se anotou com 0=carro, 1=moto, 2=ônibus, 3=caminhão, rode depois: python extrair_quadros.py remapear --pasta", saida)


def remapear(pasta: Path) -> None:
    arquivos = list((pasta / "labels").rglob("*.txt"))
    if not arquivos:
        sys.exit(f"ERRO: nenhum .txt encontrado em {pasta / 'labels'}")
    for arquivo in arquivos:
        novas = []
        for linha in arquivo.read_text(encoding="utf-8").splitlines():
            partes = linha.split()
            if not partes:
                continue
            classe = int(partes[0])
            if classe not in MAPA_CLASSES:
                sys.exit(f"ERRO: classe {classe} em {arquivo.name}. Use só 0, 1, 2 e 3 antes de remapear "
                         "(remapear só pode ser rodado UMA vez).")
            novas.append(" ".join([str(MAPA_CLASSES[classe])] + partes[1:]))
        arquivo.write_text("\n".join(novas) + ("\n" if novas else ""), encoding="utf-8")
    print(f"{len(arquivos)} arquivos convertidos (0->2 carro, 1->3 moto, 2->5 ônibus, 3->7 caminhão).")


def main() -> None:
    analisador = argparse.ArgumentParser(description="Prepara o conjunto de avaliação do ViaVision.")
    sub = analisador.add_subparsers(dest="comando", required=True)

    a = sub.add_parser("extrair", help="Extrai quadros do vídeo e cria as pastas e o dataset.yaml.")
    a.add_argument("--video", required=True, type=Path)
    a.add_argument("--saida", default=Path("dataset_avaliacao"), type=Path)
    a.add_argument("--intervalo", type=float, default=1.0, help="Segundos entre quadros (padrão: 1).")
    a.add_argument("--maximo", type=int, default=120, help="Máximo de quadros (padrão: 120).")

    b = sub.add_parser("remapear", help="Converte rótulos 0-3 para os números do COCO.")
    b.add_argument("--pasta", required=True, type=Path)

    args = analisador.parse_args()
    if args.comando == "extrair":
        extrair(args.video, args.saida, args.intervalo, args.maximo)
    else:
        remapear(args.pasta)


if __name__ == "__main__":
    main()
