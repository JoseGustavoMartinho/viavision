"""
train.py - Treinamento (ajuste fino) de um modelo YOLO com um dataset anotado do grupo.

IMPORTANTE: este script NÃO acompanha dataset e NÃO traz resultados de treinamento.
Você precisa fornecer um dataset no formato YOLO (veja dataset_exemplo.yaml e o README).

Exemplo (PowerShell, na pasta do projeto, com o ambiente virtual ativo):
    python train.py --data C:\\caminho\\meu_dataset.yaml --epochs 30

Ao final, os pesos ficam em runs\\viavision_treino\\weights\\best.pt (o caminho exato é impresso).
Use esse best.pt na barra lateral do app, em "Modelo customizado".
"""

import argparse
import sys
from pathlib import Path


def ler_argumentos():
    analisador = argparse.ArgumentParser(description="Treina um modelo YOLO com um dataset anotado.")
    analisador.add_argument("--data", required=True, help="Caminho do arquivo .yaml do dataset (formato YOLO).")
    analisador.add_argument("--model", default="yolo11n.pt",
                            help="Pesos de partida (padrão: yolo11n.pt, baixado na primeira vez).")
    analisador.add_argument("--epochs", type=int, default=30, help="Número de épocas (padrão: 30).")
    analisador.add_argument("--imgsz", type=int, default=640, help="Tamanho da imagem de treino (padrão: 640).")
    analisador.add_argument("--batch", type=int, default=8, help="Tamanho do lote (padrão: 8; reduza se faltar memória).")
    analisador.add_argument("--device", default="cpu", help="Dispositivo: 'cpu' (padrão) ou '0' para a primeira GPU.")
    analisador.add_argument("--workers", type=int, default=0,
                            help="Processos de leitura de dados (padrão: 0, o mais estável no Windows).")
    analisador.add_argument("--project", default="runs", help="Pasta onde os resultados serão salvos.")
    analisador.add_argument("--name", default="viavision_treino", help="Nome da execução.")
    return analisador.parse_args()


def main():
    args = ler_argumentos()

    if not Path(args.data).is_file():
        print(f"ERRO: arquivo de dataset não encontrado: {args.data}")
        print("Crie o dataset no formato YOLO e aponte para o .yaml (veja dataset_exemplo.yaml).")
        sys.exit(1)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERRO: 'ultralytics' não está instalado. Rode: python -m pip install -r requirements.txt")
        sys.exit(1)

    try:
        modelo = YOLO(args.model)
    except Exception as erro:
        print(f"ERRO ao carregar '{args.model}': {erro}")
        print("Se for a primeira execução, verifique a conexão com a internet (download dos pesos).")
        sys.exit(1)

    try:
        resultados = modelo.train(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            project=args.project,
            name=args.name,
        )
    except Exception as erro:
        print(f"ERRO durante o treinamento: {erro}")
        print("Verifique os caminhos dentro do .yaml, as pastas images/labels e o formato dos rótulos.")
        sys.exit(1)

    pasta = getattr(resultados, "save_dir", None) or Path(args.project) / args.name
    print("\nTreinamento concluído.")
    print(f"Resultados em: {pasta}")
    print(f"Melhores pesos: {Path(pasta) / 'weights' / 'best.pt'}")
    print("Avalie no conjunto de teste com: python evaluate.py --model <best.pt> --data <dataset.yaml>")


# No Windows, este bloco é obrigatório para o treinamento funcionar.
if __name__ == "__main__":
    main()
