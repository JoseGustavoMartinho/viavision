"""
evaluate.py - Avalia um modelo YOLO no conjunto de TESTE anotado do grupo.

Os números (Precision, Recall, mAP e F1) vêm do cálculo da própria biblioteca Ultralytics,
sobre o seu dataset. Nada aqui é inventado: se você não rodar com um dataset real, não há resultado.

Exemplo (PowerShell):
    python evaluate.py --model runs\\viavision_treino\\weights\\best.pt --data C:\\caminho\\meu_dataset.yaml

Para comparar com o modelo pré-treinado, rode também com --model yolo11n.pt no MESMO dataset de teste
(atenção: as classes do dataset precisam corresponder às classes do modelo).
"""

import argparse
import csv
import sys
from pathlib import Path


def f1(precisao, revocacao):
    """F1 = média harmônica entre precisão e revocação (0 se ambas forem 0)."""
    soma = precisao + revocacao
    return 0.0 if soma == 0 else 2 * precisao * revocacao / soma


def ler_argumentos():
    analisador = argparse.ArgumentParser(description="Avalia um modelo YOLO em um conjunto anotado.")
    analisador.add_argument("--model", required=True, help="Caminho dos pesos (.pt) ou nome (ex.: yolo11n.pt).")
    analisador.add_argument("--data", required=True, help="Caminho do .yaml do dataset (precisa ter a chave 'test').")
    analisador.add_argument("--split", default="test", choices=["test", "val"],
                            help="Conjunto avaliado (padrão: test).")
    analisador.add_argument("--imgsz", type=int, default=640, help="Tamanho da imagem (padrão: 640).")
    analisador.add_argument("--device", default="cpu", help="'cpu' (padrão) ou '0' para GPU.")
    analisador.add_argument("--saida", default="resultados_avaliacao.csv", help="CSV com o resumo das métricas.")
    return analisador.parse_args()


def main():
    args = ler_argumentos()

    if not Path(args.data).is_file():
        print(f"ERRO: dataset não encontrado: {args.data}")
        sys.exit(1)
    if args.model.lower().endswith(".pt") and ("/" in args.model or "\\" in args.model):
        if not Path(args.model).is_file():
            print(f"ERRO: modelo não encontrado: {args.model}")
            sys.exit(1)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERRO: 'ultralytics' não está instalado. Rode: python -m pip install -r requirements.txt")
        sys.exit(1)

    try:
        modelo = YOLO(args.model)
        metricas = modelo.val(data=args.data, split=args.split, imgsz=args.imgsz, device=args.device, plots=True)
    except Exception as erro:
        print(f"ERRO durante a avaliação: {erro}")
        print(f"Confira se o .yaml tem a chave '{args.split}' apontando para imagens com rótulos.")
        sys.exit(1)

    caixa = metricas.box
    precisao, revocacao = float(caixa.mp), float(caixa.mr)
    resumo = [
        ("Precision (média das classes)", precisao),
        ("Recall (média das classes)", revocacao),
        ("F1 (calculado de P e R médios)", f1(precisao, revocacao)),
        ("mAP50", float(caixa.map50)),
        ("mAP50-95", float(caixa.map)),
    ]

    print(f"\n=== Resultados no conjunto '{args.split}' ===")
    for nome, valor in resumo:
        print(f"{nome}: {valor:.4f}")

    # Métricas por classe (apenas classes que aparecem no conjunto avaliado).
    por_classe = []
    nomes = metricas.names
    for posicao, id_classe in enumerate(caixa.ap_class_index):
        p, r = float(caixa.p[posicao]), float(caixa.r[posicao])
        por_classe.append((nomes[int(id_classe)], p, r, f1(p, r), float(caixa.ap50[posicao]), float(caixa.ap[posicao])))

    if por_classe:
        print("\nPor classe: classe | P | R | F1 | mAP50 | mAP50-95")
        for linha in por_classe:
            print(f"{linha[0]} | " + " | ".join(f"{valor:.4f}" for valor in linha[1:]))

    with open(args.saida, "w", newline="", encoding="utf-8-sig") as arquivo:
        escritor = csv.writer(arquivo)
        escritor.writerow(["metrica", "valor"])
        escritor.writerows([(nome, f"{valor:.4f}") for nome, valor in resumo])
        escritor.writerow([])
        escritor.writerow(["classe", "precision", "recall", "f1", "map50", "map50_95"])
        for linha in por_classe:
            escritor.writerow([linha[0]] + [f"{valor:.4f}" for valor in linha[1:]])

    print(f"\nResumo salvo em: {args.saida}")
    print(f"Gráficos (curvas e matriz de confusão) salvos em: {getattr(metricas, 'save_dir', 'pasta runs')}")


if __name__ == "__main__":
    main()
