# ViaVision — Guia do projeto

Resumo de como o projeto foi feito, o que cada parte faz, as ferramentas, o uso na rua, benefícios e riscos, e os pedidos (prompts) organizados. Para instalar e executar, veja o `README.md`.

## Uso de IA neste projeto

O grupo escreveu os pedidos (prompts) e acompanhou os ajustes. O aplicativo (código, README e maquete) foi criado pelo Claude (Anthropic) a partir desses pedidos. As ferramentas principais (Python, Streamlit, YOLO, ByteTrack e OpenCV) são as indicadas pelo professor no enunciado da Opção C.

## Como foi feito, em ordem

1. O grupo escreveu um pedido detalhado com o contexto da disciplina, as restrições, as funcionalidades e o que não usar.
2. O Claude consultou a documentação para escolher as versões (Python 3.11 recomendado) e gerou o projeto: `app.py`, `vision.py`, `train.py`, `evaluate.py`, `requirements.txt`, README e `.gitignore`.
3. O Claude testou a lógica de contagem com casos simples e com um modelo simulado, e conferiu a sintaxe dos arquivos.
4. O grupo pediu uma maquete em HTML para ver no celular. Ela foi ajustada várias vezes a pedido do grupo: veículos desenhados, relógio de 24 horas, congestionamento às 6h, 12h e 17h, motos no corredor, dia reiniciando sozinho, tempo mais rápido e veículos mais realistas, com vários modelos.
5. O enunciado da Opção C foi incluído, e o projeto ganhou webcam, área restrita com alerta, alerta de EPI (com modelo próprio), gráfico de tendência e métricas em tempo real.
6. Correção: o código no GitHub é obrigatório pelo enunciado, e o README passou a trazer o passo a passo.
7. Teste real: o app foi executado no Windows, pela CPU, com o YOLO11n e o vídeo de demonstração (`exemplos/transito_demo.mp4`). A webcam também foi testada. O `.gitignore` foi ajustado para versionar só esse vídeo.
8. Pendente: medir os resultados (contagem manual × automática) e, se o grupo quiser, treinar e avaliar com um dataset próprio.

## O que cada parte faz

| Arquivo | O que faz |
| --- | --- |
| `app.py` | Interface Streamlit: barra lateral, métricas, gráficos, downloads, modos arquivo e webcam, alerta visual e sonoro. |
| `vision.py` | Carrega o modelo, detecta, rastreia, conta o cruzamento da linha, área restrita, EPI, série temporal, CSV, arquivos temporários e webcam. |
| `train.py` | Treina ou customiza o YOLO com um dataset anotado do grupo. |
| `evaluate.py` | Avalia no conjunto de teste: Precision, Recall, F1 e mAP. |
| `teste_logica.py` | Testes da contagem, área restrita, alertas e série temporal, sem usar o YOLO. |
| `requirements.txt` | Lista das dependências. |
| `README.md` | Instalação no Windows, limitações, explicação para a defesa, licenças e GitHub. |
| `dataset_exemplo.yaml` | Exemplo de configuração de dataset (não é um dataset real). |
| `extrair_quadros.py` | Prepara o conjunto de avaliação (quadros do vídeo, divisão por trecho e `dataset.yaml`). |
| `exemplos/transito_demo.mp4` | Vídeo curto de trânsito para testar o app. É o único `.mp4` versionado. |
| `docs/index.html` | Maquete visual (simulação), publicável no GitHub Pages. |

## App real e maquete

| Recurso | App real (Python) | Maquete (HTML) |
| --- | --- | --- |
| Detecção com YOLO | Sim (testado no Windows com o vídeo de demonstração) | Não, são desenhos |
| Contagem pela linha | Sim | Simulada |
| Webcam ao vivo | Sim | Não |
| Área restrita com alerta | Sim | Não |
| Alerta de EPI | Só com modelo customizado | Não |
| Gráfico de tendência | Sim, por intervalo de segundos | Gráfico por hora, simulado |
| Análise por horário do dia e fechamento diário | Não | Sim |
| Congestionamento por horário e motos no corredor | Não | Sim (comportamento programado) |
| Leitura de placa | Não | Não |

## Ferramentas usadas

| Ferramenta | Para que serve | Origem |
| --- | --- | --- |
| Python 3.11 | Linguagem do projeto | Enunciado (3.9 ou mais) |
| Streamlit | Interface web, `st.metric`, gráficos nativos | Enunciado |
| Ultralytics YOLO (yolo11n, yolov8n) | Detecção de objetos | Enunciado |
| ByteTrack | Rastreamento, integrado à Ultralytics | Enunciado |
| OpenCV | Ler vídeo e webcam, desenhar caixas | Enunciado |
| PyTorch | Motor do YOLO, instalado pela Ultralytics | Dependência |
| `lap` | Necessária para o rastreamento | Dependência |
| pandas e numpy | Tabelas e gráficos, vêm com o Streamlit | Dependência |
| Biblioteca padrão do Python | CSV, arquivos temporários, som do alerta | Pedido do grupo |
| HTML, CSS e JavaScript | Maquete visual | Pedido do grupo |
| GitHub e GitHub Pages | Entrega do código e maquete no ar | Enunciado |
| Claude (Anthropic) | Criação do código a partir dos pedidos do grupo | Apoio |

O enunciado recomenda o Plotly, mas o grupo pediu para não usá-lo. Os gráficos são os nativos do Streamlit.

## Como ajudaria na rua

- **Contagem de fluxo:** saber quantos veículos passam por hora, para planejar semáforos, obras e rotas.
- **Horários de pico:** identificar quando há mais movimento e congestionamento.
- **Áreas restritas:** avisar quando alguém entra em um local proibido, como canteiros de obra.
- **Estacionamentos:** com uma evolução, mostrar vagas livres e ocupadas.
- **Segurança do trabalho:** avisar falta de capacete ou colete, se houver um modelo treinado.

É um protótipo acadêmico: não substitui um sistema homologado, e o desempenho real só se sabe testando com vídeos reais.

## Como facilita e como prejudica

**Facilita**

- Dados objetivos sem ninguém contando à mão, 24 horas por dia.
- Ajuda a decidir melhor onde investir em trânsito e segurança.
- Pode evitar acidentes com o alerta de invasão ou de falta de EPI.

**Prejudica ou exige cuidado**

- **Privacidade:** imagens de pessoas e placas podem ser dados pessoais (LGPD). Avise que há câmeras, guarde o mínimo e não publique placas ou rostos.
- **Vigilância:** o mesmo sistema pode ser usado para monitorar pessoas além do necessário.
- **Erros:** oclusão, noite, chuva e troca de ID geram contagens erradas e alertas falsos ou perdidos.
- **Falsa segurança:** confiar demais na máquina sem revisão humana.
- **Custo:** computador e energia para rodar vídeo o tempo todo.

## Prompt organizado do app

Este é o pedido reorganizado, não o texto literal original.

```
Crie um projeto acadêmico completo e executável chamado "ViaVision — Monitoramento de Tráfego e Segurança com IA".

CONTEXTO
Disciplina Tópicos Especiais em TI, Opção C (segurança, EPI e gestão de tráfego). Obrigatório: Python 3.9+, Streamlit, Visão Computacional e Deep Learning. Haverá defesa oral.

PRIORIDADES E RESTRIÇÕES
- Simples e organizado, rodando no Windows pela CPU, sem placa de vídeo.
- Sem Docker, banco de dados, API paga, serviço externo, frontend separado ou login.
- Poucas bibliotecas. CSV pela biblioteca padrão. Sem Plotly nem pacote separado de rastreamento.
- Verificar a documentação e a compatibilidade das versões e informar o Python recomendado.

FERRAMENTAS
Python, Streamlit (interface, métricas e gráficos nativos), Ultralytics (YOLO pequeno com rastreamento integrado), OpenCV.

FUNCIONALIDADES
1. Barra lateral: fonte (arquivo ou webcam); upload JPG/JPEG/PNG/MP4; confiança mínima; seleção de modelo (pré-treinado e customizado local opcional); botão Analisar; instruções curtas em português.
2. Imagem: detectar carros, motos, ônibus e caminhões (e pessoas); caixas com categoria e confiança; contagem por categoria; baixar imagem processada e CSV.
3. Vídeo: rastreamento integrado; linha horizontal com altura ajustável; contar quando o centro da caixa cruza a linha; cada ID uma vez; total por categoria e por sentido; reiniciar o estado a cada análise; progresso e prévia sem atualizar a cada quadro; liberar recursos; arquivos temporários compatíveis com Windows; não prometer rastreamento perfeito.
4. Webcam ao vivo com a mesma análise.
5. Alertas: área restrita ajustável com alerta visual e sonoro (1 por ID); falta de EPI apenas com modelo customizado treinado, sem inventar resultados.
6. Dashboard: st.metric com total, contagem por categoria, tempo médio de inferência em ms e confiança média (nunca chamar confiança de acurácia); gráfico nativo por categoria; gráfico de tendência no tempo; métricas em tempo real.
7. Exportação: CSV de detecções, passagens e alertas; imagem processada; vídeo processado opcional.
8. Erros tratados: arquivo inválido, imagem sem veículos, falha do modelo, falta de internet no primeiro download, modelo customizado ausente ou incompatível, webcam indisponível.
9. Treino e avaliação: train.py e evaluate.py separados; dataset de exemplo identificado como exemplo; divisão treino/validação/teste por vídeo; Precision, Recall, mAP e F1 reais; comparação com contagem manual. Nunca inventar treino, resultados ou pesos.
10. Interface toda em português, com componentes nativos, e resultados que permanecem após os cliques (st.session_state).
11. README para iniciante no Windows (PowerShell, ambiente virtual, alternativa sem ativar), com limitações, problemas comuns, explicação simples para a defesa, licenças, GitHub e roteiro do relatório.
12. Estrutura: app.py, vision.py, train.py, evaluate.py, requirements.txt, README.md, .gitignore, dataset_exemplo.yaml.
13. Verificação: testar o que for possível e dizer exatamente o que foi e o que não foi testado. Entregar tudo completo, em ZIP.
```

## Prompt organizado da maquete

```
Crie uma página HTML única, para abrir no celular, que simule o ViaVision, sem IA e sem câmera.

- Visão de cima de uma rua com 4 faixas, canteiro central, faixa de pedestre e calçadas.
- Veículos desenhados e realistas, com vários modelos por tipo: carros (sedan, hatch, SUV, picape, van), motos (esportiva, scooter, entrega, com garupa), ônibus (urbano, micro, articulado) e caminhões (baú, leve, carreta, caçamba, tanque), com cores variadas, sombras e faróis acesos à noite.
- Caixas de detecção com ID e tipo; linha de contagem ajustável; cada veículo contado uma vez ao cruzar; contadores por tipo que piscam, com "+1".
- Relógio de 00:00 a 23:59:59 (o primeiro dia começa às 06:00), com tempo acelerado. No fim do dia, fechar e registrar o dia e reiniciar sozinho às 00:00. Só parar quando apertar Parar.
- Congestionamento às 6h, 12h e 17h (veículos bem próximos e lentos) e motos passando pelo corredor.
- Gráficos: passagens por hora empilhadas por tipo, com as faixas de congestionamento; rosca com a participação por tipo; tabela de fechamento de cada dia; registro das passagens do dia.
- Visual limpo, tema claro e escuro, sem aviso grande, apenas uma marca discreta "SIMULAÇÃO".
```

## Para validar antes da defesa

- Conferir imagem, vídeo e webcam no computador que será usado na defesa (o vídeo de demonstração já foi testado).
- Comparar a contagem automática com a manual em vários vídeos.
- Treinar e medir com dataset próprio, se o grupo quiser métricas reais.
- Conferir se o repositório no GitHub está com README, `requirements.txt` e o vídeo `exemplos/transito_demo.mp4`.
