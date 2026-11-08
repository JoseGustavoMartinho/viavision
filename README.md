# ViaVision — Monitoramento de Tráfego e Segurança com Inteligência Artificial

Projeto acadêmico da disciplina **Tópicos Especiais em TI** (**Opção C**: monitoramento de segurança, EPI e gestão de tráfego urbano/comercial).

O ViaVision detecta **carros, motos, ônibus, caminhões e pessoas** em imagens, vídeos e webcam, **conta as passagens** por uma linha, emite **alertas** (invasão de área restrita e falta de EPI) e mostra **métricas e gráficos de tendência**, usando Visão Computacional e Deep Learning (YOLO). A interface é feita com Streamlit e tudo roda localmente, pela CPU.

## Autores

- Ronatta Kleyton Bezerra Cunha
- Miguel Arthur Vasconcelos Soares
- Suelen de Souza
- Ana Luiza de Souza Vieira
- Joyce Fernandes de Lima
- José Gustavo Martinho Araújo de Almeida

---

## 1. Objetivo e funcionamento

| Modo | O que faz | Resultado |
|---|---|---|
| **Imagem** (JPG, JPEG, PNG) | Detecta veículos (e pessoas, se marcado) e desenha caixa, categoria e confiança. | Total, contagem por categoria, imagem processada (PNG) e CSV. |
| **Vídeo** (MP4) | Detecta e **rastreia**; conta quando o **centro da caixa cruza a linha horizontal**; alerta se alguém entra na **área restrita**. | Métricas ao vivo durante a análise, total de passagens, contagem por categoria e sentido, **gráfico de tendência no tempo**, alertas, CSVs e (opcional) vídeo processado. |
| **Webcam ao vivo** | Mesma análise do vídeo, em tempo real, com a webcam do computador. | Métricas, gráfico de tendência e alertas atualizados na tela. |

Cada identificador rastreado é contado **uma única vez** por análise. O estado do rastreador e do contador é reiniciado a cada novo clique em **Analisar**.

### 1.1 Funcionalidades esperadas da Opção C e como o projeto atende

| Esperado no enunciado | Como o ViaVision faz |
|---|---|
| Processamento de vídeo/webcam ao vivo | Fonte “Webcam ao vivo” (chave **Ligar webcam**) e upload de MP4. A webcam é a do computador que executa o app. |
| Alerta sonoro/visual de **invasão de área restrita** | Você define um retângulo (sliders na barra lateral). Se a **base da caixa** de uma pessoa (ou de veículos, se marcado) entra nele, a caixa fica vermelha, aparece o banner **ALERTA** e toca um beep (no Windows, beep do computador; fora dele, áudio do navegador). 1 alerta por ID rastreado. |
| Alerta de **não uso de EPI** | Só funciona com um **modelo customizado treinado pelo grupo** com classes como `no_helmet`, `no_vest`, `sem_capacete` ou `sem_colete`. O modelo pré-treinado (COCO) **não reconhece EPI**. Sem esse modelo, o app avisa e o recurso fica inativo. |
| Gráficos de tendência temporal do fluxo | Gráfico de linhas com as passagens por intervalo de tempo (1, 2, 5 ou 10 s, conforme a duração). Também atualizado ao vivo na webcam. |
| Métricas em tempo real (`st.metric`) | Total de passagens, contagem por categoria, tempo médio de inferência (ms), confiança média do modelo, alertas e pessoas na área, atualizados durante a análise. |
| Sidebar, painel, KPIs e exportação | Upload JPG/PNG/MP4, limiar de confiança, seleção de modelo; imagem/vídeo processado com caixas; CSV de detecções/passagens/alertas e download da imagem/vídeo processado. |

> Sobre o “acurácia média” citado nos KPIs do enunciado: o app mostra a **confiança média do modelo**, que **não é acurácia**. Os valores reais de Precision, Recall, mAP e F1 vêm do `evaluate.py`, com um dataset anotado do grupo (seção 6).

### Arquivos

| Arquivo | Função |
|---|---|
| `app.py` | Interface Streamlit (barra lateral, métricas, gráfico, downloads). |
| `vision.py` | Lógica: modelo, detecção, rastreamento, contagem, CSV. Não usa Streamlit. |
| `train.py` | Treinamento/customização com um dataset YOLO fornecido pelo grupo. |
| `evaluate.py` | Avaliação no conjunto de teste (Precision, Recall, F1, mAP). |
| `teste_logica.py` | Testes da lógica de cruzamento da linha, área restrita, alertas e série temporal (sem YOLO; os alertas usam um modelo falso). |
| `exemplos/transito_demo.mp4` | Vídeo curto de trânsito (cerca de 7 MB) para testar o app sem precisar de outro arquivo. |
| `docs/index.html` | Maquete visual em HTML (simulação, sem IA) para publicar no GitHub Pages. |
| `GUIA_DO_PROJETO.md` | Como o projeto foi feito, ferramentas, uso na rua, riscos e prompts organizados. |
| `dataset_exemplo.yaml` | **Exemplo** de configuração de dataset (não é um dataset real). |
| `requirements.txt` | Dependências. |

---

## 2. Requisitos do computador

- **Windows 10/11, 64 bits.** (O projeto foi testado no Windows; veja a seção 2.1.)
- **Python recomendado: 3.11 (64 bits).** O projeto exige Python 3.9 ou superior, mas versões novas do Streamlit e do PyTorch podem não oferecer suporte ao 3.9; o 3.11 é uma escolha conservadora, com pacotes prontos (wheels) para Windows.
- Não precisa de placa de vídeo dedicada: o app força o uso da **CPU**.
- Espaço em disco: a instalação do PyTorch (instalado automaticamente pela Ultralytics) é grande — reserve alguns GB.
- Internet: necessária para instalar as dependências e para baixar os pesos pré-treinados **na primeira execução**.
- Não há garantia de desempenho. O tempo de análise depende do seu processador, da resolução e do tamanho do vídeo; em CPU **não é tempo real**.

---

### 2.1 Onde o projeto foi testado

| Item | Valor |
|---|---|
| Sistema | Windows, execução pela CPU (Intel Core i7-4510U, 2,00 GHz) |
| Python | 3.15.2 (a recomendação conservadora continua sendo o 3.11) |
| Ultralytics | 8.4.202 |
| PyTorch | 2.15.0 (versão CPU) |
| Modelo | YOLO11n pré-treinado (COCO) |
| Execução | `streamlit run app.py`, abrindo em `http://localhost:8501` |
| Teste da lógica | `python teste_logica.py` (12 testes) |
| Webcam | **Ainda não testada** (o recurso existe no código, mas não foi validado) |

O app abriu e carregou o modelo normalmente. Este registro diz apenas **onde funcionou**; não é medida de desempenho nem de acerto. As medidas reais estão na seção 6.4 e 6.5.

---

## 3. Passo a passo para iniciantes (Windows + PowerShell)

### 3.1 Instalar o Python 3.11
1. Baixe o instalador do Python 3.11 (64 bits) em <https://www.python.org/downloads/windows/>.
2. No instalador, marque **“Add python.exe to PATH”** e clique em *Install Now*.
3. Confirme, em um PowerShell novo:
   ```powershell
   py -3.11 --version
   ```
   Deve aparecer `Python 3.11.x`.

### 3.2 Abrir a pasta no VS Code
1. Extraia o ZIP do projeto.
2. No VS Code: **File > Open Folder…** e escolha a pasta `viavision`.
3. Abra o terminal: **Terminal > New Terminal** (deve ser PowerShell e já estar dentro da pasta do projeto).

### 3.3 Criar o ambiente virtual
```powershell
py -3.11 -m venv .venv
```

### 3.4 Ativar o ambiente virtual (opcional) e instalar as dependências
**Opção A — com ativação:**
```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```
Se o PowerShell bloquear a ativação (erro sobre “execução de scripts desabilitada”), libere **apenas para esta janela**:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

**Opção B — sem ativar (usando direto o Python do ambiente virtual):**
```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 3.5 Executar
Com o ambiente ativado:
```powershell
python -m streamlit run app.py
```
Sem ativar:
```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```
O navegador abrirá em `http://localhost:8501`. **Execute sempre dentro da pasta do projeto**, pois é nela que os pesos baixados ficam salvos.

### 3.6 Usar
1. Na barra lateral, envie uma imagem ou um vídeo MP4 curto. Para testar rápido, use o vídeo de demonstração que acompanha o projeto: `exemplos/transito_demo.mp4`.
2. Ajuste a **confiança mínima** e escolha o **modelo**.
3. Em vídeos, ajuste a **altura da linha** (a pré-visualização mostra a linha amarela).
4. Clique em **Analisar** e baixe os arquivos desejados.

### 3.7 (Opcional) Testar a lógica de contagem
```powershell
python teste_logica.py
```

### 3.8 (Opcional) Congelar as versões que funcionaram
O `requirements.txt` usa **faixas** de versão. Depois que tudo funcionar no seu computador, registre as versões exatas:
```powershell
python -m pip freeze > requirements-lock.txt
```

---

## 4. Download inicial do modelo

- O modelo padrão é pré-treinado no conjunto COCO (`yolo11n.pt`, pequeno). Também há `yolov8n.pt`.
- Na **primeira análise**, a Ultralytics **baixa o arquivo `.pt` da internet** e o salva na pasta onde o app foi executado. Nas próximas vezes, o arquivo local é reutilizado, sem internet.
- Sem internet na primeira vez, o app mostra uma mensagem de erro explicando o problema. Alternativa: baixe os pesos em outro computador e coloque o `.pt` na pasta do projeto.
- **Modelo customizado (opcional):** escolha “Modelo customizado” na barra lateral e informe o caminho do `.pt` (por exemplo, o `best.pt` gerado pelo `train.py`). O app funciona normalmente sem esse arquivo. O modelo precisa ser de detecção e ter classes chamadas `car`, `motorcycle`, `bus` e/ou `truck` (ou `carro`, `moto`, `onibus`, `caminhao`).

---

## 5. Como preparar vídeos para demonstração

O repositório já traz um vídeo pronto para teste: **`exemplos/transito_demo.mp4`**. Basta enviá-lo pelo botão de upload da barra lateral. Os demais arquivos `.mp4` não vão para o GitHub (estão no `.gitignore`); se quiser usar outros vídeos, guarde-os em uma pasta local como `videos/`.

Para gravar ou escolher outros vídeos:

- Vídeos **curtos** (10 a 60 segundos).
- **Câmera parada** (tripé ou ponto fixo, como passarela ou janela), vendo a via de cima ou em ângulo.
- **Boa iluminação** e veículos bem visíveis.
- Os veículos devem **cruzar a linha horizontal** (movimento predominantemente vertical na imagem). Se a rua aparecer na horizontal, grave ou escolha outro ângulo, pois a linha é sempre horizontal.
- Formato **MP4**. Se o seu vídeo não abrir, converta para MP4 (H.264) com um conversor.
- Respeite a privacidade: evite gravar rostos e placas de forma identificável e use apenas vídeos que você tem direito de usar.
- Para análises mais rápidas, use o controle “Analisar 1 a cada N quadros” (mais rápido, porém com rastreamento menos preciso).

---

## 6. Como treinar e avaliar (opcional)

> **O treinamento depende de um dataset anotado fornecido pelo grupo.** Este projeto **não inclui dataset, pesos treinados nem resultados de treinamento**. A demonstração inicial funciona normalmente só com o modelo pré-treinado.

### 6.1 Formato do dataset
Use o formato YOLO (veja `dataset_exemplo.yaml`, que é apenas um exemplo): uma pasta `images/` e uma `labels/` com subpastas `train`, `val` e `test`. Cada imagem tem um `.txt` com o mesmo nome; cada linha: `id_da_classe x_centro y_centro largura altura` (valores de 0 a 1). Ferramentas gratuitas de anotação (por exemplo, CVAT ou Label Studio) conseguem exportar nesse formato.

### 6.2 Como separar treino, validação e teste (sem “vazamento”)
Quadros de um mesmo vídeo são quase idênticos. Se um estiver no treino e o vizinho no teste, a avaliação fica **otimista e falsa**. Por isso:

1. **Separe por vídeo (ou por gravação/câmera/dia), nunca por quadro.** Defina primeiro quais vídeos vão para treino, validação e teste; só depois extraia os quadros.
2. Uma divisão comum é aproximadamente 70% / 15% / 15% **dos vídeos**. Com poucos vídeos, reserve ao menos um vídeo inteiro para validação e um para teste.
3. Extraia quadros com **espaçamento** (por exemplo, 1 quadro a cada 1 ou 2 segundos) para evitar quadros repetidos.
4. Se possível, o teste deve ter **condições diferentes** (outro local, horário ou câmera) para refletir uso real.
5. **Use o conjunto de teste uma única vez**, no fim. Ajustes de parâmetros devem usar só a validação.
6. Não deixe o mesmo vídeo (nem cópias dele) em mais de um conjunto.

### 6.3 Treinar
Com o ambiente virtual ativo:
```powershell
python train.py --data C:\caminho\meu_dataset.yaml --epochs 30
```
Em CPU o treinamento pode demorar bastante; para testar o fluxo, use poucas épocas e poucas imagens. Ao final, o script imprime o caminho do `best.pt`.

### 6.4 Avaliar no conjunto de teste
```powershell
python evaluate.py --model runs\viavision_treino\weights\best.pt --data C:\caminho\meu_dataset.yaml
```
O script mostra e salva (`resultados_avaliacao.csv`) **Precision, Recall, F1, mAP50 e mAP50-95**, no total e por classe, **calculados pela biblioteca sobre o seu dataset real**. Observações:
- **Precision**: das detecções feitas, quantas estavam certas. **Recall**: dos veículos que existem, quantos foram encontrados.
- **F1**: combina as duas (calculado aqui a partir de P e R). **mAP**: resumo da qualidade da detecção em vários limiares de confiança/IoU (mAP50 usa IoU 0,5; mAP50-95 é mais rigoroso).
- O ponto de confiança em que Precision e Recall são reportados é definido pela Ultralytics; consulte a documentação oficial da versão que você instalou antes de interpretar os valores no relatório.
- Para comparar com o modelo pré-treinado, rode também `--model yolo11n.pt` no mesmo dataset de teste (as classes do dataset precisam corresponder às do modelo).
- Não registre no relatório nenhum número que não tenha saído dessa execução.

### 6.5 Comparar a contagem automática com a manual
1. Escolha 3 ou mais vídeos curtos de demonstração.
2. Assista a cada vídeo e **conte manualmente** quantos veículos cruzam a linha (por categoria, se quiser). Faça a contagem **antes** de ver o resultado do app, e, se possível, com duas pessoas.
3. Rode o app com a mesma altura de linha e anote o total automático.
4. Monte uma tabela:

   | Vídeo | Manual | Automático | Erro absoluto = \|Auto − Manual\| | Erro relativo = Erro absoluto ÷ Manual |
   |---|---|---|---|---|
   | v1 | | | | |

5. Use o CSV de passagens (ID, categoria, tempo e sentido) para localizar **quando** houve contagens a mais ou a menos e anote a causa provável (oclusão, troca de ID, veículo parado sobre a linha, etc.).
6. Reporte os números como foram medidos, nesses vídeos, nessas condições — sem generalizar.

---

## 7. Limitações

- **Oclusão:** quando um veículo fica escondido atrás de outro, o rastreador pode perder o objeto.
- **Troca de identificador (ID switch):** após a oclusão, o mesmo veículo pode receber um **novo ID** (e ser contado duas vezes) ou dois veículos podem trocar de ID (e um deixar de ser contado).
- O rastreamento **não é perfeito** e a contagem **pode ter duplicatas ou omissões**. Os resultados devem ser tratados como estimativa.
- Veículo que **surge já depois da linha** não é contado (não há posição anterior para detectar o cruzamento).
- Em CPU, **não há processamento em tempo real**.
- O alerta de **EPI** depende de um modelo customizado; sem ele, não há alerta de EPI.
- A **área restrita** é um retângulo e usa a base da caixa como referência; pessoas parcialmente escondidas podem falhar.
- A **webcam ao vivo** em CPU roda com poucos quadros por segundo, e mudar qualquer controle reinicia a análise.
- A linha é **horizontal**; o “sentido” é relativo à imagem (descendo = de cima para baixo; subindo = de baixo para cima).
- O modelo pré-treinado (COCO) pode errar em ângulos incomuns, noite, chuva, veículos pequenos ou distantes.
- A **confiança** é a pontuação do modelo para cada detecção; **não é acurácia** nem probabilidade de acerto garantida.
- O vídeo processado é gravado com codec `mp4v`, que costuma abrir em players como VLC e no Windows, mas pode não tocar dentro do navegador. Por isso o app só oferece o download. Se a gravação não for possível, o restante continua funcionando.

---

## 8. Problemas comuns e soluções

| Problema | Solução |
|---|---|
| `python` não é reconhecido | Reinstale o Python marcando “Add python.exe to PATH” ou use `py -3.11`. |
| Erro ao ativar o `.venv` (scripts desabilitados) | Use `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` ou a Opção B (executável direto do `.venv`). |
| `No module named streamlit` / `ultralytics` | Você está usando outro Python. Use `.\.venv\Scripts\python.exe -m pip install -r requirements.txt` e `.\.venv\Scripts\python.exe -m streamlit run app.py`. |
| Falha ao instalar `lap` | Edite o `requirements.txt` trocando `lap>=0.5.12` por `lapx>=0.5.2` e instale de novo. Prefira o Python 3.11. |
| Erro ao baixar os pesos | Sem internet na primeira execução. Conecte-se e tente de novo (ou copie o `.pt` para a pasta do projeto). Redes de faculdade podem bloquear o download. |
| Falha por arquivo de pesos incompleto | Apague o `.pt` da pasta do projeto e tente de novo. |
| Mensagem sobre `c10.dll` / `WinError 126` (PyTorch) | Instale o “Microsoft Visual C++ Redistributable” (x64) e reinicie. |
| Vídeo não abre | Converta para MP4 (H.264). |
| Análise muito lenta | Use vídeo menor/mais curto, aumente “Analisar 1 a cada N quadros” e deixe desmarcada a opção de gerar vídeo. |
| Nenhum veículo detectado | Reduza a confiança mínima; confira se há veículos visíveis e bem iluminados. |
| Contagens duplicadas | Esperado em oclusões/trocas de ID (ver Limitações). Teste outra altura de linha ou outro vídeo. |
| Modelo customizado não funciona | Confira o caminho, se é `.pt` de detecção e se as classes se chamam `car`, `motorcycle`, `bus` ou `truck`. |
| Webcam não abre | Feche Zoom/Teams/Câmera do Windows, confira em Configurações > Privacidade > Câmera, e tente outro índice (0, 1, 2). |
| Sem som no alerta | No Windows o beep sai pelo alto-falante do computador (confira o volume). Fora do Windows, o navegador pode bloquear áudio automático: clique na página uma vez. |
| Porta ocupada | `python -m streamlit run app.py --server.port 8502` |

---

## 9. Explicação simples para a defesa oral

- **Detecção:** o YOLO olha uma imagem **uma vez** e prevê, ao mesmo tempo, onde há objetos (caixas) e de que tipo são. Foi treinado com milhares de imagens anotadas (COCO), por isso “já sabe” o que é carro, moto, ônibus e caminhão. Usamos a versão pequena (“n”) para rodar na CPU.
- **Confiança:** é a nota (0 a 1) que o modelo dá a cada detecção. O controle da barra lateral é um **filtro**: descarta o que tem nota baixa. Nota alta **não garante** que está certo e **não é acurácia**; a qualidade real do sistema só é medida com avaliação em dados anotados (Precision, Recall, mAP).
- **Rastreamento:** detectar quadro a quadro não diz se o carro do quadro 10 é o mesmo do quadro 11. O rastreador (ByteTrack, integrado à Ultralytics) liga as caixas entre quadros consecutivos por proximidade/sobreposição e atribui um **ID** a cada veículo.
- **Contagem:** calculamos o **centro da caixa** de cada ID. Se, entre dois quadros seguidos, o centro passou de um lado da linha para o outro, houve uma passagem. O ID entra num conjunto de “já contados”, então nunca conta duas vezes. O sentido (descendo/subindo) vem de qual lado ele estava antes.
- **Por que há erros:** se o ID muda (oclusão), o sistema acha que é outro veículo.
- **Como mostrar o código:** `vision.py` (`sentido_de_passagem`, `ContadorDeLinha`, `processar_video`), `app.py` (interface e `st.session_state`) e `teste_logica.py` (prova do cruzamento com casos simples).
- **Por que o resultado não some no Streamlit:** o Streamlit reexecuta o script a cada clique; guardamos o resultado em `st.session_state`.

---

## 10. Entrega no GitHub (obrigatória pelo enunciado)

O enunciado exige o código em repositório **GitHub ou GitLab** (público ou privado), com **README** e **requirements.txt** atualizado. Passo a passo (PowerShell, na pasta do projeto):

```powershell
git init
git add .
git commit -m "ViaVision: versão inicial"
git branch -M main
git remote add origin https://github.com/JoseGustavoMartinho/viavision.git
git push -u origin main
```

- Crie o repositório vazio no GitHub antes, e adicione os colegas em **Settings > Collaborators** para que todos consigam enviar commits (o repositório do grupo é público, então o professor consegue ver sem convite).
- O `.gitignore` impede de subir `venv/`/`.venv/`, pesos `.pt`, `runs/` e vídeos `.mp4`, **exceto** `exemplos/transito_demo.mp4` (regra `!transito_demo.mp4`). Os pesos são baixados automaticamente na primeira execução (seção 4).
- Faça commits pequenos e frequentes, **de cada integrante**, para o histórico mostrar o trabalho do grupo.
- Rode `python -m pip freeze > requirements-lock.txt` depois que tudo funcionar, e confira se o `requirements.txt` está atualizado.

### Maquete visual no ar (GitHub Pages)

O GitHub **guarda o código**, mas **não executa o app Streamlit** (ele precisa de Python, YOLO e CPU). O que pode ficar "no ar" pelo GitHub é a **maquete visual em HTML** (simulação, sem IA), que está em `docs/index.html`:

1. No repositório: **Settings > Pages**.
2. Em **Source**, escolha **Deploy from a branch**, branch `main` e pasta `/docs`, e clique em **Save**.
3. Depois de 1 a 2 minutos, o link da maquete aparece ali mesmo, na página do Pages.

Esse link mostra apenas a simulação. A análise real de vídeo roda no computador, seguindo o passo a passo deste README.

## 11. Relatório técnico (20% da nota): roteiro

O enunciado pede um documento em formato de artigo/relatório com arquitetura, pré-processamento, métricas (Precision, Recall, mAP/F1) e resultados. Sugestão de estrutura:

1. Introdução e objetivo (Opção C).
2. Arquitetura (Streamlit + `vision.py` + YOLO + ByteTrack + OpenCV) e fluxograma do pipeline.
3. Dados e pré-processamento (origem dos vídeos/imagens, extração de quadros, anotação, divisão por vídeo).
4. Modelo (pré-treinado e, se houver, customizado) e configuração do treino.
5. Métricas **reais** do `evaluate.py` (P, R, F1, mAP50, mAP50-95) e comparação contagem automática × manual (seção 6.5).
6. Resultados do app, limitações e trabalhos futuros.
7. Referências e licenças.

**Não escreva nenhum número que não tenha saído de uma execução real.** Se o grupo não treinar um modelo, relate apenas a comparação da contagem automática com a manual feita no vídeo de demonstração, e diga claramente que o modelo é pré-treinado.

## 12. Licenças (confira sempre as fontes oficiais)

Este projeto acadêmico usa bibliotecas de terceiros, cada uma com sua licença. Antes de distribuir ou usar comercialmente, leia as fontes oficiais:

- **Ultralytics (YOLO, código e pesos):** licenciada sob **AGPL-3.0**, com opção de licença empresarial (Enterprise). A AGPL tem exigências importantes se você distribuir ou oferecer o software como serviço em rede. Consulte: <https://www.ultralytics.com/license> e <https://github.com/ultralytics/ultralytics>.
- **Streamlit:** Apache License 2.0 (conforme a página do projeto no PyPI). <https://github.com/streamlit/streamlit>
- **OpenCV / opencv-python:** consulte o repositório oficial do pacote (<https://github.com/opencv/opencv-python>) e do OpenCV (<https://opencv.org/license/>), pois a licença depende da versão e dos componentes empacotados.
- **PyTorch:** licença BSD-style; veja <https://github.com/pytorch/pytorch>.
- **Conjunto COCO** (usado no pré-treinamento dos pesos): consulte os termos em <https://cocodataset.org/#termsofuse>.
- Se você treinar com seu próprio dataset, **você é responsável** pelos direitos de uso das imagens e vídeos.

Esta lista é um ponto de partida, não aconselhamento jurídico. Na dúvida, pergunte ao professor.
