"""
app.py - Interface Streamlit do ViaVision.

Execute com:  python -m streamlit run app.py

O Streamlit reexecuta este script inteiro a cada interação. Por isso o resultado da
análise fica guardado em st.session_state["resultado"]: assim as métricas e os botões
de download continuam na tela depois de qualquer clique.
"""

import time

import pandas as pd  # já vem instalado junto com o Streamlit
import streamlit as st

import vision

st.set_page_config(page_title="ViaVision", page_icon="🚦", layout="wide")

OPCOES_MODELO = {
    "YOLO11n (pequeno, padrão)": "yolo11n.pt",
    "YOLOv8n (pequeno)": "yolov8n.pt",
    "Modelo customizado (arquivo .pt local)": None,
}
FONTE_ARQUIVO = "Arquivo (imagem ou vídeo)"
FONTE_WEBCAM = "Webcam ao vivo"


# ---------------------------------------------------------------------------
# Funções auxiliares
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Carregando o modelo (na primeira vez pode baixar os pesos)...")
def carregar(origem: str, customizado: bool):
    """Carrega o modelo uma única vez e reaproveita nas próximas execuções."""
    return vision.carregar_modelo(origem, customizado)


@st.cache_data(show_spinner=False)
def primeiro_quadro(dados: bytes):
    """Primeiro quadro do vídeo, para mostrar onde ficarão a linha e a área restrita."""
    caminho = vision.salvar_temporario(dados, ".mp4")
    try:
        return vision.ler_primeiro_quadro(caminho)
    finally:
        vision.remover_arquivo(caminho)


def metricas(rotulo_total, total, contagem, ativas, rotulo_ms, ms, conf, extras=()):
    """Desenha as métricas (st.metric) no contêiner atual."""
    colunas = st.columns(1 + len(ativas))
    colunas[0].metric(rotulo_total, total)
    for coluna, categoria in zip(colunas[1:], ativas):
        coluna.metric(categoria.capitalize(), contagem[categoria])
    itens = [(rotulo_ms, f"{ms:.0f}"),
             ("Confiança média do modelo (não é acurácia)", "—" if conf is None else f"{conf:.2f}")]
    itens.extend(extras)
    colunas2 = st.columns(len(itens))
    for coluna, (rotulo, valor) in zip(colunas2, itens):
        coluna.metric(rotulo, valor)


def tabela_tendencia(eventos, duracao, ativas):
    """DataFrame do gráfico de tendência (passagens por intervalo de tempo)."""
    intervalo = vision.intervalo_sugerido(duracao)
    linhas = vision.serie_temporal(eventos, duracao, intervalo)
    tabela = pd.DataFrame(linhas).set_index("início (s)")
    return tabela[list(ativas) + ["total"]], intervalo


def sinalizar(novos, alerta_ph, som_ativo, estado):
    """Mostra o alerta visual e toca o aviso sonoro (no máximo 1 vez a cada 2 s)."""
    if not novos:
        return
    tipos = ", ".join(sorted({a["tipo"] for a in novos}))
    alerta_ph.error(f"ALERTA: {tipos} (tempo {novos[-1]['tempo_s']:.1f} s)")
    agora = time.time()
    estado["ultimo_alerta"] = agora
    if som_ativo and agora - estado["ultimo_som"] >= 2:
        estado["ultimo_som"] = agora
        if not vision.tocar_beep():  # fora do Windows: toca pelo navegador
            estado["som_ph"].audio(vision.wav_do_alerta(), format="audio/wav", autoplay=True)


def limpar_alerta_antigo(alerta_ph, estado):
    if estado["ultimo_alerta"] and time.time() - estado["ultimo_alerta"] > 3:
        alerta_ph.empty()
        estado["ultimo_alerta"] = 0.0


def montar_resultado(nome, analisador, duracao, quadros, ativas, cfg, ao_vivo, video_bytes=None, pediu_video=False):
    """Junta tudo o que a tela de resultado precisa em um dicionário."""
    contador = analisador.contador
    eventos = contador.eventos if contador else []
    tabela, intervalo = tabela_tendencia(eventos, duracao, ativas)
    return {
        "tipo": "video", "ao_vivo": ao_vivo, "nome": nome, "ativas": list(ativas),
        "total": analisador.total, "contagem": analisador.contagem,
        "por_sentido": contador.por_sentido if contador else {},
        "ms": analisador.ms_medio, "conf": analisador.conf_media,
        "quadros": quadros, "duracao": duracao,
        "csv": vision.csv_video(eventos), "tabela": eventos,
        "csv_alertas": vision.csv_alertas(analisador.alertas), "alertas": list(analisador.alertas),
        "monitorando": cfg["area_ativa"] or cfg["epi_ativo"],
        "tendencia": tabela, "intervalo": intervalo,
        "video": video_bytes, "pediu_video": pediu_video,
    }


def executar_imagem(arquivo, modelo, mapa, epi, confianca, ativas, cfg):
    imagem = vision.decodificar_imagem(arquivo.getvalue())
    r = vision.analisar_imagem(modelo, mapa, epi, imagem, confianca)
    return {
        "tipo": "imagem", "nome": arquivo.name, "ativas": list(ativas),
        "total": r.total, "contagem": r.contagem, "ms": r.ms_inferencia, "conf": r.conf_media,
        "epi_ativo": cfg["epi_ativo"], "violacoes_epi": r.violacoes_epi,
        "imagem": r.imagem, "png": vision.codificar_png(r.imagem),
        "csv": vision.csv_imagem(r.deteccoes), "tabela": r.deteccoes,
    }


def executar_video(arquivo, modelo, mapa, epi, cfg, ativas):
    caminho_entrada = vision.salvar_temporario(arquivo.getvalue(), ".mp4")
    caminho_saida = vision.criar_caminho_temporario(".mp4") if cfg["gerar_video"] else None

    barra = st.progress(0.0, text="Iniciando...")
    alerta_ph = st.empty()
    metricas_ph = st.empty()
    previa = st.empty()
    estado = {"ultimo_alerta": 0.0, "ultimo_som": 0.0, "som_ph": st.empty()}
    ultima = [0.0]

    def ao_progresso(indice, total_quadros, quadro, analisador, novos):
        sinalizar(novos, alerta_ph, cfg["som"], estado)  # alertas: sempre na hora
        agora = time.time()
        if agora - ultima[0] < 0.5:  # interface: no máximo a cada 0,5 s, e não a cada quadro
            return
        ultima[0] = agora
        if total_quadros > 0:
            barra.progress(min(indice / total_quadros, 1.0), text=f"Quadro {indice} de {total_quadros}")
        else:
            barra.progress(0.0, text=f"Processando quadro {indice}")
        with metricas_ph.container():
            metricas("Passagens até agora", analisador.total, analisador.contagem, ativas,
                     "Tempo médio de inferência por quadro (ms)", analisador.ms_medio,
                     analisador.conf_media, extras=[("Alertas até agora", len(analisador.alertas))])
        previa.image(quadro, channels="BGR", caption="Prévia do processamento")
        limpar_alerta_antigo(alerta_ph, estado)

    try:
        resultado = vision.processar_video(
            modelo, mapa, epi, caminho_entrada, cfg["confianca"], cfg["linha_rel"], passo=cfg["passo"],
            area=cfg["area"], alvos_area=cfg["alvos"], caminho_saida=caminho_saida, ao_progresso=ao_progresso,
        )
        video_bytes = None
        if resultado.video_gerado:
            with open(caminho_saida, "rb") as arquivo_video:
                video_bytes = arquivo_video.read()
    finally:
        vision.remover_arquivo(caminho_entrada)
        vision.remover_arquivo(caminho_saida)
        barra.empty()
        previa.empty()
        metricas_ph.empty()
        alerta_ph.empty()

    tabela, intervalo = tabela_tendencia(resultado.eventos, resultado.duracao_s, ativas)
    return {
        "tipo": "video", "ao_vivo": False, "nome": arquivo.name, "ativas": list(ativas),
        "total": resultado.total, "contagem": resultado.contagem, "por_sentido": resultado.por_sentido,
        "ms": resultado.ms_inferencia_medio, "conf": resultado.conf_media,
        "quadros": resultado.quadros_analisados, "duracao": resultado.duracao_s,
        "csv": vision.csv_video(resultado.eventos), "tabela": resultado.eventos,
        "csv_alertas": vision.csv_alertas(resultado.alertas), "alertas": resultado.alertas,
        "monitorando": cfg["area_ativa"] or cfg["epi_ativo"],
        "tendencia": tabela, "intervalo": intervalo,
        "video": video_bytes, "pediu_video": cfg["gerar_video"],
    }


def executar_webcam(indice_camera, modelo, mapa, epi, cfg, ativas):
    """
    Loop ao vivo. Para parar, basta desligar a chave "Ligar webcam": o Streamlit reexecuta
    o script e interrompe este loop. O último resultado fica guardado em session_state.
    """
    st.session_state.pop("resultado", None)
    captura = vision.abrir_webcam(indice_camera)
    analisador = vision.AnalisadorDeQuadros(
        modelo, mapa, epi, cfg["confianca"], cfg["linha_rel"], cfg["area"], cfg["alvos"]
    )
    st.info("Ao vivo. Para parar, desligue a chave “Ligar webcam” na barra lateral. "
            "Mudar qualquer controle reinicia a análise.")
    alerta_ph = st.empty()
    metricas_ph = st.empty()
    quadro_ph = st.empty()
    grafico_ph = st.empty()
    estado = {"ultimo_alerta": 0.0, "ultimo_som": 0.0, "som_ph": st.empty()}
    inicio = time.time()
    indice = 0
    ultima_ui = 0.0
    ultimo_grafico = 0.0
    duracao = 0.0

    try:
        while True:
            ok, quadro = captura.read()
            if not ok:
                st.warning("A webcam parou de enviar imagens.")
                break
            duracao = time.time() - inicio
            novos = analisador.processar(quadro, indice, duracao)
            quadro_ph.image(quadro, channels="BGR")
            sinalizar(novos, alerta_ph, cfg["som"], estado)

            if duracao - ultima_ui >= 0.5:  # métricas em tempo real
                ultima_ui = duracao
                with metricas_ph.container():
                    metricas("Passagens (ao vivo)", analisador.total, analisador.contagem, ativas,
                             "Tempo médio de inferência por quadro (ms)", analisador.ms_medio,
                             analisador.conf_media,
                             extras=[("Na área restrita agora", analisador.em_area_agora),
                                     ("Alertas até agora", len(analisador.alertas))])
                limpar_alerta_antigo(alerta_ph, estado)
            if duracao - ultimo_grafico >= 2.0:  # gráfico de tendência
                ultimo_grafico = duracao
                tabela, _ = tabela_tendencia(analisador.contador.eventos, duracao, ativas)
                grafico_ph.line_chart(tabela)
            indice += 1
    finally:
        captura.release()
        analisador.encerrar()
        try:  # guarda o que foi medido para continuar na tela depois de parar
            if analisador.contador is not None:
                st.session_state["resultado"] = montar_resultado(
                    "Webcam ao vivo", analisador, duracao, indice, ativas, cfg, ao_vivo=True)
        except Exception:
            pass


def mostrar_resultado(r):
    st.subheader(f"Resultado: {r['nome']}")
    ativas = r["ativas"]

    if r["tipo"] == "imagem":
        extras = [("Violações de EPI", r["violacoes_epi"])] if r["epi_ativo"] else []
        metricas("Total detectado (na imagem)", r["total"], r["contagem"], ativas,
                 "Tempo de inferência (ms)", r["ms"], r["conf"], extras)
        st.image(r["imagem"], channels="BGR", caption="Imagem processada (categoria e confiança do modelo)")
        if r["total"] == 0 and r["violacoes_epi"] == 0:
            st.info("Nada foi detectado. Tente reduzir a confiança mínima ou use outra imagem.")
        if r["violacoes_epi"]:
            st.error(f"{r['violacoes_epi']} violação(ões) de EPI detectada(s).")
    else:
        rotulo = "Total de passagens pela linha (ao vivo)" if r["ao_vivo"] else "Total de passagens pela linha (no vídeo)"
        metricas(rotulo, r["total"], r["contagem"], ativas,
                 "Tempo médio de inferência por quadro (ms)", r["ms"], r["conf"],
                 extras=[("Alertas registrados", len(r["alertas"]))])
        st.caption(
            f"{r['quadros']} quadros analisados em {r['duracao']:.1f} s. A contagem depende do "
            "rastreamento: oclusões e trocas de ID podem gerar passagens a mais ou a menos."
        )
        if r["alertas"]:
            st.error(f"{len(r['alertas'])} alerta(s) registrado(s) nesta análise.")
            st.dataframe(pd.DataFrame(r["alertas"]))
        elif r["monitorando"]:
            st.success("Nenhum alerta registrado nesta análise.")
        if r["total"] == 0:
            st.info("Nenhuma passagem foi contada. Ajuste a altura da linha ou reduza a confiança mínima.")

    st.markdown("**Contagem por categoria**")
    grafico = pd.DataFrame({"Quantidade": [r["contagem"][c] for c in ativas]}, index=ativas)
    if ativas:
        st.bar_chart(grafico)

    if r["tipo"] == "video":
        st.markdown(f"**Tendência do fluxo no tempo** (passagens a cada {r['intervalo']} s)")
        st.line_chart(r["tendencia"])
        st.markdown("**Passagens por sentido** (em relação à imagem: de cima para baixo ou de baixo para cima)")
        sentidos = pd.DataFrame(r["por_sentido"]).T.reindex(ativas)[list(vision.SENTIDOS)]
        st.dataframe(sentidos)

    with st.expander("Ver dados da exportação"):
        if r["tabela"]:
            st.dataframe(pd.DataFrame(r["tabela"]))
        else:
            st.write("Sem registros.")

    st.markdown("**Downloads**")
    base = r["nome"].rsplit(".", 1)[0].replace(" ", "_")
    colunas = st.columns(3)
    if r["tipo"] == "imagem":
        colunas[0].download_button("Baixar imagem processada (PNG)", r["png"],
                                   file_name=f"{base}_processada.png", mime="image/png")
        colunas[1].download_button("Baixar CSV das detecções", r["csv"],
                                   file_name=f"{base}_deteccoes.csv", mime="text/csv")
    else:
        colunas[0].download_button("Baixar CSV das passagens", r["csv"],
                                   file_name=f"{base}_passagens.csv", mime="text/csv")
        if r["monitorando"]:
            colunas[1].download_button("Baixar CSV dos alertas", r["csv_alertas"],
                                       file_name=f"{base}_alertas.csv", mime="text/csv")
        if r.get("video") is not None:
            colunas[2].download_button("Baixar vídeo processado (MP4)", r["video"],
                                       file_name=f"{base}_processado.mp4", mime="video/mp4")
        elif r.get("pediu_video"):
            st.warning("Não foi possível gravar o vídeo processado neste computador. "
                       "A contagem e os CSVs não foram afetados.")


# ---------------------------------------------------------------------------
# Barra lateral
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Configurações")

    fonte = st.radio("Fonte", [FONTE_ARQUIVO, FONTE_WEBCAM])
    arquivo = None
    if fonte == FONTE_ARQUIVO:
        arquivo = st.file_uploader("Imagem ou vídeo", type=["jpg", "jpeg", "png", "mp4"])
    else:
        indice_camera = st.number_input("Índice da webcam", min_value=0, max_value=5, value=0, step=1,
                                        help="0 é a webcam padrão do computador.")
        ligar = st.toggle("Ligar webcam")

    confianca = st.slider(
        "Confiança mínima da detecção", 0.05, 0.95, 0.35, 0.05,
        help="Detecções com confiança do modelo abaixo deste valor são descartadas. "
             "Confiança não é o mesmo que acurácia.",
    )

    escolha = st.selectbox("Modelo", list(OPCOES_MODELO.keys()))
    caminho_customizado = ""
    if OPCOES_MODELO[escolha] is None:
        caminho_customizado = st.text_input(
            "Caminho do arquivo .pt", placeholder=r"C:\caminho\para\best.pt",
            help="Opcional. O app funciona normalmente sem modelo customizado. Para o alerta de EPI, "
                 "o modelo precisa ter classes como no_helmet, no_vest, sem_capacete ou sem_colete.",
        )

    eh_video = arquivo is not None and arquivo.name.lower().endswith(".mp4")
    usa_linha = eh_video or fonte == FONTE_WEBCAM

    linha_pct, passo, gerar_video = 50, 1, False
    area_ativa, alertar_veiculos, area_x, area_y = False, False, (60, 95), (50, 95)
    if usa_linha:
        st.divider()
        st.subheader("Contagem e alertas")
        linha_pct = st.slider("Altura da linha de contagem (%)", 5, 95, 50,
                              help="0% é o topo da imagem e 100% é a base.")
        area_ativa = st.checkbox("Ativar área restrita (alerta de invasão)")
        if area_ativa:
            area_x = st.slider("Área restrita: horizontal (%)", 0, 100, (60, 95))
            area_y = st.slider("Área restrita: vertical (%)", 0, 100, (50, 95))
            alertar_veiculos = st.checkbox("Alertar também para veículos", value=False)
    incluir_pessoas = st.checkbox("Detectar e contar pessoas", value=area_ativa, disabled=area_ativa,
                                  help="Fica ligado automaticamente quando há área restrita.")
    som = st.checkbox("Alerta sonoro", value=True)
    if eh_video:
        passo = st.slider("Analisar 1 a cada N quadros", 1, 5, 1,
                          help="Valores maiores deixam a análise mais rápida, mas o rastreamento fica menos preciso.")
        gerar_video = st.checkbox("Gerar vídeo processado para download (mais lento)")

    if fonte == FONTE_ARQUIVO:
        analisar = st.button("Analisar", type="primary", disabled=arquivo is None)
    else:
        analisar = False

    st.divider()
    st.markdown(
        "**Como usar**\n"
        "1. Escolha a fonte: arquivo (JPG/PNG/MP4 curto) ou webcam.\n"
        "2. Ajuste a confiança mínima e escolha o modelo.\n"
        "3. Ajuste a linha de contagem e, se quiser, a área restrita.\n"
        "4. Clique em **Analisar** (ou ligue a webcam).\n\n"
        "Na primeira execução, os pesos pré-treinados são baixados da internet. "
        "Depois disso ficam salvos e são reutilizados."
    )


# ---------------------------------------------------------------------------
# Página principal
# ---------------------------------------------------------------------------

st.title("ViaVision — Monitoramento de Tráfego e Segurança com IA")
st.caption(
    "Detecção de veículos e pessoas com YOLO (Ultralytics), contagem de passagens com rastreamento, "
    "alerta de invasão de área restrita e de falta de EPI (com modelo customizado). "
    "Projeto acadêmico, Opção C."
)

# Se o usuário trocou de arquivo ou de fonte, descarta o resultado antigo.
identificador = (fonte, arquivo.name, arquivo.size) if arquivo is not None else (fonte, None, None)
if st.session_state.get("arquivo_id") != identificador:
    st.session_state["arquivo_id"] = identificador
    st.session_state.pop("resultado", None)

cfg = {
    "confianca": confianca, "linha_rel": linha_pct / 100.0, "passo": passo, "gerar_video": gerar_video,
    "area_ativa": area_ativa, "som": som,
    "area": (area_x[0] / 100.0, area_y[0] / 100.0, area_x[1] / 100.0, area_y[1] / 100.0) if area_ativa else None,
    "alvos": [vision.PESSOA] + (vision.CATEGORIAS if alertar_veiculos else []),
    "epi_ativo": False,
}


def preparar_modelo():
    """Carrega o modelo e devolve (modelo, mapa filtrado, classes de EPI, categorias ativas)."""
    origem = OPCOES_MODELO[escolha]
    customizado = origem is None
    if customizado:
        origem = caminho_customizado
        if not origem.strip():
            raise vision.ErroDeModelo("Informe o caminho do arquivo .pt do modelo customizado.")
    modelo, mapa, epi = carregar(origem, customizado)
    mapa_ativo = {i: c for i, c in mapa.items() if c != vision.PESSOA or incluir_pessoas or area_ativa}
    if area_ativa and vision.PESSOA not in mapa_ativo.values():
        raise vision.ErroDeModelo("A área restrita precisa de um modelo que detecte pessoas (classe person).")
    if not mapa_ativo and not epi:
        raise vision.ErroDeModelo("Este modelo não tem nenhuma classe para detectar com as opções escolhidas.")
    cfg["epi_ativo"] = bool(epi)
    ativas = [c for c in vision.CATEGORIAS_TODAS if c in mapa_ativo.values()]
    return modelo, mapa_ativo, epi, ativas


try:
    if fonte == FONTE_ARQUIVO and analisar and arquivo is not None:
        st.session_state.pop("resultado", None)
        modelo, mapa_ativo, epi, ativas = preparar_modelo()
        if eh_video:
            novo = executar_video(arquivo, modelo, mapa_ativo, epi, cfg, ativas)
        else:
            novo = executar_imagem(arquivo, modelo, mapa_ativo, epi, confianca, ativas, cfg)
        st.session_state["resultado"] = novo
    elif fonte == FONTE_WEBCAM and ligar:
        modelo, mapa_ativo, epi, ativas = preparar_modelo()
        executar_webcam(int(indice_camera), modelo, mapa_ativo, epi, cfg, ativas)
except vision.ErroDeModelo as erro:
    st.error(f"Erro no modelo: {erro}")
except vision.ErroDeArquivo as erro:
    st.error(f"Erro: {erro}")
except Exception as erro:  # qualquer imprevisto: mostra mensagem em vez de quebrar a tela
    st.error(f"Erro inesperado ({type(erro).__name__}): {erro}")

resultado = st.session_state.get("resultado")
webcam_ligada = fonte == FONTE_WEBCAM and ligar

if resultado is not None and not (fonte == FONTE_WEBCAM and ligar):
    mostrar_resultado(resultado)
elif webcam_ligada:
    pass
elif fonte == FONTE_WEBCAM:
    st.info("Ligue a webcam na barra lateral para começar a análise ao vivo. "
            "A webcam é a do computador onde o app está rodando.")
elif arquivo is None:
    st.info("Envie uma imagem ou um vídeo MP4 na barra lateral para começar.")
elif eh_video:
    quadro = primeiro_quadro(arquivo.getvalue())
    if quadro is None:
        st.error("Não foi possível ler este vídeo. Verifique se é um MP4 válido.")
    else:
        quadro = quadro.copy()
        if area_ativa:
            vision.desenhar_area(quadro, cfg["area"])
        vision.desenhar_linha(quadro, cfg["linha_rel"])
        st.image(quadro, channels="BGR",
                 caption="Pré-visualização: linha amarela = contagem; retângulo laranja = área restrita.")
        st.write("Clique em **Analisar** na barra lateral para iniciar.")
else:
    try:
        st.image(vision.decodificar_imagem(arquivo.getvalue()), channels="BGR", caption="Imagem enviada")
        st.write("Clique em **Analisar** na barra lateral para iniciar.")
    except vision.ErroDeArquivo as erro:
        st.error(f"Erro no arquivo: {erro}")

st.divider()
with st.expander("Limitações importantes"):
    st.markdown(
        "- O rastreamento pode trocar o identificador de um objeto quando ele fica escondido atrás de outro "
        "(oclusão). Isso pode causar contagem duplicada ou perdida.\n"
        "- A **confiança** é uma pontuação interna do modelo para cada detecção; "
        "ela **não** é a acurácia do sistema. A qualidade real só é medida com o `evaluate.py` em dados anotados.\n"
        "- O alerta de **EPI** só funciona com um modelo customizado treinado para isso; o modelo "
        "pré-treinado não reconhece capacete nem colete.\n"
        "- A área restrita usa a base da caixa (os “pés”) como referência.\n"
        "- Em CPU, a análise não é em tempo real: a webcam ao vivo roda com poucos quadros por segundo.\n"
        "- A webcam é a do computador que executa o app, e o beep sonoro é do Windows."
    )
