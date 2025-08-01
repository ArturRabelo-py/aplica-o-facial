# --- ETAPA 0: CONFIGURAÇÃO DE AMBIENTE E IMPORTS ---
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import dash
from dash import dcc, html, Input, Output, State, ALL
import dash_bootstrap_components as dbc
from PIL import Image
import numpy as np
import cv2
from deepface import DeepFace
from scipy.spatial.distance import cosine
import base64
import io
import time
import glob
from tqdm import tqdm
import imgaug.augmenters as iaa

# --- 1. CONFIGURAÇÕES GLOBAIS ---
# Todas as configurações principais da aplicação estão centralizadas aqui.

# Define o caminho absoluto para a pasta de cadastros, garantindo que funcione em qualquer diretório.
PASTA_CADASTROS = os.path.abspath('./cadastros')
MODELO_RECONHECIMENTO = 'ArcFace'
# O sistema tentará os detectores nesta ordem se o anterior falhar.
DETECTORES_A_TENTAR = ['mtcnn', 'opencv', 'ssd']
LIMIAR_VERIFICACAO = 0.4
NUMERO_DE_AUMENTACOES_POR_IMAGEM = 10

# --- Sequência de Aumentação de Dados ---
seq = iaa.Sequential([
    iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 0.2))),
    iaa.LinearContrast((0.85, 1.15)),
    iaa.Multiply((0.9, 1.1), per_channel=0.2),
], random_order=True)

# --- "Aquecimento" dos Modelos ---
print("=" * 50)
print("INICIALIZANDO SERVIDOR WEB...")
print(f"A procurar por ficheiros de identidade em: {PASTA_CADASTROS}")
try:
    # Aquece o modelo de reconhecimento e o primeiro detector da lista
    DeepFace.represent(img_path=np.zeros((100, 100, 3), dtype=np.uint8), model_name=MODELO_RECONHECIMENTO,
                       detector_backend=DETECTORES_A_TENTAR[0], enforce_detection=False)
    print("Modelos de reconhecimento prontos.")
    print("=" * 50)
except Exception as e:
    print(f"ERRO FATAL ao carregar modelos: {e}")
    exit()


# --- 2. FUNÇÕES AUXILIARES ---
def parse_contents(contents):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    if 'image' in content_type:
        image = Image.open(io.BytesIO(decoded))
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    elif 'csv' in content_type or 'text' in content_type:
        return decoded
    return None


def get_lista_identidades():
    # Garante que a pasta de cadastros existe
    if not os.path.exists(PASTA_CADASTROS):
        os.makedirs(PASTA_CADASTROS)
    ficheiros_csv = glob.glob(os.path.join(PASTA_CADASTROS, '*.csv'))
    opcoes = []
    for f in ficheiros_csv:
        nome_ficheiro = os.path.basename(f)
        nome_utilizador = nome_ficheiro.replace('assinatura_', '').replace('.csv', '')
        opcoes.append({'label': nome_utilizador.capitalize(), 'value': nome_ficheiro})
    return opcoes


# --- 3. INICIALIZAÇÃO E LAYOUT DA APLICAÇÃO DASH ---
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SOLAR, dbc.icons.BOOTSTRAP])

server = app.server

# --- Layout da Aba de Cadastro ---
rótulos_cadastro = [
    "Sério", "Sorrindo", "Rindo", "Olhos bem abertos", "Olhos bem fechados",
    "Olhando para a esquerda", "Olhando para a direita", "Olhando para cima",
    "Olhando para baixo", "Foto afastada"
]
caixas_upload_cadastro = [
    dbc.Col([
        html.Label(rótulo, style={'font-size': '0.9em'}),
        dcc.Upload(id={'type': 'upload-cadastro', 'index': i},
                   children=html.Div(['+'], style={'font-size': '2em', 'color': 'gray'}),
                   style={'width': '100px', 'height': '100px', 'lineHeight': '100px', 'borderWidth': '1px',
                          'borderStyle': 'dashed', 'borderRadius': '5px', 'textAlign': 'center', 'margin': 'auto'},
                   multiple=False)
    ], width="auto", className="text-center mb-3") for i, rótulo in enumerate(rótulos_cadastro)
]
tab_cadastro = dbc.Card(dbc.CardBody([
    html.H4("Cadastrar Nova Identidade a partir de Fotos", className="card-title"),
    dbc.Input(id="input-nome-cadastro", placeholder="Digite o nome do utilizador", type="text", className="mb-3"),
    html.P("Envie uma foto para cada uma das 10 expressões faciais abaixo:"),
    dbc.Row(caixas_upload_cadastro, justify="center"),
    dbc.Button("Iniciar Cadastro por Fotos", id="btn-cadastrar-fotos", color="primary", className="mt-3", n_clicks=0),
    dcc.Loading(id="loading-cadastro-fotos", type="circle",
                children=html.Div(id='output-cadastro-fotos-status', className="mt-3")),
    html.Hr(),
    html.H4("Cadastrar Nova Identidade a partir de Ficheiro CSV", className="card-title mt-4"),
    dbc.Input(id="input-nome-csv", placeholder="Digite o nome para esta identidade", type="text", className="mb-3"),
    dcc.Upload(id='upload-csv-cadastro',
               children=html.Div(['Arraste e solte ou ', html.A('Selecione um Ficheiro CSV')]),
               style={'width': '100%', 'height': '60px', 'lineHeight': '60px', 'borderWidth': '1px',
                      'borderStyle': 'dashed', 'borderRadius': '5px', 'textAlign': 'center', 'margin': '10px 0'},
               multiple=False),
    dbc.Button("Iniciar Cadastro por CSV", id="btn-cadastrar-csv", color="success", className="mt-3", n_clicks=0),
    dcc.Loading(id="loading-cadastro-csv", type="circle",
                children=html.Div(id='output-cadastro-csv-status', className="mt-3"))
]))

# --- Layout da Aba de Verificação ---
tab_verificacao = dbc.Card(dbc.CardBody([
    html.H4("Verificar Identidade", className="card-title"),
    html.P("Selecione uma identidade registada e envie uma foto para verificação."),
    dcc.Dropdown(id='dropdown-identidades', placeholder="Selecione uma identidade...", options=get_lista_identidades()),
    html.Hr(),
    dcc.Upload(id='upload-image-verificacao',
               children=html.Div(['Arraste e solte ou ', html.A('Selecione uma Imagem')]),
               style={'width': '100%', 'height': '60px', 'lineHeight': '60px', 'borderWidth': '1px',
                      'borderStyle': 'dashed', 'borderRadius': '5px', 'textAlign': 'center', 'margin': '10px 0'},
               multiple=False),
    dcc.Loading(id="loading-verificacao", type="circle",
                children=html.Div(id='output-verificacao-deteccao', className="mt-4")),
    html.Div(id='output-verificacao-resultado', className="mt-3")
]))

# --- Layout Principal com Abas ---
app.layout = dbc.Container([
    dcc.Store(id='store-verificacao-embeddings'),  # Armazenamento para os embeddings da verificação
    html.H1("Sistema de Reconhecimento Facial", className="text-center text-primary my-4"),
    dbc.Tabs([dbc.Tab(tab_verificacao, label="Verificação"), dbc.Tab(tab_cadastro, label="Cadastramento")])
], fluid=True)


# --- 4. LÓGICA DE BACKEND (CALLBACKS) ---

# Callback para feedback visual do upload de CSV
@app.callback(Output('upload-csv-cadastro', 'children'), Input('upload-csv-cadastro', 'filename'),
              prevent_initial_call=True)
def update_csv_upload_label(filename):
    if filename:
        return html.Div([html.I(className="bi bi-check-circle-fill me-2", style={'color': 'green'}),
                         f"Ficheiro selecionado: {filename}"])
    return html.Div(['Arraste e solte ou ', html.A('Selecione um Ficheiro CSV')])


# Callback para atualizar a lista de identidades
@app.callback(Output('dropdown-identidades', 'options'),
              [Input('output-cadastro-fotos-status', 'children'), Input('output-cadastro-csv-status', 'children')])
def update_dropdown_options(status_fotos, status_csv):
    return get_lista_identidades()


# Callback para CADASTRO POR FOTOS
@app.callback(Output('output-cadastro-fotos-status', 'children'), Input('btn-cadastrar-fotos', 'n_clicks'),
              [State('input-nome-cadastro', 'value'), State({'type': 'upload-cadastro', 'index': ALL}, 'contents')],
              prevent_initial_call=True)
def processar_cadastro_fotos(n_clicks, nome_utilizador, lista_contents):
    if not nome_utilizador or not all(lista_contents):
        return dbc.Alert("Por favor, preencha o nome e envie todas as 10 imagens.", color="warning")
    nome_ficheiro = f"assinatura_{nome_utilizador.lower().replace(' ', '_')}.csv"
    caminho_saida = os.path.join(PASTA_CADASTROS, nome_ficheiro)
    lista_embeddings, sucessos, falhas = [], 0, 0
    for contents in tqdm(lista_contents, desc=f"Processando cadastro de {nome_utilizador}"):
        imagem = parse_contents(contents)
        imagens_para_processar = [imagem] + seq(images=[imagem] * NUMERO_DE_AUMENTACOES_POR_IMAGEM)
        for img_proc in imagens_para_processar:
            try:
                resultado = DeepFace.represent(img_path=img_proc, model_name=MODELO_RECONHECIMENTO,
                                               detector_backend=DETECTORES_A_TENTAR[0], enforce_detection=True)
                lista_embeddings.append(resultado[0]["embedding"])
                sucessos += 1
            except:
                falhas += 1
    if not lista_embeddings:
        return dbc.Alert("Falha no cadastro. Nenhum rosto foi detetado.", color="danger")
    np.savetxt(caminho_saida, np.asarray(lista_embeddings), delimiter=',')
    return dbc.Alert(f"Cadastro por fotos concluído para '{nome_utilizador}'. Sucessos: {sucessos}, Falhas: {falhas}.",
                     color="success")


# Callback para CADASTRO POR CSV
@app.callback(Output('output-cadastro-csv-status', 'children'), Input('btn-cadastrar-csv', 'n_clicks'),
              [State('input-nome-csv', 'value'), State('upload-csv-cadastro', 'contents')], prevent_initial_call=True)
def processar_cadastro_csv(n_clicks, nome_utilizador, contents):
    if not nome_utilizador or not contents:
        return dbc.Alert("Por favor, preencha o nome e envie um ficheiro CSV.", color="warning")
    nome_ficheiro = f"assinatura_{nome_utilizador.lower().replace(' ', '_')}.csv"
    caminho_saida = os.path.join(PASTA_CADASTROS, nome_ficheiro)
    try:
        dados_csv = parse_contents(contents)
        with open(caminho_saida, 'wb') as f:
            f.write(dados_csv)
        return dbc.Alert(f"Identidade '{nome_utilizador}' cadastrada com sucesso a partir do ficheiro CSV!",
                         color="success")
    except Exception as e:
        return dbc.Alert(f"Ocorreu um erro ao guardar o ficheiro CSV: {e}", color="danger")


# --- Callbacks da Aba de Verificação ---

# Callback 1 (Verificação): Deteta todos os rostos e mostra para seleção.
@app.callback(
    [Output('output-verificacao-deteccao', 'children'),
     Output('store-verificacao-embeddings', 'data')],
    Input('upload-image-verificacao', 'contents'),
    prevent_initial_call=True
)
def detectar_rostos_para_verificacao(contents):
    if contents is None:
        return [], None
    imagem_teste = parse_contents(contents)

    # Lógica de fallback para deteção de rosto
    resultados = None
    for detector in DETECTORES_A_TENTAR:
        try:
            print(f"[INFO] A tentar detetar rostos com o detector: {detector}...")
            resultados = DeepFace.represent(
                img_path=imagem_teste,
                model_name=MODELO_RECONHECIMENTO,
                detector_backend=detector,
                enforce_detection=True
            )
            print(f"[INFO] Rosto(s) detetado(s) com sucesso usando {detector}.")
            break
        except Exception as e:
            if 'Face could not be detected' in str(e):
                print(f"[AVISO] Detector '{detector}' não encontrou um rosto. A tentar o próximo...")
                continue
            else:
                return dbc.Alert(f"Ocorreu um erro inesperado: {e}", color="danger"), None

    if not resultados:
        return dbc.Alert(
            "Não foi possível detetar um rosto na imagem enviada, mesmo com múltiplos detectores. Por favor, tente com uma foto mais nítida e bem iluminada.",
            color="warning"), None

    embeddings_encontrados, miniaturas_para_layout = [], []
    for i, rosto in enumerate(resultados):
        embeddings_encontrados.append(rosto['embedding'])
        r = rosto['facial_area']
        miniatura_rosto = imagem_teste[r['y']:r['y'] + r['h'], r['x']:r['x'] + r['w']]
        _, buffer = cv2.imencode('.png', miniatura_rosto)
        miniatura_b64 = base64.b64encode(buffer).decode('utf-8')
        miniaturas_para_layout.append(
            dbc.Button(
                html.Img(src=f'data:image/png;base64,{miniatura_b64}', style={'height': '100px', 'width': '100px'}),
                id={'type': 'btn-verificar-rosto', 'index': i}, n_clicks=0, className="m-1", color="secondary")
        )
    layout_deteccao = dbc.Row([
        dbc.Col(html.Img(src=contents, style={'max-height': '400px', 'max-width': '100%'}), md=8),
        dbc.Col(
            [html.H5("Rostos Detetados:"), html.P("Clique no rosto a verificar."), html.Div(miniaturas_para_layout)],
            md=4)
    ], className="align-items-center mt-4")
    return layout_deteccao, embeddings_encontrados


# Callback 2 (Verificação): Realiza a verificação quando uma miniatura é clicada OU limpa o resultado quando uma nova foto é enviada.
@app.callback(
    Output('output-verificacao-resultado', 'children'),
    [Input({'type': 'btn-verificar-rosto', 'index': ALL}, 'n_clicks'),
     Input('upload-image-verificacao', 'contents')],
    [State('dropdown-identidades', 'value'),
     State('store-verificacao-embeddings', 'data')],
    prevent_initial_call=True
)
def processar_verificacao_selecionada(n_clicks, contents, identidade_selecionada, stored_embeddings):
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id']

    if 'upload-image-verificacao' in triggered_id:
        return []

    if not any(n_clicks) or not identidade_selecionada or not stored_embeddings:
        if 'btn-verificar-rosto' in triggered_id:
            return dbc.Alert("Por favor, selecione primeiro uma identidade na lista.", color="warning")
        return dash.no_update

    button_id = triggered_id.split('.')[0]
    index_clicado = eval(button_id)['index']
    assinatura_selecionada = np.asarray(stored_embeddings[index_clicado])

    try:
        caminho_assinatura = os.path.join(PASTA_CADASTROS, identidade_selecionada)
        assinaturas_cadastradas = np.loadtxt(caminho_assinatura, delimiter=',')
        if assinaturas_cadastradas.ndim == 1:
            assinaturas_cadastradas = np.expand_dims(assinaturas_cadastradas, axis=0)
        vetor_medio_cadastrado = np.mean(assinaturas_cadastradas, axis=0)

        distancia = cosine(vetor_medio_cadastrado, assinatura_selecionada)
        if distancia <= LIMIAR_VERIFICACAO:
            resultado_texto, cor_alerta = "IDENTIDADE VERIFICADA", "success"
        else:
            resultado_texto, cor_alerta = "IDENTIDADE NÃO CORRESPONDE", "danger"
        return dbc.Alert([html.H5(resultado_texto), f"Distância de similaridade: {distancia:.4f}"], color=cor_alerta,
                         className="mt-3")
    except Exception as e:
        return dbc.Alert(f"Ocorreu um erro durante a verificação: {e}", color="danger")


# --- 5. EXECUTAR O SERVIDOR ---
if __name__ == '__main__':
    app.run(debug=True, port=8050)
