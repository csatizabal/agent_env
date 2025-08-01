# ==============================================================================
# 0. Importaciones y Configuración Inicial
# ==============================================================================
import os
import logging
import json
import base64
from io import BytesIO
from PIL import Image

# Cargar variables de entorno (opcional, pero buena práctica)
from dotenv import load_dotenv
load_dotenv()

# --- Librerías de Backend y Comunicación ---
from flask import Flask, render_template_string, request
from flask_socketio import SocketIO, emit

# --- Librerías de Langchain ---
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

# --- Integración con Google Cloud (Vertex AI y Text-to-Speech) ---
from langchain_google_vertexai import ChatVertexAI
from google.cloud import texttospeech

# --- Agente de Langchain ---
from langchain.agents import AgentExecutor, create_tool_calling_agent

# --- Configuración de Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ==============================================================================
# 1. Configuración de Servicios de Google Cloud y Langchain
# ==============================================================================

# --- Configuración del Proyecto ---
GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT", "gen-lang-client-0004230584")
VERTEX_AI_LOCATION = os.getenv("VERTEX_AI_LOCATION", "us-central1")
GEMINI_MODEL_NAME = "gemini-2.5-pro" # Modelo multimodal recomendado

# --- Inicialización del LLM (Modelo de Lenguaje) ---
try:
    vertex_ai_llm = ChatVertexAI(
        model_name=GEMINI_MODEL_NAME,
        project=GOOGLE_CLOUD_PROJECT,
        location=VERTEX_AI_LOCATION,
        temperature=0.7,
        # Habilitamos el streaming para futuras mejoras de UX
        streaming=True 
    )
    logging.info(f"ChatVertexAI inicializado exitosamente con el modelo: {GEMINI_MODEL_NAME}")
except Exception as e:
    logging.error("Error fatal al inicializar ChatVertexAI. Verifica las credenciales de ADC y la configuración del proyecto.", exc_info=True)
    vertex_ai_llm = None

# --- Inicialización del Cliente de Text-to-Speech (TTS) ---
try:
    tts_client = texttospeech.TextToSpeechClient()
    logging.info("Cliente de Google Cloud Text-to-Speech inicializado.")
except Exception as e:
    logging.error("Error al inicializar el cliente de TTS.", exc_info=True)
    tts_client = None

# ==============================================================================
# 2. Definición de Herramientas (Tools) para el Agente de IA
# ==============================================================================
# Estas son las funciones que el LLM puede decidir ejecutar para cumplir con la solicitud del usuario.

# --- Base de datos simulada ---
mock_seguros_db = {
    "seguro_auto_estandar": {"nombre": "Póliza Estándar de Auto", "id_html": "poliza-auto-estandar", "precio_mes": 25000, "coberturas": ["Robo", "Incendio"], "descripcion": "Cobertura básica para tu vehículo."},
    "seguro_auto_premium": {"nombre": "Póliza Premium de Auto", "id_html": "poliza-auto-premium", "precio_mes": 35000, "coberturas": ["Robo", "Incendio", "Asistencia 24/7", "Conductor Elegido"], "descripcion": "Cobertura completa con beneficios adicionales."},
    "seguro_hogar_basico": {"nombre": "Seguro de Hogar Básico", "id_html": "poliza-hogar-basico", "precio_mes": 15000, "coberturas": ["Incendio", "Robo"], "descripcion": "Protección esencial contra incendios y robos."}
}
mock_deducible_info = "El deducible es la cantidad fija que tú pagas de tu bolsillo antes de que el seguro comience a cubrir los gastos. Por ejemplo, en un siniestro de $1000 con un deducible de $200, tú pagas $200 y la aseguradora los $800 restantes."

@tool
def obtener_info_producto(nombre_producto: str) -> str:
    """Busca información detallada sobre un producto de seguro específico (ej. 'seguro_auto_estandar'). Utiliza esta herramienta si el usuario pregunta sobre precios, coberturas o detalles de una póliza que ve en pantalla."""
    logging.info(f"Tool 'obtener_info_producto' llamada con: {nombre_producto}")
    producto_key = nombre_producto.lower().replace(" ", "_")
    producto = mock_seguros_db.get(producto_key)
    return json.dumps(producto) if producto else f"No se encontró información para el producto '{nombre_producto}'."

@tool
def obtener_info_deducible() -> str:
    """Proporciona información general sobre qué es el 'deducible' en un seguro. Útil cuando el usuario tiene dudas sobre terminología de seguros."""
    logging.info("Tool 'obtener_info_deducible' llamada.")
    return mock_deducible_info

@tool
def calcular_cotizacion(tipo_seguro: str, datos_usuario: dict) -> str:
    """Calcula el precio de una cotización de seguro basándose en el tipo de seguro y los datos del usuario. Por ejemplo, para un seguro de auto, los datos pueden incluir 'modelo_vehiculo' y 'año'."""
    logging.info(f"Tool 'calcular_cotizacion' llamada con: {tipo_seguro}, {datos_usuario}")
    base_price = mock_seguros_db.get(tipo_seguro, {}).get("precio_mes", 30000)
    # Lógica de cotización simplificada
    factor_ajuste = 1.0
    if datos_usuario.get("año", 2024) < 2020:
        factor_ajuste += 0.15 # 15% más caro para autos más antiguos
    precio_final = base_price * factor_ajuste
    return json.dumps({"tipo_seguro": tipo_seguro, "precio_calculado": f"${precio_final:,.0f} COP/mes"})

@tool
def interactuar_con_ui(accion: str, selector_css: str, texto: str = None) -> str:
    """
    Ejecuta una acción en la interfaz de usuario del frontend.
    Acciones válidas: 'resaltar', 'mostrar_tooltip'.
    'selector_css' es el identificador del elemento en el HTML (ej. '#poliza-auto-premium').
    'texto' es opcional, usado por 'mostrar_tooltip'.
    """
    logging.info(f"Tool 'interactuar_con_ui' llamada con accion: {accion}, selector: {selector_css}")
    # ¡Clave! En lugar de retornar un string, emitimos un evento al frontend.
    socketio.emit('accion_ui', {
        'accion': accion,
        'selector': selector_css,
        'texto': texto
    })
    return f"Acción '{accion}' ejecutada en el elemento '{selector_css}' de la UI."

# Lista de todas las herramientas disponibles para el agente
tools = [obtener_info_producto, obtener_info_deducible, calcular_cotizacion, interactuar_con_ui]

# ==============================================================================
# 3. Configuración del Agente de Langchain (Cerebro de la IA)
# ==============================================================================

# --- Plantilla de Prompt ---
# Este es el "alma" del asistente. Define su personalidad, objetivos e instrucciones.
# ¡NUEVO! Le indicamos explícitamente que es un asistente multimodal.
agent_prompt_template = ChatPromptTemplate.from_messages([
    ("system", """
    Eres un Asistente Cognitivo Multimodal experto en seguros para Indra. Tu nombre es 'IndraBot'.
    Tu misión es guiar al usuario de forma proactiva y amigable a través del proceso de selección y compra de seguros.

    Recibirás dos tipos de información del usuario en cada turno:
    1.  Un comando de texto (transcrito de su voz o escrito).
    2.  Una imagen (captura de la pantalla actual del usuario).

    Tu proceso de pensamiento debe ser:
    1.  **Analiza la imagen** para entender en qué parte de la página web está el usuario (página de inicio, comparación de productos, formulario, etc.) e identificar los elementos visibles (póizas, botones, campos de texto).
    2.  **Analiza el texto del usuario** para comprender su pregunta o intención directa.
    3.  **Combina ambos contextos**. La imagen te da el 'dónde' y el texto te da el 'qué'.
    4.  **Usa tus herramientas** para responder preguntas, obtener información, interactuar con la UI (resaltando elementos) o calcular cotizaciones.
    5.  **Responde de forma clara y concisa**. Si realizas una acción en la UI, menciónalo (ej. "Claro, te resalto la póliza premium en la pantalla.").
    
    Sé siempre servicial y profesional.
    """),
    MessagesPlaceholder(variable_name="chat_history"),
    # Aquí se insertará la entrada multimodal (texto + imagen)
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# --- Creación del Agente y el Executor ---
# El 'AgentExecutor' es el motor que ejecuta el ciclo: LLM -> Tool -> LLM -> ...
if vertex_ai_llm:
    agent = create_tool_calling_agent(vertex_ai_llm, tools, agent_prompt_template)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
else:
    agent_executor = None

# --- Gestión de Memoria (Historial de Conversación) ---
message_history_dict = {}

def get_chat_history(session_id: str):
    if session_id not in message_history_dict:
        message_history_dict[session_id] = ChatMessageHistory()
    return message_history_dict[session_id]

# Envolvemos el agente con la gestión de memoria
if agent_executor:
    chain_con_agente_y_memoria = RunnableWithMessageHistory(
        agent_executor,
        get_chat_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )
else:
    chain_con_agente_y_memoria = None
    logging.error("AgentExecutor no pudo ser configurado. El asistente no funcionará.")

# ==============================================================================
# 4. Aplicación Flask y Lógica de WebSockets
# ==============================================================================

app = Flask(__name__)
app.config['SECRET_KEY'] = 'indra-hackathon-2024-secret!'
socketio = SocketIO(app, cors_allowed_origins="*", max_http_buffer_size=10 * 1024 * 1024) # Aumentar buffer para imágenes

@app.route('/')
@app.route('/')
def index():
    try:
        # 1. Obtiene la ruta absoluta del directorio donde se encuentra este script (app.py)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 2. Construye la ruta completa hacia frontend.html
        html_path = os.path.join(script_dir, "templates/frontend.html")
        
        # 3. Abre el archivo usando la ruta absoluta y completa
        logging.info(f"Intentando abrir la plantilla desde la ruta: {html_path}")
        with open(html_path, "r", encoding="utf-8") as f:
            return render_template_string(f.read())
            
    except FileNotFoundError:
        logging.error(f"¡CRÍTICO! No se pudo encontrar 'frontend.html' en la ruta esperada: {html_path}")
        return "Error 500: No se pudo encontrar el archivo de la interfaz. Verifique los logs del servidor.", 500

@socketio.on('mensaje_usuario')
def handle_mensaje_usuario(data):
    # ¡Punto clave de la multimodalidad!
    mensaje_texto = data.get('mensaje')
    screenshot_b64 = data.get('screenshot') # La imagen viene como string Base64
    session_id = data.get('session_id', 'default_session')
    
    logging.info(f"[{session_id}] Mensaje recibido: '{mensaje_texto}' con screenshot.")

    if not chain_con_agente_y_memoria:
        emit('mensaje_asistente', {"respuesta_texto": "Lo siento, el asistente no está disponible en este momento."}, room=request.sid)
        return

    try:
        # Construimos el input multimodal para Langchain
        # Esto es lo que permite a Gemini "ver" y "leer" al mismo tiempo.
        input_multimodal = HumanMessage(
            content=[
                {"type": "text", "text": mensaje_texto},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"},
                },
            ]
        )
        
        input_data = {"input": input_multimodal}
        config = {"configurable": {"session_id": session_id}}
        
        # Invocamos al agente
        result = chain_con_agente_y_memoria.invoke(input_data, config=config)
        respuesta_final_agente = result.get("output", "No he podido procesar tu solicitud. Intenta de nuevo.")

        # --- Lógica de Text-to-Speech (TTS) ---
        audio_content_b64 = None
        if tts_client and respuesta_final_agente:
            synthesis_input = texttospeech.SynthesisInput(text=respuesta_final_agente)
            voice = texttospeech.VoiceSelectionParams(language_code="es-US", ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL)
            audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
            
            response_tts = tts_client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
            audio_content_b64 = base64.b64encode(response_tts.audio_content).decode('utf-8')
            logging.info(f"[{session_id}] Audio generado para la respuesta.")

        # Emitimos la respuesta completa al frontend
        emit('mensaje_asistente', {
            "respuesta_texto": respuesta_final_agente,
            "audio_b64": audio_content_b64
        }, room=request.sid)

    except Exception as e:
        logging.error(f"[{session_id}] Error ejecutando el agente Langchain: {e}", exc_info=True)
        emit('mensaje_asistente', {"respuesta_texto": "Ha ocurrido un error inesperado al procesar tu solicitud."}, room=request.sid)

# ==============================================================================
# 5. Ejecución de la Aplicación
# ==============================================================================
if __name__ == '__main__':
    if not GOOGLE_CLOUD_PROJECT:
        logging.warning("ADVERTENCIA: La variable de entorno GOOGLE_CLOUD_PROJECT no está configurada. El script podría fallar.")
    
    if vertex_ai_llm is None or tts_client is None:
        logging.error("El servidor no puede iniciarse; uno o más servicios de Google Cloud no se inicializaron correctamente.")
    else:
        logging.info("Iniciando servidor Flask con Socket.IO en http://localhost:5000")
        socketio.run(app, debug=True, host='0.0.0.0', port=5000)
