# agent_env — Asistente Cognitivo Multimodal para Hackday 2025 (Indra)

> **Repositorio de soporte para la construcción de un agente conversacional multimodal**, desarrollado como prototipo funcional en el marco del Hackday 2025 de Indra. El agente integra visión por computadora, procesamiento de lenguaje natural, síntesis de voz y manipulación de interfaz de usuario en tiempo real.

---

## Tabla de Contenidos

- [Descripción General](#descripción-general)
- [Arquitectura del Sistema](#arquitectura-del-sistema)
- [Stack Tecnológico](#stack-tecnológico)
- [Estructura del Repositorio](#estructura-del-repositorio)
- [Requisitos Previos](#requisitos-previos)
- [Instalación y Configuración](#instalación-y-configuración)
- [Variables de Entorno](#variables-de-entorno)
- [Ejecución](#ejecución)
- [Herramientas del Agente (Tools)](#herramientas-del-agente-tools)
- [Flujos de Conversación](#flujos-de-conversación)
- [Comunicación en Tiempo Real (WebSockets)](#comunicación-en-tiempo-real-websockets)
- [Consideraciones de Seguridad](#consideraciones-de-seguridad)
- [Contribuciones](#contribuciones)
- [Autor](#autor)

---

## Descripción General

**IndraBot** es un asistente virtual de seguros construido sobre una arquitectura de agente reactivo (*ReAct Agent*) con capacidades multimodales. El sistema es capaz de:

- **Comprender texto e imágenes simultáneamente**: recibe capturas de pantalla del navegador del usuario junto con su mensaje de texto, lo que le permite contextualizar visualmente la solicitud.
- **Razonar y ejecutar herramientas**: gracias al paradigma *tool-calling* de LangChain, el agente decide autónomamente qué función ejecutar en cada turno de conversación.
- **Interactuar con la UI en tiempo real**: mediante WebSockets, el agente puede resaltar elementos, rellenar campos de formularios y navegar entre vistas de la aplicación web sin intervención manual del usuario.
- **Responder con voz sintetizada**: la respuesta textual del agente se convierte a audio MP3 mediante Google Cloud Text-to-Speech y se reproduce automáticamente en el cliente.

El dominio de negocio del prototipo es la **venta y consulta de pólizas de seguros** (auto y hogar), pero la arquitectura es agnóstica al dominio y puede adaptarse a otros casos de uso empresarial.

---

## Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────────────┐
│                        CLIENTE (Browser)                    │
│   ┌───────────────┐   WebSocket    ┌──────────────────────┐ │
│   │  HTML/JS/CSS  │ ◄────────────► │  Flask + Socket.IO   │ │
│   │  (frontend)   │                │     (Backend)        │ │
│   └───────────────┘                └──────────┬───────────┘ │
│        ▲  Captura screenshot                  │             │
│        │  Envía texto + imagen                ▼             │
│        │                          ┌──────────────────────┐  │
│        │                          │  LangChain Agent     │  │
│        │                          │  (ReAct + Tools)     │  │
│        │                          └──────────┬───────────┘  │
│        │                                     │              │
│        │                    ┌────────────────┼───────────┐  │
│        │                    ▼                ▼           ▼  │
│        │          ┌──────────────┐  ┌──────────────┐  ┌──┐ │
│        │          │ Vertex AI    │  │ Google Cloud │  │DB│ │
│        │          │ Gemini 2.5   │  │ Text-to-     │  │  │ │
│        │          │ (Multimodal) │  │ Speech       │  │  │ │
│        │          └──────────────┘  └──────────────┘  └──┘ │
│        │                                                     │
│        └─────────── Audio MP3 + Texto + Acción UI ──────────┘
└─────────────────────────────────────────────────────────────┘
```

El flujo de una interacción completa es el siguiente:

1. El cliente captura un screenshot de la página actual y lo codifica en Base64.
2. Texto + imagen se envían al servidor vía WebSocket (`mensaje_usuario`).
3. El backend construye un `HumanMessage` multimodal y lo entrega al `AgentExecutor` de LangChain.
4. El agente razona con **Gemini 2.5 Pro** (Vertex AI) y decide qué herramienta ejecutar.
5. Las herramientas pueden devolver datos o emitir eventos WebSocket hacia el cliente para manipular la UI.
6. El LLM genera la respuesta final en texto.
7. El texto pasa por **Google Cloud TTS** y se devuelve al cliente como audio MP3 en Base64.
8. El cliente muestra el texto, reproduce el audio y ejecuta las acciones de UI instruidas.

---

## Stack Tecnológico

| Capa | Tecnología | Versión recomendada |
|---|---|---|
| Backend Web | Flask + Flask-SocketIO | ≥ 3.x / ≥ 5.x |
| LLM Multimodal | Google Gemini 2.5 Pro vía Vertex AI | `gemini-2.5-pro` |
| Orquestación de Agente | LangChain | ≥ 0.2 |
| Integración GCP | `langchain-google-vertexai` | ≥ 2.x |
| Síntesis de Voz | Google Cloud Text-to-Speech | SDK oficial |
| Comunicación RT | WebSockets (Socket.IO) | - |
| Procesamiento de Imágenes | Pillow (PIL) | ≥ 10.x |
| Entorno Python | Python 3.13 (venv) | 3.13 |

---

## Estructura del Repositorio

```
agent_env/
│
├── app.py                      # Aplicación principal: agente, herramientas, Flask y WebSockets
├── vertex.py                   # Utilidades auxiliares para Vertex AI
├── verify_install.py           # Script de verificación del entorno de instalación
├── agente_web_multimodal       # Documentación o artefactos complementarios del agente
├── requirements.txt            # Dependencias del proyecto
├── pyvenv.cfg                  # Configuración del entorno virtual Python
├── .gitignore                  # Exclusiones de Git
│
├── templates/
│   └── frontend.html           # Interfaz web del asistente (HTML + JS + CSS)
│
├── static/
│   └── js/                     # Scripts JavaScript auxiliares
│
├── Include/
│   └── site/python3.13/        # Headers de extensiones nativas (greenlet)
│
└── Lib/
    └── site-packages/          # Paquetes instalados en el entorno virtual
```

> **Nota:** Las carpetas `Include/`, `Lib/` y `Scripts/` corresponden al entorno virtual Python incluido en el repositorio. En un flujo de trabajo convencional se recomienda excluirlas con `.gitignore` y reproducir el entorno mediante `requirements.txt`.

---

## Requisitos Previos

Antes de clonar y ejecutar el proyecto, asegúrese de contar con:

1. **Python 3.10+** (el proyecto fue desarrollado con 3.13).
2. **Cuenta de Google Cloud Platform** con los siguientes servicios habilitados:
   - Vertex AI API
   - Cloud Text-to-Speech API
3. **Autenticación con Google Cloud (ADC)** configurada localmente:
   ```bash
   gcloud auth application-default login
   ```
4. **Permisos IAM** sobre el proyecto GCP: roles `Vertex AI User` y `Cloud Text-to-Speech User`.

---

## Instalación y Configuración

```bash
# 1. Clonar el repositorio
git clone https://github.com/csatizabal/agent_env.git
cd agent_env

# 2. Crear y activar un entorno virtual (si no usa el incluido)
python -m venv venv
# En Linux/macOS:
source venv/bin/activate
# En Windows:
venv\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Verificar que el entorno está correctamente configurado
python verify_install.py
```

---

## Variables de Entorno

El proyecto utiliza `python-dotenv` para gestionar la configuración sensible. Cree un archivo `.env` en la raíz del proyecto con las siguientes variables:

```env
# ID del proyecto en Google Cloud Platform
GOOGLE_CLOUD_PROJECT=your-gcp-project-id

# Región de despliegue de Vertex AI
VERTEX_AI_LOCATION=us-central1
```

> **Advertencia de seguridad:** Nunca incluya el archivo `.env` en el control de versiones. Verifique que esté listado en `.gitignore`.

Si las variables no se definen, el sistema utiliza los siguientes valores por defecto (definidos en `app.py`):
- `GOOGLE_CLOUD_PROJECT`: `gen-lang-client-0004230584`
- `VERTEX_AI_LOCATION`: `us-central1`

---

## Ejecución

```bash
python app.py
```

El servidor se inicia en `http://localhost:5000`. La consola mostrará los logs de inicialización de los servicios GCP y confirmará si el agente quedó correctamente configurado.

Para verificar que todo está operativo antes de iniciar, ejecute:

```bash
python verify_install.py
```

---

## Herramientas del Agente (Tools)

El agente dispone de las siguientes herramientas (`@tool` de LangChain), que el LLM invoca autónomamente según el contexto de la conversación:

| Herramienta | Descripción |
|---|---|
| `obtener_info_producto(nombre_producto)` | Consulta la base de datos interna y retorna detalles (coberturas, precio, descripción) de una póliza específica. |
| `obtener_info_deducible()` | Retorna una explicación conceptual sobre qué es el deducible en un seguro. |
| `calcular_cotizacion(tipo_seguro, datos_usuario)` | Calcula el precio mensual de una póliza aplicando factores de ajuste según los datos del usuario (ej. antigüedad del vehículo). |
| `interactuar_con_ui(accion, selector_css, texto)` | Emite un evento WebSocket para resaltar o interactuar con elementos del DOM del cliente. |
| `navegar_a_formulario_contratacion()` | Emite un evento para cambiar la vista activa del frontend al formulario de contratación. |
| `rellenar_campo_formulario(selector_css, valor)` | Emite un evento para rellenar programáticamente un campo del formulario con los datos proporcionados por el usuario. |

---

## Flujos de Conversación

El agente sigue flujos de trabajo definidos en el *system prompt*, priorizados de mayor a menor:

1. **Flujo de Información**: activado cuando el usuario solicita conocer detalles de un producto. Resalta la sección correspondiente en la UI.
2. **Flujo de Cotización**: activado cuando el usuario expresa intención de cotizar. Resalta el botón de acción principal.
3. **Flujo de Navegación al Formulario**: activado cuando el usuario confirma querer contratar. Cambia la vista a la sección del formulario.
4. **Flujo de Relleno de Formulario**: guía al usuario campo por campo (nombre, modelo de vehículo, año, email, teléfono), rellenando el DOM automáticamente.
5. **Flujo General / Fallback**: resuelve preguntas conceptuales sobre seguros usando las herramientas de información.

---

## Comunicación en Tiempo Real (WebSockets)

La comunicación bidireccional entre cliente y servidor se establece con **Flask-SocketIO**. Los eventos definidos son:

| Evento | Dirección | Descripción |
|---|---|---|
| `mensaje_usuario` | Cliente → Servidor | Envía texto del usuario + screenshot en Base64 + `session_id`. |
| `mensaje_asistente` | Servidor → Cliente | Retorna la respuesta textual del agente + audio MP3 en Base64. |
| `accion_ui` | Servidor → Cliente | Instruye al cliente a ejecutar acciones sobre el DOM (resaltar, rellenar). |
| `cambiar_vista` | Servidor → Cliente | Instruye al cliente a mostrar u ocultar secciones de la página. |

---

## Consideraciones de Seguridad

- Las credenciales de GCP deben gestionarse exclusivamente mediante **Application Default Credentials (ADC)** o variables de entorno. No deben incluirse en el código fuente.
- La `SECRET_KEY` de Flask definida en `app.py` debe rotarse y externalizarse en producción.
- El parámetro `cors_allowed_origins="*"` en Socket.IO debe restringirse al dominio de despliegue en ambientes productivos.
- El buffer HTTP de Socket.IO está configurado en 10 MB para soportar imágenes; ajuste según las necesidades de seguridad y rendimiento.

---

## Contribuciones

Este repositorio fue desarrollado como prototipo de hackathon. Si desea contribuir, por favor abra un *issue* o un *pull request* describiendo la mejora propuesta.

---

## Autor

**César Satizábal**  
Ingeniero de Infraestructura Informática | Estudiante de Maestría en Ciencia de Datos  
[github.com/csatizabal](https://github.com/csatizabal)

---

*Desarrollado para Hackday 2025 — Indra Colombia*
