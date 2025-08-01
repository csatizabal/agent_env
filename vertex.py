from langchain_google_vertexai import VertexAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Configura tu proyecto y ubicación
PROJECT_ID = "gen-lang-client-0004230584"
LOCATION = "us-central1"

# Inicializa el modelo de Vertex AI
llm = VertexAI(model_name="gemini-2.5-pro", project=PROJECT_ID, location=LOCATION)

# Define un prompt template
prompt_template = "Traduce la siguiente frase al francés: {texto}"
prompt = PromptTemplate(template=prompt_template, input_variables=["texto"])

# Crea una cadena
chain = LLMChain(llm=llm, prompt=prompt)

# Ejecuta la cadena
texto_a_traducir = "niña"
traduccion = chain.run(texto=texto_a_traducir)

print(traduccion)