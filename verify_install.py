# verify_install.py
import sys

print("--- Python Executable ---")
print(sys.executable)
print("\n--- Python Path ---")
for p in sys.path:
    print(p)

try:
    import langchain_core
    print("\n--- Langchain Core ---")
    print("Version:", langchain_core.__version__)
    print("Path:", langchain_core.__file__)
except ImportError:
    print("\nERROR: langchain-core no está instalado.")
except AttributeError:
    print("\nERROR: La versión de langchain-core no tiene el atributo __version__.")

print("\n--- Intentando Importar ChatMessageHistory ---")

try:
    from langchain_core.chat_history import ChatMessageHistory
    print("ÉXITO: Se importó 'ChatMessageHistory' desde 'langchain_core.chat_history'")
except ImportError as e:
    print(f"FALLO: No se pudo importar desde 'langchain_core.chat_history'. Error: {e}")

try:
    from langchain.memory import ChatMessageHistory
    print("ÉXITO: Se importó 'ChatMessageHistory' desde 'langchain.memory'")
except ImportError as e:
    print(f"FALLO: No se pudo importar desde 'langchain.memory'. Error: {e}")

try:
    from langchain_community.chat_message_histories import ChatMessageHistory
    print("ÉXITO: Se importó 'ChatMessageHistory' desde 'langchain_community.chat_message_histories'")
except ImportError as e:
    print(f"FALLO: No se pudo importar desde 'langchain_community.chat_message_histories'. Error: {e}")