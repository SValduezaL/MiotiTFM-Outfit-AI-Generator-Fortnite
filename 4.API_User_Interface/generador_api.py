import websocket  # Se instala con: pip install websocket-client
import uuid
import json
import urllib.request
import urllib.parse
import random
import requests
import io
import os
from dotenv import load_dotenv

# Cargar variables de entorno desde .env
load_dotenv()

# --- 1. CONFIGURACIÓN ---
# Dirección del servidor de ComfyUI (desde variable de entorno)
COMFYUI_SERVER_ADDRESS = os.getenv("COMFYUI_SERVER_ADDRESS", "127.0.0.1:8188")
# ID de cliente único para la sesión
CLIENT_ID = str(uuid.uuid4())
# Ruta al archivo JSON de tu workflow (desde variable de entorno)
WORKFLOW_FILE_PATH = os.getenv(
    "WORKFLOW_FILE_PATH",
    "API-ComfyUI-FT_Humanoid_5e_vF__LoRA_Food_LoRA_NiceHands__SDXL_Refiner.json",
)
# URL de la API a la que quieres enviar la imagen generada (desde variable de entorno)
TARGET_API_URL = os.getenv("TARGET_API_URL", "https://httpbin.org/post")

# --- 2. FUNCIONES DE COMUNICACIÓN CON COMFYUI ---


def queue_prompt(prompt_workflow):
    """Envía el workflow a la cola de ComfyUI para su procesamiento."""
    try:
        p = {"prompt": prompt_workflow, "client_id": CLIENT_ID}
        data = json.dumps(p).encode("utf-8")
        req = urllib.request.Request(
            f"http://{COMFYUI_SERVER_ADDRESS}/prompt", data=data
        )
        response = urllib.request.urlopen(req)
        return json.loads(response.read())
    except Exception as e:
        print(f"Error al poner el prompt en la cola: {e}")
        return None


def get_image(filename, subfolder, folder_type):
    """Obtiene la imagen generada desde el servidor de ComfyUI."""
    try:
        data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        url_values = urllib.parse.urlencode(data)
        with urllib.request.urlopen(
            f"http://{COMFYUI_SERVER_ADDRESS}/view?{url_values}"
        ) as response:
            return response.read()
    except Exception as e:
        print(f"Error al obtener la imagen: {e}")
        return None


def get_history(prompt_id):
    """Obtiene el historial de ejecución de un prompt específico."""
    try:
        with urllib.request.urlopen(
            f"http://{COMFYUI_SERVER_ADDRESS}/history/{prompt_id}"
        ) as response:
            return json.loads(response.read())
    except Exception as e:
        print(f"Error al obtener el historial: {e}")
        return None


def track_progress_and_get_images(prompt_workflow):
    """
    Función principal que maneja la comunicación con ComfyUI a través de WebSockets.
    Pone el workflow en cola, espera a que termine y devuelve los datos de las imágenes.
    """
    prompt_id = queue_prompt(prompt_workflow)["prompt_id"]

    ws_url = f"ws://{COMFYUI_SERVER_ADDRESS}/ws?clientId={CLIENT_ID}"
    ws = websocket.WebSocket()
    ws.connect(ws_url)

    print(f"Workflow enviado. Prompt ID: {prompt_id}")

    while True:
        try:
            out_json = ws.recv()
            if isinstance(out_json, str):
                message = json.loads(out_json)

                if message["type"] == "progress":
                    progress = message["data"]
                    print(f"Progreso: {progress['value']}/{progress['max']} pasos")

                elif message["type"] == "executing":
                    data = message["data"]
                    # Si el nodo es None, es el final de la cola
                    if data["node"] is None and data["prompt_id"] == prompt_id:
                        print("Ejecución finalizada.")
                        ws.close()
                        break
        except websocket.WebSocketConnectionClosedException:
            print("Conexión WebSocket cerrada.")
            break
        except Exception as e:
            print(f"Error en WebSocket: {e}")
            break

    # Una vez que la ejecución termina, obtenemos el historial para encontrar los nombres de archivo
    history = get_history(prompt_id)[prompt_id]
    image_data_list = []

    for node_id in history["outputs"]:
        node_output = history["outputs"][node_id]
        if "images" in node_output:
            for image in node_output["images"]:
                print(f"Imagen generada: {image['filename']}")
                image_data = get_image(
                    image["filename"], image["subfolder"], image["type"]
                )
                if image_data:
                    image_data_list.append((image["filename"], image_data))

    return image_data_list


# --- 3. FUNCIÓN PARA ENVIAR A LA API EXTERNA ---


def send_image_to_api(filename, image_bytes):
    """Envía los bytes de una imagen a una API de destino."""
    print(f"Enviando '{filename}' a la API en {TARGET_API_URL}...")
    try:
        # 'files' es el diccionario donde preparamos el archivo para el envío multipart/form-data
        # La clave 'image' es el nombre del campo que la API de destino espera.
        files = {"image": (filename, io.BytesIO(image_bytes), "image/png")}

        # Puedes añadir datos adicionales si tu API lo requiere
        payload = {"source": "ComfyUI_Python_Script", "model": "FortniteHumanoidFood"}

        response = requests.post(TARGET_API_URL, files=files, data=payload)

        # Comprobar si la petición fue exitosa
        response.raise_for_status()

        print("¡Imagen enviada con éxito!")
        print("Respuesta de la API:")
        print(response.json())

    except requests.exceptions.RequestException as e:
        print(f"Error al enviar la imagen a la API: {e}")


# --- 4. SCRIPT PRINCIPAL ---

if __name__ == "__main__":
    # Cargar el workflow desde el archivo JSON
    with open(WORKFLOW_FILE_PATH, "r") as f:
        prompt_workflow = json.load(f)

    # --- MODIFICACIÓN DINÁMICA DEL WORKFLOW ---
    # Aquí es donde puedes cambiar los valores antes de ejecutar.
    # Los nodos se identifican por su 'id' en el archivo JSON.

    # ID 13: Positive Prompt (Text)
    prompt_workflow["13"]["inputs"][
        "text"
    ] = "fortnite_food_character, nice_hands, a knight made of cheese, holding a sword made of bread, fantasy art, epic, cinematic lighting"

    # ID 14: Negative Prompt (Text)
    prompt_workflow["14"]["inputs"][
        "text"
    ] = "ugly, blurry, bad anatomy, text, watermark"

    # ID 10: KSampler (Base) - Cambiamos la semilla (seed) a un valor aleatorio
    prompt_workflow["10"]["inputs"]["seed"] = random.randint(1, 1_000_000_000)

    # ID 19: SaveImage - Cambiamos el prefijo del nombre de archivo
    prompt_workflow["19"]["inputs"]["filename_prefix"] = "queso_knight_api"

    print("Workflow modificado. Iniciando generación...")

    # Ejecutar el workflow y obtener las imágenes generadas
    generated_images = track_progress_and_get_images(prompt_workflow)

    # Procesar y enviar cada imagen generada
    if generated_images:
        print(f"\nSe generaron {len(generated_images)} imágenes. Enviando a la API...")
        for filename, image_data in generated_images:
            send_image_to_api(filename, image_data)
    else:
        print("No se generaron imágenes o hubo un error.")
