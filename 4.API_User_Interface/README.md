# API y Interfaz de Usuario - SkinCraft

Esta carpeta contiene el sistema completo de API y interfaz de usuario para la generación de skins estilo Fortnite mediante IA. El sistema integra OpenAI para clasificación automática de personajes y ComfyUI para la generación de imágenes.

## 📋 Índice

1. [Arquitectura General](#arquitectura-general)
2. [Componentes Principales](#componentes-principales)
3. [Configuración](#configuración)
4. [Uso de la API](#uso-de-la-api)
5. [Interfaz Web](#interfaz-web)
6. [Workflows de ComfyUI](#workflows-de-comfyui)
7. [Flujo de Generación](#flujo-de-generación)

---

## 🏗️ Arquitectura General

El sistema está compuesto por tres componentes principales:

```
┌─────────────────┐
│   WebUI (HTML)  │  ← Interfaz de usuario
└────────┬────────┘
         │ HTTP Requests
         ▼
┌─────────────────┐
│  FastAPI (main) │  ← API REST con clasificación IA
└────────┬────────┘
         │
         ├──► OpenAI API (traducción y clasificación)
         │
         └──► ComfyClient ──► ComfyUI Server (generación de imágenes)
```

---

## 📦 Componentes Principales

### 1. `main.py` - API FastAPI

**Descripción:** Servidor API REST que gestiona las peticiones de generación de skins.

**Funcionalidades principales:**

-   **Clasificación automática con IA:**

    -   Utiliza OpenAI GPT-4o para traducir prompts a inglés
    -   Clasifica automáticamente el personaje en categorías: `Animal`, `Robot`, `Star Wars`, `Fuzzy Bear`, `Humanoid`, `Food`
    -   Selecciona automáticamente el workflow de ComfyUI apropiado según la categoría

-   **Endpoints disponibles:**

    -   `GET /` - Health check y estado del servicio
    -   `GET /workflows` - Lista de workflows disponibles
    -   `POST /generate` - Iniciar generación de skin
    -   `GET /status/{task_id}` - Consultar estado de una generación
    -   `GET /trace/{task_id}` - Ver página de traza de generación
    -   `GET /api/trace/{task_id}` - Obtener datos JSON de la traza
    -   `GET /image/{task_id}` - Descargar imagen generada
    -   `GET /tasks` - Listar todas las tareas
    -   `DELETE /task/{task_id}` - Eliminar tarea

-   **Características:**
    -   Procesamiento asíncrono con background tasks
    -   Trazabilidad completa de cada generación
    -   Sistema de progreso en tiempo real
    -   Manejo de errores robusto

**Dependencias:**

-   `fastapi` - Framework web
-   `openai` - Cliente de OpenAI
-   `python-dotenv` - Gestión de variables de entorno
-   `comfy_client` - Cliente personalizado para ComfyUI

---

### 2. `comfy_client.py` - Cliente ComfyUI

**Descripción:** Cliente Python para interactuar con el servidor ComfyUI.

**Funcionalidades:**

-   **Gestión de workflows:**

    -   Carga workflows desde archivos JSON
    -   Personaliza workflows con prompts, dimensiones, steps, CFG scale, seed
    -   Soporte para múltiples workflows según categoría de personaje

-   **Generación de imágenes:**

    -   Envío de prompts a ComfyUI
    -   Monitoreo del progreso de generación
    -   Descarga automática de imágenes generadas
    -   Callbacks para actualización de progreso

-   **Utilidades:**
    -   Verificación de conexión con ComfyUI
    -   Listado de workflows disponibles
    -   Consulta del estado de la cola
    -   Interrupción de generaciones

**Configuración:**

-   URL de ComfyUI configurable mediante variable de entorno `COMFY_URL` (por defecto: `http://localhost:3000`)

---

### 3. `WebUI/skingen/index.html` - Interfaz Web

**Descripción:** Interfaz de usuario web responsive para generar skins.

**Características:**

-   **Diseño moderno:**

    -   Interfaz responsive optimizada para móviles y desktop
    -   Diseño estilo Fortnite con colores característicos (#FFD60A, #FFB000)
    -   Animaciones suaves y feedback visual

-   **Funcionalidades:**

    -   Formulario de generación con campos:
        -   Prompt descriptivo (soporta múltiples idiomas)
        -   Prompt negativo (opcional)
        -   Parámetros de generación (dimensiones, steps, CFG, seed)
        -   Selección forzada de categoría (opcional)
        -   Opción de saltar traducción
    -   Visualización de progreso en tiempo real
    -   Descarga de imágenes generadas
    -   Compartir URLs y Task IDs
    -   Código QR para acceso rápido

-   **Integración:**
    -   Detección automática de URL de API (local o RunPod)
    -   Polling automático del estado de generación
    -   Manejo de errores con mensajes claros

---

### 4. Archivos JSON - Workflows de ComfyUI

**Descripción:** Definiciones de workflows para diferentes categorías de personajes.

**Workflows disponibles:**

1. **`API-ComfyUI-FT_Humanoid_5e_vF__SDXL_Refiner.json`**

    - Categoría: `Humanoid`
    - Para personajes humanos estándar (Jonesy, soldados, etc.)

2. **`API-ComfyUI-FT_Humanoid_5e_vF__LoRA_Animal_LoRA_NiceHands__SDXL_Refiner.json`**

    - Categoría: `Animal`
    - Para personajes animales con LoRA especializado

3. **`API-ComfyUI-FT_Humanoid_5e_vF__LoRA_Robots_LoRA_NiceHands__SDXL_Refiner.json`**

    - Categoría: `Robot`
    - Para personajes robóticos y mecánicos

4. **`API-ComfyUI-FT_Humanoid_5e_vF__LoRA_StarWars_LoRA_NiceHands__SDXL_Refiner.json`**

    - Categoría: `Star Wars`
    - Para personajes del universo Star Wars

5. **`API-ComfyUI-FT_Humanoid_5e_vF__LoRA_FuzzyBear_LoRA_NiceHands__SDXL_Refiner.json`**

    - Categoría: `Fuzzy Bear`
    - Para personajes tipo oso peludo

6. **`API-ComfyUI-FT_Humanoid_5e_vF__LoRA_Food_LoRA_NiceHands__SDXL_Refiner.json`**
    - Categoría: `Food`
    - Para personajes con temática de comida

**Características comunes:**

-   Todos los workflows incluyen refiner SDXL para alta calidad
-   LoRAs especializados para diferentes estilos
-   Optimización para manos (NiceHands)
-   Configuración de prompts y negative prompts

---

### 5. `trace_viewer.html` - Visor de Trazas

**Descripción:** Página HTML para visualizar la traza completa de una generación.

**Funcionalidades:**

-   Muestra información detallada de cada generación:
    -   Prompt original y traducido
    -   Categoría detectada
    -   Workflow utilizado
    -   Prompt final enviado a ComfyUI
    -   Parámetros de generación
    -   Timestamps de cada etapa
    -   Historial de progreso
-   Diseño profesional con código de colores
-   Formato JSON legible

---

## ⚙️ Configuración

### Variables de Entorno

El sistema utiliza variables de entorno almacenadas en el archivo `.env` en la raíz del proyecto:

```env
# OpenAI API Key (requerida)
OPENAI_API_KEY=tu_api_key_aqui

# ComfyUI Server URL (opcional, por defecto: http://localhost:3000)
COMFY_URL=http://localhost:3000
```

**⚠️ Importante:**

-   El archivo `.env` está incluido en `.gitignore` y no debe subirse al repositorio
-   Las API keys hardcodeadas han sido eliminadas del código
-   Todas las claves deben estar en el archivo `.env`

### Instalación de Dependencias

```bash
pip install fastapi uvicorn openai python-dotenv requests
```

### Requisitos del Sistema

1. **ComfyUI Server:** Debe estar ejecutándose y accesible

    - Por defecto en `http://localhost:3000`
    - Configurable mediante `COMFY_URL`

2. **OpenAI API Key:** Necesaria para traducción y clasificación

    - Obtener en: https://platform.openai.com/api-keys

3. **Workflows JSON:** Los archivos de workflow deben estar en el mismo directorio que `main.py`

---

## 🚀 Uso de la API

### Iniciar el servidor

```bash
python main.py
```

El servidor se iniciará en `http://0.0.0.0:8188` (o el puerto especificado en `PORT`).

### Ejemplo de petición POST a `/generate`

```python
import requests

response = requests.post("http://localhost:8188/generate", json={
    "prompt": "Un soldado futurista con armadura dorada",
    "negative_prompt": "(worst quality:1.4, low quality:1.4)",
    "width": 1024,
    "height": 1024,
    "steps": 32,
    "cfg": 7.0,
    "seed": None,  # None para seed aleatorio
    "force_category": None,  # None para detección automática
    "skip_translation": False
})

result = response.json()
print(f"Task ID: {result['task_id']}")
print(f"Categoría detectada: {result['detected_category']}")
```

### Consultar estado

```python
task_id = "tu-task-id-aqui"
response = requests.get(f"http://localhost:8188/status/{task_id}")
status = response.json()

print(f"Estado: {status['status']}")
print(f"Progreso: {status['progress']}%")
if status['status'] == 'completed':
    print(f"Imagen: {status['image_url']}")
```

### Descargar imagen

```python
response = requests.get(f"http://localhost:8188/image/{task_id}")
with open("skin_generada.png", "wb") as f:
    f.write(response.content)
```

---

## 🌐 Interfaz Web

### Acceso

1. Abrir `WebUI/skingen/index.html` en un navegador
2. O servir mediante un servidor web:

    ```bash
    # Python
    python -m http.server 5000

    # Node.js
    npx http-server -p 5000
    ```

### Uso

1. **Ingresar prompt:** Describe el personaje que deseas generar (en cualquier idioma)
2. **Ajustar parámetros (opcional):**
    - Dimensiones de imagen
    - Número de steps
    - CFG scale
    - Seed (para reproducibilidad)
3. **Forzar categoría (opcional):** Selecciona una categoría específica
4. **Generar:** Haz clic en "Generar personaje"
5. **Esperar:** El sistema traducirá, clasificará y generará automáticamente
6. **Descargar:** Una vez completado, descarga la imagen generada

### Características de la UI

-   **Progreso en tiempo real:** Barra de progreso animada
-   **Feedback visual:** Mensajes de estado claros
-   **Responsive:** Funciona en móviles y tablets
-   **Código QR:** Para compartir acceso fácilmente
-   **Task ID:** Para rastrear generaciones específicas

---

## 🔄 Flujo de Generación

```
1. Usuario envía prompt
   │
   ├─► [Si skip_translation=False]
   │   └─► OpenAI traduce prompt a inglés
   │
   ├─► [Si force_category=None]
   │   └─► OpenAI clasifica en categoría (Animal/Robot/etc.)
   │
   ├─► Selección de workflow según categoría
   │
   ├─► Construcción de prompt final:
   │   trigger + prompt_traducido + GENERAL_POSITIVE_SUFFIX
   │
   ├─► ComfyClient personaliza workflow JSON
   │
   ├─► Envío a ComfyUI Server
   │
   ├─► Monitoreo de progreso (polling)
   │
   └─► Descarga y retorno de imagen generada
```

### Ejemplo de Prompt Final

```
Input: "Un soldado futurista con armadura dorada"

1. Traducción: "A futuristic soldier with golden armor"
2. Clasificación: "Humanoid"
3. Trigger: "" (vacío para Humanoid)
4. Suffix: "\nfortnite style, clean empty background, ..."

Prompt Final:
"A futuristic soldier with golden armor
fortnite style, clean empty background, show only one character, perfect anatomy, anatomically correct hands with five distinct fingers on each hand, realistic skin texture, natural joint structure,
best quality, ultra high resolution, ultra-detailed, crisp details, stylized game art, natural lighting"
```

---

## 📊 Categorías y Workflows

| Categoría  | Workflow                                                                          | Trigger                                       | Descripción                 |
| ---------- | --------------------------------------------------------------------------------- | --------------------------------------------- | --------------------------- |
| Humanoid   | `API-ComfyUI-FT_Humanoid_5e_vF__SDXL_Refiner.json`                                | (vacío)                                       | Personajes humanos estándar |
| Animal     | `API-ComfyUI-FT_Humanoid_5e_vF__LoRA_Animal_LoRA_NiceHands__SDXL_Refiner.json`    | `fortnite_animal_character, nice_hands, `     | Personajes animales         |
| Robot      | `API-ComfyUI-FT_Humanoid_5e_vF__LoRA_Robots_LoRA_NiceHands__SDXL_Refiner.json`    | `fortnite_robots_character, nice_hands, `     | Personajes robóticos        |
| Star Wars  | `API-ComfyUI-FT_Humanoid_5e_vF__LoRA_StarWars_LoRA_NiceHands__SDXL_Refiner.json`  | `fortnite_star_wars_character, nice_hands, `  | Personajes Star Wars        |
| Fuzzy Bear | `API-ComfyUI-FT_Humanoid_5e_vF__LoRA_FuzzyBear_LoRA_NiceHands__SDXL_Refiner.json` | `fortnite_fuzzy_bear_character, nice_hands, ` | Personajes tipo oso         |
| Food       | `API-ComfyUI-FT_Humanoid_5e_vF__LoRA_Food_LoRA_NiceHands__SDXL_Refiner.json`      | `fortnite_food_character, nice_hands, `       | Personajes temática comida  |

---

## 🔍 Trazabilidad

Cada generación incluye una traza completa accesible en `/trace/{task_id}` que contiene:

-   **Prompt original:** Texto ingresado por el usuario
-   **Prompt traducido:** Versión en inglés generada por OpenAI
-   **Categoría detectada:** Clasificación automática
-   **Trigger aplicado:** Tags específicos de la categoría
-   **Workflow utilizado:** Archivo JSON empleado
-   **Prompt final:** Prompt completo enviado a ComfyUI
-   **Parámetros de generación:** Dimensiones, steps, CFG, seed
-   **Timestamps:** Tiempos de cada etapa del proceso
-   **Historial de progreso:** Actualizaciones de estado en tiempo real

---

## 🐛 Solución de Problemas

### Error: "OPENAI_API_KEY no definida"

-   **Solución:** Verifica que el archivo `.env` existe y contiene `OPENAI_API_KEY=tu_key_aqui`

### Error: "Error conectando a ComfyUI"

-   **Solución:**
    -   Verifica que ComfyUI está ejecutándose
    -   Comprueba la URL en `COMFY_URL` (por defecto: `http://localhost:3000`)
    -   Revisa los logs de ComfyUI

### La generación se queda en "processing"

-   **Solución:**
    -   Verifica los logs de ComfyUI
    -   Comprueba que el workflow JSON es válido
    -   Revisa la conexión de red con ComfyUI

### Categoría incorrecta detectada

-   **Solución:**
    -   Usa `force_category` para forzar una categoría específica
    -   Mejora la descripción del prompt
    -   Verifica que la categoría existe en `WORKFLOW_CONFIG`

---

## 📝 Notas Importantes

-   **Seguridad:** Todas las API keys están ahora en variables de entorno, nunca hardcodeadas
-   **Rendimiento:** Las generaciones pueden tardar 30-120 segundos dependiendo de la complejidad
-   **Límites:** Respeta los límites de rate limiting de OpenAI y ComfyUI
-   **Almacenamiento:** Las imágenes generadas se mantienen en memoria hasta descargarse
-   **Escalabilidad:** El sistema está diseñado para manejar múltiples generaciones concurrentes

---

## 🔗 Enlaces Útiles

-   [Documentación FastAPI](https://fastapi.tiangolo.com/)
-   [OpenAI API Documentation](https://platform.openai.com/docs)
-   [ComfyUI Documentation](https://github.com/comfyanonymous/ComfyUI)

---

**Versión:** 1.1.0  
**Última actualización:** Dic 2025
