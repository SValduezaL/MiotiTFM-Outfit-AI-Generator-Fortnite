import os
import logging
import requests
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import io
import uuid
import time
from dotenv import load_dotenv

from openai import OpenAI, OpenAIError

from comfy_client import ComfyClient

# Cargar variables de entorno desde .env
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="SkinCraft API with AI Classification",
    description="API para generar skins estilo Fortnite con clasificación automática usando OpenAI",
    version="2.3.0",  # Versión actualizada para reflejar trazabilidad completa
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_openai_api_key() -> str:
    """Obtiene la API key de OpenAI desde las variables de entorno."""
    from dotenv import load_dotenv

    load_dotenv()

    key = os.getenv("OPENAI_API_KEY")

    if not key:
        logger.error("OPENAI_API_KEY no definida")
        raise RuntimeError(
            "Define la variable de entorno OPENAI_API_KEY en el archivo .env"
        )
    return key


openai_client = OpenAI(api_key=get_openai_api_key())


def translate_text(text, model="gpt-4o", temperature=0.0, max_tokens=1000):
    system_prompt = (
        "You are a high‑quality translation assistant that translates "
        "any language into clear, fluent English."
    )
    user_prompt = (
        "Translate the following text to English, "
        f"preserving meaning and style:\n\n{text}"
    )

    resp = openai_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        n=1,
    )
    return resp.choices[0].message.content.strip()


def classify_text(text, model="gpt-4o", temperature=0.0):
    system_prompt = (
        "You are a text classification assistant. "
        "Given a character name or description, you will classify it into one of the following categories: "
        '["Animal", "Robot", "Star Wars", "Fuzzy Bear", "Humanoid", "Food"]. '  # Categoría 'Food' añadida
        "Respond ONLY with the category name. No additional text, explanations, or context."
    )
    user_prompt = f"Classify the following character:\n\n{text}"

    resp = openai_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=10,
        n=1,
    )
    return resp.choices[0].message.content.strip()


GENERAL_POSITIVE_SUFFIX = "\nfortnite style, clean empty background, show only one character, perfect anatomy, anatomically correct hands with five distinct fingers on each hand, realistic skin texture, natural joint structure, \nbest quality, ultra high resolution, ultra-detailed, crisp details, stylized game art, natural lighting"

WORKFLOW_CONFIG = {
    "Humanoid": {
        "workflow_file": "API-ComfyUI-FT_Humanoid_5e_vF__SDXL_Refiner.json",
        "description": "Personajes humanoides como Jonesy, soldados, etc.",
        "trigger": "",
    },
    "Animal": {
        "workflow_file": "API-ComfyUI-FT_Humanoid_5e_vF__LoRA_Animal_LoRA_NiceHands__SDXL_Refiner.json",
        "description": "Personajes animales como perros, gatos, etc.",
        "trigger": "fortnite_animal_character, nice_hands, ",
    },
    "Robot": {
        "workflow_file": "API-ComfyUI-FT_Humanoid_5e_vF__LoRA_Robots_LoRA_NiceHands__SDXL_Refiner.json",
        "description": "Personajes robóticos y mecánicos",
        "trigger": "fortnite_robots_character, nice_hands, ",
    },
    "Star Wars": {
        "workflow_file": "API-ComfyUI-FT_Humanoid_5e_vF__LoRA_StarWars_LoRA_NiceHands__SDXL_Refiner.json",
        "description": "Personajes del universo Star Wars",
        "trigger": "fortnite_star_wars_character, nice_hands, ",
    },
    "Fuzzy Bear": {
        "workflow_file": "API-ComfyUI-FT_Humanoid_5e_vF__LoRA_FuzzyBear_LoRA_NiceHands__SDXL_Refiner.json",
        "description": "Personajes tipo oso peludo",
        "trigger": "fortnite_fuzzy_bear_character, nice_hands, ",
    },
    "Food": {
        "workflow_file": "API-ComfyUI-FT_Humanoid_5e_vF__LoRA_Food_LoRA_NiceHands__SDXL_Refiner.json",
        "description": "Personajes con temática de comida",
        "trigger": "fortnite_food_character, nice_hands, ",
    },
}

comfy_client = ComfyClient()


class GenerationRequest(BaseModel):
    prompt: str = Field(
        ..., description="Texto descriptivo para generar la skin (en cualquier idioma)"
    )
    negative_prompt: Optional[str] = Field(
        "(worst quality:1.4, low quality:1.4), (bad anatomy:1.2, malformed hands:1.2, fused fingers:1.2, extra fingers:1.2), (bad proportions:1.1, gross proportions, double torso, long neck), (deformed, disfigured, malformed limbs, extra limbs), multiple characters, blurry, out of frame, cropped, watermark, signature, text, jpeg artifacts, nsfw, clone, duplicate, poorly drawn face, asymmetrical eyes, unrealistic eyes, bad shadow, bad lighting, oversaturated, underexposed, lowres, draft, sketch, monochrome, grayscale",
        description="Prompt negativo",
    )
    width: Optional[int] = Field(1024, description="Ancho de la imagen")
    height: Optional[int] = Field(1024, description="Alto de la imagen")
    steps: Optional[int] = Field(32, description="Número de pasos de sampling")
    cfg: Optional[float] = Field(7.0, description="CFG Scale")
    seed: Optional[int] = Field(None, description="Seed para reproducibilidad")
    force_category: Optional[str] = Field(
        None, description="Forzar una categoría específica"
    )
    skip_translation: Optional[bool] = Field(
        False, description="Saltar traducción automática"
    )


class GenerationResponse(BaseModel):
    task_id: str
    status: str
    message: str
    detected_category: Optional[str] = None
    translated_prompt: Optional[str] = None
    workflow_used: Optional[str] = None


class StatusResponse(BaseModel):
    task_id: str
    status: str
    progress: Optional[int] = None
    message: Optional[str] = None
    image_url: Optional[str] = None
    detected_category: Optional[str] = None
    translated_prompt: Optional[str] = None
    workflow_used: Optional[str] = None


class WorkflowInfo(BaseModel):
    category: str
    workflow_file: str
    description: str


tasks: Dict[str, Dict[str, Any]] = {}


async def translate_and_classify_prompt(
    prompt: str, force_category: Optional[str] = None
) -> tuple[str, str]:
    try:
        # Si se fuerza una categoría, solo traducimos
        if force_category and force_category in WORKFLOW_CONFIG:
            logger.info(
                f"Usando categoría forzada: {force_category}. Solo se traducirá el prompt."
            )
            translation = translate_text(text=prompt)
            logger.info(f"Traducción completada: '{prompt}' -> '{translation}'")
            return translation, force_category

        # Si no, traducimos y luego clasificamos la traducción
        logger.info(f"Enviando a OpenAI para traducir y clasificar: '{prompt}'")

        # 1. Traducir
        translation = translate_text(text=prompt)
        logger.info(f"Respuesta de OpenAI - Traducción: '{translation}'")

        # 2. Clasificar el texto ya traducido
        category = classify_text(text=translation)
        logger.info(f"Respuesta de OpenAI - Categoría: '{category}'")

        if category not in WORKFLOW_CONFIG:
            logger.warning(
                f"Categoría desconocida '{category}', usando 'Humanoid' por defecto"
            )
            category = "Humanoid"

        return translation, category

    except OpenAIError as e:
        logger.error(f"Error en la llamada a la API de OpenAI: {e}")
        # En caso de error, devolvemos el prompt original y la categoría por defecto
        return prompt, "Humanoid"
    except Exception as e:
        logger.error(f"Error inesperado en traducción/clasificación: {e}")
        return prompt, "Humanoid"


@app.get("/", summary="Health Check")
async def health_check():
    return {
        "status": "healthy",
        "service": "SkinCraft API with Integrated AI Classification",
        "version": app.version,
        "comfy_status": comfy_client.check_connection(),
        "openai_status": "integrated",
        "available_categories": list(WORKFLOW_CONFIG.keys()),
    }


@app.get("/workflows", summary="Obtener información de workflows disponibles")
async def get_workflows():
    return {
        "workflows": [
            WorkflowInfo(
                category=category,
                workflow_file=config["workflow_file"],
                description=config["description"],
            )
            for category, config in WORKFLOW_CONFIG.items()
        ]
    }


@app.post(
    "/generate",
    response_model=GenerationResponse,
    summary="Generar skin con clasificación automática",
)
async def generate_skin(request: GenerationRequest, background_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())

    if request.skip_translation:
        translated_prompt = request.prompt
        detected_category = request.force_category or "Humanoid"
    else:
        translated_prompt, detected_category = await translate_and_classify_prompt(
            request.prompt, request.force_category
        )

    workflow_info = WORKFLOW_CONFIG[detected_category]
    workflow_file = workflow_info["workflow_file"]
    trigger = workflow_info["trigger"]

    final_prompt = trigger + translated_prompt + GENERAL_POSITIVE_SUFFIX

    logger.info(f"=== INICIO TRAZA GENERACIÓN: {task_id} ===")

    tasks[task_id] = {
        "status": "pending",
        "progress": 0,
        "message": "Tarea iniciada",
        "request": request.dict(),
        "detected_category": detected_category,
        "translated_prompt": translated_prompt,
        "workflow_used": workflow_file,
        "progress_history": [],
        "trace": {
            "original_prompt": request.prompt,
            "translated_prompt": translated_prompt,
            "detected_category": detected_category,
            "trigger": trigger if trigger else "None",
            "workflow_file": workflow_file,
            "final_prompt": final_prompt,
            "negative_prompt": request.negative_prompt,
            "generation_params": {
                "width": request.width,
                "height": request.height,
                "steps": request.steps,
                "cfg": request.cfg,
                "seed": request.seed,
            },
            "timestamps": {
                "created": time.time(),
                "classification_completed": time.time(),
                "comfy_started": None,
                "comfy_completed": None,
            },
            "output_details": None,  # Inicializar como nulo
        },
    }

    background_tasks.add_task(
        process_generation, task_id, request, final_prompt, workflow_file
    )

    return GenerationResponse(
        task_id=task_id,
        status="accepted",
        message=f"Generación iniciada con categoría '{detected_category}'",
        detected_category=detected_category,
        translated_prompt=translated_prompt,
        workflow_used=workflow_file,
    )


@app.get(
    "/status/{task_id}",
    response_model=StatusResponse,
    summary="Estado de la generación",
)
async def get_generation_status(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Tarea no encontrada")

    task = tasks[task_id]
    return StatusResponse(
        task_id=task_id,
        status=task["status"],
        progress=task.get("progress"),
        message=task.get("message"),
        image_url=task.get("image_url"),
        detected_category=task.get("detected_category"),
        translated_prompt=task.get("translated_prompt"),
        workflow_used=task.get("workflow_used"),
    )


@app.get(
    "/trace/{task_id}",
    response_class=HTMLResponse,
    summary="Mostrar página de traza de generación",
)
async def show_trace_page(task_id: str):
    if task_id not in tasks:
        html_content = f"""
        <!DOCTYPE html><html><head><title>Error 404</title></head>
        <body><h1>Error 404: Tarea no encontrada</h1><p>No se encontró la tarea con ID: {task_id}</p></body></html>
        """
        return HTMLResponse(content=html_content, status_code=404)
    try:
        with open("trace_viewer.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        raise HTTPException(
            status_code=500, detail="trace_viewer.html no encontrado en el servidor."
        )


@app.get("/api/trace/{task_id}", summary="Obtener datos de la traza de generación")
async def get_generation_trace_data(task_id: str):
    logger.info(f"API request for trace data. Task ID: {task_id}")
    if task_id not in tasks:
        logger.error(f"Task ID {task_id} not found in memory.")
        raise HTTPException(status_code=404, detail=f"Tarea {task_id} no encontrada")

    task = tasks[task_id]
    return {
        "task_id": task_id,
        "status": task["status"],
        "trace": task.get("trace", {}),
        "progress_history": task.get("progress_history", []),
    }


@app.get("/image/{task_id}", summary="Descargar imagen generada")
async def download_image(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Tarea no encontrada")

    task = tasks[task_id]

    if task["status"] != "completed":
        raise HTTPException(status_code=400, detail="La generación no está completa")

    if "image_data" not in task:
        raise HTTPException(status_code=404, detail="Imagen no encontrada")

    category = task.get("detected_category", "unknown")
    filename = f"{category.lower()}_character_{task_id[:8]}.png"

    return StreamingResponse(
        io.BytesIO(task["image_data"]),
        media_type="image/png",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@app.delete("/task/{task_id}", summary="Cancelar/eliminar tarea")
async def delete_task(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Tarea no encontrada")

    del tasks[task_id]
    return {"message": "Tarea eliminada correctamente"}


@app.get("/tasks", summary="Listar todas las tareas")
async def list_tasks():
    return {
        "tasks": [
            {
                "task_id": task_id,
                "status": task["status"],
                "progress": task.get("progress", 0),
                "message": task.get("message", ""),
                "detected_category": task.get("detected_category"),
                "workflow_used": task.get("workflow_used"),
            }
            for task_id, task in tasks.items()
        ]
    }


async def process_generation(
    task_id: str, request: GenerationRequest, final_prompt: str, workflow_file: str
):
    try:
        tasks[task_id]["status"] = "processing"
        tasks[task_id]["trace"]["timestamps"]["comfy_started"] = time.time()
        update_task_progress(task_id, 10, f"Preparando workflow '{workflow_file}'...")

        image_data = await comfy_client.generate_image(
            prompt=final_prompt,
            negative_prompt=request.negative_prompt,
            width=request.width,
            height=request.height,
            steps=request.steps,
            cfg=request.cfg,
            seed=request.seed,
            task_id=task_id,
            workflow_file=workflow_file,
            update_callback=lambda p, m: update_task_progress(task_id, p, m),
        )

        if image_data:
            task = tasks[task_id]
            task["status"] = "completed"
            task["image_data"] = image_data
            image_url = f"/image/{task_id}"
            task["image_url"] = image_url

            # Establecer timestamp de finalización
            task["trace"]["timestamps"]["comfy_completed"] = time.time()

            # Generar detalles del archivo
            category = task.get("detected_category", "unknown")
            filename = f"{category.lower()}_character_{task_id[:8]}.png"

            # Guardar detalles en la traza
            task["trace"]["output_details"] = {
                "filename": filename,
                "image_url": image_url,
            }
            update_task_progress(task_id, 100, "Generación completada")

            # Calcular tiempo total al final
            total_time = (
                task["trace"]["timestamps"]["comfy_completed"]
                - task["trace"]["timestamps"]["created"]
            )
            logger.info(
                f"Generación completada en {total_time:.2f} segundos para task {task_id}"
            )
        else:
            tasks[task_id]["status"] = "failed"
            update_task_progress(
                task_id, tasks[task_id].get("progress", 0), "Error generando imagen"
            )

    except Exception as e:
        logger.error(f"Error procesando generación {task_id}: {str(e)}")
        tasks[task_id]["status"] = "failed"
        update_task_progress(
            task_id, tasks[task_id].get("progress", 0), f"Error: {str(e)}"
        )


def update_task_progress(task_id: str, progress: int, message: str):
    if task_id in tasks:
        task = tasks[task_id]
        task["progress"] = progress
        task["message"] = message
        task["progress_history"].append(
            {"progress": progress, "message": message, "timestamp": time.time()}
        )


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8188))
    uvicorn.run(app, host="0.0.0.0", port=port)
