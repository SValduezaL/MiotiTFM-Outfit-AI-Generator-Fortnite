import requests
import time
import json
import logging
import asyncio
import os
from typing import Optional, Callable, Dict, Any
import random
from dotenv import load_dotenv

# Cargar variables de entorno desde .env
load_dotenv()

logger = logging.getLogger(__name__)


class ComfyClient:
    def __init__(
        self,
        base_url: str = None,
        default_workflow: str = "API-ComfyUI-FT_Humanoid_5e_vF__SDXL_Refiner.json",
    ):
        self.base_url = base_url or os.environ.get("COMFY_URL", "http://localhost:3000")
        self.default_workflow = default_workflow
        self.session = requests.Session()
        self.session.timeout = 30

    def check_connection(self) -> bool:
        try:
            response = self.session.get(f"{self.base_url}/system_stats", timeout=10)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Error conectando a ComfyUI: {e}")
            return False

    def load_workflow(self, workflow_file: str = None) -> Dict[str, Any]:
        workflow_path = workflow_file or self.default_workflow

        try:
            with open(workflow_path, "r", encoding="utf-8") as f:
                logger.info(f"Cargando workflow: {workflow_path}")
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"Archivo de workflow no encontrado: {workflow_path}")
            if workflow_path != self.default_workflow:
                logger.info(
                    f"Intentando con workflow por defecto: {self.default_workflow}"
                )
                return self.load_workflow(self.default_workflow)
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Error parseando JSON en {workflow_path}: {e}")
            raise

    def customize_workflow(self, workflow: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        custom_workflow = workflow.copy()

        if "prompt" in kwargs and kwargs["prompt"]:
            for node_id in ["6", "15"]:
                if node_id in custom_workflow and "inputs" in custom_workflow[node_id]:
                    custom_workflow[node_id]["inputs"]["text"] = kwargs["prompt"]
                    logger.info(
                        f"Prompt actualizado en nodo {node_id}: {kwargs['prompt'][:100]}..."
                    )

        if "negative_prompt" in kwargs and kwargs["negative_prompt"]:
            for node_id in ["7", "16"]:
                if node_id in custom_workflow and "inputs" in custom_workflow[node_id]:
                    custom_workflow[node_id]["inputs"]["text"] = kwargs[
                        "negative_prompt"
                    ]
                    logger.info(f"Negative prompt actualizado en nodo {node_id}")

        if "5" in custom_workflow and "inputs" in custom_workflow["5"]:
            if "width" in kwargs:
                custom_workflow["5"]["inputs"]["width"] = kwargs["width"]
            if "height" in kwargs:
                custom_workflow["5"]["inputs"]["height"] = kwargs["height"]
            logger.info(
                f"Dimensiones actualizadas: {kwargs.get('width', 1024)}x{kwargs.get('height', 1024)}"
            )

        for node_id in ["10", "11"]:
            if node_id in custom_workflow and "inputs" in custom_workflow[node_id]:
                if "steps" in kwargs:
                    custom_workflow[node_id]["inputs"]["steps"] = kwargs["steps"]
                if "cfg" in kwargs:
                    custom_workflow[node_id]["inputs"]["cfg"] = kwargs["cfg"]

        if "10" in custom_workflow and "inputs" in custom_workflow["10"]:
            if "seed" in kwargs and kwargs["seed"] is not None:
                custom_workflow["10"]["inputs"]["noise_seed"] = kwargs["seed"]
                logger.info(f"Seed fijo usado: {kwargs['seed']}")
            else:
                seed = random.randint(0, 2**32 - 1)
                custom_workflow["10"]["inputs"]["noise_seed"] = seed
                logger.info(f"Seed aleatorio generado: {seed}")

        logger.info(f"=== PARÁMETROS DE GENERACIÓN ===")
        logger.info(
            f"Workflow personalizado - Steps: {kwargs.get('steps', 32)}, "
            f"CFG: {kwargs.get('cfg', 7.0)}, "
            f"Size: {kwargs.get('width', 1024)}x{kwargs.get('height', 1024)}"
        )
        logger.info(f"================================")

        return custom_workflow

    async def generate_image(
        self,
        prompt: str,
        negative_prompt: str = "(worst quality:1.4, low quality:1.4), (bad anatomy:1.2, malformed hands:1.2, fused fingers:1.2, extra fingers:1.2), (bad proportions:1.1, gross proportions, double torso, long neck), (deformed, disfigured, malformed limbs, extra limbs), multiple characters, blurry, out of frame, cropped, watermark, signature, text, jpeg artifacts, nsfw, clone, duplicate, poorly drawn face, asymmetrical eyes, unrealistic eyes, bad shadow, bad lighting, oversaturated, underexposed, lowres, draft, sketch, monochrome, grayscale",
        width: int = 1024,
        height: int = 1024,
        steps: int = 32,
        cfg: float = 7.0,
        seed: Optional[int] = None,
        task_id: str = None,
        workflow_file: str = None,  # <- PARÁMETRO AÑADIDO
        update_callback: Optional[Callable] = None,
    ) -> Optional[bytes]:
        try:

            def update_progress(progress: int, message: str):
                if update_callback:
                    update_callback(progress, message)

            update_progress(15, "Cargando workflow...")

            workflow = self.load_workflow(workflow_file)

            custom_workflow = self.customize_workflow(
                workflow,
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                steps=steps,
                cfg=cfg,
                seed=seed,
            )

            update_progress(25, f"Enviando a ComfyUI...")

            response = self.session.post(
                f"{self.base_url}/prompt", json={"prompt": custom_workflow}
            )
            response.raise_for_status()

            prompt_id = response.json().get("prompt_id")
            if not prompt_id:
                raise ValueError("No se recibió prompt_id válido")

            logger.info(f"Prompt enviado con ID: {prompt_id}")
            update_progress(35, f"Procesando...")

            return await self._wait_for_result(prompt_id, update_callback)

        except Exception as e:
            logger.error(f"Error generando imagen: {e}")
            if update_callback:
                update_callback(0, f"Error: {str(e)}")
            raise

    async def _wait_for_result(
        self, prompt_id: str, update_callback: Optional[Callable] = None
    ) -> Optional[bytes]:
        max_attempts = 120

        for attempt in range(max_attempts):
            try:
                progress = 35 + (attempt * 60 // max_attempts)

                if update_callback:
                    update_callback(
                        progress, f"Generando imagen... ({attempt + 1}/{max_attempts})"
                    )

                response = self.session.get(f"{self.base_url}/history/{prompt_id}")

                if response.status_code != 200:
                    await asyncio.sleep(1)
                    continue

                data = response.json()

                if prompt_id not in data:
                    await asyncio.sleep(1)
                    continue

                outputs = data[prompt_id].get("outputs", {})

                for node_id, node_data in outputs.items():
                    images = node_data.get("images", [])

                    if images:
                        img_info = images[0]
                        filename = img_info["filename"]
                        subfolder = img_info.get("subfolder", "")
                        img_type = img_info.get("type", "output")

                        if update_callback:
                            update_callback(95, "Descargando imagen...")

                        img_url = f"{self.base_url}/view"
                        params = {
                            "filename": filename,
                            "subfolder": subfolder,
                            "type": img_type,
                        }

                        img_response = self.session.get(img_url, params=params)
                        img_response.raise_for_status()

                        logger.info(f"Imagen generada exitosamente: {filename}")
                        return img_response.content

                await asyncio.sleep(1)

            except Exception as e:
                logger.error(f"Error en intento {attempt + 1}: {e}")
                await asyncio.sleep(2)

        logger.error(f"Timeout esperando resultado para {prompt_id}")
        return None

    def list_available_workflows(self, workflow_dir: str = ".") -> list:
        try:
            import glob

            workflow_files = glob.glob(os.path.join(workflow_dir, "*.json"))
            workflows = []

            for file_path in workflow_files:
                filename = os.path.basename(file_path)
                try:
                    with open(file_path, "r") as f:
                        workflow_data = json.load(f)

                    info = {
                        "filename": filename,
                        "path": file_path,
                        "nodes": len(workflow_data),
                        "valid": True,
                    }
                    workflows.append(info)
                except:
                    workflows.append(
                        {"filename": filename, "path": file_path, "valid": False}
                    )

            return workflows
        except Exception as e:
            logger.error(f"Error listando workflows: {e}")
            return []

    def get_queue_status(self) -> Dict[str, Any]:
        try:
            response = self.session.get(f"{self.base_url}/queue")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error obteniendo estado de cola: {e}")
            return {"queue_running": [], "queue_pending": []}

    def interrupt_generation(self) -> bool:
        try:
            response = self.session.post(f"{self.base_url}/interrupt")
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Error interrumpiendo generación: {e}")
            return False
