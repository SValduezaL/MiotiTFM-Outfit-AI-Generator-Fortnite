# Descarga de Modelos Base

Este documento contiene las instrucciones para descargar los modelos base grandes necesarios para ejecutar el proyecto. Estos modelos son demasiado grandes para incluir en el repositorio de GitHub (SDXL base ~7GB, refiner ~6GB).

## 📦 Modelos Requeridos

### 1. Stable Diffusion XL Base 1.0

**Archivo:** `sd_xl_base_1.0.safetensors` (~7GB)

**Ubicación en el proyecto:** `4.Modelos_seleccionados/sd_xl_base_1.0.safetensors`

**Descarga:**

-   **Hugging Face (Recomendado):** https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0
-   **Directo:** https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors

**Verificación:**

-   Tamaño: ~6.9 GB
-   SHA256: Verificar en la página de Hugging Face

### 2. Stable Diffusion XL Refiner 1.0

**Archivo:** `sd_xl_refiner_1.0.safetensors` (~6GB)

**Ubicación en el proyecto:** `4.Modelos_seleccionados/sd_xl_refiner_1.0.safetensors`

**Descarga:**

-   **Hugging Face (Recomendado):** https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0
-   **Directo:** https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/sd_xl_refiner_1.0.safetensors

**Verificación:**

-   Tamaño: ~6.0 GB
-   SHA256: Verificar en la página de Hugging Face

### 3. Gufeng Anime XL v10

**Archivo:** `gufengAnimeXL_v10.safetensors` (~7GB)

**Ubicación en el proyecto:** `4.Modelos_seleccionados/gufengAnimeXL_v10.safetensors`

**Descarga:**

-   **Civitai:** https://civitai.com/models/[ID_DEL_MODELO]
-   **Hugging Face (si está disponible):** Buscar "gufengAnimeXL_v10"

**Nota:** Este modelo es opcional y se usa como referencia. Verificar la fuente oficial del modelo.

### 4. Modelo Fine-tuned Fortnite Humanoid

**Carpeta:** `humanoid_05/` (~7GB)

**Ubicación en el proyecto:** `4.Modelos_seleccionados/humanoid_05/`

**Descarga:**

-   Este modelo fue entrenado específicamente para este proyecto mediante fine-tuning del SDXL base sobre datos de Fortnite.
-   **Opción 1:** Contactar al autor del proyecto para obtener acceso al modelo.
-   **Opción 2:** Entrenar el modelo siguiendo las instrucciones en `2.Finetuning_Humanoids/` (si están disponibles).
-   **Opción 3:** Usar directamente SDXL base (con resultados menos especializados).

**Estructura esperada:**

```
humanoid_05/
├── model_index.json
├── scheduler/
│   └── scheduler_config.json
├── text_encoder/
│   ├── config.json
│   └── model.safetensors
├── text_encoder_2/
│   ├── config.json
│   └── model.safetensors
├── tokenizer/
│   ├── merges.txt
│   ├── special_tokens_map.json
│   ├── tokenizer_config.json
│   └── vocab.json
├── tokenizer_2/
│   ├── merges.txt
│   ├── special_tokens_map.json
│   ├── tokenizer_config.json
│   └── vocab.json
├── unet/
│   ├── config.json
│   └── diffusion_pytorch_model.safetensors
└── vae/
    ├── config.json
    └── diffusion_pytorch_model.safetensors
```

## 📥 Instrucciones de Descarga

### Método 1: Descarga Manual desde Hugging Face

1. Visita las páginas de Hugging Face indicadas arriba
2. Haz clic en "Files and versions"
3. Descarga el archivo `.safetensors` correspondiente
4. Coloca el archivo en la ubicación correcta según la estructura del proyecto

### Método 2: Usando `huggingface-cli`

```bash
# Instalar huggingface-hub si no está instalado
pip install huggingface-hub

# Descargar SDXL Base
huggingface-cli download stabilityai/stable-diffusion-xl-base-1.0 sd_xl_base_1.0.safetensors --local-dir 4.Modelos_seleccionados/

# Descargar SDXL Refiner
huggingface-cli download stabilityai/stable-diffusion-xl-refiner-1.0 sd_xl_refiner_1.0.safetensors --local-dir 4.Modelos_seleccionados/
```

### Método 3: Usando Python

```python
from huggingface_hub import hf_hub_download
import os

# Directorio de destino
model_dir = "4.Modelos_seleccionados"

# Descargar SDXL Base
hf_hub_download(
    repo_id="stabilityai/stable-diffusion-xl-base-1.0",
    filename="sd_xl_base_1.0.safetensors",
    local_dir=model_dir
)

# Descargar SDXL Refiner
hf_hub_download(
    repo_id="stabilityai/stable-diffusion-xl-refiner-1.0",
    filename="sd_xl_refiner_1.0.safetensors",
    local_dir=model_dir
)
```

## ✅ Verificación

Después de descargar los modelos, verifica que:

1. Los archivos están en las ubicaciones correctas
2. Los tamaños de archivo coinciden con los esperados
3. Los archivos no están corruptos (puedes intentar cargarlos en ComfyUI o similar)

## 📝 Notas Importantes

-   **Espacio en disco:** Asegúrate de tener al menos 30GB de espacio libre para todos los modelos
-   **Tiempo de descarga:** Dependiendo de tu conexión, la descarga puede tardar varias horas
-   **Modelo Fine-tuned:** El modelo `humanoid_05` es específico de este proyecto y puede no estar disponible públicamente. Contacta al autor si necesitas acceso.

## 🔗 Enlaces Útiles

-   [Stable Diffusion XL en Hugging Face](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
-   [Documentación de Hugging Face Hub](https://huggingface.co/docs/huggingface_hub)
-   [ComfyUI - Para probar los modelos](https://github.com/comfyanonymous/ComfyUI)
