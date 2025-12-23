# TFM Outfit AI Generator Fortnite

## 🏆 Reconocimientos

Este proyecto ha sido galardonado por **MIOTI TECH & BUSINESS SCHOOL** como:

-   🥇 **Mejor TFM del Máster Deep Learning** - Promoción 2025
-   🏆 **Mejor Proyecto Académico de toda la Escuela** - Año académico 2024-2025

---

Proyecto de Fin de Máster (TFM) para la generación de outfits/skins de Fortnite mediante Inteligencia Artificial, utilizando una arquitectura híbrida de fine-tuning y LoRAs especializados.

## 📋 Descripción del Proyecto

Este proyecto implementa un sistema completo de generación de imágenes de personajes estilo Fortnite mediante IA, combinando:

-   **Fine-tuning** del modelo base Stable Diffusion XL sobre datos de Fortnite
-   **LoRAs especializados** por categoría temática (Animal, Food, Robots, Star Wars, Fuzzy Bear)
-   **API REST** con clasificación automática mediante OpenAI
-   **Interfaz web** para generación interactiva de skins

### Objetivo

Generar imágenes de outfits de Fortnite manteniendo la identidad visual característica del juego mientras se especializa en diferentes categorías temáticas mediante adaptadores LoRA ligeros y modulares.

## 🏗️ Arquitectura del Proyecto

```
┌─────────────────────────────────────────────────────────┐
│  1. Data Preparation                                    │
│  - Obtención de datos desde API de Fortnite            │
│  - Limpieza y procesamiento de imágenes                │
│  - Generación de etiquetas y captions                  │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│  2. Fine-tuning                                         │
│  - Fine-tuning de SDXL sobre datos Fortnite           │
│  - Modelo base: v1x0_fortnite_humanoid_sdxl1_vae_fix   │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│  3. LoRAs Especializados                               │
│  - Animal, Food, Robots, Star Wars, Fuzzy Bear         │
│  - Entrenamiento con KOHYA_ss                          │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│  4. API & User Interface                                │
│  - FastAPI con clasificación automática (OpenAI)       │
│  - Integración con ComfyUI                              │
│  - Interfaz web interactiva                            │
└─────────────────────────────────────────────────────────┘
```

## 📁 Estructura del Proyecto

```
TFM_Outfit_AI_Generator_Fortnite/
├── 0.Presentacion/              # Presentación del proyecto
│   ├── TFM_Generacion_Personajes_Fortnite_vF.pdf
│   └── TFM_Generacion_Personajes_Fortnite_vF.pptx
│
├── 1.Data_prep/          # Pipeline de preparación de datos
│   ├── 0.1.get_items_from_api.ipynb
│   ├── 0.2.1.data_cleaning.ipynb
│   ├── 0.2.2.1.data_augmentation.ipynb
│   ├── 0.2.2.2.data_transformation_256x512.ipynb
│   ├── 0.2.4.etiquetas.ipynb
│   ├── 0.2.5.mejorar-etiquetas.ipynb
│   └── README.md
│
├── 2.Finetuning_Humanoids/      # Fine-tunings del modelo base
|   ├── humanoid_02              # Segundo entrenamiento realizado
|   ├── humanoid_03              # Tercer entrenamiento realizado
|   ├── humanoid_04              # Cuarto entrenamiento realizado
|   ├── humanoid_05              # Quinto entrenamiento realizado (seleccionado)
|   ├── humanoid_06              # Sexto entrenamiento realizado
|   ├── humanoid_07              # Séptimo entrenamiento realizado
│   └── README.md
│
├── 3.LoRAs/                     # Entrenamiento de LoRAs
│   ├── 1.Datasets LoRAs/        # Datasets por categoría
│   ├── 2.Entrenamientos/        # Configuraciones y checkpoints
│   ├── 3.Inferencias LoRAs seleccionados/  # Resultados de inferencia
│   └── README.md
│
├── 4.Modelos_seleccionados/     # Modelos base y LoRAs entrenados
│
├── 5.API_User_Interface/        # API y interfaz web
│   ├── main.py                  # API FastAPI
│   ├── comfy_client.py          # Cliente ComfyUI
│   ├── generador_api.py         # Generador de workflows
│   ├── WebUI/                   # Interfaz web HTML
│   └── README.md
│
├── Guias KOHYA Trainings/       # Documentación de entrenamiento
│
├── requirements.txt             # Dependencias Python
├── SETUP_venv.md                # Instrucciones de configuración
├── MODELS_DOWNLOAD.md           # Instrucciones de descarga de modelos
├── VAST_AI_SETUP.md             # Guía técnica: Vast.ai, KOHYA y ComfyUI
└── README.md                    # Este archivo
```

## 🚀 Inicio Rápido

### Prerrequisitos

**Para desarrollo local:**

-   Python 3.11
-   Git
-   Git LFS (para archivos grandes)
-   ComfyUI instalado y configurado
-   GPU con al menos 8GB VRAM (recomendado 12GB+)

**Para entrenamiento e inferencia en la nube:**

-   Cuenta en [Vast.ai](https://vast.ai)
-   Acceso SSH configurado
-   Consulta **[VAST_AI_SETUP.md](VAST_AI_SETUP.md)** para guía completa de despliegue en la nube

### Instalación

1. **Clonar el repositorio:**

```bash
git clone https://github.com/SValduezaL/TFM_Outfit_AI_Generator_Fortnite.git
cd TFM_Outfit_AI_Generator_Fortnite
```

2. **Instalar Git LFS (si no está instalado):**

```bash
# Windows (con Chocolatey)
choco install git-lfs

# Linux
sudo apt install git-lfs

# macOS
brew install git-lfs

# Inicializar Git LFS
git lfs install
```

3. **Crear y activar entorno virtual:**

```bash
# Windows (PowerShell)
python -m venv .venv_tfm_skin_ai
.\.venv_tfm_skin_ai\Scripts\Activate.ps1

# Linux/Mac
python -m venv .venv_tfm_skin_ai
source .venv_tfm_skin_ai/bin/activate
```

4. **Instalar dependencias:**

```bash
pip install -r requirements.txt
```

5. **Descargar modelos base grandes:**

**⚠️ IMPORTANTE:** Los modelos base grandes (~20GB total) no están incluidos en el repositorio. Debes descargarlos manualmente.

Consulta [MODELS_DOWNLOAD.md](MODELS_DOWNLOAD.md) para instrucciones detalladas de descarga.

**Modelos requeridos:**

-   `sd_xl_base_1.0.safetensors` (~7GB)
-   `sd_xl_refiner_1.0.safetensors` (~6GB)
-   `humanoid_05/` (~7GB) - Modelo fine-tuned

6. **Configurar variables de entorno:**

Crea un archivo `.env` en la raíz del proyecto:

```env
# API Keys
FORTNITE_API_KEY=tu_api_key_aqui
GOOGLE_GEMINI_API_KEY=tu_api_key_aqui
OPENAI_API_KEY=tu_api_key_aqui

# ComfyUI
COMFYUI_URL=http://127.0.0.1:8188
COMFYUI_OUTPUT_DIR=path/to/comfyui/output

# Configuración de la API
API_HOST=0.0.0.0
API_PORT=8000
```

### Ejecutar la API

```bash
# Desde el directorio 4.API_User_Interface/
cd 4.API_User_Interface
python main.py
```

La API estará disponible en `http://localhost:8000`

La interfaz web estará disponible en `http://localhost:8000/static/skingen.html`

## 📚 Documentación

Cada módulo del proyecto tiene su propia documentación:

### Documentación Principal

-   **[VAST_AI_SETUP.md](VAST_AI_SETUP.md)** - ⭐ **Guía técnica completa** para entrenamiento e inferencia con Vast.ai, KOHYA y ComfyUI
    -   Alquiler y configuración de GPUs en Vast.ai
    -   Uso de templates preconfigurados (Kohya's GUI y ComfyUI)
    -   Entrenamiento de fine-tuning y LoRAs con A100 40GB/80GB
    -   Despliegue de ComfyUI con RTX 4060/4090
    -   Integración con API REST

### Documentación por Módulo

-   **[1.Data_prep/README.md](1.Data_prep/README.md)** - Pipeline de preparación de datos
-   **[2.Finetuning_Humanoids/README.md](2.Finetuning_Humanoids/README.md)** - Fine-tuning del modelo base
-   **[3.LoRAs/README.md](3.LoRAs/README.md)** - Entrenamiento de LoRAs especializados
-   **[5.API_User_Interface/README.md](5.API_User_Interface/README.md)** - API y interfaz de usuario
-   **[MODELS_DOWNLOAD.md](MODELS_DOWNLOAD.md)** - Instrucciones de descarga de modelos
-   **[SETUP_venv.md](SETUP_venv.md)** - Configuración del entorno local

### Presentación del Proyecto

-   **[0.Presentacion/](0.Presentacion/)** - Presentación del TFM
    -   `TFM_Generacion_Personajes_Fortnite_vF.pdf` - Presentación en PDF
    -   `TFM_Generacion_Personajes_Fortnite_vF.pptx` - Presentación en PowerPoint

## 🎯 Características Principales

### 1. Preparación de Datos

-   Obtención automática de datos desde API de Fortnite
-   Limpieza y procesamiento de imágenes
-   Aumento de datos (data augmentation)
-   Generación automática de etiquetas con IA

### 2. Fine-tuning y LoRAs

-   Fine-tuning de SDXL sobre datos Fortnite
-   5 LoRAs especializados por categoría temática
-   Configuraciones optimizadas de entrenamiento
-   Documentación completa de parámetros

### 3. API y Clasificación Automática

-   Clasificación automática de personajes con OpenAI GPT-4o
-   Selección automática de workflow según categoría
-   Traducción automática de prompts
-   Procesamiento asíncrono con seguimiento de progreso

### 4. Modelos Seleccionados

-   Modelo Base Stable Diffusion XL Base 1.0 [a descargar según MODELS_DOWNLOAD.md]
-   Modelo Stable Diffusion XL Refiner 1.0 [a descargar según MODELS_DOWNLOAD.md]
-   [Opcional] Modelo Gufeng Anime XL v10 [a descargar según MODELS_DOWNLOAD.md]
-   Modelo Fine-tuned Fortnite Humanoid seleccionado [a descargar según MODELS_DOWNLOAD.md]
-   Modelos LoRA seleccionados para cada categoría temática
-   Modelo Nice Hands para refinar esa parte del cuerpo de los LoRA

### 5. Interfaz Web

-   Interfaz HTML interactiva
-   Visualización en tiempo real del progreso
-   Descarga de imágenes generadas
-   Historial de generaciones

## 🔧 Tecnologías Utilizadas

-   **Python 3.11**
-   **FastAPI** - Framework web para la API
-   **Stable Diffusion XL** - Modelo base de generación
-   **KOHYA_ss** - Framework de entrenamiento LoRA
-   **ComfyUI** - Interfaz y backend de generación
-   **OpenAI API** - Clasificación y traducción
-   **OpenCV, Pillow** - Procesamiento de imágenes
-   **Pandas** - Manipulación de datos
-   **Jupyter Notebooks** - Análisis y experimentación

## 📊 Modelos y LoRAs entrenados en este TFM

### Modelo Base Fine-tuned

-   **humanoid_05**: Modelo fine-tuned sobre SDXL base con datos de Fortnite

### LoRAs Entrenados

-   **Animal**: 42 imágenes, especializado en personajes animales
-   **Food**: 27 imágenes, especializado en personajes de comida
-   **Fuzzy Bear**: 8 imágenes, especializado en osos peludos
-   **Robots**: 15 imágenes, especializado en robots
-   **Star Wars**: 19 imágenes, especializado en temática Star Wars

Todos los LoRAs están incluidos en el repositorio mediante Git LFS.

## 🐛 Solución de Problemas

### Error: "Modelo no encontrado"

-   Verifica que has descargado los modelos base según [MODELS_DOWNLOAD.md](MODELS_DOWNLOAD.md)
-   Asegúrate de que los archivos están en las ubicaciones correctas

### Error: "Git LFS no funciona"

-   Verifica que Git LFS está instalado: `git lfs version`
-   Inicializa Git LFS: `git lfs install`
-   Si clonaste antes de instalar LFS, ejecuta: `git lfs pull`

### Error: "ComfyUI no responde"

-   Verifica que ComfyUI está ejecutándose
-   Comprueba la URL en `.env`: `COMFYUI_URL`
-   Verifica que los modelos están en las rutas correctas de ComfyUI

## 📝 Licencia

Este proyecto es parte de un Trabajo de Fin de Máster (TFM) para el Máster de Deep Learning en **MIOTI Tech & Business School**.

**Reconocimientos:**

-   🥇 Mejor TFM del Máster Deep Learning - Promoción 2025
-   🏆 Mejor Proyecto Académico de toda la Escuela - Año académico 2024-2025

Consulta la licencia en el repositorio.

## 👤 Autores

-   Odreman Ferrer Diaz
-   Sergio Valdueza Lozano

## 🙏 Agradecimientos

-   **MOITI TECH & BUSINESS SCHOOL**.
-   **Diego García Morate**, Tutor del Proyecto (diegogm@faculty.mioti.es)
-   **Stability AI** por Stable Diffusion XL
-   **KOHYA_ss** por el framework de entrenamiento
-   **ComfyUI** por la interfaz de generación
-   **Fortnite API** por los datos de outfits

## 📞 Contacto

Para preguntas o problemas, abre un issue en el repositorio de GitHub.

---

**Nota:** Este proyecto requiere recursos computacionales significativos (GPU con VRAM suficiente) y acceso a APIs externas (Fortnite API, OpenAI API). Asegúrate de tener los recursos necesarios antes de comenzar.
