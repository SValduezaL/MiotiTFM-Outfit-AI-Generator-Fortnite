# Guía Técnica: Entrenamiento e Inferencia con Vast.ai, KOHYA y ComfyUI

## 1. Introducción

Este documento proporciona una guía técnica paso a paso para ingenieros de Machine Learning que necesitan:

-   **Alquilar y configurar GPUs en Vast.ai** para entrenamiento e inferencia
-   **Usar templates preconfigurados** (Kohya's GUI y ComfyUI) para acelerar el despliegue
-   **Ejecutar entrenamientos de fine-tuning y LoRA con KOHYA_ss** sobre modelos Stable Diffusion XL usando **A100 40GB/80GB**
-   **Desplegar ComfyUI en GPU remota** para generación de imágenes usando **RTX 4060 o RTX 4090**
-   **Integrar la generación de imágenes con una API REST** mediante FastAPI

### Flujos Cubiertos

1. **Fine-tuning completo de SDXL**: Entrenamiento del modelo base sobre datos de Fortnite
2. **Entrenamiento de LoRAs**: Creación de adaptadores especializados por categoría temática
3. **Inferencia con ComfyUI**: Generación de imágenes mediante workflows JSON
4. **API de generación**: Integración REST para producción

### Audiencia

Este documento está dirigido a:

-   Ingenieros de Machine Learning con experiencia en PyTorch
-   Data Scientists familiarizados con fine-tuning de modelos generativos
-   Desarrolladores que necesitan desplegar pipelines de inferencia

---

## 2. Requisitos Previos

### 2.1 Cuenta en Vast.ai

1. **Crear cuenta**: Registrarse en [vast.ai](https://vast.ai)
2. **Verificar identidad**: Completar verificación KYC si es requerida
3. **Añadir método de pago**: Tarjeta de crédito o PayPal
4. **Verificar saldo**: Asegurar fondos suficientes (mínimo $5-10 recomendado)

### 2.2 Acceso SSH

**Generar par de claves SSH** (si no existe):

```bash
# Linux/Mac
ssh-keygen -t ed25519 -C "vast-ai-key" -f ~/.ssh/vast_ai_key

# Windows (PowerShell)
ssh-keygen -t ed25519 -C "vast-ai-key" -f $env:USERPROFILE\.ssh\vast_ai_key
```

**Añadir clave pública a Vast.ai**:

1. Copiar contenido de `~/.ssh/vast_ai_key.pub` (Linux/Mac) o `%USERPROFILE%\.ssh\vast_ai_key.pub` (Windows)
2. En Vast.ai: Settings → SSH Keys → Add SSH Key
3. Pegar la clave pública

### 2.3 Templates Preconfigurados en Vast.ai

Vast.ai ofrece instancias con templates preconfigurados que incluyen el software necesario ya instalado y configurado. Esto elimina la necesidad de instalar manualmente KOHYA o ComfyUI.

**Para Entrenamientos (Fine-tuning y LoRAs)**:

-   **Template**: "Kohya's GUI" o "Kohya_ss"
-   **Incluye**: KOHYA_ss con interfaz web, PyTorch, CUDA, xformers, y todas las dependencias
-   **Acceso**: Interfaz web en puerto 7860 (típicamente)
-   **Ventaja**: No requiere instalación manual, listo para usar inmediatamente

**Para Inferencias (ComfyUI)**:

-   **Template**: "ComfyUI"
-   **Incluye**: ComfyUI con todas las dependencias, PyTorch, CUDA
-   **Acceso**: Interfaz web en puerto 8188 (por defecto)
-   **Ventaja**: Despliegue inmediato, sin configuración adicional

**Este repositorio** (workflows y API):

```bash
git clone https://github.com/SValduezaL/TFM_Outfit_AI_Generator_Fortnite.git
cd TFM_Outfit_AI_Generator_Fortnite
```

**Nota**: Si prefieres instalar manualmente, puedes usar instancias sin template y seguir los pasos de las secciones 5 y 7. Sin embargo, usar templates preconfigurados es más rápido y recomendado.

### 2.4 Conocimientos Previos

-   **Python 3.11+**: Sintaxis, entornos virtuales, pip
-   **PyTorch**: Conceptos básicos de tensores, CUDA
-   **Linux**: Comandos básicos de terminal, SSH, gestión de procesos
-   **Git**: Clonación, commits básicos
-   **Stable Diffusion**: Conceptos de difusión, latents, VAE, U-Net

---

## 3. Selección de GPU en Vast.ai

### 3.1 Criterios de Selección

Los requisitos de hardware varían según la tarea:

| Tarea                  | VRAM Mínima | VRAM Óptima | GPU Recomendada      | CPU      | Disco  | Coste/hora (aprox.) |
| ---------------------- | ----------- | ----------- | -------------------- | -------- | ------ | ------------------- |
| **Fine-tuning SDXL**   | 40GB        | 80GB        | A100 40GB, A100 80GB | 8+ cores | 100GB+ | $1.00-3.00          |
| **LoRA SDXL**          | 40GB        | 80GB        | A100 40GB, A100 80GB | 4+ cores | 50GB+  | $1.00-3.00          |
| **Inferencia ComfyUI** | 8GB         | 16GB+       | RTX 4060, RTX 4090   | 2+ cores | 30GB+  | $0.20-0.80          |

### 3.2 Fine-tuning de SDXL

**Requisitos técnicos**:

-   **VRAM**: 40GB mínimo, 80GB recomendado
    -   **Razón**: SDXL Base (~7GB) + optimizador AdamW (~14GB) + batch size 2-4 + overhead = ~40GB mínimo
    -   **Con batch size 8**: Requiere 80GB VRAM (A100 80GB)
-   **GPU**: **A100 40GB o A100 80GB** (recomendado para este proyecto)
    -   **A100 40GB**: Suficiente para batch_size 2-4, entrenamiento estable
    -   **A100 80GB**: Permite batch_size 8+, entrenamiento más rápido
-   **CPU**: 8+ cores para data loading paralelo
-   **Disco**: 100GB+ (modelo base 7GB + dataset + checkpoints intermedios)
-   **Red**: 100+ Mbps para descarga de modelos

**Filtros en Vast.ai**:

```
GPU: A100
VRAM: >= 40GB
Price: < $3.00/hour
Disk: >= 100GB
Internet: Up
Template: Kohya's GUI (opcional pero recomendado)
```

**Estimación de coste**:

-   **A100 40GB**: ~$1.50-2.50/hora
-   **A100 80GB**: ~$2.00-3.00/hora
-   **Tiempo de entrenamiento**: 3-8 horas para 1000-1500 imágenes, 5 epochs
-   **Coste total**: $4.50-24.00 por entrenamiento completo (según GPU y duración)

### 3.3 Entrenamiento de LoRA

**Requisitos técnicos**:

-   **VRAM**: 40GB mínimo, 80GB recomendado
    -   **Razón**: Modelo base (~7GB) + LoRA adapters (~2GB) + optimizador (~6GB) + batch size 4-8 = ~40GB mínimo
    -   **Con batch size 8+**: Requiere 80GB VRAM (A100 80GB)
-   **GPU**: **A100 40GB o A100 80GB** (mismas GPU que fine-tuning)
    -   **A100 40GB**: Suficiente para batch_size 4-6, entrenamiento estable
    -   **A100 80GB**: Permite batch_size 8+, entrenamiento más rápido
-   **CPU**: 4+ cores suficiente
-   **Disco**: 50GB+ (modelo base + dataset pequeño + LoRA output ~10-50MB)
-   **Red**: 50+ Mbps

**Filtros en Vast.ai**:

```
GPU: A100
VRAM: >= 40GB
Price: < $3.00/hour
Disk: >= 50GB
Internet: Up
Template: Kohya's GUI (opcional pero recomendado)
```

**Estimación de coste**:

-   **A100 40GB**: ~$1.50-2.50/hora
-   **A100 80GB**: ~$2.00-3.00/hora
-   **Tiempo de entrenamiento**: 1-2 horas para 20-50 imágenes, 10-20 epochs
-   **Coste total**: $1.50-6.00 por LoRA (según GPU y duración)

### 3.4 Inferencia con ComfyUI

**Requisitos técnicos**:

-   **VRAM**: 8GB mínimo, 16GB+ recomendado
    -   **Razón**: Modelo base (~7GB) + LoRA en memoria (~50MB) + latents temporales = ~8GB mínimo
    -   **Con múltiples LoRAs o resolución alta**: 16GB+ recomendado
-   **GPU**: **RTX 4060 o RTX 4090** (usadas en este proyecto)
    -   **RTX 4060 (16GB)**: Suficiente para inferencia estándar, buena relación coste/rendimiento
    -   **RTX 4090 (24GB)**: Máximo rendimiento, permite batch processing y resoluciones altas
-   **CPU**: 2+ cores suficiente
-   **Disco**: 30GB+ (modelos + ComfyUI)
-   **Red**: 50+ Mbps

**Filtros en Vast.ai**:

```
GPU: RTX 4060, RTX 4090
VRAM: >= 8GB
Price: < $0.80/hour
Disk: >= 30GB
Internet: Up
Template: ComfyUI (opcional pero recomendado)
```

**Estimación de coste**:

-   **RTX 4060**: ~$0.20-0.40/hora
-   **RTX 4090**: ~$0.50-0.80/hora
-   **Uso continuo**: $0.20-0.80/hora según GPU
-   **1000 inferencias/día**: ~$4.80-19.20/día (asumiendo 1 minuto por inferencia)

### 3.5 Cómo Usar Filtros en Vast.ai

1. **Acceder a "Create"** en el dashboard de Vast.ai
2. **Configurar filtros**:
    - **GPU**: Seleccionar modelo específico o rango
    - **VRAM**: Mínimo requerido
    - **Price**: Máximo por hora
    - **Disk**: Espacio mínimo
    - **Internet**: Marcar "Up" para conexión activa
3. **Ordenar por precio** o disponibilidad
4. **Verificar especificaciones** antes de alquilar:
    - Click en instancia → Verificar VRAM real, CPU, disco disponible

---

## 4. Provisionamiento

### 4.1 Crear Oferta / Alquilar Instancia

**Método 1: Alquilar instancia existente con template** (más rápido y recomendado):

1. En Vast.ai: **"Create"** → Aplicar filtros → Seleccionar instancia
2. **Configurar**:
    - **Template**:
        - Para entrenamientos: Seleccionar **"Kohya's GUI"** o **"Kohya_ss"**
        - Para inferencias: Seleccionar **"ComfyUI"**
    - **Disk**: Ajustar según necesidad (mínimo según sección 3)
    - **Jupyter**: Opcional (útil para debugging)
3. **Click "Rent"**
4. **Esperar aprovisionamiento**: 1-5 minutos

**Ventajas de usar templates**:

-   Software preinstalado y configurado
-   Sin necesidad de instalación manual
-   Acceso inmediato a interfaz web
-   Configuración optimizada para la tarea

**Método 2: Alquilar instancia sin template** (si prefieres instalación manual):

1. En Vast.ai: **"Create"** → Aplicar filtros → Seleccionar instancia
2. **Configurar**:
    - **Image**: `pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime` (o similar con CUDA)
    - **Disk**: Ajustar según necesidad
    - **Jupyter**: Opcional
3. **Click "Rent"**
4. **Esperar aprovisionamiento**: 1-5 minutos
5. **Seguir pasos de instalación manual** (secciones 5 y 7)

**Método 2: Crear oferta** (más económico, puede tardar más):

1. En Vast.ai: **"Create"** → **"Create Offer"**
2. **Configurar requisitos**:
    - GPU, VRAM, precio máximo, disco
3. **Esperar aceptación**: Puede tardar minutos u horas

### 4.2 Obtener IP Pública y Credenciales

**Después del aprovisionamiento**:

1. **IP pública**: Visible en el dashboard de la instancia
    - Formato: `ssh.paperspacegradient.com` o IP directa
2. **Puerto SSH**: Generalmente `22` (verificar en detalles de instancia)
3. **Usuario**: Típicamente `root` o `paperspace` (verificar en detalles)
4. **Clave SSH**: La clave privada que añadiste a Vast.ai

### 4.3 Conexión SSH

**Comando de conexión**:

```bash
# Linux/Mac
ssh -i ~/.ssh/vast_ai_key root@<IP_PUBLICA> -p <PUERTO>

# Windows (PowerShell)
ssh -i $env:USERPROFILE\.ssh\vast_ai_key root@<IP_PUBLICA> -p <PUERTO>
```

**Ejemplo real**:

```bash
ssh -i ~/.ssh/vast_ai_key root@ssh.paperspacegradient.com -p 22
```

**Primera conexión**:

-   Aceptar fingerprint SSH (escribir `yes`)
-   Si falla: Verificar que la clave privada tiene permisos correctos:
    ```bash
    chmod 600 ~/.ssh/vast_ai_key
    ```

### 4.4 Configuración de Seguridad

**Firewall/puertos**:

Vast.ai gestiona el firewall automáticamente. Para aplicaciones personalizadas:

1. **Puerto SSH (22)**: Abierto por defecto
2. **Puerto Jupyter (8888)**: Abierto si habilitaste Jupyter
3. **Puerto ComfyUI (8188)**: Necesita configuración manual si accedes externamente

**Abrir puerto para ComfyUI** (si es necesario):

```bash
# En la instancia Vast.ai
# Verificar si ufw está activo
ufw status

# Si está activo, abrir puerto 8188
ufw allow 8188/tcp
ufw reload
```

**⚠️ Advertencia**: Abrir puertos expone servicios. Usa autenticación o VPN si es posible.

### 4.5 Verificación Inicial

**Verificar GPU**:

```bash
nvidia-smi
```

**Salida esperada** (ejemplo con A100 40GB):

```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.xx.xx    Driver Version: 525.xx.xx    CUDA Version: 12.0  |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  A100-SXM4-40GB      Off   | 00000000:00:04.0 Off |                  Off |
| N/A   45C    P0    45W / 400W |      0MiB / 40960MiB |      0%      Default |
+-----------------------------------------------------------------------------+
```

**Para inferencia (RTX 4060 o RTX 4090)**, la salida mostrará:

-   RTX 4060: `40960MiB` (16GB) o similar
-   RTX 4090: `24576MiB` (24GB) o similar

**Verificar CUDA**:

```bash
nvcc --version
# O
python3 -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

**Verificar disco**:

```bash
df -h
```

**Verificar CPU y RAM**:

```bash
lscpu
free -h
```

---

## 5. Configuración del Servidor para KOHYA

### 5.1 Usando Template Preconfigurado (Recomendado)

Si alquilaste una instancia con template **"Kohya's GUI"**, el software ya está instalado y configurado:

1. **Acceder a la interfaz web**:

    - URL proporcionada en el dashboard de Vast.ai (típicamente puerto 7860)
    - O mediante túnel SSH: `ssh -L 7860:localhost:7860 -i ~/.ssh/vast_ai_key root@<IP_PUBLICA>`
    - Acceder desde navegador: `http://localhost:7860`

2. **Verificar instalación**:

    ```bash
    # Conectar por SSH
    ssh -i ~/.ssh/vast_ai_key root@<IP_PUBLICA>

    # Verificar que KOHYA está instalado
    which kohya_gui.py
    # O verificar proceso
    ps aux | grep kohya
    ```

3. **Ubicación típica de archivos**:
    - KOHYA: `/workspace/kohya_ss` o `/root/kohya_ss`
    - Modelos: `/workspace/models` o `/root/models`
    - Output: `/workspace/output` o `/root/output`

**Ventajas del template**:

-   No requiere instalación manual
-   Entorno optimizado y probado
-   Interfaz web lista para usar
-   Actualizaciones gestionadas automáticamente

### 5.2 Instalación Manual (Solo si no usas template)

Si prefieres instalar manualmente o la instancia no tiene template:

### 5.2.1 Entorno Python / Conda

**Opción 1: Entorno virtual Python** (recomendado para KOHYA):

```bash
# Actualizar sistema
apt-get update && apt-get install -y python3-pip python3-venv git wget

# Crear entorno virtual
python3 -m venv /root/kohya_env
source /root/kohya_env/bin/activate

# Verificar Python
python --version  # Debe ser 3.10 o 3.11
```

**Opción 2: Conda** (alternativa):

```bash
# Instalar Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
source $HOME/miniconda3/bin/activate

# Crear entorno
conda create -n kohya python=3.11 -y
conda activate kohya
```

### 5.2.2 Instalación de Dependencias Base

**CUDA y Drivers**:

Los drivers NVIDIA ya están instalados en imágenes de Vast.ai. Verificar versión:

```bash
nvidia-smi | grep "Driver Version"
```

**PyTorch con CUDA**:

```bash
# Activar entorno virtual
source /root/kohya_env/bin/activate

# Instalar PyTorch (ajustar según CUDA disponible)
# Para CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Para CUDA 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verificar instalación
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

**Dependencias adicionales**:

```bash
# xformers (optimización de atención)
pip install xformers

# accelerate (distributed training)
pip install accelerate

# Otros paquetes comunes
pip install transformers diffusers datasets pillow numpy
```

### 5.2.3 Clonación y Configuración de KOHYA

```bash
# Clonar repositorio
cd /root
git clone https://github.com/kohya-ss/sd-scripts.git
cd sd-scripts

# Instalar dependencias
pip install -r requirements.txt

# Instalar KOHYA
pip install -e .
```

**Configurar accelerate** (para distributed training):

```bash
accelerate config
```

**Respuestas recomendadas**:

-   **Infer the config**: `No`
-   **Which type of machine**: `This machine`
-   **Do you want to run your training on CPU only**: `No`
-   **Do you want to use DeepSpeed**: `No`
-   **Do you want to use FullyShardedDataParallel**: `No`
-   **Do you want to use Megatron-LM**: `No`
-   **How many different machines will you use**: `1`
-   **What is the rank of this machine**: `0`
-   **What is the IP address of the machine**: `localhost`
-   **What is the port**: `29500`
-   **Do you want to optimize your script with torch dynamo**: `No`
-   **Do you want to use fp16 or bf16 mixed precision**: `fp16` (o `bf16` si GPU soporta)

### 5.4 Configuración de Dataset

**Estructura de directorios**:

```bash
# Crear estructura
mkdir -p /root/training_data
mkdir -p /root/training_data/images
mkdir -p /root/training_data/reg_images  # Si usas regularización
```

**Subir dataset**:

**Opción 1: SCP desde máquina local**:

```bash
# Desde tu máquina local
scp -i ~/.ssh/vast_ai_key -r /ruta/local/dataset root@<IP_PUBLICA>:/root/training_data/
```

**Opción 2: Git LFS** (si dataset está en repositorio):

```bash
# En la instancia Vast.ai
cd /root/training_data
git lfs install
git clone https://github.com/USERNAME/dataset-repo.git
```

**Opción 3: wget/curl** (si dataset está en URL pública):

```bash
cd /root/training_data
wget https://example.com/dataset.zip
unzip dataset.zip
```

**Estructura final esperada**:

```
/root/training_data/
├── images/
│   ├── image1.png
│   ├── image1.txt          # Caption
│   ├── image2.png
│   └── image2.txt
└── reg_images/             # Opcional
    ├── reg1.png
    └── reg2.png
```

### 5.5 Archivo de Configuración TOML

**Crear archivo de configuración** (`/root/training_data/config.toml`):

```toml
[general]
enable_bucket = true
shuffle_caption = true
keep_tokens = 1

[[datasets]]
resolution = 1024
batch_size = 2

  [[datasets.subsets]]
  image_dir = '/root/training_data/images'
  caption_extension = '.txt'
  num_repeats = 10

  # Regularización (opcional)
  [[datasets.subsets]]
  is_reg = true
  image_dir = '/root/training_data/reg_images'
  class_tokens = 'fortnite character'
  num_repeats = 1
```

**Explicación de parámetros**:

-   **`resolution`**: Resolución de entrenamiento. Para SDXL: 1024. Afecta VRAM: 1024×1024 requiere ~24GB, 512×512 requiere ~12GB.
-   **`batch_size`**: Imágenes por batch. Afecta VRAM: batch_size 1 = ~20GB, batch_size 2 = ~24GB, batch_size 4 = ~32GB.
-   **`num_repeats`**: Repeticiones por imagen. Si tienes 100 imágenes y num_repeats=10, el modelo verá 1000 imágenes por epoch.
-   **`enable_bucket`**: Aspect Ratio Bucketing. Permite entrenar con diferentes aspect ratios sin recortar.

### 5.6 Ejecución de Fine-tuning

**Comando base**:

```bash
# Activar entorno
source /root/kohya_env/bin/activate
cd /root/sd-scripts

# Fine-tuning completo de SDXL
accelerate launch --num_cpu_threads_per_process=8 train_network.py \
  --pretrained_model_name_or_path=/root/models/sd_xl_base_1.0.safetensors \
  --dataset_config=/root/training_data/config.toml \
  --output_dir=/root/output/finetune \
  --output_name=humanoid_06 \
  --save_model_as=safetensors \
  --save_precision=fp16 \
  --save_every_n_epochs=1 \
  --mixed_precision=fp16 \
  --full_fp16 \
  --gradient_checkpointing \
  --learning_rate=1e-6 \
  --lr_scheduler=cosine \
  --lr_warmup_steps=100 \
  --max_train_epochs=5 \
  --optimizer_type=AdamW8bit \
  --xformers \
  --cache_latents \
  --cache_latents_to_disk \
  --seed=42 \
  --max_token_length=225 \
  --clip_skip=2 \
  --logging_dir=/root/logs \
  --log_prefix=ft_humanoid_
```

**Explicación de parámetros críticos**:

-   **`--pretrained_model_name_or_path`**: Ruta al modelo base SDXL. Debe estar descargado previamente.
-   **`--learning_rate`**: 1e-6 es conservador. Para fine-tuning completo: 1e-6 a 5e-6. Para LoRA: 1e-4 a 5e-4.
-   **`--max_train_epochs`**: Número de epochs. Para fine-tuning: 3-10 epochs. Para LoRA: 10-50 epochs.
-   **`--full_fp16`**: Entrenamiento en fp16 completo. Reduce VRAM pero puede afectar estabilidad.
-   **`--gradient_checkpointing`**: Trade-off memoria/velocidad. Reduce VRAM ~30% pero aumenta tiempo ~20%.
-   **`--cache_latents`**: Cachea latents en RAM. Reduce VRAM y acelera entrenamiento.

**Descargar modelo base** (si no está disponible):

```bash
mkdir -p /root/models
cd /root/models

# Opción 1: Hugging Face (requiere token)
huggingface-cli download stabilityai/stable-diffusion-xl-base-1.0 --local-dir sd_xl_base_1.0

# Opción 2: wget directo (si hay URL pública)
wget https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors
```

### 5.7 Monitoreo y Logs

**TensorBoard** (recomendado):

```bash
# En otra terminal SSH (o con screen/tmux)
source /root/kohya_env/bin/activate
tensorboard --logdir=/root/logs --port=6006 --host=0.0.0.0
```

**Acceder a TensorBoard**:

-   URL: `http://<IP_PUBLICA>:6006`
-   O mediante túnel SSH: `ssh -L 6006:localhost:6006 -i ~/.ssh/vast_ai_key root@<IP_PUBLICA>`

**Verificar uso de GPU durante entrenamiento**:

```bash
# Monitoreo continuo
watch -n 1 nvidia-smi

# O en otra terminal
watch -n 1 'nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv'
```

**Logs de entrenamiento**:

Los logs se guardan en `/root/logs/`. Revisar:

```bash
tail -f /root/logs/ft_humanoid_*/events.out.tfevents.*
```

**Verificar checkpoints guardados**:

```bash
ls -lh /root/output/finetune/
```

### 5.8 Verificación de GPU en Uso

**Durante entrenamiento, `nvidia-smi` debe mostrar**:

```
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI   PID   Type   Process name                  GPU Memory     |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A  1234    C   python3                          22000MiB     |
+-----------------------------------------------------------------------------+
```

**Si la GPU no se usa**:

-   Verificar que PyTorch detecta CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
-   Verificar que el modelo se carga en GPU (revisar logs)
-   Verificar que `accelerate config` está correcto

---

## 6. Entrenamiento Fine-Tuning y LoRA

### 6.1 Fine-Tuning Completo de SDXL

**Workflow completo**:

```bash
# 1. Preparar entorno
source /root/kohya_env/bin/activate
cd /root/sd-scripts

# 2. Verificar dataset
ls -la /root/training_data/images/ | head -10

# 3. Ejecutar entrenamiento
accelerate launch --num_cpu_threads_per_process=8 train_network.py \
  --pretrained_model_name_or_path=/root/models/sd_xl_base_1.0.safetensors \
  --dataset_config=/root/training_data/config.toml \
  --output_dir=/root/output/finetune \
  --output_name=humanoid_06 \
  --save_model_as=safetensors \
  --save_precision=fp16 \
  --save_every_n_epochs=1 \
  --mixed_precision=fp16 \
  --gradient_checkpointing \
  --learning_rate=1e-6 \
  --lr_scheduler=cosine \
  --lr_warmup_steps=100 \
  --max_train_epochs=5 \
  --optimizer_type=AdamW8bit \
  --xformers \
  --cache_latents \
  --seed=42 \
  --max_token_length=225 \
  --clip_skip=2 \
  --logging_dir=/root/logs \
  --log_prefix=ft_humanoid_

# 4. Monitorear progreso
# En otra terminal: tensorboard --logdir=/root/logs
```

**Archivos generados**:

-   **Checkpoint final**: `/root/output/finetune/humanoid_06.safetensors`
-   **Checkpoints intermedios**: `/root/output/finetune/humanoid_06-000001.safetensors`, etc.
-   **Logs**: `/root/logs/ft_humanoid_*/`

**Adaptación de valores según VRAM (A100)**:

| GPU       | VRAM | batch_size | resolution | gradient_checkpointing | full_fp16 |
| --------- | ---- | ---------- | ---------- | ---------------------- | --------- |
| A100 40GB | 40GB | 2-4        | 1024       | Opcional               | Sí        |
| A100 80GB | 80GB | 4-8        | 1024       | No                     | Sí        |

**Nota**: Con A100 40GB puedes usar batch_size 2-4 cómodamente. Con A100 80GB puedes usar batch_size 8+ para entrenamiento más rápido.

**Razón técnica**:

-   `batch_size` afecta memoria linealmente: cada imagen adicional requiere ~4-6GB VRAM.
-   `resolution` afecta memoria cuadráticamente: 1024×1024 requiere 4× más memoria que 512×512.
-   `gradient_checkpointing` reduce memoria ~30% pero aumenta tiempo ~20% (trade-off).

### 6.2 Entrenamiento de LoRA

**Comando para LoRA**:

```bash
source /root/kohya_env/bin/activate
cd /root/sd-scripts

accelerate launch --num_cpu_threads_per_process=4 train_network.py \
  --pretrained_model_name_or_path=/root/models/humanoid_05 \
  --dataset_config=/root/training_data/lora_animal/config.toml \
  --output_dir=/root/output/lora \
  --output_name=animal_v10 \
  --save_model_as=safetensors \
  --save_precision=fp16 \
  --save_every_n_epochs=2 \
  --mixed_precision=fp16 \
  --gradient_checkpointing \
  --learning_rate=1e-4 \
  --lr_scheduler=cosine \
  --lr_warmup_steps=50 \
  --max_train_epochs=20 \
  --optimizer_type=AdamW8bit \
  --xformers \
  --cache_latents \
  --network_module=networks.lora \
  --network_dim=128 \
  --network_alpha=64 \
  --network_args "rank_dropout=0.0" "module_dropout=0.0" "algo=loragon" \
  --seed=42 \
  --max_token_length=225 \
  --clip_skip=2 \
  --logging_dir=/root/logs \
  --log_prefix=lora_animal_
```

**Parámetros específicos de LoRA**:

-   **`--network_module=networks.lora`**: Habilita LoRA (no fine-tuning completo).
-   **`--network_dim`**: Dimensión del rank. Valores comunes: 32, 64, 128, 256.
    -   **32-64**: LoRAs pequeños (~5-10MB), menos capacidad.
    -   **128**: Balance recomendado (~20MB), buena capacidad.
    -   **256**: LoRAs grandes (~40MB), máxima capacidad pero riesgo de overfitting.
-   **`--network_alpha`**: Escala de LoRA. Típicamente `network_dim / 2` o igual a `network_dim`.
    -   **Razón**: Controla la fuerza de la adaptación. Alpha mayor = adaptación más fuerte.
-   **`--learning_rate`**: Para LoRA: 1e-4 a 5e-4 (10-50× mayor que fine-tuning completo).
    -   **Razón**: LoRA solo entrena adaptadores pequeños, necesita LR más alto para converger.

**Adaptación según dataset**:

| Tamaño Dataset | epochs | num_repeats | learning_rate |
| -------------- | ------ | ----------- | ------------- |
| 10-20 imágenes | 30-50  | 20-30       | 1e-4          |
| 20-50 imágenes | 20-30  | 10-20       | 1e-4          |
| 50+ imágenes   | 10-20  | 5-10        | 5e-4          |

**Razón técnica**:

-   Datasets pequeños necesitan más epochs y repeats para evitar overfitting y asegurar convergencia.
-   Datasets grandes pueden usar menos epochs y repeats, pero requieren LR más alto para converger rápido.

### 6.3 Archivos JSON de Configuración

**KOHYA no genera JSON directamente**, pero puedes crear scripts de configuración:

**Ejemplo: `train_finetune.sh`**:

```bash
#!/bin/bash
source /root/kohya_env/bin/activate
cd /root/sd-scripts

accelerate launch --num_cpu_threads_per_process=8 train_network.py \
  --pretrained_model_name_or_path=/root/models/sd_xl_base_1.0.safetensors \
  --dataset_config=/root/training_data/config.toml \
  --output_dir=/root/output/finetune \
  --output_name=humanoid_06 \
  --save_model_as=safetensors \
  --save_precision=fp16 \
  --save_every_n_epochs=1 \
  --mixed_precision=fp16 \
  --gradient_checkpointing \
  --learning_rate=1e-6 \
  --lr_scheduler=cosine \
  --lr_warmup_steps=100 \
  --max_train_epochs=5 \
  --optimizer_type=AdamW8bit \
  --xformers \
  --cache_latents \
  --seed=42 \
  --max_token_length=225 \
  --clip_skip=2 \
  --logging_dir=/root/logs \
  --log_prefix=ft_humanoid_
```

**Hacer ejecutable**:

```bash
chmod +x train_finetune.sh
./train_finetune.sh
```

### 6.4 Descarga de Modelos Entrenados

**Desde instancia Vast.ai a máquina local**:

```bash
# Desde tu máquina local
scp -i ~/.ssh/vast_ai_key root@<IP_PUBLICA>:/root/output/finetune/humanoid_06.safetensors ./

# Para LoRA
scp -i ~/.ssh/vast_ai_key root@<IP_PUBLICA>:/root/output/lora/animal_v10.safetensors ./
```

**O mediante rsync** (más eficiente para múltiples archivos):

```bash
rsync -avz -e "ssh -i ~/.ssh/vast_ai_key" root@<IP_PUBLICA>:/root/output/ ./output/
```

---

## 7. Provisionamiento de GPU para Inferencia

### 7.1 Selección de GPU para Inferencia

**GPU suficiente para inferencia**:

-   **RTX 4060 (16GB)**: Recomendado para este proyecto, buena relación coste/rendimiento
    -   Suficiente para inferencia estándar (1024×1024, 32 steps)
    -   Permite múltiples LoRAs en memoria simultáneamente
-   **RTX 4090 (24GB)**: Máximo rendimiento
    -   Permite batch processing y resoluciones altas
    -   Ideal para producción con alto volumen

**Razón**: Inferencia requiere menos VRAM que entrenamiento. Modelo base (~7GB) + LoRA (~50MB) + latents temporales (~1-2GB) = ~8-9GB mínimo. RTX 4060 con 16GB ofrece margen cómodo.

**Filtros Vast.ai para inferencia**:

```
GPU: RTX 4060, RTX 4090
VRAM: >= 8GB
Price: < $0.80/hour
Disk: >= 30GB
Internet: Up
Template: ComfyUI (opcional pero recomendado)
```

### 7.2 Despliegue de ComfyUI

**Opción 1: Usando Template Preconfigurado (Recomendado)**

Si alquilaste una instancia con template **"ComfyUI"**, el software ya está instalado:

1. **Acceder a la interfaz web**:

    - URL proporcionada en el dashboard de Vast.ai (típicamente puerto 8188)
    - O mediante túnel SSH: `ssh -L 8188:localhost:8188 -i ~/.ssh/vast_ai_key root@<IP_PUBLICA>`
    - Acceder desde navegador: `http://localhost:8188`

2. **Verificar instalación**:

    ```bash
    # Conectar por SSH
    ssh -i ~/.ssh/vast_ai_key root@<IP_PUBLICA>

    # Verificar que ComfyUI está corriendo
    ps aux | grep comfy
    # O verificar puerto
    netstat -tulpn | grep 8188
    ```

3. **Ubicación típica de archivos**:
    - ComfyUI: `/workspace/ComfyUI` o `/root/ComfyUI`
    - Modelos: `/workspace/ComfyUI/models` o `/root/ComfyUI/models`

**Opción 2: Instalación Manual** (Solo si no usas template)

```bash
# Actualizar sistema
apt-get update && apt-get install -y python3-pip git wget

# Clonar ComfyUI
cd /root
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI

# Instalar dependencias
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

**Descargar modelos necesarios**:

```bash
# Crear directorios
mkdir -p /root/ComfyUI/models/checkpoints
mkdir -p /root/ComfyUI/models/loras
mkdir -p /root/ComfyUI/models/vae

# Descargar modelos base (ajustar URLs según disponibilidad)
cd /root/ComfyUI/models/checkpoints

# SDXL Base
wget https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors

# SDXL Refiner (opcional)
wget https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/sd_xl_refiner_1.0.safetensors

# Modelo fine-tuned (subir desde máquina local)
# scp -i ~/.ssh/vast_ai_key humanoid_05.safetensors root@<IP>:/root/ComfyUI/models/checkpoints/

# LoRAs (subir desde máquina local)
# scp -i ~/.ssh/vast_ai_key animal_v10.safetensors root@<IP>:/root/ComfyUI/models/loras/
```

**Configurar rutas de modelos**:

ComfyUI busca modelos en:

-   **Checkpoints**: `ComfyUI/models/checkpoints/`
-   **LoRAs**: `ComfyUI/models/loras/`
-   **VAE**: `ComfyUI/models/vae/`

No requiere configuración adicional si los modelos están en estas rutas.

### 7.3 Ejecución de ComfyUI

**Modo desarrollo (con UI web)**:

```bash
cd /root/ComfyUI
python main.py --listen 0.0.0.0 --port 8188
```

**Acceder a UI**:

-   URL: `http://<IP_PUBLICA>:8188`
-   O mediante túnel SSH: `ssh -L 8188:localhost:8188 -i ~/.ssh/vast_ai_key root@<IP_PUBLICA>`

**Modo headless (solo API)**:

```bash
cd /root/ComfyUI
python main.py --listen 0.0.0.0 --port 8188 --disable-smart-memory
```

**Ejecutar en background (screen/tmux)**:

```bash
# Instalar screen
apt-get install -y screen

# Crear sesión
screen -S comfyui

# Ejecutar ComfyUI
cd /root/ComfyUI
python main.py --listen 0.0.0.0 --port 8188

# Desconectar: Ctrl+A, luego D
# Reconectar: screen -r comfyui
```

### 7.4 Configuración de Workflows

**Subir workflows JSON**:

```bash
# Desde máquina local
scp -i ~/.ssh/vast_ai_key API-ComfyUI-FT_Humanoid_5e_vF__SDXL_Refiner.json root@<IP_PUBLICA>:/root/ComfyUI/
```

**Estructura de workflow JSON**:

Los workflows de ComfyUI son JSON que definen nodos y conexiones. Ejemplo mínimo:

```json
{
    "1": {
        "inputs": {
            "ckpt_name": "humanoid_05.safetensors"
        },
        "class_type": "CheckpointLoaderSimple"
    },
    "2": {
        "inputs": {
            "text": "fortnite character, epic, cinematic",
            "clip": ["1", 0]
        },
        "class_type": "CLIPTextEncode"
    },
    "3": {
        "inputs": {
            "text": "low quality, blurry",
            "clip": ["1", 0]
        },
        "class_type": "CLIPTextEncode"
    },
    "4": {
        "inputs": {
            "seed": 42,
            "steps": 32,
            "cfg": 7.0,
            "sampler_name": "euler",
            "scheduler": "normal",
            "denoise": 1.0,
            "model": ["1", 0],
            "positive": ["2", 0],
            "negative": ["3", 0],
            "latent_image": ["5", 0]
        },
        "class_type": "KSampler"
    },
    "5": {
        "inputs": {
            "width": 1024,
            "height": 1024,
            "batch_size": 1
        },
        "class_type": "EmptyLatentImage"
    },
    "6": {
        "inputs": {
            "samples": ["4", 0],
            "vae": ["1", 2]
        },
        "class_type": "VAEDecode"
    },
    "7": {
        "inputs": {
            "filename_prefix": "ComfyUI",
            "images": ["6", 0]
        },
        "class_type": "SaveImage"
    }
}
```

**Modificar workflows programáticamente**:

Ver sección 8.2 para integración con API.

### 7.5 Docker vs Entorno Local

**Docker** (opcional, más complejo):

```bash
# Crear Dockerfile
cat > /root/ComfyUI/Dockerfile << EOF
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y python3-pip git
WORKDIR /app
COPY . .
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip install -r requirements.txt

EXPOSE 8188
CMD ["python", "main.py", "--listen", "0.0.0.0", "--port", "8188"]
EOF

# Construir imagen
docker build -t comfyui .

# Ejecutar
docker run --gpus all -p 8188:8188 comfyui
```

**Entorno local** (recomendado, más simple):

Seguir pasos de sección 7.2 (sin Docker).

---

## 8. Generación de Imágenes y API

### 8.1 Inferencia con ComfyUI desde Línea de Comandos

**API REST de ComfyUI**:

ComfyUI expone API REST en `http://<IP>:8188/`.

**Enviar workflow**:

```bash
# Cargar workflow
WORKFLOW=$(cat /root/ComfyUI/API-ComfyUI-FT_Humanoid_5e_vF__SDXL_Refiner.json)

# Enviar prompt
curl -X POST http://localhost:8188/prompt \
  -H "Content-Type: application/json" \
  -d "{\"prompt\": $WORKFLOW, \"client_id\": \"test\"}"
```

**Obtener resultado**:

```bash
# Obtener historial
PROMPT_ID="<ID_RETORNADO_ANTERIORMENTE>"
curl http://localhost:8188/history/$PROMPT_ID
```

**Descargar imagen**:

```bash
# Obtener imagen
curl "http://localhost:8188/view?filename=<FILENAME>&subfolder=&type=output" \
  -o output.png
```

### 8.2 Integración con API FastAPI

**Cliente ComfyUI en Python**:

El repositorio incluye `comfy_client.py`. Ejemplo de uso:

```python
from comfy_client import ComfyClient

client = ComfyClient(
    base_url="http://<IP_PUBLICA>:8188",
    default_workflow="API-ComfyUI-FT_Humanoid_5e_vF__SDXL_Refiner.json"
)

# Generar imagen
image_bytes = await client.generate_image(
    prompt="fortnite character, epic, cinematic",
    negative_prompt="low quality, blurry",
    width=1024,
    height=1024,
    steps=32,
    cfg=7.0,
    seed=42
)
```

**API FastAPI completa**:

Ver `5.API_User_Interface/main.py` para implementación completa.

**Desplegar API en instancia Vast.ai**:

```bash
# Clonar repositorio
cd /root
git clone https://github.com/USERNAME/TFM_Outfit_AI_Generator_Fortnite.git
cd TFM_Outfit_AI_Generator_Fortnite/5.API_User_Interface

# Instalar dependencias
pip install -r ../../requirements.txt

# Configurar .env
cat > .env << EOF
COMFYUI_URL=http://localhost:8188
OPENAI_API_KEY=tu_key_aqui
API_HOST=0.0.0.0
API_PORT=8000
EOF

# Ejecutar API
python main.py
```

**Acceder a API**:

-   URL: `http://<IP_PUBLICA>:8000`
-   Docs: `http://<IP_PUBLICA>:8000/docs`

### 8.3 Formatos de Entrada/Salida

**Entrada (workflow JSON)**:

-   **Prompt**: String de texto
-   **Negative prompt**: String de texto
-   **Dimensiones**: width, height (múltiplos de 64)
-   **Parámetros**: steps, cfg, seed

**Salida**:

-   **Formato**: PNG (por defecto)
-   **Resolución**: Según parámetros de entrada
-   **Bytes**: Array de bytes de imagen

**Ejemplo de request a API**:

```bash
curl -X POST http://<IP_PUBLICA>:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "fortnite character, epic",
    "negative_prompt": "low quality",
    "width": 1024,
    "height": 1024,
    "steps": 32,
    "cfg": 7.0
  }' \
  --output result.png
```

---

## 9. Recomendaciones de Coste y Rendimiento

### 9.1 Cálculo de Coste Estimado

**Fórmula general**:

```
Coste Total = (Tiempo de Ejecución en horas) × (Precio por hora) + (Overhead de setup)
```

**Ejemplo: Fine-tuning SDXL**:

-   **GPU**: A100 40GB a $2.00/hora
-   **Tiempo**: 6 horas (1500 imágenes, 5 epochs, batch_size=4)
-   **Coste**: 6 × $2.00 = **$12.00**

**Ejemplo: LoRA**:

-   **GPU**: A100 40GB a $2.00/hora
-   **Tiempo**: 1.5 horas (30 imágenes, 20 epochs, batch_size=6)
-   **Coste**: 1.5 × $2.00 = **$3.00**

**Ejemplo: Inferencia continua**:

-   **GPU**: RTX 4060 a $0.30/hora
-   **Uso**: 24 horas/día
-   **Coste diario**: 24 × $0.30 = **$7.20/día**
-   **Coste mensual**: $7.20 × 30 = **$216/mes**

### 9.2 Consejos para Reducir Coste

**1. Optimizar batch size**:

-   **Aumentar batch_size** reduce tiempo total (más imágenes por iteración).
-   **Trade-off**: Requiere más VRAM (GPU más cara).
-   **Ejemplo**: batch_size 4 vs 2: ~2× más rápido, pero requiere GPU con +8GB VRAM.

**2. Usar cache de latents**:

-   **`--cache_latents`**: Cachea latents en RAM, reduce tiempo de carga de datos.
-   **Ahorro**: ~10-20% de tiempo total.
-   **Requisito**: RAM suficiente (16GB+ recomendado).

**3. Entrenar en resolución menor**:

-   **512×512 vs 1024×1024**: ~4× más rápido, pero menor calidad.
-   **Uso**: Para prototipado rápido, luego refinar en 1024×1024.

**4. Usar spot instances** (si disponible):

-   **Ahorro**: 50-70% de coste.
-   **Riesgo**: Instancia puede terminar abruptamente.
-   **Mitigación**: Guardar checkpoints frecuentemente (`--save_every_n_epochs=1`).

**5. Apagar instancia cuando no se usa**:

-   **Inferencia**: Apagar cuando no hay requests.
-   **Entrenamiento**: Dejar corriendo hasta completar.

### 9.3 Tiempos de Entrenamiento Estimados

**Fine-tuning SDXL** (1500 imágenes, 1024×1024):

| GPU       | VRAM | batch_size | Tiempo/epoch | 5 epochs |
| --------- | ---- | ---------- | ------------ | -------- |
| A100 40GB | 40GB | 4          | ~1.2h        | ~6h      |
| A100 80GB | 80GB | 8          | ~0.6h        | ~3h      |

**LoRA** (30 imágenes, 1024×1024):

| GPU       | VRAM | batch_size | Tiempo/epoch | 20 epochs |
| --------- | ---- | ---------- | ------------ | --------- |
| A100 40GB | 40GB | 6          | ~0.8min      | ~16min    |
| A100 80GB | 80GB | 8          | ~0.6min      | ~12min    |

**Inferencia** (1024×1024, 32 steps):

| GPU      | VRAM | Tiempo por imagen |
| -------- | ---- | ----------------- |
| RTX 4060 | 16GB | ~20-30s           |
| RTX 4090 | 24GB | ~12-18s           |

**Factores que afectan tiempo**:

-   **Resolución**: 1024×1024 es ~4× más lento que 512×512.
-   **Steps**: 32 steps es ~2× más lento que 16 steps.
-   **Batch size**: Afecta entrenamiento, no inferencia (inferencia siempre batch_size=1 típicamente).

---

## 10. Solución de Problemas Comunes

### 10.1 Problemas de CUDA

**Error: "CUDA out of memory"**:

```bash
# Verificar uso de GPU
nvidia-smi

# Soluciones:
# 1. Reducir batch_size
# 2. Reducir resolution
# 3. Habilitar gradient_checkpointing
# 4. Habilitar --cache_latents
# 5. Usar --full_fp16
```

**Error: "CUDA driver version is insufficient"**:

```bash
# Verificar versión de driver
nvidia-smi | grep "Driver Version"

# Verificar versión de CUDA requerida por PyTorch
python -c "import torch; print(torch.version.cuda)"

# Solución: Usar imagen Docker con drivers más recientes o actualizar drivers
```

**Error: "No CUDA GPUs are available"**:

```bash
# Verificar que GPU es visible
nvidia-smi

# Verificar que PyTorch detecta CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Solución: Reinstalar PyTorch con CUDA correcto
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 10.2 Fallos de Memoria

**Error: "RuntimeError: out of memory"**:

**Diagnóstico**:

```bash
# Monitorear memoria durante entrenamiento
watch -n 1 nvidia-smi
```

**Soluciones progresivas**:

1. **Reducir batch_size**: De 4 a 2, o de 2 a 1.
2. **Reducir resolution**: De 1024×1024 a 512×512.
3. **Habilitar gradient_checkpointing**: `--gradient_checkpointing`
4. **Habilitar cache_latents**: `--cache_latents` (reduce VRAM de VAE)
5. **Usar fp16**: `--mixed_precision=fp16 --full_fp16`
6. **Cerrar otros procesos**: Verificar con `nvidia-smi` si hay otros procesos usando GPU.

**Error: "OSError: [Errno 28] No space left on device"**:

```bash
# Verificar espacio en disco
df -h

# Limpiar espacio
# 1. Eliminar checkpoints antiguos
rm /root/output/finetune/humanoid_06-000001.safetensors  # Mantener solo el último

# 2. Limpiar cache de pip
pip cache purge

# 3. Limpiar logs antiguos
rm -rf /root/logs/ft_humanoid_*/events.out.tfevents.*
```

### 10.3 Conexión SSH

**Error: "Permission denied (publickey)"**:

```bash
# Verificar que la clave privada tiene permisos correctos
chmod 600 ~/.ssh/vast_ai_key

# Verificar que la clave pública está en Vast.ai
# Settings → SSH Keys → Verificar que está añadida

# Intentar conexión con verbose para debug
ssh -v -i ~/.ssh/vast_ai_key root@<IP_PUBLICA>
```

**Error: "Connection timed out"**:

```bash
# Verificar que la IP es correcta
# Verificar que el puerto es correcto (generalmente 22)
# Verificar firewall local (si aplica)

# Probar con telnet
telnet <IP_PUBLICA> <PUERTO>
```

**Error: "Host key verification failed"**:

```bash
# Eliminar entrada antigua de known_hosts
ssh-keygen -R <IP_PUBLICA>

# O editar ~/.ssh/known_hosts manualmente
```

### 10.4 Permisos en Vast.ai

**Error: "Insufficient funds"**:

-   Verificar saldo en Vast.ai dashboard
-   Añadir método de pago
-   Verificar límites de gasto diario

**Error: "Instance terminated unexpectedly"**:

-   **Causa común**: Saldo insuficiente o límite de gasto alcanzado
-   **Solución**: Verificar saldo y límites, añadir fondos

**Error: "Cannot connect to instance"**:

-   Verificar que la instancia está "Running" en dashboard
-   Verificar que la IP no ha cambiado (puede cambiar si se reinicia)
-   Verificar logs de la instancia en dashboard de Vast.ai

### 10.5 Problemas Específicos de KOHYA

**Error: "ModuleNotFoundError: No module named 'xformers'"**:

```bash
# Instalar xformers
pip install xformers

# O deshabilitar xformers en comando (usar --mem_eff_attn en su lugar)
```

**Error: "ValueError: num_samples should be a positive integer"**:

-   **Causa**: Dataset vacío o rutas incorrectas
-   **Solución**: Verificar que `image_dir` en `config.toml` es correcto y contiene imágenes

**Error: "RuntimeError: Expected all tensors to be on the same device"**:

-   **Causa**: Modelo o datos no están en GPU
-   **Solución**: Verificar que `accelerate config` está correcto, reiniciar entrenamiento

### 10.6 Problemas Específicos de ComfyUI

**Error: "Model not found"**:

```bash
# Verificar que el modelo está en la ruta correcta
ls -la /root/ComfyUI/models/checkpoints/

# Verificar que el nombre en workflow JSON coincide con el archivo
```

**Error: "Port 8188 already in use"**:

```bash
# Encontrar proceso usando el puerto
lsof -i :8188
# O
netstat -tulpn | grep 8188

# Matar proceso
kill -9 <PID>
```

**Error: "CUDA out of memory" en inferencia**:

-   **Causa**: Modelo demasiado grande para GPU
-   **Soluciones**:
    1. Usar GPU con más VRAM
    2. Reducir resolución de inferencia
    3. Usar `--disable-smart-memory` (puede ayudar en algunos casos)

---

## 11. Referencias y Recursos Adicionales

### 11.1 Documentación Oficial

-   **KOHYA_ss**: https://github.com/kohya-ss/sd-scripts
-   **ComfyUI**: https://github.com/comfyanonymous/ComfyUI
-   **Vast.ai**: https://vast.ai/help
-   **Stable Diffusion XL**: https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0

### 11.2 Comunidades y Foros

-   **KOHYA Discord**: https://discord.gg/kohya-ss
-   **ComfyUI Discord**: https://discord.gg/comfyanonymous
-   **Reddit**: r/StableDiffusion, r/LocalLLaMA

### 11.3 Herramientas Útiles

-   **TensorBoard**: Monitoreo de entrenamiento
-   **Weights & Biases**: Alternativa a TensorBoard con más features
-   **Screen/Tmux**: Gestión de sesiones SSH persistentes

---

## 12. Checklist de Despliegue

### 12.1 Antes de Empezar

-   [ ] Cuenta en Vast.ai creada y verificada
-   [ ] Método de pago añadido
-   [ ] Claves SSH generadas y añadidas a Vast.ai
-   [ ] Repositorios clonados localmente (para referencia)

### 12.2 Configuración Inicial

-   [ ] Instancia GPU alquilada en Vast.ai con template apropiado
    -   [ ] Para entrenamientos: Template "Kohya's GUI" seleccionado
    -   [ ] Para inferencias: Template "ComfyUI" seleccionado
-   [ ] Conexión SSH establecida
-   [ ] GPU verificada (`nvidia-smi`)
    -   [ ] Para entrenamientos: A100 40GB o A100 80GB confirmada
    -   [ ] Para inferencias: RTX 4060 o RTX 4090 confirmada
-   [ ] CUDA verificada (`python -c "import torch; print(torch.cuda.is_available())"`)
-   [ ] Si usas template: Interfaz web accesible
-   [ ] Si instalación manual: Entorno Python/Conda creado y KOHYA/ComfyUI instalado

### 12.3 Para Entrenamiento

-   [ ] Dataset preparado y subido
-   [ ] Archivo `config.toml` creado
-   [ ] Modelo base descargado
-   [ ] Comando de entrenamiento preparado
-   [ ] TensorBoard configurado (opcional)
-   [ ] Sesión screen/tmux iniciada (para persistencia)

### 12.4 Para Inferencia

-   [ ] ComfyUI instalado y configurado
-   [ ] Modelos (base + LoRAs) descargados
-   [ ] Workflows JSON subidos
-   [ ] ComfyUI ejecutándose y accesible
-   [ ] API FastAPI desplegada (si aplica)
-   [ ] Prueba de generación exitosa

### 12.5 Post-Entrenamiento

-   [ ] Checkpoints descargados a máquina local
-   [ ] Logs descargados (si necesario)
-   [ ] Instancia apagada o liberada
-   [ ] Coste verificado en dashboard de Vast.ai

---

**Última actualización**: 2025-01-XX

**Mantenido por**: Equipo TFM Outfit AI Generator Fortnite
