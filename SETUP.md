# Configuración del Entorno Virtual

## Entorno Virtual

El proyecto utiliza un entorno virtual de Python 3.11 con el nombre: **`.venv_tfm_skin_ai`**

## Instalación

### 1. Crear y activar el entorno virtual

**Windows (PowerShell):**

```powershell
# Crear el entorno virtual (ya creado)
python -m venv .venv_tfm_skin_ai

# Activar el entorno virtual
.\.venv_tfm_skin_ai\Scripts\Activate.ps1
```

**Windows (CMD):**

```cmd
.venv_tfm_skin_ai\Scripts\activate.bat
```

**Linux/Mac:**

```bash
source .venv_tfm_skin_ai/bin/activate
```

### 2. Instalar dependencias

Una vez activado el entorno virtual, instala todas las dependencias:

```bash
pip install -r requirements.txt
```

### 3. Configurar variables de entorno

Asegúrate de tener el archivo `.env` en el directorio raíz con tus API keys:

```env
FORTNITE_API_KEY=tu_api_key_aqui
GOOGLE_GEMINI_API_KEY=tu_api_key_aqui
```

## Verificación

Para verificar que el entorno está activado correctamente:

```bash
python --version  # Debe mostrar Python 3.11.x
pip list          # Debe mostrar todas las dependencias instaladas
```

## Desactivar el entorno virtual

Cuando termines de trabajar, puedes desactivar el entorno virtual:

```bash
deactivate
```

## Configurar Jupyter Notebooks

Para usar el entorno virtual en Jupyter Notebooks, el kernel ya está registrado con el nombre **"Python (TFM Skin AI)"**.

### Seleccionar el kernel en Jupyter

1. Abre tu notebook en Jupyter
2. Ve a **Kernel** → **Change Kernel** → **Python (TFM Skin AI)**
3. El notebook ahora usará el entorno virtual `.venv_tfm_skin_ai`

### Re-registrar el kernel (si es necesario)

Si necesitas volver a registrar el kernel:

```bash
# Activar el entorno virtual primero
.\.venv_tfm_skin_ai\Scripts\Activate.ps1

# Registrar el kernel
python -m ipykernel install --user --name=venv_tfm_skin_ai --display-name "Python (TFM Skin AI)"
```

## Notas

-   El entorno virtual está incluido en `.gitignore` y no se subirá al repositorio
-   El archivo `requirements.txt` contiene todas las dependencias con versiones específicas, incluyendo `ipykernel` para Jupyter
-   Asegúrate de activar el entorno virtual antes de ejecutar cualquier script o notebook
-   El kernel de Jupyter está configurado para usar este entorno virtual automáticamente
