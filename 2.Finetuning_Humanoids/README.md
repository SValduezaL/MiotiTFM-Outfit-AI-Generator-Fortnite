# Fine-tuning de Modelo Base para Generación de Outfits Fortnite - TFM Skin AI Gen Fortnite

## 📋 Overview del Proyecto

Este documento describe el proceso de fine-tuning del modelo base Stable Diffusion XL (SDXL) sobre datos de Fortnite para establecer la base estilística necesaria para la generación de outfits. Este fine-tuning constituye la primera etapa del pipeline, previa al entrenamiento de LoRAs especializados.

### Contexto Académico

**Problema abordado:** Establecer una base estilística coherente que capture las características visuales distintivas de Fortnite (proporciones, iluminación, texturas, anatomía) mediante fine-tuning completo del modelo base SDXL.

**Solución propuesta:** Fine-tuning del modelo completo SDXL Base 1.0 sobre un dataset curado de imágenes de personajes de Fortnite, seguido de iteraciones experimentales para optimizar hiperparámetros y lograr el balance óptimo entre fidelidad al estilo Fortnite y capacidad de generalización.

---

## 🏗️ Arquitectura y Enfoque

### Modelo Base

El fine-tuning se realizó sobre **Stable Diffusion XL Base 1.0** (`sd_xl_base_1.0.safetensors`), modelo generativo de difusión desarrollado por Stability AI.

**Justificación del enfoque:**

1. **Fine-tuning completo vs. LoRA:** A diferencia de los LoRAs especializados posteriores, el fine-tuning completo modifica todos los pesos del modelo, estableciendo una base estilística profunda que afecta todas las generaciones subsiguientes.
2. **Base para especialización:** Este modelo fine-tuned (`humanoid_05`) sirve como base para todos los LoRAs especializados, permitiendo que estos se enfoquen en variaciones temáticas sin perder la identidad visual de Fortnite.
3. **Ventajas técnicas:**
    - **Coherencia estilística:** El modelo base aprende características generales del arte de Fortnite
    - **Preservación de capacidades:** Mantiene la capacidad de generar personajes humanos estándar
    - **Base estable:** Proporciona una base sólida para entrenamientos LoRA posteriores

### Rol de KOHYA

**KOHYA_ss** (`kohya-ss/sd-scripts`) es el framework utilizado para el fine-tuning. Proporciona:

-   Implementación optimizada de fine-tuning para Stable Diffusion XL
-   Gestión avanzada de datasets con Aspect Ratio Bucketing
-   Configuración granular de parámetros de entrenamiento
-   Integración con TensorBoard y W&B para monitoreo
-   Soporte para múltiples optimizadores y schedulers de learning rate
-   Fine-tuning selectivo de componentes (U-Net, Text Encoder)

---

## 📊 Dataset y Preparación de Datos

### Estructura del Dataset

El dataset utilizado para el fine-tuning consta de:

-   **Número de imágenes:** 1486 imágenes únicas
-   **Resolución:** 1024×1024 píxeles
-   **Formato de imagen:** RGB (3 canales)
-   **Contenido:** Personajes distintos de Fortnite
-   **Formato de captions:** Archivos `.txt` asociados a cada imagen (mismo nombre base)

**Características del dataset:**

-   Imágenes de personajes de Fortnite procesadas desde assets originales
-   Resolución nativa de SDXL (1024×1024), óptima para calidad y eficiencia
-   Cada imagen tiene un caption descriptivo asociado en formato texto plano
-   Dataset diverso que cubre diferentes tipos de personajes y outfits de Fortnite

**⚠️ NOTA:** La siguiente información debe extraerse de los archivos JSON de configuración de entrenamiento:

-   Número de repeats por imagen utilizado en cada checkpoint
-   Uso de imágenes de regularización (si aplica)
-   Estructura específica de los captions (convenciones de etiquetado)

### Procesamiento de Datos

**Pipeline de preparación:**

1. **Obtención de datos:** Imágenes de personajes de Fortnite desde assets originales
2. **Preprocesamiento:** Redimensionamiento y normalización a 1024×1024 píxeles
3. **Captioning:** Generación de archivos `.txt` con descripciones de cada personaje
4. **Validación:** Verificación de correspondencia entre imágenes y captions

**Características técnicas:**

-   Resolución fija 1024×1024 (sin Aspect Ratio Bucketing necesario al ser resolución uniforme)
-   Formato RGB estándar (3 canales de color)
-   Captions en formato texto plano (`.txt`)

### Captioning

Los captions para fine-tuning incluyen:

-   Descripción de características visuales clave del personaje
-   Tags de estilo Fortnite
-   Anatomía y estructura del personaje
-   Elementos de outfit y accesorios
-   Características distintivas que permiten al modelo aprender el estilo visual de Fortnite

---

## 🎯 Estrategia de Fine-tuning

### Pipeline Completo

```
SDXL Base 1.0
    ↓
Fine-tuning Iterativo
    ├── humanoid_02 (experimento inicial)
    ├── humanoid_03 (ajuste de hiperparámetros)
    ├── humanoid_04 (optimización)
    ├── humanoid_05 ✅ (seleccionado como modelo final)
    ├── humanoid_06 (experimento adicional)
    └── humanoid_07 (experimento adicional)
    ↓
Modelo Base para LoRAs Especializados
```

### Parámetros de Entrenamiento

**⚠️ NOTA:** Los hiperparámetros específicos deben extraerse de los archivos JSON de configuración de entrenamiento. La siguiente tabla muestra parámetros típicos para fine-tuning SDXL, pero **deben verificarse con los archivos de configuración reales**.

| Parámetro                  | Valor Típico (SDXL Fine-tuning) | Justificación                                             |
| -------------------------- | ------------------------------- | --------------------------------------------------------- |
| **Resolución**             | 1024×1024                       | Resolución nativa de SDXL, óptima para calidad            |
| **Optimizer**              | AdamW8bit                       | Balance entre precisión y uso de memoria                  |
| **LR Scheduler**           | Cosine                          | Decaimiento suave del learning rate                       |
| **Mixed Precision**        | fp16                            | Reducción de memoria sin pérdida significativa de calidad |
| **Noise Offset**           | 0.05-0.1                        | Mejora contraste y saturación de colores                  |
| **Caption Dropout**        | 0.05-0.1                        | Regularización para evitar overfitting a captions         |
| **Flip Augmentation**      | true                            | Aumento de datos mediante volteo horizontal               |
| **XFormers**               | true                            | Optimización de atención para eficiencia                  |
| **Aspect Ratio Bucketing** | enabled                         | Permite diferentes aspect ratios dentro de buckets        |
| **Min Bucket Reso**        | 512                             | Resolución mínima para buckets                            |
| **Max Bucket Reso**        | 2048                            | Resolución máxima para buckets                            |
| **Bucket Reso Steps**      | 64                              | Intervalo de resolución para buckets                      |
| **Max Token Length**       | 225                             | Soporte para captions largos                              |
| **Clip Skip**              | 1                               | Uso de última capa de CLIP (estándar SDXL)                |
| **Loss Type**              | L2                              | Función de pérdida estándar                               |
| **Huber Schedule**         | SNR                             | Weighting basado en Signal-to-Noise Ratio                 |
| **Save Format**            | safetensors                     | Formato seguro y eficiente                                |

**⚠️ IMPORTANTE:** Los valores reales de los siguientes parámetros **DEBEN extraerse de los archivos JSON de configuración:**

-   Learning rate (U-Net)
-   Learning rate (Text Encoder)
-   Batch size
-   Gradient accumulation steps
-   Epochs / Total steps
-   Entrenamiento del Text Encoder (habilitado/deshabilitado, porcentaje de entrenamiento)
-   Regularizaciones específicas (noise offset exacto, caption dropout exacto)

---

## 🔬 Proceso Iterativo de Fine-tuning

### Evolución de Experimentos

Se observan múltiples checkpoints (`humanoid_02` a `humanoid_07`), indicando un proceso iterativo de experimentación. Sin acceso a los archivos JSON de configuración, no es posible documentar los cambios específicos entre iteraciones.

**Información requerida de los archivos JSON:**

Para cada checkpoint (`humanoid_02` a `humanoid_07`), se necesita extraer:

1. **Hiperparámetros de entrenamiento:**

    - Learning rate (U-Net)
    - Learning rate (Text Encoder)
    - Batch size
    - Gradient accumulation steps
    - Epochs
    - Total steps

2. **Configuración del dataset:**

    - Número de imágenes
    - Repeats por imagen
    - Resolución de entrenamiento

3. **Regularizaciones:**

    - Noise offset
    - Caption dropout rate
    - Otras técnicas de regularización aplicadas

4. **Problemas detectados y soluciones:**

    - Overfitting
    - Pérdida de identidad Fortnite
    - Incoherencia visual
    - Otros problemas observados

5. **Justificación de cambios:**
    - Por qué se modificaron ciertos hiperparámetros
    - Qué problemas se buscaban corregir
    - Cómo las decisiones afectaron la calidad del modelo

### Metodología de Evaluación

La evaluación de cada iteración se realizó mediante:

1. **Generación de muestras durante entrenamiento:** Cada epoch o conjunto de steps generaba imágenes de prueba con prompts estándar
2. **Análisis visual cualitativo:**
    - Coherencia con estilo Fortnite
    - Calidad anatómica
    - Presencia de artefactos o deformaciones
    - Fidelidad a características visuales de Fortnite
3. **Detección de problemas:**
    - **Overfitting:** Generaciones demasiado similares a imágenes de entrenamiento
    - **Underfitting:** Falta de características distintivas de Fortnite
    - **Pérdida de identidad Fortnite:** Desviación excesiva del estilo base
    - **Ruido estilístico:** Inconsistencias visuales entre generaciones

---

## 📈 Selección del Modelo Final

### Modelo Seleccionado: `humanoid_05`

**Justificación de la selección:**

El modelo `humanoid_05` fue seleccionado como modelo base final para los entrenamientos LoRA posteriores. Esta selección se basó en:

1. **Balance óptimo:** Equilibrio entre fidelidad al estilo Fortnite y capacidad de generalización
2. **Estabilidad:** Comportamiento estable en generaciones de prueba
3. **Base para LoRAs:** Capacidad demostrada de servir como base sólida para especializaciones temáticas mediante LoRAs

**⚠️ NOTA:** La justificación detallada y los parámetros específicos de entrenamiento de `humanoid_05` deben extraerse de los archivos JSON de configuración correspondientes.

### Características del Modelo Final

El modelo `humanoid_05` aprendió una distribución visual que captura:

1. **Estilo Fortnite:**

    - Proporciones características de personajes
    - Iluminación y sombreado distintivos
    - Texturas y materiales del juego
    - Anatomía coherente con el arte de Fortnite

2. **Coherencia de outfit:**

    - Generación coherente de prendas y accesorios
    - Integración adecuada de elementos del outfit
    - Mantenimiento de estilo consistente

3. **Base para especialización:**
    - Capacidad de servir como base para LoRAs temáticos
    - Preservación de características generales mientras permite especialización

### Uso Posterior

Este modelo base se utiliza como punto de partida para todos los entrenamientos LoRA especializados documentados en `3.LoRAs/README.md`. Los LoRAs actúan como adaptadores que modulan el comportamiento del modelo base hacia categorías temáticas específicas (Animal, Food, Robots, Star Wars, Fuzzy Bear) sin comprometer la identidad base establecida por el fine-tuning.

---

## ⚠️ Limitaciones y Trabajo Futuro

### Limitaciones Identificadas

1. **Documentación incompleta:** Los archivos JSON de configuración de entrenamiento no están disponibles en la estructura de directorios actual, limitando la documentación precisa de hiperparámetros y decisiones de ingeniería.

2. **Dependencia del dataset:** La calidad del modelo final depende críticamente de la calidad y diversidad del dataset de entrenamiento.

3. **Recursos computacionales:** El fine-tuning completo requiere recursos significativos (GPU con VRAM suficiente) y tiempo de entrenamiento extenso.

### Trabajo Futuro

1. **Documentación completa:** Recuperar y documentar los archivos JSON de configuración de entrenamiento para cada checkpoint experimental.

2. **Análisis comparativo:** Realizar análisis comparativo detallado entre los diferentes checkpoints (`humanoid_02` a `humanoid_07`) para entender la evolución del proceso.

3. **Métricas cuantitativas:** Implementar métricas objetivas (FID, CLIP Score) además de evaluación cualitativa para futuros fine-tunings.

4. **Optimización de hiperparámetros:** Automatizar búsqueda de hiperparámetros óptimos mediante técnicas de optimización bayesiana o grid search sistemático.

5. **Expansión del dataset:** Evaluar el impacto de expandir el dataset de entrenamiento en la calidad y generalización del modelo.

---

## 📚 Referencias Técnicas

-   **KOHYA_ss:** [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts)
-   **Stable Diffusion XL:** Stability AI - [Hugging Face](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
-   **Fine-tuning Techniques:** Documentación de KOHYA sobre fine-tuning de modelos completos
-   **Aspect Ratio Bucketing:** NovelAI implementation

---

## 📝 Notas de Implementación

**⚠️ NOTA:** La siguiente información debe completarse con datos de los archivos JSON de configuración:

-   **Hardware utilizado:** [A extraer de archivos JSON]
-   **Tiempo de entrenamiento:** [A extraer de archivos JSON]
-   **Framework:** KOHYA_ss con soporte para SDXL
-   **Formato de salida:** SafeTensors (fp16)

---

## 🔍 Información Requerida para Completar Documentación

Para completar esta documentación con precisión técnica, se requieren los siguientes archivos JSON de configuración de entrenamiento de KOHYA:

1. **Archivos de configuración por checkpoint:**

    - `humanoid_02_*.json` (o equivalente)
    - `humanoid_03_*.json`
    - `humanoid_04_*.json`
    - `humanoid_05_*.json` ✅ (checkpoint seleccionado)
    - `humanoid_06_*.json`
    - `humanoid_07_*.json`

2. **Información a extraer de cada archivo:**

    - Modelo base utilizado
    - Resolución de entrenamiento
    - Batch size
    - Gradient accumulation steps
    - Learning rate (U-Net)
    - Learning rate (Text Encoder)
    - Epochs / Total steps
    - Optimizador
    - Scheduler de learning rate
    - Entrenamiento del Text Encoder (habilitado/deshabilitado, porcentaje)
    - Regularizaciones (noise offset, caption dropout, etc.)
    - Configuración del dataset (número de imágenes, repeats, etc.)

3. **Logs de entrenamiento (opcional pero recomendado):**
    - Pérdidas durante entrenamiento
    - Imágenes de muestra generadas durante entrenamiento
    - Observaciones y problemas detectados

---

**Autores:** Odreman Ferrer y Sergio Valdueza - TFM Deep Learning MIOTI  
**Licencia:** CC BY-NC-SA 4.0  
**Última actualización:** Diciembre 2025

**Estado del documento:** ⚠️ **INCOMPLETO** - Requiere acceso a archivos JSON de configuración de entrenamiento para completar información técnica precisa.
