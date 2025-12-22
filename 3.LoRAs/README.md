# Entrenamiento de LoRAs Especializados - TFM Skin AI Gen Fortnite

## 📋 Overview del Proyecto

Este proyecto de Fin de Máster (TFM) tiene como objetivo la generación de imágenes de outfits de Fortnite mediante Inteligencia Artificial, utilizando una estrategia de fine-tuning inicial seguida de entrenamientos LoRA especializados para homogeneizar y controlar estilos temáticos específicos.

### Contexto Académico

**Problema abordado:** Control estilístico y coherencia visual en la generación de outfits de Fortnite mediante modelos generativos. El desafío principal radica en mantener la identidad visual característica de Fortnite mientras se especializa en diferentes categorías temáticas (animales, comida, robots, Star Wars, Fuzzy Bear).

**Solución propuesta:** Arquitectura híbrida que combina fine-tuning del modelo base SDXL sobre datos de Fortnite, seguido de entrenamientos LoRA especializados por categoría temática. Esta aproximación permite mantener la coherencia del estilo base mientras se añade especialización estilística mediante adaptadores ligeros.

---

## 🏗️ Arquitectura y Enfoque General

### Modelo Base

Todos los LoRAs se entrenan sobre el modelo fine-tuned **`v1x0_fortnite_humanoid_sdxl1_vae_fix-000005`**, que a su vez fue entrenado sobre **Stable Diffusion XL Base 1.0** (`sd_xl_base_1.0.safetensors`).

**Justificación del enfoque Fine-tuning + LoRA:**

1. **Fine-tuning inicial:** Establece la base estilística de Fortnite, capturando características generales del arte del juego (proporciones, iluminación, texturas, anatomía).
2. **LoRAs especializados:** Permiten especialización temática sin comprometer la identidad base. Cada LoRA actúa como un adaptador que modula el comportamiento del modelo base hacia una categoría específica.
3. **Ventajas técnicas:**
   - **Eficiencia:** Los LoRAs (~10-50MB) son mucho más ligeros que reentrenar el modelo completo (~7GB).
   - **Modularidad:** Cada categoría temática puede actualizarse independientemente.
   - **Combinabilidad:** Múltiples LoRAs pueden combinarse para estilos híbridos.
   - **Preservación:** El modelo base mantiene su capacidad de generar personajes humanos estándar.

### Rol de KOHYA

**KOHYA_ss** (`kohya-ss/sd-scripts`) es el framework utilizado para el entrenamiento de LoRAs. Proporciona:

- Implementación optimizada de LoRA para Stable Diffusion XL
- Gestión avanzada de datasets con Aspect Ratio Bucketing
- Configuración granular de parámetros de entrenamiento
- Integración con TensorBoard y W&B para monitoreo
- Soporte para múltiples optimizadores y schedulers de learning rate

---

## 📊 Dataset y Preparación de Datos

### Estructura de Datasets

Cada LoRA se entrenó con datasets específicos ubicados en `1.Datasets LoRAs/`:

| LoRA | Imágenes | Resolución | Formato Captions |
|------|----------|------------|------------------|
| **Animal** | 42 | 1024×1024 | `.txt` |
| **Food** | 27 | 1024×1024 | `.txt` |
| **FuzzyBear** | 8 | 1024×1024 | `.txt` |
| **Robots** | 15 | 1024×1024 | `.txt` |
| **StarWars** | 19 | 1024×1024 | `.txt` |

**Características del dataset:**
- Imágenes procesadas desde assets originales de Fortnite
- Captions manuales con etiquetas descriptivas específicas
- Formato: imagen `.png` + caption `.txt` con mismo nombre base
- Sin regularización explícita (no se utilizaron imágenes de regularización)

### Captioning

Los captions fueron creados manualmente siguiendo convenciones específicas:
- Inclusión de triggers temáticos (ej: `fortnite_animal_character`)
- Descripción de características visuales clave
- Tags de calidad y estilo Fortnite
- Anatomía y estructura del personaje

---

## 🎯 Estrategia de Entrenamiento

### Pipeline Completo

```
SDXL Base 1.0
    ↓
Fine-tuning (v1x0_fortnite_humanoid_sdxl1_vae_fix-000005)
    ↓
LoRA Animal ──┐
LoRA Food ────┤
LoRA FuzzyBear├──► Inferencia con ComfyUI
LoRA Robots ──┤
LoRA StarWars ─┘
```

### Parámetros Comunes de Entrenamiento

Todos los entrenamientos comparten la siguiente configuración base:

| Parámetro | Valor | Justificación |
|-----------|-------|---------------|
| **Resolución** | 1024×1024 | Resolución nativa de SDXL, óptima para calidad |
| **Optimizer** | AdamW8bit | Balance entre precisión y uso de memoria |
| **LR Scheduler** | Cosine | Decaimiento suave del learning rate |
| **Mixed Precision** | fp16 | Reducción de memoria sin pérdida significativa de calidad |
| **Noise Offset** | 0.05 | Mejora contraste y saturación de colores |
| **Caption Dropout** | 0.1 | Regularización para evitar overfitting a captions |
| **Flip Augmentation** | true | Aumento de datos mediante volteo horizontal |
| **XFormers** | true | Optimización de atención para eficiencia |
| **Aspect Ratio Bucketing** | enabled | Permite diferentes aspect ratios dentro de buckets |
| **Min Bucket Reso** | 512 | Resolución mínima para buckets |
| **Max Bucket Reso** | 2048 | Resolución máxima para buckets (2048 para mayoría, 1024 para StarWars) |
| **Bucket Reso Steps** | 64 | Intervalo de resolución para buckets |
| **Max Token Length** | 225 | Soporte para captions largos |
| **Clip Skip** | 1 | Uso de última capa de CLIP (estándar SDXL) |
| **Loss Type** | L2 | Función de pérdida estándar |
| **Huber Schedule** | SNR | Weighting basado en Signal-to-Noise Ratio |
| **Save Format** | safetensors | Formato seguro y eficiente |

---

## 🔬 Entrenamiento de LoRAs con KOHYA

### 1. LoRA Animal

**Versión final seleccionada:** `v3`

#### Evolución de Parámetros

| Versión | Epochs | Steps | Batch | Rank | Alpha | LR (U-Net) | LR (Text Encoder) | Observaciones |
|---------|--------|-------|-------|------|-------|------------|-------------------|--------------|
| **v1** | 20 | 4200 | 4 | 16 | 8 | 8e-5 | 4e-5 | Entrenamiento inicial, 20 repeats |
| **v2** | 40 | 4200 | 4 | 16 | 8 | 8e-5 | 4e-5 | Aumento de epochs para mayor convergencia, 10 repeats |
| **v3** ✅ | 15 | 4725 | 4 | 16 | 8 | 8e-5 | 4e-5 | Reducción de epochs, aumento de steps (30 repeats) |

**Parámetros finales (v3):**
- **Dataset:** 42 imágenes, 30 repeats
- **Batch Size:** 4
- **Epochs:** 15
- **Total Steps:** 4725
- **Network Rank (Dimension):** 16
- **Network Alpha:** 8 (ratio efectivo: 0.5)
- **Learning Rate (U-Net):** 8e-5
- **Learning Rate (Text Encoder):** 4e-5 (50% del U-Net)
- **Max Bucket Reso:** 2048

**Justificación de la selección:**
- Balance óptimo entre convergencia y generalización
- Ratio alpha/rank de 0.5 proporciona suficiente capacidad sin overfitting
- Learning rate conservador (8e-5) evita inestabilidad
- 15 epochs con 30 repeats proporciona exposición suficiente sin memorización

**Problemas detectados y soluciones:**
- **v1 → v2:** Overfitting temprano detectado → Aumento de epochs con menos repeats para mejor generalización
- **v2 → v3:** Pérdida de identidad Fortnite en algunas generaciones → Reducción de epochs, aumento de repeats para mayor exposición por imagen

---

### 2. LoRA Food

**Versión final seleccionada:** `v8`

#### Evolución de Parámetros

| Versión | Epochs | Steps | Batch | Rank | Alpha | LR (U-Net) | LR (Text Encoder) | Observaciones |
|---------|--------|-------|-------|------|-------|------------|-------------------|--------------|
| **v1** | 12 | 6480 | 2 | 32 | 16 | 1e-4 | 5e-5 | Rank alto, learning rate estándar |
| **v2** | 20 | 4050 | 4 | 8 | 4 | 1e-4 | 5e-5 | Reducción drástica de rank |
| **v3** | 20 | 4050 | 4 | 32 | 16 | 1e-4 | 5e-5 | Vuelta a rank alto |
| **v4** | 40 | 3240 | 4 | 8 | 4 | 1.5e-5 | 5e-5 | LR muy bajo, rank bajo |
| **v5** | 40 | 3240 | 4 | 16 | 8 | 1.5e-5 | 5e-5 | Aumento de rank, LR bajo |
| **v6** | 40 | 3240 | 4 | 8 | 4 | 5e-5 | 5e-5 | LR intermedio |
| **v7** | 40 | 3240 | 4 | 8 | 4 | 5e-4 | 5e-5 | LR muy alto (experimental) |
| **v8** ✅ | 30 | 4050 | 4 | 16 | 8 | 8e-5 | 4e-5 | Configuración final balanceada |
| **v9** | 10 | 4050 | 2 | 16 | 8 | 8e-5 | 4e-5 | Batch reducido, epochs mínimos |

**Parámetros finales (v8):**
- **Dataset:** 27 imágenes, 20 repeats
- **Batch Size:** 4
- **Epochs:** 30
- **Total Steps:** 4050
- **Network Rank (Dimension):** 16
- **Network Alpha:** 8 (ratio efectivo: 0.5)
- **Learning Rate (U-Net):** 8e-5
- **Learning Rate (Text Encoder):** 4e-5
- **Max Bucket Reso:** 2048

**Justificación de la selección:**
- Configuración que balancea capacidad (rank 16) con estabilidad (LR 8e-5)
- 30 epochs proporcionan convergencia adecuada sin overfitting
- Ratio alpha/rank de 0.5 mantiene el efecto del LoRA controlado

**Problemas detectados y soluciones:**
- **v1 → v2:** Rank 32 causaba overfitting y artefactos → Reducción a rank 8
- **v2 → v3:** Rank 8 insuficiente para capturar detalles complejos de comida → Vuelta a rank 32
- **v3 → v4-v7:** Experimentación con learning rates extremos → Inestabilidad o convergencia lenta
- **v4-v7 → v8:** Configuración balanceada con rank 16 y LR 8e-5 → Resultados estables y coherentes
- **v8 → v9:** Prueba con menos epochs → Insuficiente convergencia, se mantiene v8

---

### 3. LoRA FuzzyBear

**Versión final seleccionada:** `v5`

#### Evolución de Parámetros

| Versión | Epochs | Steps | Batch | Rank | Alpha | LR (U-Net) | LR (Text Encoder) | Observaciones |
|---------|--------|-------|-------|------|-------|------------|-------------------|--------------|
| **v1** | 20 | 4000 | 2 | 32 | 16 | 1e-4 | 5e-5 | Rank alto inicial |
| **v2** | 10 | 2000 | 2 | 32 | 16 | 1e-4 | 5e-5 | Reducción de epochs |
| **v3** | 15 | 3000 | 2 | 32 | 16 | 1e-4 | 5e-5 | Epochs intermedios |
| **v4** | 20 | 4000 | 2 | 16 | 16 | 1e-4 | 5e-5 | Reducción de rank, alpha igual |
| **v5** ✅ | 20 | 4000 | 2 | 8 | 4 | 1e-4 | 5e-5 | Rank y alpha reducidos |

**Parámetros finales (v5):**
- **Dataset:** 8 imágenes, 50 repeats
- **Batch Size:** 2
- **Epochs:** 20
- **Total Steps:** 4000
- **Network Rank (Dimension):** 8
- **Network Alpha:** 4 (ratio efectivo: 0.5)
- **Learning Rate (U-Net):** 1e-4
- **Learning Rate (Text Encoder):** 5e-5
- **Max Bucket Reso:** 2048

**Justificación de la selección:**
- Dataset pequeño (8 imágenes) requiere rank bajo para evitar overfitting
- 50 repeats proporcionan exposición suficiente pese al tamaño reducido del dataset
- Rank 8 con alpha 4 mantiene capacidad suficiente para el estilo FuzzyBear
- Learning rate estándar (1e-4) funciona bien con batch size 2

**Problemas detectados y soluciones:**
- **v1-v3:** Rank 32 causaba overfitting severo con dataset pequeño → Reducción progresiva de rank
- **v4:** Rank 16 con alpha 16 (ratio 1.0) → Efecto del LoRA demasiado fuerte, pérdida de coherencia
- **v4 → v5:** Reducción a rank 8 y alpha 4 → Balance óptimo para dataset pequeño

---

### 4. LoRA Robots

**Versión final seleccionada:** `v2`

#### Evolución de Parámetros

| Versión | Epochs | Steps | Batch | Rank | Alpha | LR (U-Net) | LR (Text Encoder) | Observaciones |
|---------|--------|-------|-------|------|-------|------------|-------------------|--------------|
| **v1** | 25 | 3750 | 4 | 16 | 8 | 1e-4 | 5e-5 | Configuración inicial estándar |
| **v2** ✅ | 15 | 4500 | 2 | 8 | 4 | 1e-4 | 5e-5 | Reducción de rank y batch |

**Parámetros finales (v2):**
- **Dataset:** 15 imágenes, 40 repeats
- **Batch Size:** 2
- **Epochs:** 15
- **Total Steps:** 4500
- **Network Rank (Dimension):** 8
- **Network Alpha:** 4 (ratio efectivo: 0.5)
- **Learning Rate (U-Net):** 1e-4
- **Learning Rate (Text Encoder):** 5e-5
- **Max Bucket Reso:** 2048

**Justificación de la selección:**
- Dataset mediano (15 imágenes) se beneficia de rank bajo para evitar overfitting
- Batch size 2 permite mayor granularidad en el entrenamiento
- 15 epochs con 40 repeats proporcionan exposición adecuada
- Rank 8 es suficiente para capturar características robóticas sin memorizar detalles específicos

**Problemas detectados y soluciones:**
- **v1:** Rank 16 con batch 4 → Overfitting a detalles específicos de robots del dataset
- **v1 → v2:** Reducción a rank 8 y batch 2 → Mayor generalización, mejor coherencia estilística

---

### 5. LoRA StarWars

**Versión final seleccionada:** `v1`

#### Parámetros Finales

- **Dataset:** 19 imágenes, 40 repeats
- **Batch Size:** 4
- **Epochs:** 20
- **Total Steps:** 3800
- **Network Rank (Dimension):** 16
- **Network Alpha:** 16 (ratio efectivo: 1.0)
- **Learning Rate (U-Net):** 1e-4
- **Learning Rate (Text Encoder):** 5e-5
- **Max Bucket Reso:** 1024 (diferente a otros LoRAs)

**Justificación de la configuración:**
- Único LoRA con alpha igual a rank (ratio 1.0), maximizando el efecto del adaptador
- Max bucket reso de 1024 (vs 2048 en otros) para mantener coherencia con el estilo Star Wars
- Rank 16 proporciona capacidad suficiente para detalles característicos (armaduras, cascos, etc.)
- 20 epochs con 40 repeats aseguran convergencia adecuada

**Nota:** Este LoRA fue entrenado en una sola iteración, sin necesidad de ajustes adicionales debido a la configuración inicial óptima.

---

## 📈 Análisis Comparativo de Parámetros

### Network Rank (Dimension)

| LoRA | Rank Final | Justificación |
|------|-----------|--------------|
| Animal | 16 | Dataset grande (42), necesita capacidad para detalles animales |
| Food | 16 | Dataset mediano (27), balance entre capacidad y generalización |
| FuzzyBear | 8 | Dataset pequeño (8), rank bajo previene overfitting |
| Robots | 8 | Dataset mediano (15), rank bajo mejora generalización |
| StarWars | 16 | Dataset mediano (19), necesita capacidad para detalles complejos |

### Network Alpha / Rank Ratio

| LoRA | Alpha | Rank | Ratio | Efecto |
|------|-------|------|-------|--------|
| Animal | 8 | 16 | 0.5 | Efecto moderado, balanceado |
| Food | 8 | 16 | 0.5 | Efecto moderado, balanceado |
| FuzzyBear | 4 | 8 | 0.5 | Efecto moderado, balanceado |
| Robots | 4 | 8 | 0.5 | Efecto moderado, balanceado |
| StarWars | 16 | 16 | 1.0 | Efecto máximo, adaptador completo |

**Observación:** Todos los LoRAs excepto StarWars utilizan ratio 0.5, que es un estándar común. StarWars utiliza ratio 1.0 para maximizar el impacto del adaptador, posiblemente debido a la necesidad de capturar características muy específicas del universo Star Wars.

### Learning Rate

| LoRA | LR U-Net | LR Text Encoder | Ratio TE/U-Net |
|------|----------|-----------------|----------------|
| Animal | 8e-5 | 4e-5 | 0.5 |
| Food | 8e-5 | 4e-5 | 0.5 |
| FuzzyBear | 1e-4 | 5e-5 | 0.5 |
| Robots | 1e-4 | 5e-5 | 0.5 |
| StarWars | 1e-4 | 5e-5 | 0.5 |

**Patrón observado:**
- Animal y Food utilizan LR más conservador (8e-5) → Mayor estabilidad
- FuzzyBear, Robots y StarWars utilizan LR estándar (1e-4) → Convergencia más rápida
- Todos mantienen ratio Text Encoder / U-Net de 0.5 → Text Encoder se entrena más lentamente para evitar overfitting

---

## 🎯 Selección Final de Modelos

### Resumen de LoRAs Finales

| LoRA | Versión | Archivo | Tamaño Dataset | Epochs | Steps | Rank | Alpha | LR |
|------|---------|--------|----------------|--------|-------|------|-------|-----|
| **Animal** | v3 | `FT_Humanoid_5e_vF_LoRA_Animal_v3-000012.safetensors` | 42 | 15 | 4725 | 16 | 8 | 8e-5 |
| **Food** | v8 | `FT_Humanoid_5e_vF_LoRA_Food_v8-000008.safetensors` | 27 | 30 | 4050 | 16 | 8 | 8e-5 |
| **FuzzyBear** | v5 | `FT_Humanoid_5e_vF_LoRA_FuzzyBear_v5-000020.safetensors` | 8 | 20 | 4000 | 8 | 4 | 1e-4 |
| **Robots** | v2 | `FT_Humanoid_5e_vF_LoRA_Robots_v2-000008.safetensors` | 15 | 15 | 4500 | 8 | 4 | 1e-4 |
| **StarWars** | v1 | `FT_Humanoid_5e_vF_LoRA_StarWars_v1-000013.safetensors` | 19 | 20 | 3800 | 16 | 16 | 1e-4 |

### Triggers por Categoría

Cada LoRA utiliza triggers específicos en los prompts para activar el estilo:

- **Animal:** `fortnite_animal_character, nice_hands, `
- **Food:** `fortnite_food_character, nice_hands, `
- **FuzzyBear:** `fortnite_fuzzy_bear_character, nice_hands, `
- **Robots:** `fortnite_robots_character, nice_hands, `
- **StarWars:** `fortnite_star_wars_character, nice_hands, `

### Diferencias Clave entre LoRAs

1. **Especialización:**
   - **Animal y Food:** Rank 16, mayor capacidad para detalles complejos
   - **FuzzyBear y Robots:** Rank 8, enfoque en generalización sobre memorización
   - **StarWars:** Rank 16 con alpha 16, máximo impacto del adaptador

2. **Robustez:**
   - **Animal:** Mayor robustez debido a dataset grande (42 imágenes)
   - **FuzzyBear:** Menor robustez pero suficiente para dataset pequeño (8 imágenes)
   - **Food, Robots, StarWars:** Robustez intermedia

3. **Coherencia:**
   - Todos mantienen coherencia con el estilo base Fortnite
   - StarWars tiene mayor desviación estilística permitida (alpha/rank = 1.0)
   - FuzzyBear y Robots priorizan coherencia sobre especialización extrema

---

## 🔍 Proceso Iterativo y Ajustes

### Metodología de Evaluación

La evaluación de cada iteración se realizó mediante:

1. **Generación de muestras durante entrenamiento:** Cada epoch generaba imágenes de prueba con prompts estándar
2. **Análisis visual cualitativo:**
   - Coherencia con estilo Fortnite
   - Calidad anatómica
   - Presencia de artefactos o deformaciones
   - Fidelidad a la categoría temática
3. **Detección de problemas:**
   - **Overfitting:** Generaciones demasiado similares a imágenes de entrenamiento
   - **Underfitting:** Falta de características temáticas distintivas
   - **Pérdida de identidad Fortnite:** Desviación excesiva del estilo base
   - **Ruido estilístico:** Inconsistencias visuales entre generaciones

### Ajustes Comunes Realizados

1. **Reducción de Rank:** Cuando se detectaba overfitting, se reducía el rank para limitar la capacidad del adaptador
2. **Ajuste de Learning Rate:** Learning rates muy altos causaban inestabilidad; muy bajos, convergencia lenta
3. **Modificación de Epochs/Repeats:** Balance entre exposición suficiente y evitar memorización
4. **Cambio de Batch Size:** Batch más pequeños permiten mayor granularidad pero requieren más epochs

---

## 📊 Resultados y Ejemplos

Los LoRAs finales se utilizan en el sistema de generación mediante ComfyUI, integrados en workflows específicos que combinan:

- Modelo base fine-tuned: `v1x0_fortnite_humanoid_sdxl1_vae_fix-000005`
- LoRA especializado según categoría
- LoRA NiceHands para mejora de anatomía de manos
- SDXL Refiner para post-procesamiento de alta calidad

Ejemplos de generaciones se encuentran en `4.Inferencias LoRAs seleccionados/`.

---

## ⚠️ Limitaciones y Trabajo Futuro

### Limitaciones Identificadas

1. **Tamaño de datasets:** Algunos datasets (especialmente FuzzyBear con 8 imágenes) son pequeños y limitan la generalización
2. **Overfitting en detalles específicos:** Algunos LoRAs tienden a memorizar características específicas de imágenes de entrenamiento
3. **Combinabilidad limitada:** Los LoRAs no están optimizados para combinarse entre sí
4. **Dependencia del modelo base:** Cambios en el modelo base requieren reentrenamiento de LoRAs

### Trabajo Futuro

1. **Expansión de datasets:** Aumentar el tamaño de datasets, especialmente para FuzzyBear y Robots
2. **Regularización explícita:** Incorporar imágenes de regularización para mejorar generalización
3. **LoRAs combinables:** Investigar técnicas para permitir combinación de múltiples LoRAs
4. **Fine-tuning de triggers:** Optimizar triggers mediante análisis de activación
5. **Métricas cuantitativas:** Implementar métricas objetivas (FID, CLIP Score) además de evaluación cualitativa
6. **Hiperparámetros adaptativos:** Automatizar búsqueda de hiperparámetros según tamaño y características del dataset

---

## 📚 Referencias Técnicas

- **KOHYA_ss:** [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts)
- **LoRA Paper:** "LoRA: Low-Rank Adaptation of Large Language Models" (Hu et al., 2021)
- **Stable Diffusion XL:** Stability AI
- **Aspect Ratio Bucketing:** NovelAI implementation

---

## 📝 Notas de Implementación

- **Hardware utilizado:** NVIDIA A100 (80GB VRAM) para mayoría de entrenamientos, A100 (40GB VRAM) para algunos
- **Tiempo de entrenamiento:** Variable según dataset y configuración, típicamente 2-6 horas por LoRA
- **Framework:** KOHYA_ss con soporte para SDXL
- **Formato de salida:** SafeTensors (fp16)

---

**Autores:** Odreman Ferrer y Sergio Valdueza - TFM Deep Learning MIOTI  
**Licencia:** CC BY-NC-SA 4.0  
**Última actualización:** Diciembre 2025

