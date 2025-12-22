# Pipeline de Preparación de Datos - TFM Skin AI Generation Fortnite

Este directorio contiene el pipeline completo de preparación de datos para el proyecto de generación de skins de Fortnite mediante IA. Los notebooks y scripts están organizados en orden secuencial según el flujo de procesamiento.

## 📋 Índice

1. [Obtención de Datos](#1-obtención-de-datos)
2. [Limpieza de Imágenes](#2-limpieza-de-imágenes)
3. [Reducción de Tamaño](#3-reducción-de-tamaño)
4. [Aumento de Datos](#4-aumento-de-datos)
5. [Transformación de Dimensiones](#5-transformación-de-dimensiones)
6. [Generación de Etiquetas](#6-generación-de-etiquetas)
7. [Mejora de Etiquetas](#7-mejora-de-etiquetas)

---

## 1. Obtención de Datos

### `0.1.get_items_from_api.ipynb`

**Objetivo:** Obtener los datos de los items cosméticos de Fortnite desde la API oficial.

**Funcionalidad:**

-   Se conecta a la API de Fortnite (`fortniteapi.io`) utilizando la API key almacenada en variables de entorno
-   Extrae información de todos los items cosméticos disponibles (skins, emotes, etc.)
-   Normaliza y estructura los datos JSON en DataFrames de pandas
-   Genera archivos CSV de salida:
    -   `items.csv`: Contiene todos los items cosméticos con sus metadatos
    -   `emotes.csv`: Contiene información específica de emotes
    -   `outfit.csv`: Contiene información de outfits/skins
    -   `outfits_with_details.csv`: Contiene información de los outfits con más detalles
-   Descarga las imágenes en 1024x1024 y 4 canales RGBA.

**Salidas:**

-   `items.csv`
-   `emotes.csv`
-   `outfit.csv`
-   `outfits_with_details.csv`
-   Carpeta `outfits_originales_1024_rgba` con las imágenes originales de la API de Fortnite.

**Dependencias:**

-   API key de Fortnite (almacenada en `.env` como `FORTNITE_API_KEY`)
-   `pandas`, `requests`, `python-dotenv`

---

## 2. Limpieza de Imágenes

### `0.2.1.data_cleaning.ipynb`

**Objetivo:** Preprocesar y limpiar las imágenes originales de los personajes de Fortnite.

**Funcionalidad:**

-   **Entrada:** Imágenes originales en formato RGBA (1024x1024 píxeles) desde `outfits_originales_1024_rgba/`
-   **Procesamiento:**
    1. Limpieza de fondos: Elimina píxeles semi-transparentes estableciendo un umbral de opacidad (200)
    2. Aislamiento del objeto principal: Detecta y extrae el contorno más grande (el personaje principal)
    3. Eliminación de artefactos: Remueve objetos secundarios y ruido
-   **Salidas:**
    -   `outfits_procesados_1024_rgba/`: Versiones procesadas manteniendo canal alfa
    -   `outfits_procesados_1024_rgb/`: Versiones procesadas sin canal alfa (RGB)

**Características técnicas:**

-   Utiliza OpenCV para procesamiento de imágenes
-   Detecta contornos para aislar el objeto principal
-   Aplica máscaras binarias para limpieza de fondos

---

## 3. Reducción de Tamaño

### `0.2.3.size_reduction con PIL.py`

**Objetivo:** Reducir el tamaño de las imágenes procesadas usando PIL con antialiasing de alta calidad.

**Funcionalidad:**

-   **Entrada:** Imágenes procesadas de 1024x1024 píxeles desde `outfits_procesados_1024_rgb/`
-   **Procesamiento:**
    1. Reducción a 512x512 píxeles usando filtro LANCZOS (alta calidad)
    2. Reducción adicional a 256x256 píxeles desde la versión 512x512
-   **Salidas:**
    -   `outfits_procesados_512_rgb/`: Imágenes de 512x512
    -   `outfits_procesados_256_rgb/`: Imágenes de 256x256

**Ventajas:**

-   Filtro LANCZOS preserva mejor la nitidez durante el downscaling
-   Ideal para mantener calidad visual en datasets

### `0.2.3.size_reduction con CV2.py`

**Objetivo:** Alternativa al script anterior usando OpenCV para reducción de tamaño.

**Funcionalidad:**

-   Similar al script PIL pero utiliza `cv2.INTER_AREA` para interpolación
-   Más rápido que PIL pero con calidad ligeramente inferior
-   Mismas entradas y salidas que el script PIL

**Cuándo usar:**

-   Cuando la velocidad es prioritaria sobre la calidad máxima
-   Para procesamiento en lotes grandes

---

## 4. Aumento de Datos

### `0.2.2.1.data_augmentation.ipynb`

**Objetivo:** Generar múltiples variaciones de cada imagen mediante técnicas de data augmentation.

**Funcionalidad:**

-   **Entrada:** Imágenes procesadas de 1024x1024 RGB desde `outfits_procesados_1024_rgb/`
-   **Técnicas aplicadas:**
    1. **Transformaciones de color:**
        - Ajuste de brillo y contraste aleatorio
        - Cambio de tono y saturación
        - Desplazamiento RGB
    2. **Transformaciones geométricas:**
        - Volteo horizontal
        - Rotación, escala y traslación
        - Detección de personajes "grounded" (pegados al suelo) o "ceiled" (pegados al techo) para aplicar transformaciones específicas
-   **Salidas:**
    -   `outfits_augmented_1024_rgb/`: Múltiples variaciones de cada imagen original
    -   Genera aproximadamente 34 aumentaciones por imagen original (objetivo: 50,000 imágenes totales)

**Características especiales:**

-   Detecta si el personaje está "grounded" o "ceiled" para aplicar transformaciones apropiadas
-   Descarta aumentaciones que resulten en personajes "ceiled"
-   Preserva el fondo negro durante las transformaciones

---

## 5. Transformación de Dimensiones

### `0.2.2.2.data_transformation_256x512.ipynb`

**Objetivo:** Transformar imágenes de 512x512 a formato 256x512 (aspecto vertical).

**Funcionalidad:**

-   **Entrada:** Imágenes de 512x512 desde `outfits_procesados_512_rgb/`
-   **Procesamiento:**
    1. Detección de márgenes: Identifica las columnas con contenido visible
    2. Recorte inteligente: Elimina márgenes laterales vacíos
    3. Redimensionamiento proporcional: Ajusta el ancho a 256 píxeles manteniendo proporción
    4. Padding superior: Añade padding negro en la parte superior si la altura es menor a 512px
-   **Salidas:**
    -   `outfits_procesados_256x512_rgb/`: Imágenes en formato 256x512

**Características técnicas:**

-   Utiliza PIL con filtro LANCZOS para redimensionamiento de alta calidad
-   Detecta automáticamente el área visible para centrar el contenido
-   Asegura dimensiones finales exactas de 256x512

---

## 6. Generación de Etiquetas

### `0.2.4.etiquetas.ipynb`

**Objetivo:** Generar etiquetas descriptivas para cada imagen del dataset.

**Funcionalidad:**

-   **Entrada:**
    -   Imágenes procesadas desde `outfits_procesados_1024_rgb/`
    -   Datos del CSV (`outfits_with_details.csv`) con metadatos de los personajes
-   **Procesamiento:**
    1. **Generación con BLIP:** Utiliza el modelo BLIP (Bootstrapping Language-Image Pre-training) para generar descripciones automáticas de las imágenes
    2. **Enriquecimiento con metadatos:** Combina las descripciones de BLIP con información del CSV:
        - Nombre del personaje
        - Nombre del set/colección
        - Serie (Marvel, Star Wars, DC Comics, etc.)
    3. **Tags de estilo:** Añade tags específicos de estilo Fortnite
    4. **Manejo de imágenes no encontradas:** Para imágenes sin match en el CSV, utiliza solo BLIP
-   **Salidas:**
    -   Archivos `.txt` con etiquetas para cada imagen en `tags/`
    -   Archivos `_store_tags.txt` con solo los tags del CSV en `tags_store/`

**Modelo utilizado:**

-   BLIP (Salesforce/blip-image-captioning-base) para generación automática de descripciones

---

## 7. Mejora de Etiquetas

### `0.2.5.mejorar-etiquetas.ipynb`

**Objetivo:** Refinar y mejorar las etiquetas generadas usando inteligencia artificial avanzada.

**Funcionalidad:**

-   **Entrada:**
    -   Etiquetas generadas previamente (archivos `.txt`)
    -   Datos del CSV con información adicional (nombre, descripción, colaboraciones)
-   **Procesamiento:**
    1. Lee las etiquetas originales generadas por BLIP
    2. Utiliza Google Gemini API para:
        - Refinar y mejorar las descripciones
        - Añadir contexto adicional basado en metadatos del CSV
        - Optimizar el formato y estructura de las etiquetas
    3. Genera versiones mejoradas de las etiquetas
-   **Salidas:**
    -   Archivos `.txt` mejorados en el directorio de salida

**Dependencias:**

-   Google Gemini API (almacenada en `.env` como `GOOGLE_GEMINI_API_KEY`)
-   `google-generativeai`, `pandas`

**Características:**

-   Procesamiento por lotes con logging
-   Manejo de errores y reintentos
-   Preserva información relevante mientras mejora la calidad descriptiva

---

## 🔄 Flujo de Procesamiento Completo

```
1. Obtener datos de API
   └─> 0.1.get_items_from_api.ipynb
       └─> Genera: items.csv, emotes.csv, outfit.csv, outfits_with_details.csv
       └─> outfits_originales_1024_rgba/

2. Limpiar imágenes originales
   └─> 0.2.1.data_cleaning.ipynb
       └─> outfits_originales_1024_rgba/
           └─> outfits_procesados_1024_rgb/

3. Reducir tamaño de imágenes
   └─> 0.2.3.size_reduction con PIL.py (o CV2.py)
       └─> outfits_procesados_1024_rgb/
           └─> outfits_procesados_512_rgb/
           └─> outfits_procesados_256_rgb/

4. Aumentar datos
   └─> 0.2.2.1.data_augmentation.ipynb
       └─> outfits_procesados_1024_rgb/
           └─> outfits_augmented_1024_rgb/

5. Transformar dimensiones
   └─> 0.2.2.2.data_transformation_256x512.ipynb
       └─> outfits_procesados_512_rgb/
           └─> outfits_procesados_256x512_rgb/

6. Generar etiquetas
   └─> 0.2.4.etiquetas.ipynb
       └─> outfits_procesados_1024_rgb/ + outfits_with_details.csv
           └─> tags/*.txt

7. Mejorar etiquetas
   └─> 0.2.5.mejorar-etiquetas.ipynb
       └─> tags/*.txt + outfits_with_details.csv
           └─> tags mejorados
```

---

## 📦 Dependencias Principales

-   **Procesamiento de imágenes:** `opencv-python`, `Pillow`, `numpy`
-   **Aumento de datos:** `albumentations`
-   **Procesamiento de datos:** `pandas`
-   **APIs:** `requests`, `google-generativeai`
-   **IA/ML:** `transformers` (para BLIP)
-   **Utilidades:** `tqdm`, `python-dotenv`

---

## ⚙️ Configuración

Antes de ejecutar los notebooks, asegúrate de:

1. **Configurar variables de entorno** en el archivo `.env` en la raíz del proyecto:

    ```
    FORTNITE_API_KEY=tu_api_key_aqui
    GOOGLE_GEMINI_API_KEY=tu_api_key_aqui
    ```

2. **Verificar rutas de directorios** en cada notebook según tu estructura de carpetas

3. **Instalar dependencias** desde `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```

---

## 📝 Notas Importantes

-   Los notebooks están diseñados para ejecutarse en orden secuencial
-   Algunos notebooks pueden tardar considerablemente (especialmente data augmentation y generación de etiquetas)
-   Se recomienda verificar las salidas de cada paso antes de continuar al siguiente
-   Los scripts de reducción de tamaño (PIL vs CV2) son alternativas entre sí, no es necesario ejecutar ambos

---

## 🐛 Solución de Problemas

-   **Error de API keys:** Verifica que el archivo `.env` esté en la raíz del proyecto y contenga las claves correctas
-   **Rutas no encontradas:** Ajusta las rutas de entrada/salida en cada notebook según tu estructura de directorios
-   **Memoria insuficiente:** Para datasets grandes, considera procesar en lotes más pequeños
-   **Modelos no encontrados:** Los modelos de BLIP se descargan automáticamente en la primera ejecución

---

**Última actualización:** Dic2025
