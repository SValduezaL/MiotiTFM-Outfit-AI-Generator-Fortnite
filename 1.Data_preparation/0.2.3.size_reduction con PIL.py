import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

def resize_pil_antialias(image_bgr, size):
    """
    Redimensiona una imagen BGR a un tamaño cuadrado dado, utilizando el filtro de antialiasing
    LANCZOS de la librería PIL para preservar la máxima nitidez posible durante el downscaling.

    Este método es especialmente útil cuando se requiere alta calidad visual, como en
    generación de datasets, publicaciones o presentación de resultados.

    Parámetros:
    ----------
    image_bgr : np.ndarray
        Imagen original en formato BGR (como la carga OpenCV).

    size : int
        Tamaño final (ancho y alto en píxeles) al que se desea redimensionar la imagen.

    Retorna:
    -------
    result_bgr : np.ndarray
        Imagen redimensionada en formato BGR con dimensiones (size, size, 3).
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    pil_resized = pil_image.resize((size, size), Image.Resampling.LANCZOS)  # Filtro LANCZOS = alta calidad
    result_bgr = cv2.cvtColor(np.array(pil_resized), cv2.COLOR_RGB2BGR)
    return result_bgr


def process_and_resize_images(
    input_folder: str,
    output_folder_512: str,
    output_folder_256: str
) -> None:
    """
    Procesa imágenes de un directorio de entrada, las reduce a 512x512 y luego
    a 256x256, guardando los resultados en los directorios de salida especificados.
    La reducción de tamaño se realiza usando redimensionado antialiasing con PIL.

    Args:
        input_folder (str): Ruta al directorio que contiene las imágenes originales.
        output_folder_512 (str): Ruta al directorio donde se guardarán las imágenes de 512x512.
        output_folder_256 (str): Ruta al directorio donde se guardarán las imágenes de 256x256.
    """
    os.makedirs(output_folder_512, exist_ok=True)
    os.makedirs(output_folder_256, exist_ok=True)

    try:
        image_filenames = [
            f for f in os.listdir(input_folder)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
        ]
    except FileNotFoundError:
        print(f"❌ Error: El directorio de entrada '{input_folder}' no fue encontrado.")
        return

    if not image_filenames:
        print(f"⚠️ No se encontraron imágenes en '{input_folder}'.")
        return

    print(f"🔍 Encontradas {len(image_filenames)} imágenes en '{input_folder}' para procesar.")

    for filename in tqdm(image_filenames, desc="Procesando imágenes"):
        input_file_path = os.path.join(input_folder, filename)
        output_file_path_512 = os.path.join(output_folder_512, filename).replace('1024_3c', '512_3c')
        output_file_path_256 = os.path.join(output_folder_256, filename).replace('1024_3c', '256_3c')

        original_image = cv2.imread(input_file_path, cv2.IMREAD_UNCHANGED)

        if original_image is None:
            tqdm.write(f"⚠️ Advertencia: No se pudo cargar '{input_file_path}'. Saltando.")
            continue

        try:
            # 1. Reducción a 512x512 usando antialiasing de PIL
            image_512 = resize_pil_antialias(original_image, 512)
            cv2.imwrite(output_file_path_512, image_512)

            # 2. Reducción desde 512x512 a 256x256
            image_256 = resize_pil_antialias(image_512, 256)
            cv2.imwrite(output_file_path_256, image_256)

        except Exception as e:
            tqdm.write(f"❌ Error procesando '{filename}': {e}")
            continue

    print("\n✅ Proceso de reducción de imágenes completado.")


if __name__ == "__main__":
    # 📂 Rutas (ajusta según tu estructura)
    # Asumiendo que este script está en un directorio y las carpetas de outfits están relativas a él
    # o usa rutas absolutas.
    INPUT_DIR = "./outfits_procesados_1024_rgb/"
    OUTPUT_DIR_512 = "./outfits_procesados_512_rgb/"
    OUTPUT_DIR_256 = "./outfits_procesados_256_rgb/"

    process_and_resize_images(INPUT_DIR, OUTPUT_DIR_512, OUTPUT_DIR_256)
