import os
import cv2
import numpy as np
from tqdm import tqdm

def resize_image_array(
    image_array: np.ndarray,
    new_width: int,
    new_height: int,
    interpolation_method: int = cv2.INTER_AREA
) -> np.ndarray | None:
    """
    Reduce o amplía un array de imagen NumPy al tamaño especificado.

    Args:
        image_array (np.ndarray): El array de la imagen de entrada (cargada por OpenCV).
        new_width (int): Ancho deseado para la imagen redimensionada.
        new_height (int): Alto deseado para la imagen redimensionada.
        interpolation_method (int, optional): Método de interpolación de OpenCV.
            Recomendado cv2.INTER_AREA para reducir, cv2.INTER_CUBIC o
            cv2.INTER_LINEAR para ampliar. Defaults to cv2.INTER_AREA.

    Returns:
        np.ndarray | None: El array de la imagen redimensionada,
            o None si el array de entrada es inválido.
    """
    if not isinstance(image_array, np.ndarray) or image_array.size == 0:
        # print("Error: El array de imagen proporcionado es inválido.") # Opcional: log de error
        return None

    resized_image = cv2.resize(
        image_array,
        (new_width, new_height),
        interpolation=interpolation_method
    )
    return resized_image


def process_and_resize_images(
    input_folder: str,
    output_folder_512: str,
    output_folder_256: str
) -> None:
    """
    Procesa imágenes de un directorio de entrada, las reduce a 512x512 y luego
    a 256x256, guardando los resultados en los directorios de salida especificados.

    Args:
        input_folder (str): Ruta al directorio que contiene las imágenes originales.
        output_folder_512 (str): Ruta al directorio donde se guardarán las imágenes de 512x512.
        output_folder_256 (str): Ruta al directorio donde se guardarán las imágenes de 256x256.
    """
    # Crear carpetas de salida si no existen
    os.makedirs(output_folder_512, exist_ok=True)
    os.makedirs(output_folder_256, exist_ok=True)

    # Obtener la lista de archivos de imagen del directorio de entrada
    try:
        image_filenames = [
            f for f in os.listdir(input_folder)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
        ]
    except FileNotFoundError:
        print(f"Error: El directorio de entrada '{input_folder}' no fue encontrado.")
        return

    if not image_filenames:
        print(f"No se encontraron imágenes en '{input_folder}'.")
        return

    print(f"🔍 Encontradas {len(image_filenames)} imágenes en '{input_folder}' para procesar.")

    for filename in tqdm(image_filenames, desc="Procesando imágenes"):
        input_file_path = os.path.join(input_folder, filename)
        output_file_path_512 = os.path.join(output_folder_512, filename)
        output_file_path_512 = output_file_path_512.replace('1024_3c', '512_3c')
        output_file_path_256 = os.path.join(output_folder_256, filename)
        output_file_path_256 = output_file_path_256.replace('1024_3c', '256_3c')

        # Cargar la imagen original una vez
        # IMREAD_UNCHANGED para preservar canal alfa si existe (ej. en PNGs)
        original_image = cv2.imread(input_file_path, cv2.IMREAD_UNCHANGED)

        if original_image is None:
            tqdm.write(f"Advertencia: No se pudo cargar la imagen '{input_file_path}'.")
            tqdm.write("Saltando esta imagen.")
            continue

        # 1. Reducir a 512x512
        image_512 = resize_image_array(original_image, 512, 512, cv2.INTER_AREA)
        if image_512 is not None:
            # Guardar la imagen de 512x512
            # El tercer parámetro para imwrite son flags de compresión, opcionales.
            # Para PNG, cv2.IMWRITE_PNG_COMPRESSION (0-9, default 3).
            # Para JPG, cv2.IMWRITE_JPEG_QUALITY (0-100, default 95).
            cv2.imwrite(output_file_path_512, image_512)

            # 2. Reducir la imagen de 512x512 (ya en memoria) a 256x256
            image_256 = resize_image_array(image_512, 256, 256, cv2.INTER_AREA)
            if image_256 is not None:
                # Guardar la imagen de 256x256
                cv2.imwrite(output_file_path_256, image_256)
            else:
                tqdm.write(f"Advertencia: No se pudo reducir '{filename}' a 256x256.")
                tqdm.write("Saltando esta reducción.")
        else:
            tqdm.write(f"Advertencia: No se pudo reducir '{filename}' a 512x512.")
            tqdm.write("Saltando esta reducción.")

    print("\nProceso de reducción de imágenes completado.")


if __name__ == "__main__":
    # 📂 Rutas (ajusta según tu estructura)
    # Asumiendo que este script está en un directorio y las carpetas de outfits están relativas a él
    # o usa rutas absolutas.
    INPUT_DIR = "./outfits_procesados_1024_rgb/"
    OUTPUT_DIR_512 = "./outfits_procesados_512_rgb/"
    OUTPUT_DIR_256 = "./outfits_procesados_256_rgb/"

    process_and_resize_images(INPUT_DIR, OUTPUT_DIR_512, OUTPUT_DIR_256)
