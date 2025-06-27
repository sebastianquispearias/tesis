import sys
import cv2
import numpy as np
import pandas as pd
from skimage.measure import label, regionprops_table

def analizar_mascaras(directorio_mascaras):
    """
    Analiza todas las máscaras en un directorio para extraer estadísticas
    de forma de los defectos. python analizar_morfologia_mascaras.py "data/all_results/regimes/C75/train/masks_labeled"
    """
    # Usamos glob para encontrar todas las imágenes .png
    from glob import glob
    rutas_mascaras = glob(f"{directorio_mascaras}/*.png")

    if not rutas_mascaras:
        print(f"Error: No se encontraron máscaras en '{directorio_mascaras}'")
        return

    print(f"Analizando {len(rutas_mascaras)} máscaras...")

    all_props = []
    for ruta in rutas_mascaras:
        # Cargar la máscara y binarizar
        mask = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        
        # Binarizar la máscara (asumimos que el defecto es cualquier cosa > 0)
        mask_binaria = (mask > 0).astype(np.uint8)

        # Etiquetar regiones conectadas (cada mancha de defecto)
        mascara_etiquetada = label(mask_binaria)

        # Extraer propiedades de cada región
        # Solo calculamos para máscaras que no están completamente vacías
        if mascara_etiquetada.max() > 0:
            props = regionprops_table(
                mascara_etiquetada,
                properties=('area', 'equivalent_diameter', 
                            'major_axis_length', 'minor_axis_length', 'orientation')
            )
            all_props.append(pd.DataFrame(props))

    if not all_props:
        print("No se encontraron defectos en ninguna de las máscaras analizadas.")
        return

    # Combinar todas las propiedades en un único DataFrame
    df = pd.concat(all_props, ignore_index=True)

    # Calcular el ratio de aspecto (elongación)
    # Añadimos un valor pequeño para evitar división por cero si un defecto es una línea perfecta
    df['aspect_ratio'] = df['major_axis_length'] / (df['minor_axis_length'] + 1e-6)

    # Convertir orientación de radianes a grados
    df['orientation_deg'] = np.rad2deg(df['orientation'])

    # --- Resultados ---
    print("\n--- ANÁLISIS ESTADÍSTICO DE FORMAS DE DEFECTOS ---\n")

    # 1. Análisis de TAMAÑO (en píxeles)
    print("1. Distribución de Tamaño (Diámetro Equivalente):")
    p25, p50, p75 = np.percentile(df['equivalent_diameter'], [25, 50, 75])
    print(f"   - Percentil 25 (P25): {p25:.2f} px")
    print(f"   - Mediana (P50):      {p50:.2f} px")
    print(f"   - Percentil 75 (P75): {p75:.2f} px\n")

    # 2. Análisis de FORMA (Elongación)
    print("2. Distribución de Forma (Ratio de Aspecto):")
    mediana_ar = df['aspect_ratio'].median()
    print(f"   - Mediana del Ratio de Aspecto: {mediana_ar:.2f}")
    print("     (Un valor de 1.0 es un círculo perfecto. Valores > 1 son elípticos/alargados)\n")

    # 3. Análisis de ORIENTACIÓN (en grados)
    print("3. Distribución de Orientación:")
    # Histograma para ver la distribución de ángulos
    hist, bin_edges = np.histogram(df['orientation_deg'], bins=12, range=(-90, 90))
    print("   - Histograma de Orientaciones (de -90° a 90°):")
    for i in range(len(hist)):
        print(f"     - Rango {bin_edges[i]:.0f}° a {bin_edges[i+1]:.0f}°: {hist[i]} defectos")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Uso: python analizar_morfologia_mascaras.py <ruta_al_directorio_de_mascaras>")
        sys.exit(1)
    
    directorio = sys.argv[1]
    analizar_mascaras(directorio)