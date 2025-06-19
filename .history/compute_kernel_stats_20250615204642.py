# compute_kernel_stats.py
import json
import glob
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

def main():
    radii = []
    # Usamos las máscaras rotuladas para medir pitting real
    for mask_path in glob.glob('data/all_results/regimes/C75/train/masks_labeled/*.png'):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) > 0
        dt   = distance_transform_edt(mask)
        # Para cada isla, tomo el mayor distance transform como radio
        if np.any(mask):
            radii.append(dt[mask].max())
    # Calculo percentiles y redondeo a entero
    p25, p50, p75 = np.percentile(radii, [25,50,75]).round().astype(int)
    stats = {"radii": [int(p25), int(p50), int(p75)]}
    with open('data/kernel_stats.json','w') as f:
        json.dump(stats, f)
    print("Kernel radii saved:", stats)

if __name__ == "__main__":
    main()
