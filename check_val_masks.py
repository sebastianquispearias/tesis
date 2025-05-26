import os, cv2

mask_dir = r'C:\Users\User\Desktop\tesis\data\320x320\Val\masks'

total = empty = 0
for fn in os.listdir(mask_dir):
    if not fn.lower().endswith(('.png','.jpg','.jpeg')):
        continue
    total += 1
    m = cv2.imread(os.path.join(mask_dir, fn), cv2.IMREAD_GRAYSCALE)
    if m is None: continue
    if m.sum() == 0:
        empty += 1

print(f"Total de máscaras:           {total}")
print(f"Máscaras vacías (solo 0):    {empty} ({empty/total*100:.1f} %)")
print(f"Máscaras con algo de corrosión: {total-empty} ({(total-empty)/total*100:.1f} %)")
