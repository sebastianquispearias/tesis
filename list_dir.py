from pathlib import Path

# Cambia esta ruta por la tuya
base_dir = Path(r"C:\Users\User\Desktop\tesis\data\Corrosion Condition State Classification")

for p in base_dir.rglob("*"):
    tipo = "Directorio" if p.is_dir() else "Archivo"
    print(f"{p} — {tipo} — {p.name}")
