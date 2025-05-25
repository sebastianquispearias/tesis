import os

base = r"C:\Users\User\Desktop\tesis\data"

print("Contenido de:", base)
for entry in os.listdir(base):
    path = os.path.join(base, entry)
    if os.path.isdir(path):
        print(f"\n-- {entry} (carpeta) --")
        try:
            subs = os.listdir(path)
            print("   Subcarpetas/archivos:", subs)
        except PermissionError:
            print("   (sin permiso para listar)")
