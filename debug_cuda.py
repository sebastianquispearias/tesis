# debug_cuda.py
import ctypes, os

print("=== PATH ===")
print(os.environ.get("PATH", "").split(os.pathsep))

for lib in ("nvcuda.dll", "cudart64_110.dll", "cudnn64_8.dll"):
    try:
        ctypes.WinDLL(lib)
        print(f"{lib}: ✅ cargada")
    except Exception as e:
        print(f"{lib}: ❌ no se pudo cargar ({e})")
