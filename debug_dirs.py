from pathlib import Path

# Rutas que queremos inspeccionar
to_check = {
    "VT Train Images": Path(r"C:\Users\User\Desktop\tesis\data\Corrosion Condition State Classification\512x512\Train\images_512"),
    "VT Train Masks":  Path(r"C:\Users\User\Desktop\tesis\data\Corrosion Condition State Classification\512x512\Train\mask_512"),
    "VT Test Images":  Path(r"C:\Users\User\Desktop\tesis\data\Corrosion Condition State Classification\512x512\Test\images_512"),
    "VT Test Masks":   Path(r"C:\Users\User\Desktop\tesis\data\Corrosion Condition State Classification\512x512\Test\mask_512"),
    "Ameli Images":    Path(r"C:\Users\User\Desktop\tesis\data\corrosion images\corrosion images\images"),
    "Ameli JSON":      Path(r"C:\Users\User\Desktop\tesis\data\corrosion images\corrosion images\Annotation json format"),
    "Ameli TXT":       Path(r"C:\Users\User\Desktop\tesis\data\corrosion images\corrosion images\Annotation txt format"),
}

for name, p in to_check.items():
    print(f"\n{name}: {p}")
    print("  Exists:", p.exists())
    if p.exists() and p.is_dir():
        entries = sorted(p.iterdir())
        print("  Entries:", [e.name for e in entries][:10], "…" if len(entries)>10 else "")
    else:
        print("  (no es un directorio válido)")
