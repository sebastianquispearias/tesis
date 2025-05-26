import os
import pandas as pd
import re

def summarize_excels():
    excel_dir = os.path.join(os.getcwd(), 'callbacks')
    arches = ['baseline', 'pspnet', 'fpn']
    rows = []
    for arch in arches:
        path = os.path.join(excel_dir, f'Unet_EfficientnetB3_final_{arch}.xlsx')
        if not os.path.isfile(path):
            print(f"[¡Error!] No encontré el Excel para '{arch}': {path}")
            continue
        df = pd.read_excel(path)
        last = df.iloc[-1]
        rows.append({
            'Arquitectura': arch,
            'Train IoU':    last.get('Train_IoU'),
            'Val IoU':      last.get('Val_IoU'),
            'Train F1':     last.get('Train_F1'),
            'Val F1':       last.get('Val_F1'),
        })
    if rows:
        summary_df = pd.DataFrame(rows)
        print("\n=== Resumen desde los Excels ===")
        print(summary_df.to_string(index=False))
    else:
        print("\n[¡Error!] No se pudo armar ningún resumen de Excels.")

def summarize_log():
    log_path = 'entrenamiento_log.txt'
    if not os.path.isfile(log_path):
        print(f"\n[¡Error!] No encontré el log en {log_path}")
        return
    print("\n=== Resumen final del entrenamiento (desde el log) ===")
    pattern = re.compile(r'^(baseline|pspnet|fpn):.*$')
    with open(log_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            m = pattern.match(line)
            if m:
                print(line)

if __name__ == '__main__':
    summarize_excels()
    summarize_log()
