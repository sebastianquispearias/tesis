# metrics_utils.py

def evaluate_and_log(
        model, loader, writer, epoch,
        loss_sup, loss_cons, loss_total
    ):
    """
    model.evaluate devuelve:
      [val_loss, val_iou, val_precision, val_recall, val_f1]
    Y además tú le pasas las pérdidas:
      loss_sup, loss_cons, loss_total
    Escribe TODO en el CSV.
    """

    # 1) Calcula las métricas de validación
    metrics = model.evaluate(loader, verbose=0)
    val_loss, val_iou, val_precision, val_recall, val_f1 = map(float, metrics)

    # 2) Cabecera (solo en la época 0)
    if epoch == 0:
        writer.writerow([
            'epoch',
            'loss_sup', 'loss_cons', 'loss_total',
            'val_loss', 'val_iou',
            'val_precision', 'val_recall', 'val_f1'
        ])

    # 3) Fila de datos
    writer.writerow([
        epoch,
        loss_sup, loss_cons, loss_total,
        val_loss, val_iou,
        val_precision, val_recall, val_f1
    ])

    # 4) Devuelve lo que necesites para logging en consola
    return val_loss, val_iou, val_precision, val_recall, val_f1
