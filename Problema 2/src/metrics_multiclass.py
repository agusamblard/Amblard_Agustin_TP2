import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def confusion_matrix_multiclass(y_true, y_pred, labels):
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for yt, yp in zip(y_true, y_pred):
        cm[labels.index(yt)][labels.index(yp)] += 1
    return cm

def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)


def precision_macro(y_true, y_pred, labels):
    precisions = []
    for label in labels:
        TP = np.sum((y_true == label) & (y_pred == label))
        FP = np.sum((y_true != label) & (y_pred == label))
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        precisions.append(precision)
    return np.mean(precisions)

def recall_macro(y_true, y_pred, labels):
    recalls = []
    for label in labels:
        TP = np.sum((y_true == label) & (y_pred == label))
        FN = np.sum((y_true == label) & (y_pred != label))
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        recalls.append(recall)
    return np.mean(recalls)

def f1_macro(y_true, y_pred, labels):
    f1s = []
    for label in labels:
        TP = np.sum((y_true == label) & (y_pred == label))
        FP = np.sum((y_true != label) & (y_pred == label))
        FN = np.sum((y_true == label) & (y_pred != label))

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        f1s.append(f1)
    return np.mean(f1s)


def roc_curve_multi(y_true, y_scores):
    thresholds = np.sort(np.unique(y_scores))[::-1]
    tpr_list, fpr_list = [], []

    for thresh in thresholds:
        y_pred = (y_scores >= thresh).astype(int)
        TP = np.sum((y_true == 1) & (y_pred == 1))
        TN = np.sum((y_true == 0) & (y_pred == 0))
        FP = np.sum((y_true == 0) & (y_pred == 1))
        FN = np.sum((y_true == 1) & (y_pred == 0))

        TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
        FPR = FP / (FP + TN) if (FP + TN) > 0 else 0

        tpr_list.append(TPR)
        fpr_list.append(FPR)

    return np.array(fpr_list), np.array(tpr_list)

def pr_curve_multi(y_true, y_scores):
    thresholds = np.sort(np.unique(y_scores))[::-1]
    precision_list = [1]
    recall_list = [0]

    for thresh in thresholds:
        y_pred = (y_scores >= thresh).astype(int)
        TP = np.sum((y_true == 1) & (y_pred == 1))
        FP = np.sum((y_true == 0) & (y_pred == 1))
        FN = np.sum((y_true == 1) & (y_pred == 0))

        prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        rec = TP / (TP + FN) if (TP + FN) > 0 else 0.0


        if rec == 0:
            precision_list.pop(0)
            recall_list.pop(0)

        precision_list.append(prec)
        recall_list.append(rec)

    return np.array(recall_list), np.array(precision_list)



def auc(x, y):
    return np.trapz(y, x)





def report_metrics_multiclass(y_true, y_pred, y_scores, clase_objetivo, class_index_override=None):
    labels = sorted([label for label in set(y_true) | set(y_pred)])
    cm = confusion_matrix_multiclass(y_true, y_pred, labels)
    acc = accuracy_score(y_true, y_pred)
    prec_macro = precision_macro(y_true, y_pred, labels)
    rec_macro = recall_macro(y_true, y_pred, labels)
    f1_macro_val = f1_macro(y_true, y_pred, labels)
    print("Matriz de Confusión:")
    print(cm)

    print("\nMétricas:")
    print(f"Accuracy:        {acc:.4f}")
    print(f"Precision macro: {prec_macro:.4f}")
    print(f"Recall macro:    {rec_macro:.4f}")
    print(f"F1 Score macro:  {f1_macro_val:.4f}")
    # Binaria para clase objetivo
    y_true_bin = (y_true == clase_objetivo).astype(int)
    y_pred_bin = (y_pred == clase_objetivo).astype(int)
    if class_index_override is not None:
        y_scores_bin = y_scores[:, class_index_override]
    else:
        class_index = labels.index(clase_objetivo)
        y_scores_bin = y_scores[:, class_index]
    # Curvas y AUC
    fpr, tpr = roc_curve_multi(y_true_bin, y_scores_bin)
    recall_vals, precision_vals = pr_curve_multi(y_true_bin, y_scores_bin)
    auc_roc = auc(fpr, tpr)
    auc_pr = auc(recall_vals, precision_vals)

    print(f"\nAUC-ROC (clase {clase_objetivo}): {auc_roc:.4f}")
    print(f"AUC-PR  (clase {clase_objetivo}): {auc_pr:.4f}")

    # Graficar
    plt.figure(figsize=(18, 5))

    plt.subplot(1, 3, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title("Matriz de Confusión")
    plt.xlabel("Predicción")
    plt.ylabel("Valor real")

    plt.subplot(1, 3, 2)
    plt.plot(fpr, tpr, label=f"AUC = {auc_roc:.4f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title(f"Curva ROC - Clase {clase_objetivo}")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(recall_vals, precision_vals, label=f"AUC = {auc_pr:.4f}")
    plt.title(f"Curva PR - Clase {clase_objetivo}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()

    plt.tight_layout()
    plt.show()


def macro_average_curves(y_true, y_scores, labels):


        # AUC-ROC y AUC-PR promedio
    aucs_roc = []
    aucs_pr = []

    # Usamos una única fuente de verdad para curvas y AUCs
    for i, clase in enumerate(labels):
        y_true_bin = (y_true == clase).astype(int)
        y_score_bin = y_scores[:, i]

        fpr, tpr = roc_curve_multi(y_true_bin, y_score_bin)
        recall_vals, precision_vals = pr_curve_multi(y_true_bin, y_score_bin)

        aucs_roc.append(auc(fpr, tpr))
        aucs_pr.append(auc(recall_vals, precision_vals))

    auc_roc_macro = np.mean(aucs_roc)
    auc_pr_macro = np.mean(aucs_pr)
    # Gráfica
    plt.figure(figsize=(18, 5))
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, label=f"AUC-ROC Macro = {auc_roc_macro:.4f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title("Curva ROC - Promedio Macro")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(recall_vals, precision_vals, label=f"AUC-PR Macro = {auc_pr_macro:.4f}")
    plt.title("Curva PR - Promedio Macro")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.tight_layout()
    plt.show()


def report_metrics_multiclass_global(y_true, y_pred, y_scores):
    labels = sorted(list(set(y_true)))

    cm = confusion_matrix_multiclass(y_true, y_pred, labels)
    acc = accuracy_score(y_true, y_pred)
    prec_macro = precision_macro(y_true, y_pred, labels)
    rec_macro = recall_macro(y_true, y_pred, labels)
    f1_macro_val = f1_macro(y_true, y_pred, labels)

    print("📊 Matriz de Confusión:")
    print(cm)

    print("\n✅ Métricas Globales:")
    print(f"Accuracy:        {acc:.4f}")
    print(f"Precision Macro: {prec_macro:.4f}")
    print(f"Recall Macro:    {rec_macro:.4f}")
    print(f"F1 Score Macro:  {f1_macro_val:.4f}")
    
    # AUC-ROC y AUC-PR promedio
    aucs_roc = []
    aucs_pr = []

    # Usamos una única fuente de verdad para curvas y AUCs
    for i, clase in enumerate(labels):
        y_true_bin = (y_true == clase).astype(int)
        y_score_bin = y_scores[:, i]

        fpr, tpr = roc_curve_multi(y_true_bin, y_score_bin)
        recall_vals, precision_vals = pr_curve_multi(y_true_bin, y_score_bin)

        aucs_roc.append(auc(fpr, tpr))
        aucs_pr.append(auc(recall_vals, precision_vals))

    auc_roc_macro = np.mean(aucs_roc)
    auc_pr_macro = np.mean(aucs_pr)

    print(f"\n📈 AUC-ROC Macro (curva promediada): {auc_roc_macro:.4f}")
    print(f"📈 AUC-PR  Macro (curva promediada): {auc_pr_macro:.4f}")

    macro_average_curves(y_true, y_scores, labels)



def comparar_modelos_macro_roc_pr(modelos, nombres_modelos):
    """
    Grafica las curvas ROC y PR promedio macro de múltiples modelos.
    
    Parámetros:
        modelos: lista de tuplas (y_true, y_scores) por modelo
        nombres_modelos: lista de nombres de modelos
    """
    plt.figure(figsize=(16, 6))

    # ROC promedio
    plt.subplot(1, 2, 1)
    for (y_true, y_scores), nombre in zip(modelos, nombres_modelos):
        labels = sorted(list(set(y_true)))
        aucs_roc = []
        for i, clase in enumerate(labels):
            y_true_bin = (y_true == clase).astype(int)
            y_scores_bin = y_scores[:, i]
            fpr, tpr = roc_curve_multi(y_true_bin, y_scores_bin)
            aucs_roc.append(auc(fpr, tpr))
        auc_roc_macro = np.mean(aucs_roc)
        # Solo graficamos una curva representativa (la última) + promedio
        plt.plot(fpr, tpr, label=f"{nombre} (AUC ROC macro = {auc_roc_macro:.4f})")
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.title("Curva ROC - Macro promedio")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.grid()
    plt.legend()

    # PR promedio
    plt.subplot(1, 2, 2)
    for (y_true, y_scores), nombre in zip(modelos, nombres_modelos):
        labels = sorted(list(set(y_true)))
        aucs_pr = []
        for i, clase in enumerate(labels):
            y_true_bin = (y_true == clase).astype(int)
            y_scores_bin = y_scores[:, i]
            recall, precision = pr_curve_multi(y_true_bin, y_scores_bin)
            aucs_pr.append(auc(recall, precision))
        auc_pr_macro = np.mean(aucs_pr)
        plt.plot(recall, precision, label=f"{nombre} (AUC PR macro = {auc_pr_macro:.4f})")
    plt.title("Curva Precision-Recall - Macro promedio")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.show()
