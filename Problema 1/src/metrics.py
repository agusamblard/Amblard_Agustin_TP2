import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import Counter
from random import randint, uniform


def confusion_matrix(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    return TP, TN, FP, FN

def plot_confusion_matrix(y_true, y_pred, labels=[0, 1], title="Matriz de Confusión"):
    # Crear la matriz de confusión manualmente
    cm = np.zeros((2, 2), dtype=int)
    for yt, yp in zip(y_true, y_pred):
        cm[int(yt)][int(yp)] += 1

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)

    plt.xlabel("Predicción")
    plt.ylabel("Valor real")
    plt.title(title)
    plt.show()

def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)

def precision_score(y_true, y_pred):
    TP, _, FP, _ = confusion_matrix(y_true, y_pred)
    return TP / (TP + FP) if (TP + FP) > 0 else 0.0

def recall_score(y_true, y_pred):
    TP, _, _, FN = confusion_matrix(y_true, y_pred)
    return TP / (TP + FN) if (TP + FN) > 0 else 0.0

def f1_score(y_true, y_pred):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

def roc_curve(y_true, y_scores):
    thresholds = np.sort(np.unique(y_scores))[::-1]
    tpr_list = []
    fpr_list = []

    for thresh in thresholds:
        y_pred = (y_scores >= thresh).astype(int)
        TP, TN, FP, FN = confusion_matrix(y_true, y_pred)
        TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
        FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
        tpr_list.append(TPR)
        fpr_list.append(FPR)

    return np.array(fpr_list), np.array(tpr_list)

def auc(x, y):
    # AUC usando regla del trapecio
    return np.trapz(y, x)

def pr_curve(y_true, y_scores):
    thresholds = np.sort(np.unique(y_scores))[::-1]
    precision_list = []
    recall_list = []

    for thresh in thresholds:
        y_pred = (y_scores >= thresh).astype(int)
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        precision_list.append(prec)
        recall_list.append(rec)

    return np.array(recall_list), np.array(precision_list)

def compute_f1(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    return f1





def report_metrics(y_true, y_pred, y_scores):
    TP, TN, FP, FN = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print("Matriz de Confusión:")
    print(f"TP: {TP}, FP: {FP}")
    print(f"FN: {FN}, TN: {TN}")
    print("\nMétricas:")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    fpr, tpr = roc_curve(y_true, y_scores)
    auc_roc = auc(fpr, tpr)
    print(f"AUC-ROC:   {auc_roc:.4f}")

    recall_vals, precision_vals = pr_curve(y_true, y_scores)
    auc_pr = auc(recall_vals, precision_vals)
    print(f"AUC-PR:    {auc_pr:.4f}")

    plot_confusion_matrix(y_true, y_pred)

    # Mostrar curvas
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, label=f"AUC = {auc_roc:.4f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title("Curva ROC")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(recall_vals, precision_vals, label=f"AUC = {auc_pr:.4f}")
    plt.title("Curva Precision-Recall")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()

    plt.tight_layout()
    plt.show()


