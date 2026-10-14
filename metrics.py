import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

def calculate_acc(preds, labels):
    preds = np.array(preds)
    labels = np.array(labels)
    acc = accuracy_score(labels, preds)
    return acc

def get_confusion_matrix(preds, labels):
    cm = confusion_matrix(labels, preds)
    return cm
