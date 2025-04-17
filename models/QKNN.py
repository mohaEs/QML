import pennylane as qml
from pennylane import numpy as np
from pennylane.templates import RandomLayers
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os
from google.colab import drive
import cv2
from sklearn.model_selection import train_test_split
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, Sampler
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, transforms
from PIL import Image
import time
import os
import copy
from torch.optim.lr_scheduler import CosineAnnealingLR
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, confusion_matrix, roc_curve,
    precision_recall_curve, auc
)
from sklearn.preprocessing import label_binarize
import random
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.preprocessing import LabelBinarizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import pickle
import time
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_curve, auc, precision_score, recall_score, f1_score

os.environ["OMP_NUM_THREADS"] = "1"


num_qubits = 4
batch_size = 4
device = qml.device("default.qubit", wires=num_qubits)


def quantum_feature_map(x, num_qubits=4):
    device = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(device)
    def feature_map_circuit():
        for i in range(min(len(x), num_qubits)):
            qml.RX(x[i], wires=i)
        return [qml.expval(qml.PauliZ(i)) for i in range(min(len(x), num_qubits))]

    return feature_map_circuit()

def qknn_fit(X_train, y_train, k=5, num_qubits=4):
    quantum_X_train = np.array([quantum_feature_map(x, num_qubits) for x in X_train])

    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(quantum_X_train, y_train)

    return knn

def qknn_predict(knn, X_test, num_qubits=4):
    quantum_X_test = np.array([quantum_feature_map(x, num_qubits) for x in X_test])

    return knn.predict(quantum_X_test)

def qknn_predict_proba(knn, X_test, num_qubits=4):
    quantum_X_test = np.array([quantum_feature_map(x, num_qubits) for x in X_test])

    return knn.predict_proba(quantum_X_test)

def run_qknn(X_train, y_train, X_val, y_val, X_test, y_test, num_qubits=4):
    print("Training Quantum KNN...")
    knn = qknn_fit(X_train, y_train, k=5, num_qubits=num_qubits)

    print("Computing validation predictions...")
    val_pred = qknn_predict(knn, X_val, num_qubits)
    val_accuracy = accuracy_score(y_val, val_pred)
    val_precision = precision_score(y_val, val_pred, average='weighted')
    val_recall = recall_score(y_val, val_pred, average='weighted')
    val_f1 = f1_score(y_val, val_pred, average='weighted')

    print("Computing test predictions...")
    y_pred = qknn_predict(knn, X_test, num_qubits)

    y_prob = qknn_predict_proba(knn, X_test, num_qubits)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    return {
        'y_pred': y_pred,
        'y_prob': y_prob,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'val_accuracy': val_accuracy,
        'val_precision': val_precision,
        'val_recall': val_recall,
        'val_f1': val_f1
    }