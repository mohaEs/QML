import pennylane as qml
from pennylane import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

num_qubits = 4
batch_size = 4
device = qml.device("default.qubit", wires=num_qubits)

def run_qsvm(X_train, y_train, X_val, y_val, X_test, y_test, num_qubits=4):
    dev = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(dev)
    def qksvm_kernel_cirq(a, b):
        qml.AngleEmbedding(a, wires=range(num_qubits))
        qml.adjoint(qml.AngleEmbedding(b, wires=range(num_qubits)))
        return qml.probs(wires=range(num_qubits))

    def quantum_kernel_pca(A, B):
        return np.array([[qksvm_kernel_cirq(a, b)[0] for b in B] for a in A])

    svm = SVC(kernel='precomputed', class_weight='balanced', probability=True)

    kernel_train = quantum_kernel_pca(X_train, X_train)
    kernel_val = quantum_kernel_pca(X_train, X_val)
    kernel_test = quantum_kernel_pca(X_train, X_test)

    svm.fit(kernel_train, y_train)

    val_pred = svm.predict(kernel_val.T)
    val_accuracy = accuracy_score(y_val, val_pred)
    val_precision = precision_score(y_val, val_pred, average='weighted')
    val_recall = recall_score(y_val, val_pred, average='weighted')
    val_f1 = f1_score(y_val, val_pred, average='weighted')

    y_pred = svm.predict(kernel_test.T)
    y_prob = svm.predict_proba(kernel_test.T)

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
