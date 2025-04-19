import pennylane as qml
from pennylane import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

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
