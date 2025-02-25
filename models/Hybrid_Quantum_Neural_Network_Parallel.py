import pennylane as qml
from pennylane import numpy as np
from pennylane.templates import RandomLayers
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, Sampler
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, transforms


step = 0.0004               # Learning rate
batch_size = 4              # Number of samples for each training step
q_delta = 0.01              # Initial spread of random quantum weights
start_time = time.time()    # Start of the computation timer

n_qubits = 5
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def qnode(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

# Define the QLayer
n_layers = 3
weight_shapes = {"weights": (n_layers, n_qubits)}


class QuantumHybridModel(nn.Module):
    def __init__(self, num_classes):
      super(QuantumHybridModel, self).__init__()
      self.conv1 = nn.Conv2d(3, 16, 5, stride=1, padding=2)
      self.bn1 = nn.BatchNorm2d(16)
      self.pool = nn.MaxPool2d(2, 2)
      self.conv2 = nn.Conv2d(16, 32, 5, stride=1, padding=2)
      self.bn2 = nn.BatchNorm2d(32)
      self.qlayer1 = qml.qnn.TorchLayer(qnode, weight_shapes)
      self.qlayer2 = qml.qnn.TorchLayer(qnode, weight_shapes)
      self.qlayer3 = qml.qnn.TorchLayer(qnode, weight_shapes)
      self.qlayer4 = qml.qnn.TorchLayer(qnode, weight_shapes)
      self.fc1 = nn.Linear(32 * 56 * 56, 120)
      self.fc2 = nn.Linear(120, 20)
      self.fc3 = nn.Linear(20, num_classes)

    def forward(self, x):
        # Propagate the input through the CNN layers
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        # Flatten the output from the convolutional layers
        x = x.view(x.size(0), -1)
        # Pass the output to the quantum layer
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x_1, x_2, x_3, x_4 = torch.split(x, 5, dim=1)
        x_1 = self.qlayer1(x_1)
        x_2 = self.qlayer2(x_2)
        x_3 = self.qlayer3(x_3)
        x_4 = self.qlayer4(x_4)
        x = torch.cat([x_1, x_2, x_3, x_4], axis=1)
        x = self.fc3(x)
        return x
