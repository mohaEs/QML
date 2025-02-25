import pennylane as qml
from pennylane import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
import copy
import random
from models.Classical_CNN import NormalModel
from models.Hybrid_Quantum_Neural_Network_Parallel import QuantumHybridModel
from trainers_testers import train_validate_model, test 
from utils import plot, set_seed, load_data

seeds = 5
n_epochs = 100
num_classes = 5
batch_size = 32
dataset = "IDRID"
model_name_hybrid = 'quantum_hybrid_model'
model_name_normal = 'normal_model'

for i in range(seeds):
    seed = random.randint(1, 100)
    print(f"Experiment with Seed {seed}:")
    set_seed(seed)
    train_loader, valid_loader, test_loader = load_data(batch_size=batch_size, dataset=dataset)
    model = NormalModel(num_classes=5)
    hybrid_model = QuantumHybridModel(num_classes=5)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name_hybrid = 'quantum_hybrid_model'
    model_name_normal = 'normal_model'
    model.to(device)
    hybrid_model.to(device)
    
    model_normal, normal_train_losses, normal_train_accs, normal_val_losses, normal_val_accs, normal_train_aucs, normal_val_aucs = train_validate_model(
        n_epochs=n_epochs,
        model=model,
        model_name=model_name_normal,
        train_loader=train_loader,
        valid_loader=valid_loader,
        seed=seed
    )
    
    model_hybrid, hybrid_train_losses, hybrid_train_accs, hybrid_val_losses, hybrid_val_accs, hybrid_train_aucs, hybrid_val_aucs = train_validate_model(
        n_epochs=n_epochs,
        model=hybrid_model,
        model_name=model_name_hybrid,
        train_loader=train_loader,
        valid_loader=valid_loader,
        seed=seed,
        quantum=True
    )
    
    plot(hybrid_train_losses, normal_train_losses, hybrid_train_accs, normal_train_accs, hybrid_train_aucs, normal_train_aucs)
    
    plot(hybrid_val_losses, normal_val_losses, hybrid_val_accs, normal_val_accs, hybrid_val_aucs, normal_val_aucs)
    
    test(model_hybrid, model_normal, test_loader, num_classes=num_classes)

  
