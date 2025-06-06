import os
import torch
from RETFound_MAE import models_vit
from kernel_methods_utils import prepare_model, feature_extract
from kernel_methods_trainers_testers import run
import numpy as np
from models.KNN import run_knn
from models.SVM import run_svm
from models.QSVM import run_qsvm
from models.QKNN import run_qknn
from kernel_methods_trainers_testers import load_all_results, compare_accuracy_bar_plots, compare_auc_bar_plots, plot_roc_curves, create_performance_tables


os.environ["OMP_NUM_THREADS"] = "1"

MODEL_SAVE_PATH = "/content/drive/MyDrive/H/QMLExperiments/G1020_models/"
data_dir = "/content/drive/MyDrive/H/qmlData"
extract_dir = "/content/drive/MyDrive/H/QMLExperiments/G1020"
dataset = "G1020"

# download pre-trained RETFound and extract features on dataset with model

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])


chkpt_dir = './RETFound_MAE/RETFound_cfp_weights.pth'
model_ = prepare_model(chkpt_dir, 'vit_large_patch16')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_.to(device)
print('Model loaded.')

feature_extract(extract_dir, model_)

#run models
qsvm_result = run(run_qsvm, dataset, num_qubits=4)
svm_result = run(run_svm, dataset)
knn_result = run(run_knn, dataset)
qknn_result = run(run_qknn, dataset, num_qubits=4)
     
#plot
data_dir = data_dir
results = load_all_results(data_dir)
compare_accuracy_bar_plots(results, data_dir)
compare_auc_bar_plots(results, data_dir)
plot_roc_curves(results, data_dir)
create_performance_tables(results, data_dir)
