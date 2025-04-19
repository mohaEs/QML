import pickle
import numpy as np
import os
from PIL import Image
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.utils import shuffle

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])

def prepare_model(chkpt_dir, arch='vit_large_patch16'):
    # build model
    model = models_vit.__dict__[arch](
        img_size=224,
        num_classes=5,
        drop_path_rate=0,
        global_pool=True,
    )
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    return model

def run_one_image(img, model):
    x = torch.tensor(img)
    x = x.unsqueeze(dim=0)
    x = torch.einsum('nhwc->nchw', x)

    x = x.to(device, non_blocking=True)
    latent = model.forward_features(x.float())
    latent = torch.squeeze(latent)

    return latent
     
# download pre-trained RETFound
chkpt_dir = './RETFound_MAE/RETFound_cfp_weights.pth'
model_ = prepare_model(chkpt_dir, 'vit_large_patch16')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_.to(device)
print('Model loaded.')

def feature_extract(data_path, model_):
    model_.eval()

    feature_dicts = {
        'train': {'names': [], 'features': [], 'labels': []},
        'val': {'names': [], 'features': [], 'labels': []},
        'test': {'names': [], 'features': [], 'labels': []}
    }

    for split in ['train', 'val', 'test']:
        split_path = os.path.join(data_path, split)

        for class_name in os.listdir(split_path):
            class_folder = os.path.join(split_path, class_name)

            for img_name in os.listdir(class_folder):
                img_path = os.path.join(class_folder, img_name)

                img = Image.open(img_path).resize((224, 224))
                img_array = np.array(img) / 255.0
                img_normalized = (img_array - imagenet_mean) / imagenet_std

                x = torch.tensor(img_normalized)
                x = x.unsqueeze(dim=0)
                x = torch.einsum('nhwc->nchw', x)
                x = x.to(device, non_blocking=True)

                with torch.no_grad():
                    latent = model_.forward_features(x.float())
                    latent = torch.squeeze(latent)

                feature_dicts[split]['names'].append(img_name)
                feature_dicts[split]['features'].append(latent.detach().cpu().numpy())
                feature_dicts[split]['labels'].append(class_name)

    pickle.dump(feature_dicts['train'], open('APTOS_train.pkl', 'wb'))
    pickle.dump(feature_dicts['val'], open('APTOS_val.pkl', 'wb'))
    pickle.dump(feature_dicts['test'], open('APTOS_test.pkl', 'wb'))

    print("Feature extraction completed.")
    return feature_dicts

def load_features(pkl_path='Feature_latent_multiclass.pkl'):
    with open(pkl_path, 'rb') as f:
        features = pickle.load(f)
    return features


def load_and_shuffle_data(seed=42, num_qubits=4):
    with open(dataset+'_train.pkl', 'rb') as f:
        train_data = pickle.load(f)
    with open(dataset+'_val.pkl', 'rb') as f:
        val_data = pickle.load(f)
    with open(dataset+'_test.pkl', 'rb') as f:
        test_data = pickle.load(f)

    train_features = np.array(train_data['features'])
    val_features = np.array(val_data['features'])
    test_features = np.array(test_data['features'])

    train_labels = np.array(train_data['labels'])
    val_labels = np.array(val_data['labels'])
    test_labels = np.array(test_data['labels'])

    train_features, train_labels = shuffle(train_features, train_labels, random_state=seed)
    val_features, val_labels = shuffle(val_features, val_labels, random_state=seed)
    test_features, test_labels = shuffle(test_features, test_labels, random_state=seed)

    label_encoder = LabelEncoder()
    label_encoder.fit(np.concatenate([train_labels, val_labels, test_labels]))

    train_labels_encoded = label_encoder.transform(train_labels)
    val_labels_encoded = label_encoder.transform(val_labels)
    test_labels_encoded = label_encoder.transform(test_labels)

    pca = PCA(n_components=num_qubits)
    pca.fit(train_features)

    train_features_reduced = pca.transform(train_features)
    val_features_reduced = pca.transform(val_features)
    test_features_reduced = pca.transform(test_features)

    scaler = StandardScaler()
    scaler.fit(train_features_reduced)

    train_features_scaled = scaler.transform(train_features_reduced)
    val_features_scaled = scaler.transform(val_features_reduced)
    test_features_scaled = scaler.transform(test_features_reduced)

    return (train_features_scaled, train_labels_encoded,
            val_features_scaled, val_labels_encoded,
            test_features_scaled, test_labels_encoded,
            label_encoder.classes_)

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from itertools import cycle
import seaborn as sns

def load_all_results(data_dir="qmlData"):
    all_results = {}

    pkl_files = [f for f in os.listdir(data_dir) if f.endswith(".pkl")]

    for pkl_file in pkl_files:
        file_parts = pkl_file.split('_')

        model = file_parts[0]

        dataset = '_'.join(file_parts[1:-1])

        file_path = os.path.join(data_dir, pkl_file)
        try:
            with open(file_path, 'rb') as f:
                results = pickle.load(f)

            if dataset not in all_results:
                all_results[dataset] = {}

            all_results[dataset][model] = results
            print(f"Loaded: {model} model for {dataset} dataset")

        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    return all_results

def compare_accuracy_bar_plots(all_results, data_dir="qmlData", compare_pairs=[("svm", "qsvm"), ("knn", "qknn")]):
    for model1, model2 in compare_pairs:
        datasets = []
        acc1 = []
        acc2 = []

        for dataset, models in all_results.items():
            if model1 in models and model2 in models:
                datasets.append(dataset)
                acc1.append(models[model1]['accuracy'])
                acc2.append(models[model2]['accuracy'])

        plt.figure(figsize=(12, 6))

        bar_width = 0.35
        index = np.arange(len(datasets))

        color1 = 'steelblue'  # Classical methods
        color2 = 'lightblue'  # Quantum methods

        plt.bar(index, acc1, bar_width, label=model1.upper(), color=color1)
        plt.bar(index + bar_width, acc2, bar_width, label=model2.upper(), color=color2)

        plt.xlabel('Dataset', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title(f'Accuracy Comparison: {model1.upper()} vs {model2.upper()}', fontsize=14)
        plt.xticks(index + bar_width/2, datasets, rotation=45, ha='right')
        plt.legend()

        for i, v in enumerate(acc1):
            plt.text(i, v + 0.01, f'{v:.3f}', ha='center')

        for i, v in enumerate(acc2):
            plt.text(i + bar_width, v + 0.01, f'{v:.3f}', ha='center')

        plt.ylim(0, 1.1)
        plt.tight_layout()
        plt.savefig(os.path.join(data_dir, f'accuracy_comparison_{model1}_vs_{model2}.png'), dpi=300, bbox_inches='tight')
        plt.show()

def calculate_auc_values(all_results):
    auc_values = {}

    for dataset, models in all_results.items():
        auc_values[dataset] = {}

        for model_name, results in models.items():
            y_test = results['y_test']
            y_prob = results['y_prob']

            n_classes = len(np.unique(y_test))

            fpr = dict()
            tpr = dict()
            roc_auc = dict()

            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve((y_test == i).astype(int), y_prob[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])

            all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
            mean_tpr = np.zeros_like(all_fpr)

            for i in range(n_classes):
                mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

            mean_tpr /= n_classes
            macro_auc = auc(all_fpr, mean_tpr)

            auc_values[dataset][model_name] = macro_auc

    return auc_values

def compare_auc_bar_plots(all_results, data_dir="qmlData", compare_pairs=[("svm", "qsvm"), ("knn", "qknn")]):
    auc_values = calculate_auc_values(all_results)

    for model1, model2 in compare_pairs:
        datasets = []
        dataset_labels = ['APTOS-2019 (Diabetic Retinopathy)', 'MESSIDOR (Diabetic Retinopathy)', 'IDRID (Diabetic Retinopathy)', 'PAPILA (Glaucoma)', 'Glaucoma Fundus (Glaucoma)', 'G1020 (Glaucoma)']
        auc1 = []
        auc2 = []

        for dataset, models in auc_values.items():
            if model1 in models and model2 in models:
                datasets.append(dataset)
                auc1.append(models[model1])
                auc2.append(models[model2])

        plt.figure(figsize=(12, 6))

        bar_width = 0.35
        index = np.arange(len(datasets))

        color1 = 'steelblue'  # Classical methods
        color2 = 'lightblue'  # Quantum methods

        plt.bar(index, auc1, bar_width, label=model1.upper(), color=color1, alpha=0.7)
        plt.bar(index + bar_width, auc2, bar_width, label=model2.upper(), color=color2, alpha=0.7)

        plt.xlabel('Dataset', fontsize=12)
        plt.ylabel('AUC', fontsize=12)
        plt.title(f'AUC Comparison: {model1.upper()} vs {model2.upper()}', fontsize=14)
        plt.xticks(index + bar_width/2, dataset_labels, rotation=45, ha='right')
        plt.legend()

        for i, v in enumerate(auc1):
            plt.text(i, v + 0.01, f'{v:.3f}', ha='center')

        for i, v in enumerate(auc2):
            plt.text(i + bar_width, v + 0.01, f'{v:.3f}', ha='center')

        plt.ylim(0, 1.1)
        plt.tight_layout()
        plt.savefig(os.path.join(data_dir, f'auroc_comparison_{model1}_vs_{model2}.png'), dpi=300, bbox_inches='tight')
        plt.show()

        mean_auc1 = np.mean(auc1)
        mean_auc2 = np.mean(auc2)

        plt.figure(figsize=(6, 8))
        plt.bar(0, mean_auc1, bar_width*2, label=model1.upper(), color=color1, alpha = 0.7)
        plt.bar(1, mean_auc2, bar_width*2, label=model2.upper(), color=color2, alpha = 0.7)

        plt.ylabel('Mean AUC', fontsize=12)
        plt.title(f'Mean AUC Across Datasets: {model1.upper()} vs {model2.upper()}', fontsize=14)
        plt.xticks([0, 1], [model1.upper(), model2.upper()])
        plt.legend()

        plt.text(0, mean_auc1 + 0.01, f'{mean_auc1:.3f}', ha='center')
        plt.text(1, mean_auc2 + 0.01, f'{mean_auc2:.3f}', ha='center')

        plt.ylim(0, 1.1)
        plt.tight_layout()
        plt.savefig(os.path.join(data_dir, f'mean_auc_comparison_{model1}_vs_{model2}.png'), dpi=300, bbox_inches='tight')
        plt.show()


def plot_roc_curves(all_results, data_dir="qmlData"):
    plt.rcParams.update({'font.size': 16})

    for dataset, models in all_results.items():
        plt.figure(figsize=(8, 7))

        first_model = next(iter(models.values()))
        class_names = first_model['class_names']
        n_classes = len(class_names)

        colors = cycle(['blue', 'green', 'orange', 'red'])
        line_styles = cycle(['-', '--', ':', '-.'])

        for model_name, results in models.items():
            y_test = results['y_test']
            y_prob = results['y_prob']

            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve((y_test == i).astype(int), y_prob[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])

            all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(n_classes):
                mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
            mean_tpr /= n_classes

            macro_auc = auc(all_fpr, mean_tpr)

            color = next(colors)
            linestyle = next(line_styles)
            plt.plot(
                all_fpr, mean_tpr,
                label=f'{model_name.upper()} (AUC = {macro_auc:.2f})',
                color=color, linestyle=linestyle, linewidth=2
            )

        plt.plot([0, 1], [0, 1], 'k--')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate', fontsize=18)
        plt.ylabel('True Positive Rate', fontsize=18)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.title(f'ROC Curves for {dataset} Dataset', fontsize=22)
        plt.legend(loc="lower right", fontsize=14)

        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(data_dir, f'a_roc_curves_{dataset}.pdf'), format='pdf', bbox_inches='tight')
        plt.show()

        for model_name, results in models.items():
            plt.figure(figsize=(8, 7))

            y_test = results['y_test']
            y_prob = results['y_prob']

            for i, class_name in enumerate(class_names):
                fpr, tpr, _ = roc_curve((y_test == i).astype(int), y_prob[:, i])
                roc_auc = auc(fpr, tpr)

                plt.plot(
                    fpr, tpr,
                    label=f'Class {class_name} (AUC = {roc_auc:.2f})',
                    linewidth=2
                )

            fpr = dict()
            tpr = dict()
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve((y_test == i).astype(int), y_prob[:, i])

            all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(n_classes):
                mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
            mean_tpr /= n_classes

            macro_auc = auc(all_fpr, mean_tpr)

            plt.plot(
                all_fpr, mean_tpr,
                label=f'Macro-average (AUC = {macro_auc:.2f})',
                color='navy', linestyle=':', linewidth=4
            )

            plt.plot([0, 1], [0, 1], 'k--')

            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.0])
            plt.xlabel('False Positive Rate', fontsize=18)
            plt.ylabel('True Positive Rate', fontsize=18)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.title(f'ROC Curves: {model_name.upper()} on {dataset}', fontsize=22)
            plt.legend(loc="lower right", fontsize=14)

            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(data_dir, f'a_roc_curves_{model_name}_{dataset}.pdf'), format='pdf', bbox_inches='tight')
            plt.show()


def create_performance_tables(all_results, data_dir="qmlData"):
    overall_metrics = []

    for dataset, models in all_results.items():
        for model_name, results in models.items():
            overall_metrics.append({
                'Dataset': dataset,
                'Model': model_name.upper(),
                'Accuracy': results['accuracy'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'F1 Score': results['f1'],
                'Is Quantum': 'Yes' if results['is_quantum'] else 'No'
            })

    metrics_df = pd.DataFrame(overall_metrics)

    metrics_df = metrics_df.sort_values(['Dataset', 'Model'])

    print("\n===== OVERALL PERFORMANCE METRICS =====")
    print(metrics_df.to_string(index=False))

    metrics_df.to_csv(os.path.join(data_dir, 'performance_metrics.csv'), index=False)
