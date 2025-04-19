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