# 1.1 Установка зависимостей
# !pip install -q rdkit
# !pip install -q torch_geometric

# Импорт библиотек
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import networkx as nx
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import roc_auc_score, accuracy_score

import torch
import torch.nn as nn
import torch.nn.functional as F

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem

import torch_geometric
from torch_geometric.data import Data, Dataset, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
import kagglehub
import pandas as pd
import os

# 1. Аутентификация (только один раз)
kagglehub.login()

# 2. Скачивание данных
print("Скачиваю данные соревнования...")
download_path = kagglehub.competition_download('rascar-cocrystal-challenge')
print(f"Данные скачаны в: {download_path}")

# 3. Загрузка в pandas
train_df = pd.read_csv(os.path.join(download_path, 'train.csv'))
test_df = pd.read_csv(os.path.join(download_path, 'test.csv'))

# 4. Проверка
print("\nПример данных из train.csv:")
print(train_df.head())

# 5. Теперь можно использовать ваш код с этими данными!
# Например, подставьте train_df в ваш CocrystalDataset
train_dataset = CocrystalDataset(train_df)


def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch_geometric.seed_everything(seed)

seed_everything()

# 1.2 Загрузка данных
df = pd.read_csv('/kaggle/input/competitions/rascar-ai-chem-hack/train.csv')
print(f"Размер датасета: {df.shape}")
print("Пример данных:")
# display(df.head())

# 1.3 Визуализация
def plot_molecule_pair(smiles1, smiles2, label):
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    if mol1 and mol2:
        img = Draw.MolsToGridImage([mol1, mol2], molsPerRow=2,
                                   subImgSize=(300, 300),
                                   legends=["Molecule A", "Molecule B"])
        print(f"Label: {label} ({'Сокристалл образуется' if label==1 else 'Не образуется'})")
        display(img)
    else:
        print("Ошибка в SMILES строке")

sample_row = df.iloc[0]
plot_molecule_pair(sample_row['SMILES1'], sample_row['SMILES2'], sample_row['result'])

# 1.4 Превращаем SMILES в Граф
PERMITTED_ATOM_TYPES = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P', 'Unknown']

def one_hot_encoding(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def get_atom_features(atom):
    results = one_hot_encoding(atom.GetSymbol(), PERMITTED_ATOM_TYPES)
    results += one_hot_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 'More'])
    results += one_hot_encoding(atom.GetFormalCharge(), [-1, -2, 1, 2, 0])
    results += one_hot_encoding(atom.GetHybridization(),
                                [Chem.rdchem.HybridizationType.SP,
                                 Chem.rdchem.HybridizationType.SP2,
                                 Chem.rdchem.HybridizationType.SP3])
    results += [int(atom.GetIsAromatic())]
    return torch.tensor(results, dtype=torch.float)

def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    atom_features = [get_atom_features(atom) for atom in mol.GetAtoms()]
    x = torch.stack(atom_features)

    edge_indices = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_indices.append([i, j])
        edge_indices.append([j, i])

    if len(edge_indices) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()

    return Data(x=x, edge_index=edge_index)

test_dim = smiles_to_graph('C').x.shape[1]
print(f"Размерность вектора признаков одного атома: {test_dim}")

test_smiles = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
graph = smiles_to_graph(test_smiles)
print("\n--- Пример графа (Кофеин) ---")
print(f"SMILES: {test_smiles}")
print(f"Количество атомов (nodes): {graph.num_nodes}")
print(f"Количество связей (edges): {graph.num_edges // 2}")
print(f"Размерность признаков узла (features): {graph.num_node_features}")
print("Структура объекта Data:", graph)

# 2. PyTorch Geometric Dataset
class CocrystalDataset(Dataset):
    def __init__(self, dataframe):
        super().__init__()
        self.df = dataframe.reset_index(drop=True)
        self.data_list = []

        for idx, row in tqdm(self.df.iterrows(), total=self.df.shape[0]):
            mol1 = smiles_to_graph(row['SMILES1'])
            mol2 = smiles_to_graph(row['SMILES2'])
            label = torch.tensor([row['result']], dtype=torch.float)

            if mol1 is not None and mol2 is not None:
                self.data_list.append({
                    'mol1': mol1,
                    'mol2': mol2,
                    'label': label
                })

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]

train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

train_dataset = CocrystalDataset(train_df)
val_dataset = CocrystalDataset(val_df)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 3. Архитектура
class GNNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNNEncoder, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)

        x = self.conv3(x, edge_index)

        x = global_mean_pool(x, batch)
        return x

class SiameseGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, embedding_dim=32):
        super(SiameseGNN, self).__init__()
        self.encoder = GNNEncoder(input_dim, hidden_dim, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, mol1, mol2):
        emb1 = self.encoder(mol1.x, mol1.edge_index, mol1.batch)
        emb2 = self.encoder(mol2.x, mol2.edge_index, mol2.batch)

        combined = torch.cat([emb1, emb2], dim=1)

        x = self.fc1(combined)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.fc2(x)

        return torch.sigmoid(x)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_dim = test_dim
model = SiameseGNN(input_dim=input_dim).to(device)

print(f"Модель инициализирована на {device}")
print(model)

# 4. Обучение и валидация
def get_weighted_accuracy(y_true, y_pred_probs, threshold=0.5):
    y_true = np.array(y_true)
    y_pred_classes = (np.array(y_pred_probs) > threshold).astype(int)
    weights = np.where(y_true == 1, 0.1140, 0.8860)
    weighted_acc = np.sum(weights * (y_true == y_pred_classes)) / np.sum(weights)
    return weighted_acc

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()
epochs = 15

history = {'train_loss': [], 'val_loss': [], 'val_auc': [], 'val_wa': []}

for epoch in range(epochs):
    model.train()
    train_loss = 0
    for batch in train_loader:
        mol1 = batch['mol1'].to(device)
        mol2 = batch['mol2'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        out = model(mol1, mol2)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)

    model.eval()
    val_loss = 0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for batch in val_loader:
            mol1 = batch['mol1'].to(device)
            mol2 = batch['mol2'].to(device)
            labels = batch['label'].to(device)

            out = model(mol1, mol2)
            loss = criterion(out, labels)

            val_loss += loss.item()
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(out.cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)

    try:
        val_auc = roc_auc_score(y_true, y_pred)
    except ValueError:
        val_auc = 0.5

    val_wa = get_weighted_accuracy(y_true, y_pred)

    history['train_loss'].append(avg_train_loss)
    history['val_loss'].append(avg_val_loss)
    history['val_auc'].append(val_auc)
    history['val_wa'].append(val_wa)

    print(f"Epoch {epoch+1}/{epochs} | "
          f"Train Loss: {avg_train_loss:.4f} | "
          f"Val Loss: {avg_val_loss:.4f} | "
          f"Val ROC-AUC: {val_auc:.4f} | "
          f"Val WA: {val_wa:.4f}")

# 4.2 Визуализация обучения
plt.figure(figsize=(18, 5))

plt.subplot(1, 3, 1)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.title('Loss History')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(history['val_auc'], label='Val ROC-AUC', color='orange')
plt.title('Validation ROC-AUC')
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(history['val_wa'], label='Val Weighted Acc', color='green')
plt.title('Validation Weighted Accuracy')
plt.legend()

plt.show()

# 5. Инференс и генерация сабмита
def predict_cocrystal(smiles1, smiles2, model, device):
    model.eval()

    g1 = smiles_to_graph(smiles1)
    g2 = smiles_to_graph(smiles2)

    if g1 is None or g2 is None:
        return 0.0

    g1.batch = torch.zeros(g1.num_nodes, dtype=torch.long)
    g2.batch = torch.zeros(g2.num_nodes, dtype=torch.long)

    g1 = g1.to(device)
    g2 = g2.to(device)

    with torch.no_grad():
        prob = model(g1, g2)

    return prob.item()

s1 = "O"
s2 = "CCCCCCCCCCCCCCCC"
prob = predict_cocrystal(s1, s2, model, device)

print(f"\nМолекула A: {s1}")
print(f"Молекула B: {s2}")
print(f"Вероятность: {prob:.4f}")

plot_molecule_pair(s1, s2, int(prob>0.5))

test_df = pd.read_csv('/kaggle/input/competitions/rascar-ai-chem-hack/test.csv')

submission_ids = []
submission_preds = []

model.eval()

for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
    smiles1 = row['SMILES1']
    smiles2 = row['SMILES2']
    pair_id = row['id']

    prob = predict_cocrystal(smiles1, smiles2, model, device)
    pred_class = 1 if prob > 0.5 else 0

    submission_ids.append(pair_id)
    submission_preds.append(pred_class)

submission_df = pd.DataFrame({
    'id': submission_ids,
    'result': submission_preds
})

submission_df.to_csv('submission.csv', index=False)
display(submission_df.head())