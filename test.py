# Импорт библиотек
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F

# Работа с представлениями молекул
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors, Lipinski

# Работа с графами
import torch_geometric
from torch_geometric.data import Data, Dataset, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool


# Фиксируем random seed для воспроизводимости
def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch_geometric.seed_everything(seed)

seed_everything()

# Загружаем данные
print("Загрузка данных...")
df = pd.read_csv('train.csv')
print(f"Размер датасета: {df.shape}")
print("Пример данных:")
print(df.head())

PERMITTED_ATOM_TYPES = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P', 'Unknown']


def one_hot_encoding(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def get_molecule_features(mol):
    """
    Собирает глобальные признаки для всей молекулы.
    """
    results = []
    results.append(Descriptors.MolMR(mol))
    results.append(Descriptors.LabuteASA(mol))
    results.append(Lipinski.NumHDonors(mol))
    results.append(Lipinski.NumHAcceptors(mol))
    return torch.tensor(results, dtype=torch.float)


def get_atom_features(atom):
    """
    Собирает признаки атома.
    """
    results = one_hot_encoding(atom.GetSymbol(), PERMITTED_ATOM_TYPES)
    results += one_hot_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6])
    results += one_hot_encoding(atom.GetFormalCharge(), [-1, -2, 1, 2, 0])
    results += one_hot_encoding(atom.GetHybridization(),
                                [Chem.rdchem.HybridizationType.SP,
                                 Chem.rdchem.HybridizationType.SP2,
                                 Chem.rdchem.HybridizationType.SP3])
    results += [int(atom.GetIsAromatic())]
    return torch.tensor(results, dtype=torch.float)


def get_bond_features(bond):
    """
    Собирает признаки связи (ровно 6 признаков).
    """
    bond_type = bond.GetBondType()
    result = [
        int(bond_type == Chem.rdchem.BondType.SINGLE),
        int(bond_type == Chem.rdchem.BondType.DOUBLE),
        int(bond_type == Chem.rdchem.BondType.TRIPLE),
        int(bond_type == Chem.rdchem.BondType.AROMATIC),
        int(bond.IsInRing()),
        int(bond.GetIsConjugated())
    ]
    return result


def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Узлы
    atom_features = [get_atom_features(atom) for atom in mol.GetAtoms()]
    x = torch.stack(atom_features)

    # Глобальные признаки
    mol_features = get_molecule_features(mol)

    # Ребра
    edge_indices = []
    edge_features = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        edge_indices.append([i, j])
        edge_indices.append([j, i])

        bond_feat = get_bond_features(bond)
        edge_features.append(bond_feat)
        edge_features.append(bond_feat)

    if len(edge_indices) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 6), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, global_features=mol_features)


# Проверка размерностей
test_dim = smiles_to_graph('C').x.shape[1]
print(f"Размерность вектора признаков одного атома: {test_dim}")

# Проверка на примере
test_smiles = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
graph = smiles_to_graph(test_smiles)
print(f"\n--- Пример графа ---")
print(f"Количество атомов: {graph.num_nodes}")
print(f"Количество связей: {graph.num_edges // 2}")
print(f"Признаки узлов: {graph.x.shape}")
print(f"Признаки связей: {graph.edge_attr.shape}")
print(f"Глобальные признаки: {graph.global_features.shape}")


class CocrystalDataset(Dataset):
    def __init__(self, dataframe):
        super().__init__()
        self.df = dataframe.reset_index(drop=True)
        self.data_list = []

        for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Создание датасета"):
            mol1 = smiles_to_graph(row['SMILES1'])
            mol2 = smiles_to_graph(row['SMILES2'])
            label = torch.tensor([row['result']], dtype=torch.float)

            if mol1 is not None and mol2 is not None:
                comp_features = self.calculate_complementarity(mol1, mol2)
                self.data_list.append({
                    'mol1': mol1,
                    'mol2': mol2,
                    'comp_features': comp_features,
                    'label': label
                })

    def calculate_complementarity(self, mol1, mol2):
        feat1 = mol1.global_features
        feat2 = mol2.global_features

        donors1, acceptors1 = feat1[2].item(), feat1[3].item()
        donors2, acceptors2 = feat2[2].item(), feat2[3].item()

        return torch.tensor([
            abs(donors1 - acceptors2),
            abs(acceptors1 - donors2),
            abs((donors1 + donors2) - (acceptors1 + acceptors2)),
            feat1[0] / (feat2[0] + 1e-5),
            feat1[1] / (feat2[1] + 1e-5),
        ], dtype=torch.float)

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]


# Разделяем на train/val
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

train_dataset = CocrystalDataset(train_df)
val_dataset = CocrystalDataset(val_df)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


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
        self.comp_fc = nn.Linear(5, 16)

        combined_dim = embedding_dim * 2 + 16
        self.fc1 = nn.Linear(combined_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 32)
        self.fc3 = nn.Linear(32, 1)

        self.dropout = nn.Dropout(0.2)

    def forward(self, mol1, mol2, comp_features):
        emb1 = self.encoder(mol1.x, mol1.edge_index, mol1.batch)
        emb2 = self.encoder(mol2.x, mol2.edge_index, mol2.batch)

        comp = F.relu(self.comp_fc(comp_features))

        combined = torch.cat([emb1, emb2, comp], dim=1)

        x = F.relu(self.fc1(combined))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return torch.sigmoid(x)


def get_weighted_accuracy(y_true, y_pred_probs, threshold=0.5):
    y_true = np.array(y_true)
    y_pred_classes = (np.array(y_pred_probs) > threshold).astype(int)
    weights = np.where(y_true == 1, 0.1140, 0.8860)
    weighted_acc = np.sum(weights * (y_true == y_pred_classes)) / np.sum(weights)
    return weighted_acc


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nИспользуется устройство: {device}")

model = SiameseGNN(input_dim=test_dim).to(device)
print(f"Модель создана, параметров: {sum(p.numel() for p in model.parameters()):,}")

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()
epochs = 15

history = {'train_loss': [], 'val_loss': [], 'val_auc': [], 'val_wa': []}

print("\nНачинаем обучение...")
for epoch in range(epochs):
    model.train()
    train_loss = 0
    for batch in train_loader:
        mol1 = batch['mol1'].to(device)
        mol2 = batch['mol2'].to(device)
        comp_features = batch['comp_features'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        out = model(mol1, mol2, comp_features)
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
            comp_features = batch['comp_features'].to(device)
            labels = batch['label'].to(device)

            out = model(mol1, mol2, comp_features)
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

    print(f"Epoch {epoch + 1}/{epochs} | "
          f"Train Loss: {avg_train_loss:.4f} | "
          f"Val Loss: {avg_val_loss:.4f} | "
          f"Val ROC-AUC: {val_auc:.4f} | "
          f"Val WA: {val_wa:.4f}")


def predict_cocrystal(smiles1, smiles2, model, device):
    model.eval()

    g1 = smiles_to_graph(smiles1)
    g2 = smiles_to_graph(smiles2)

    if g1 is None or g2 is None:
        return 0.0

    g1.batch = torch.zeros(g1.num_nodes, dtype=torch.long)
    g2.batch = torch.zeros(g2.num_nodes, dtype=torch.long)

    # Признаки комплементарности
    feat1 = g1.global_features
    feat2 = g2.global_features

    donors1, acceptors1 = feat1[2].item(), feat1[3].item()
    donors2, acceptors2 = feat2[2].item(), feat2[3].item()

    comp_features = torch.tensor([
        abs(donors1 - acceptors2),
        abs(acceptors1 - donors2),
        abs((donors1 + donors2) - (acceptors1 + acceptors2)),
        feat1[0] / (feat2[0] + 1e-5),
        feat1[1] / (feat2[1] + 1e-5),
    ], dtype=torch.float).unsqueeze(0)

    g1 = g1.to(device)
    g2 = g2.to(device)
    comp_features = comp_features.to(device)

    with torch.no_grad():
        prob = model(g1, g2, comp_features)

    return prob.item()


# Предсказания на тестовых данных
print("\nЗагрузка тестовых данных...")
test_df = pd.read_csv('test.csv')

submission_ids = []
submission_preds = []

model.eval()

for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Предсказание"):
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
print(f"\nСохранено {len(submission_df)} предсказаний в submission.csv")
print("\nПервые 5 строк:")
print(submission_df.head())