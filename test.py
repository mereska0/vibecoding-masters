# Импорт библиотек
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

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
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool, global_add_pool

# Фиксируем random seed для воспроизводимости
def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch_geometric.seed_everything(seed)

seed_everything()

# Определяем устройство
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Используется устройство: {device}")

# Загружаем данные
print("Загрузка данных...")
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

print(f"Размер train датасета: {train_df.shape}")
print(f"Размер test датасета: {test_df.shape}")
print("\nПример данных из train.csv:")
print(train_df.head())

# Проверка распределения классов
print("\nРаспределение классов в train:")
print(train_df['result'].value_counts())
print(f"Доля положительных классов: {train_df['result'].mean():.3f}")

# Константы для кодирования
PERMITTED_ATOM_TYPES = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P', 'Unknown']

def one_hot_encoding(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [1 if x == s else 0 for s in allowable_set]

def get_molecule_features(mol):
    """
    Собирает глобальные признаки для всей молекулы (ровно 9 признаков).
    """
    results = []
    
    # 1. Молярная рефракция
    results.append(Descriptors.MolMR(mol))
    
    # 2. Площадь поверхности по Лабуту
    results.append(Descriptors.LabuteASA(mol))
    
    # 3. Количество доноров водородных связей
    results.append(Lipinski.NumHDonors(mol))
    
    # 4. Количество акцепторов водородных связей
    results.append(Lipinski.NumHAcceptors(mol))
    
    # 5. Молекулярная масса
    results.append(Descriptors.MolWt(mol))
    
    # 6. LogP (липофильность)
    results.append(Descriptors.MolLogP(mol))
    
    # 7. Количество вращающихся связей
    results.append(Descriptors.NumRotatableBonds(mol))
    
    # 8. Количество ароматических колец
    results.append(Lipinski.NumAromaticRings(mol))
    
    # 9. Количество гетероатомов
    results.append(Lipinski.NumHeteroatoms(mol))
    
    return torch.tensor(results, dtype=torch.float)

def get_atom_features(atom):
    """
    Собирает признаки атома.
    """
    results = []
    
    # Тип атома (one-hot)
    results.extend(one_hot_encoding(atom.GetSymbol(), PERMITTED_ATOM_TYPES))
    
    # Степень - количество соседей
    results.extend(one_hot_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6]))
    
    # Явный заряд
    results.extend(one_hot_encoding(atom.GetFormalCharge(), [-2, -1, 0, 1, 2]))
    
    # Гибридизация
    hybridization = atom.GetHybridization()
    results.extend([
        int(hybridization == Chem.rdchem.HybridizationType.SP),
        int(hybridization == Chem.rdchem.HybridizationType.SP2),
        int(hybridization == Chem.rdchem.HybridizationType.SP3)
    ])
    
    # Ароматичность
    results.append(int(atom.GetIsAromatic()))
    
    # В кольце?
    results.append(int(atom.IsInRing()))
    
    # Атомная масса (нормализованная)
    results.append(atom.GetMass() / 100.0)
    
    # Число водородов
    results.append(atom.GetTotalNumHs())
    
    return torch.tensor(results, dtype=torch.float)

def get_bond_features(bond):
    """
    Собирает признаки связи (ровно 6 признаков).
    """
    bond_type = bond.GetBondType()
    
    # 4 признака типа связи
    bond_type_features = [
        int(bond_type == Chem.rdchem.BondType.SINGLE),
        int(bond_type == Chem.rdchem.BondType.DOUBLE),
        int(bond_type == Chem.rdchem.BondType.TRIPLE),
        int(bond_type == Chem.rdchem.BondType.AROMATIC)
    ]
    
    # 2 дополнительных признака
    other_features = [
        int(bond.IsInRing()),
        int(bond.GetIsConjugated())
    ]
    
    # Объединяем - всего 6 признаков
    return bond_type_features + other_features

def smiles_to_graph(smiles):
    """
    Преобразует SMILES в граф с признаками атомов, связей и глобальными признаками.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # Признаки атомов (узлы)
    atom_features = [get_atom_features(atom) for atom in mol.GetAtoms()]
    x = torch.stack(atom_features)
    
    # Глобальные признаки молекулы
    mol_features = get_molecule_features(mol)
    
    # Признаки связей (ребра)
    edge_indices = []
    edge_features = []
    
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        
        # Добавляем оба направления для неориентированного графа
        edge_indices.append([i, j])
        edge_indices.append([j, i])
        
        bond_feat = get_bond_features(bond)
        edge_features.append(bond_feat)
        edge_features.append(bond_feat)
    
    if len(edge_indices) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 6), dtype=torch.float)  # 6 признаков связи
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float)
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, global_features=mol_features)

# Проверка размерностей
test_mol = smiles_to_graph('C')
if test_mol is not None:
    node_dim = test_mol.x.shape[1]
    edge_dim = test_mol.edge_attr.shape[1] if test_mol.edge_attr is not None else 0
    global_dim = test_mol.global_features.shape[0]
    
    print(f"\nРазмерности признаков:")
    print(f"  - Признаки атомов: {node_dim}")
    print(f"  - Признаки связей: {edge_dim}")
    print(f"  - Глобальные признаки: {global_dim}")

# Создаем датасет
class CocrystalDataset(Dataset):
    def __init__(self, dataframe):
        super().__init__()
        self.df = dataframe.reset_index(drop=True)
        self.data_list = []
        
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Создание графов"):
            mol1 = smiles_to_graph(row['SMILES1'])
            mol2 = smiles_to_graph(row['SMILES2'])
            
            if mol1 is not None and mol2 is not None:
                if 'result' in row:
                    label = torch.tensor([row['result']], dtype=torch.float)
                else:
                    label = torch.tensor([0], dtype=torch.float)  # заглушка для test
                
                self.data_list.append({
                    'mol1': mol1,
                    'mol2': mol2,
                    'label': label,
                    'id': row.get('id', idx)
                })
    
    def len(self):
        return len(self.data_list)
    
    def get(self, idx):
        return self.data_list[idx]

# Разделяем на train/val
train_split, val_split = train_test_split(train_df, test_size=0.15, random_state=42, stratify=train_df['result'])

train_dataset = CocrystalDataset(train_split)
val_dataset = CocrystalDataset(val_split)
test_dataset = CocrystalDataset(test_df)

print(f"\nРазмеры датасетов:")
print(f"  - Train: {len(train_dataset)}")
print(f"  - Validation: {len(val_dataset)}")
print(f"  - Test: {len(test_dataset)}")

# Создаем загрузчики данных
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Определяем архитектуру модели
class ImprovedGNNEncoder(nn.Module):
    """
    Улучшенный энкодер для графов молекул с поддержкой признаков связей.
    """
    def __init__(self, node_dim, edge_dim, hidden_dim=128, embedding_dim=64):
        super(ImprovedGNNEncoder, self).__init__()
        
        # Начальное преобразование признаков
        self.node_embedding = nn.Linear(node_dim, hidden_dim)
        self.edge_embedding = nn.Linear(edge_dim, hidden_dim)
        
        # GNN слои с вниманием
        self.conv1 = GATConv(hidden_dim, hidden_dim // 4, heads=4, concat=True, edge_dim=hidden_dim)
        self.conv2 = GATConv(hidden_dim, hidden_dim // 4, heads=4, concat=True, edge_dim=hidden_dim)
        self.conv3 = GATConv(hidden_dim, hidden_dim // 4, heads=4, concat=True, edge_dim=hidden_dim)
        
        # BatchNorm для стабилизации
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        
        # Разные типы пулинга
        self.pool = global_mean_pool
        
        # Выходной слой
        self.fc_out = nn.Linear(hidden_dim, embedding_dim)
        
    def forward(self, x, edge_index, edge_attr, batch):
        # Начальное эмбеддинг
        x = self.node_embedding(x)
        edge_attr = self.edge_embedding(edge_attr)
        
        # Первый слой
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.bn1(x)
        x = F.dropout(x, p=0.1, training=self.training)
        
        # Второй слой
        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.bn2(x)
        x = F.dropout(x, p=0.1, training=self.training)
        
        # Третий слой
        x = self.conv3(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.bn3(x)
        
        # Пулинг
        x = self.pool(x, batch)
        
        # Выходной слой
        x = self.fc_out(x)
        
        return x

class CocrystalPredictor(nn.Module):
    """
    Модель для предсказания образования сокристаллов.
    """
    def __init__(self, node_dim, edge_dim, global_dim, hidden_dim=128, embedding_dim=64):
        super(CocrystalPredictor, self).__init__()
        
        self.global_dim = global_dim
        
        # Энкодер для графов
        self.encoder = ImprovedGNNEncoder(node_dim, edge_dim, hidden_dim, embedding_dim)
        
        # Обработка глобальных признаков
        # Входная размерность: global_dim (должно быть 9)
        self.global_fc = nn.Sequential(
            nn.Linear(global_dim, hidden_dim // 2),  # ИСПРАВЛЕНО: global_dim, а не 18!
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, embedding_dim // 2)
        )
        
        # Взаимодействие между молекулами
        combined_dim = embedding_dim * 2 + 18
        
        # Классификатор
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim // 2, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(32, 1)
        )
        
    def forward(self, mol1, mol2):
        # Получаем эмбеддинги графов
        emb1 = self.encoder(mol1.x, mol1.edge_index, mol1.edge_attr, mol1.batch)
        emb2 = self.encoder(mol2.x, mol2.edge_index, mol2.edge_attr, mol2.batch)
        
        # Обрабатываем глобальные признаки
        global1 = mol1.global_features.view(-1, self.global_dim)
        global2 = mol2.global_features.view(-1, self.global_dim)
        
        # Комбинируем глобальные признаки
        # Комбинируем глобальные признаки - используем ТОЛЬКО разницу и сумму
        global_features = torch.cat([
            torch.abs(global1 - global2),  # разница (комплементарность)
            global1 + global2,  # сумма (общая характеристика)
        ], dim=1)

# Теперь размерность: 9 + 9 = 18
        
        global_emb = self.global_fc(global_features)
        
        # Комбинируем все признаки
        combined = torch.cat([emb1, emb2, global_emb], dim=1)
        
        # Классификация
        out = self.classifier(combined)
        
        return torch.sigmoid(out)

# Создаем модель
model = CocrystalPredictor(
    node_dim=node_dim,
    edge_dim=edge_dim,
    global_dim=global_dim,
    hidden_dim=128,
    embedding_dim=64
).to(device)

print(f"\nМодель создана. Количество параметров: {sum(p.numel() for p in model.parameters())}")

# Функция для взвешенной точности
def get_weighted_accuracy(y_true, y_pred_probs, threshold=0.5):
    y_true = np.array(y_true)
    y_pred_classes = (np.array(y_pred_probs) > threshold).astype(int)
    
    # Веса из условия задачи
    weights = np.where(y_true == 1, 0.1140, 0.8860)
    weighted_acc = np.sum(weights * (y_true == y_pred_classes)) / np.sum(weights)
    return weighted_acc

# Обучение модели
def train_model(model, train_loader, val_loader, epochs=30, lr=0.001, weight_decay=1e-5):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    criterion = nn.BCELoss()
    
    history = {'train_loss': [], 'val_loss': [], 'val_auc': [], 'val_wa': [], 'val_acc': []}
    
    best_val_auc = 0
    best_model_state = None
    patience_counter = 0
    early_stop_patience = 10
    
    print("\nНачинаем обучение...")
    
    for epoch in range(epochs):
        # Обучение
        model.train()
        train_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False):
            mol1 = batch['mol1'].to(device)
            mol2 = batch['mol2'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            out = model(mol1, mol2)
            loss = criterion(out, labels)
            loss.backward()
            
            # Градиентный клиппинг
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Валидация
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
        
        # Метрики
        val_auc = roc_auc_score(y_true, y_pred)
        val_wa = get_weighted_accuracy(y_true, y_pred)
        val_acc = accuracy_score(y_true, (np.array(y_pred) > 0.5).astype(int))
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_auc'].append(val_auc)
        history['val_wa'].append(val_wa)
        history['val_acc'].append(val_acc)
        
        # Снижение learning rate
        scheduler.step(avg_val_loss)
        
        # Сохранение лучшей модели
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Печать результатов
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val ROC-AUC: {val_auc:.4f} | "
              f"Val WA: {val_wa:.4f} | "
              f"Val Acc: {val_acc:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Early stopping
        if patience_counter >= early_stop_patience:
            print(f"Early stopping на эпохе {epoch+1}")
            break
    
    # Загружаем лучшую модель
    model.load_state_dict(best_model_state)
    
    return history, best_model_state

# Обучаем модель
history, best_model_state = train_model(
    model, train_loader, val_loader, 
    epochs=30, lr=0.001, weight_decay=1e-5
)

# Визуализация обучения
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.title('Loss History')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(history['val_auc'], label='Val ROC-AUC', color='orange')
plt.title('Validation ROC-AUC')
plt.xlabel('Epoch')
plt.ylabel('ROC-AUC')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(history['val_wa'], label='Val Weighted Acc', color='green')
plt.title('Validation Weighted Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Weighted Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_history.png')
plt.show()

# Функция для предсказания
def predict(model, loader, device):
    model.eval()
    predictions = []
    ids = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Предсказание"):
            mol1 = batch['mol1'].to(device)
            mol2 = batch['mol2'].to(device)
            
            out = model(mol1, mol2)
            
            predictions.extend(out.cpu().numpy())
            if 'id' in batch:
                ids.extend(batch['id'])
    
    return np.array(predictions).flatten(), ids

# Предсказания на тестовой выборке
print("\nДелаем предсказания для test.csv...")
test_predictions, test_ids = predict(model, test_loader, device)

# Создаем submission файл
if len(test_ids) > 0:
    submission_df = pd.DataFrame({
        'id': test_ids,
        'result': (test_predictions > 0.5).astype(int)
    })
else:
    # Если нет id в данных, используем индексы
    submission_df = pd.DataFrame({
        'id': range(len(test_predictions)),
        'result': (test_predictions > 0.5).astype(int)
    })

# Сохраняем
submission_df.to_csv('submission.csv', index=False)
print(f"\nСохранено {len(submission_df)} предсказаний в submission.csv")
print("\nПервые 10 предсказаний:")
print(submission_df.head(10))

# Дополнительно сохраняем вероятности для анализа
proba_df = pd.DataFrame({
    'id': submission_df['id'],
    'probability': test_predictions,
    'prediction': submission_df['result']
})
proba_df.to_csv('predictions_with_proba.csv', index=False)
print("\nВероятности сохранены в predictions_with_proba.csv")

# Анализ распределения предсказаний
plt.figure(figsize=(10, 5))
plt.hist(test_predictions, bins=50, alpha=0.7, color='blue', edgecolor='black')
plt.axvline(x=0.5, color='red', linestyle='--', label='Порог (0.5)')
plt.xlabel('Вероятность')
plt.ylabel('Частота')
plt.title('Распределение предсказанных вероятностей на тестовой выборке')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('prediction_distribution.png')
plt.show()

print("\nСтатистика предсказаний:")
print(f"  - Средняя вероятность: {test_predictions.mean():.4f}")
print(f"  - Медианная вероятность: {np.median(test_predictions):.4f}")
print(f"  - Доля положительных классов: {(test_predictions > 0.5).mean():.4f}")
print(f"  - Минимальная вероятность: {test_predictions.min():.4f}")
print(f"  - Максимальная вероятность: {test_predictions.max():.4f}")