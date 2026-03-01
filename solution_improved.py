# -*- coding: utf-8 -*-
"""
УЛУЧШЕННОЕ РЕШЕНИЕ — RASCAR Cocrystal Challenge
================================================
Запускается одним файлом. Установка зависимостей — в первом блоке.
"""

# ╔══════════════════════════════════════════════════════════════════╗
# ║  БЛОК 0: УСТАНОВКА ЗАВИСИМОСТЕЙ                                  ║
# ║  Запустите эту ячейку, затем: Runtime → Restart & Run All        ║
# ╚══════════════════════════════════════════════════════════════════╝
import subprocess, sys

def _install(cmd):
    subprocess.check_call(cmd, shell=True,
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# torch_geometric
try:
    import torch_geometric
    print(f"torch_geometric {torch_geometric.__version__} уже установлен.")
except ImportError:
    print("Устанавливаем torch_geometric...")
    import torch
    tv   = torch.__version__
    base = tv.split("+")[0]
    cuda = tv.split("+")[1] if "+" in tv else "cpu"

    _install("pip install -q torch_geometric")

    whl = f"https://data.pyg.org/whl/torch-{base}+{cuda}.html"
    _install(f"pip install -q torch_scatter torch_sparse torch_cluster "
             f"torch_spline_conv -f {whl}")
    print("torch_geometric установлен. ПЕРЕЗАПУСТИТЕ RUNTIME и запустите снова.")
    sys.exit(0)   # выходим — нужен перезапуск ядра

# rdkit
try:
    from rdkit import Chem
except ImportError:
    print("Устанавливаем rdkit...")
    _install("pip install -q rdkit")
    from rdkit import Chem

# ╔══════════════════════════════════════════════════════════════════╗
# ║  БЛОК 1: ИМПОРТЫ                                                 ║
# ╚══════════════════════════════════════════════════════════════════╝
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_mean_pool, global_max_pool
from torch_geometric.nn.models import MLP

from rdkit.Chem import Descriptors, Lipinski, Crippen, rdMolDescriptors, AllChem
from rdkit import RDLogger

# Отключаем предупреждения RDKit (включая сообщение о депрецированной
# функции Morgan fingerprint) — безопасно для прогонов, но сохраняйте
# логирование в отладочных сценариях при необходимости.
RDLogger.DisableLog('rdApp.*')

# ╔══════════════════════════════════════════════════════════════════╗
# ║  БЛОК 2: НАСТРОЙКИ                                               ║
# ╚══════════════════════════════════════════════════════════════════╝
SEED          = 42
BATCH_SIZE    = 64
HIDDEN_DIM    = 128
EMBEDDING_DIM = 64
FP_DIM        = 128     # Morgan ECFP4 bits
EPOCHS        = 60
LR            = 3e-4
WEIGHT_DECAY  = 1e-4
PATIENCE      = 10
DEVICE        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def seed_everything(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch_geometric.seed_everything(seed)

seed_everything(SEED)
print(f"Device: {DEVICE}")

# ╔══════════════════════════════════════════════════════════════════╗
# ║  БЛОК 3: ДАННЫЕ                                                  ║
# ╚══════════════════════════════════════════════════════════════════╝
# ── Укажите пути к файлам ──────────────────────────────────────────
TRAIN_PATH = "train.csv"   # или '/kaggle/input/.../train.csv'
TEST_PATH  = "test.csv"
# ──────────────────────────────────────────────────────────────────

train_df = pd.read_csv(TRAIN_PATH)
test_df  = pd.read_csv(TEST_PATH)
print(f"Train: {train_df.shape}  |  Test: {test_df.shape}")
print(f"Баланс классов:\n{train_df['result'].value_counts()}")

# pos_weight для BCEWithLogitsLoss
n_neg = (train_df['result'] == 0).sum()
n_pos = (train_df['result'] == 1).sum()
POS_WEIGHT = torch.tensor([n_neg / n_pos], dtype=torch.float).to(DEVICE)
print(f"\npos_weight = {POS_WEIGHT.item():.2f}  (компенсируем дисбаланс {n_neg}:{n_pos})")

# ╔══════════════════════════════════════════════════════════════════╗
# ║  БЛОК 4: ПРИЗНАКИ                                                ║
# ╚══════════════════════════════════════════════════════════════════╝
PERMITTED_ATOMS = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P', 'B', 'Si', 'Unknown']

def one_hot(x, lst):
    if x not in lst:
        x = lst[-1]
    return [float(x == v) for v in lst]

def get_atom_features(atom):
    """38-мерный вектор признаков атома."""
    f  = one_hot(atom.GetSymbol(), PERMITTED_ATOMS)                          # 12
    f += one_hot(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 'More'])              # 7
    f += one_hot(atom.GetFormalCharge(), [-2, -1, 0, 1, 2])                 # 5
    f += one_hot(atom.GetHybridization(),
                 [Chem.rdchem.HybridizationType.SP,
                  Chem.rdchem.HybridizationType.SP2,
                  Chem.rdchem.HybridizationType.SP3,
                  Chem.rdchem.HybridizationType.SP3D])                       # 4
    f.append(float(atom.GetIsAromatic()))                                    # 1
    f.append(float(atom.IsInRing()))                                         # 1
    f += one_hot(atom.GetTotalNumHs(), [0, 1, 2, 3, 'More'])                # 5
    f += one_hot(str(atom.GetChiralTag()),
                 ['CHI_UNSPECIFIED',
                  'CHI_TETRAHEDRAL_CW',
                  'CHI_TETRAHEDRAL_CCW'])                                    # 3
    return torch.tensor(f, dtype=torch.float)                               # итого: 38

def get_bond_features(bond):
    """6-мерный вектор признаков связи."""
    bt = bond.GetBondType()
    return torch.tensor([
        float(bt == Chem.rdchem.BondType.SINGLE),
        float(bt == Chem.rdchem.BondType.DOUBLE),
        float(bt == Chem.rdchem.BondType.TRIPLE),
        float(bt == Chem.rdchem.BondType.AROMATIC),
        float(bond.IsInRing()),
        float(bond.GetIsConjugated()),
    ], dtype=torch.float)

def get_global_features(mol):
    """10 нормализованных физико-химических дескрипторов."""
    return torch.tensor([
        Descriptors.MolWt(mol)                              / 500.0,
        Crippen.MolLogP(mol)                                / 10.0,
        Descriptors.TPSA(mol)                               / 150.0,
        float(Lipinski.NumHDonors(mol))                     / 5.0,
        float(Lipinski.NumHAcceptors(mol))                  / 10.0,
        float(rdMolDescriptors.CalcNumRings(mol))           / 5.0,
        float(rdMolDescriptors.CalcNumAromaticRings(mol))   / 4.0,
        float(rdMolDescriptors.CalcNumRotatableBonds(mol))  / 10.0,
        Descriptors.MolMR(mol)                              / 150.0,
        Descriptors.LabuteASA(mol)                          / 200.0,
    ], dtype=torch.float)

def get_morgan_fp(mol, n_bits=FP_DIM, radius=2):
    """ECFP4 Morgan fingerprint.

    Uses the new `MorganGenerator` API when available (avoids RDKit deprecation
    warnings) and falls back to `GetMorganFingerprintAsBitVect` for older RDKit
    versions.
    """
    # Prefer the newer MorganGenerator when present
    try:
        gen_cls = rdMolDescriptors.MorganGenerator
    except Exception:
        gen_cls = None

    if gen_cls is not None:
        gen = rdMolDescriptors.MorganGenerator(radius=radius)
        bitvect = gen.GetFingerprintAsBitVect(mol, nBits=n_bits)
        # Try to get a bit string first, else fall back to iterable
        try:
            bitstr = bitvect.ToBitString()
            arr = [int(b) for b in bitstr]
        except Exception:
            try:
                arr = list(bitvect)
            except Exception:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
                arr = list(fp)
        return torch.tensor(arr, dtype=torch.float)
    else:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
        return torch.tensor(list(fp), dtype=torch.float)

def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    x = torch.stack([get_atom_features(a) for a in mol.GetAtoms()])

    edge_idx, edge_atr = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf = get_bond_features(bond)
        edge_idx += [[i, j], [j, i]]
        edge_atr += [bf, bf]

    if not edge_idx:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr  = torch.empty((0, 6),  dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_idx, dtype=torch.long).t().contiguous()
        edge_attr  = torch.stack(edge_atr)

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        global_features=get_global_features(mol),
        morgan_fp=get_morgan_fp(mol),
    )

# Размерности — нужны при инициализации модели
_s = smiles_to_graph('CCO')
NODE_DIM   = _s.x.shape[1]
EDGE_DIM   = _s.edge_attr.shape[1]
GLOBAL_DIM = _s.global_features.shape[0]
print(f"\nnode_dim={NODE_DIM}, edge_dim={EDGE_DIM}, "
      f"global_dim={GLOBAL_DIM}, fp_dim={FP_DIM}")

# ╔══════════════════════════════════════════════════════════════════╗
# ║  БЛОК 5: ДАТАСЕТ                                                 ║
# ╚══════════════════════════════════════════════════════════════════╝
class CocrystalDataset(Dataset):
    def __init__(self, dataframe):
        super().__init__()
        self.data_list = []
        for _, row in tqdm(dataframe.reset_index(drop=True).iterrows(),
                           total=len(dataframe), desc="Building graphs"):
            mol1 = smiles_to_graph(row['SMILES1'])
            mol2 = smiles_to_graph(row['SMILES2'])
            if mol1 is None or mol2 is None:
                continue
            self.data_list.append({
                'mol1':  mol1,
                'mol2':  mol2,
                'label': torch.tensor([row['result']], dtype=torch.float),
            })

    def len(self):        return len(self.data_list)
    def get(self, idx):   return self.data_list[idx]

# Стратифицированный split — сохраняет баланс классов в val
train_split, val_split = train_test_split(
    train_df, test_size=0.2, random_state=SEED, stratify=train_df['result']
)
train_dataset = CocrystalDataset(train_split)
val_dataset   = CocrystalDataset(val_split)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)

# ╔══════════════════════════════════════════════════════════════════╗
# ║  БЛОК 6: МОДЕЛЬ                                                  ║
# ╚══════════════════════════════════════════════════════════════════╝
class GINEEncoder(nn.Module):
    """GINEConv × 3 + BatchNorm + комбинированный пулинг (mean || max)."""
    def __init__(self, node_dim, edge_dim, hidden_dim, out_dim):
        super().__init__()
        self.bn0   = nn.BatchNorm1d(node_dim)
        self.conv1 = GINEConv(MLP([node_dim,  hidden_dim, hidden_dim]), edge_dim=edge_dim)
        self.bn1   = nn.BatchNorm1d(hidden_dim)
        self.conv2 = GINEConv(MLP([hidden_dim, hidden_dim, hidden_dim]), edge_dim=edge_dim)
        self.bn2   = nn.BatchNorm1d(hidden_dim)
        self.conv3 = GINEConv(MLP([hidden_dim, hidden_dim, out_dim]),    edge_dim=edge_dim)
        self.bn3   = nn.BatchNorm1d(out_dim)

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.bn0(x)
        x = self.bn1(self.conv1(x, edge_index, edge_attr)).relu()
        x = self.bn2(self.conv2(x, edge_index, edge_attr)).relu()
        x = self.bn3(self.conv3(x, edge_index, edge_attr))
        return torch.cat([global_mean_pool(x, batch),
                          global_max_pool(x, batch)], dim=1)   # -> (B, out_dim*2)


class SiameseGNN(nn.Module):
    """
    Взаимодействие молекул:
        [emb1, emb2, |emb1-emb2|, emb1*emb2, glob1, glob2, fp1, fp2]
    """
    def __init__(self, node_dim, edge_dim, global_dim, fp_dim,
                 hidden_dim=HIDDEN_DIM, emb_dim=EMBEDDING_DIM):
        super().__init__()
        self.encoder = GINEEncoder(node_dim, edge_dim, hidden_dim, emb_dim)
        graph_out = emb_dim * 2                             # mean + max
        in_dim    = graph_out * 4 + global_dim * 2 + fp_dim * 2

        self.classifier = nn.Sequential(
            nn.Linear(in_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),                       # logit (без sigmoid)
        )

    def forward(self, mol1, mol2):
        emb1 = self.encoder(mol1.x, mol1.edge_index, mol1.edge_attr, mol1.batch)
        emb2 = self.encoder(mol2.x, mol2.edge_index, mol2.edge_attr, mol2.batch)
        # Ensure global and fingerprint features have a batch dimension
        gf1 = mol1.global_features
        gf2 = mol2.global_features
        fp1 = mol1.morgan_fp
        fp2 = mol2.morgan_fp

        if gf1.dim() == 1:
            gf1 = gf1.unsqueeze(0)
        if gf2.dim() == 1:
            gf2 = gf2.unsqueeze(0)
        if fp1.dim() == 1:
            fp1 = fp1.unsqueeze(0)
        if fp2.dim() == 1:
            fp2 = fp2.unsqueeze(0)

        combined = torch.cat([
            emb1, emb2,
            (emb1 - emb2).abs(),    # разность -> похожесть
            emb1 * emb2,            # произведное -> совместимость
            gf1, gf2,
            fp1, fp2,
        ], dim=1)
        return self.classifier(combined)


model = SiameseGNN(NODE_DIM, EDGE_DIM, GLOBAL_DIM, FP_DIM).to(DEVICE)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nМодель инициализирована. Параметров: {n_params:,}")
print(model)

# ╔══════════════════════════════════════════════════════════════════╗
# ║  БЛОК 7: МЕТРИКА (ПРАВИЛЬНАЯ)                                    ║
# ╚══════════════════════════════════════════════════════════════════╝
def weighted_accuracy(y_true, y_pred_probs, threshold=0.5):
    """
    Веса из условия задачи: w=0.3 для result=1, w=0.7 для result=0.
    НЕ путать с частотами классов в датасете!
    """
    y_true  = np.array(y_true).flatten()
    y_pred  = (np.array(y_pred_probs).flatten() > threshold).astype(int)
    weights = np.where(y_true == 1, 0.3, 0.7)
    return float(np.sum(weights * (y_true == y_pred)) / np.sum(weights))

def find_best_threshold(y_true, y_pred_probs):
    """Перебираем пороги [0.2, 0.8] и выбираем лучший по WA."""
    best_thr, best_wa = 0.5, 0.0
    for thr in np.arange(0.20, 0.81, 0.01):
        wa = weighted_accuracy(y_true, y_pred_probs, thr)
        if wa > best_wa:
            best_wa, best_thr = wa, float(thr)
    return round(best_thr, 2), best_wa

# ╔══════════════════════════════════════════════════════════════════╗
# ║  БЛОК 8: ОБУЧЕНИЕ                                                ║
# ╚══════════════════════════════════════════════════════════════════╝
criterion = nn.BCEWithLogitsLoss(pos_weight=POS_WEIGHT)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=5
)

best_val_wa   = 0.0
# Инициализируем best_weights копией текущих весов — чтобы не получить
# ошибку, если улучшений не случится до ранней остановки.
best_weights  = {k: v.cpu().clone() for k, v in model.state_dict().items()}
no_improve    = 0
history       = {'train_loss': [], 'val_loss': [], 'val_auc': [], 'val_wa': []}

for epoch in range(1, EPOCHS + 1):
    # Train
    model.train()
    train_loss = 0.0
    for batch in train_loader:
        mol1   = batch['mol1'].to(DEVICE)
        mol2   = batch['mol2'].to(DEVICE)
        labels = batch['label'].to(DEVICE)

        optimizer.zero_grad()
        logits = model(mol1, mol2)
        loss   = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_loss += loss.item()

    avg_train = train_loss / len(train_loader)

    # Val
    model.eval()
    val_loss, y_true, y_pred = 0.0, [], []
    with torch.no_grad():
        for batch in val_loader:
            mol1   = batch['mol1'].to(DEVICE)
            mol2   = batch['mol2'].to(DEVICE)
            labels = batch['label'].to(DEVICE)

            logits = model(mol1, mol2)
            val_loss += criterion(logits, labels).item()

            probs = torch.sigmoid(logits).cpu().numpy()
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(probs)

    avg_val  = val_loss / len(val_loader)
    try:
        val_auc  = roc_auc_score(y_true, y_pred)
    except Exception:
        # Если в валидационной выборке все метки в одной группе,
        # roc_auc_score выбросит ошибку — просто ставим NaN.
        val_auc = float('nan')
    val_wa   = weighted_accuracy(y_true, y_pred)

    scheduler.step(val_wa)

    history['train_loss'].append(avg_train)
    history['val_loss'].append(avg_val)
    history['val_auc'].append(val_auc)
    history['val_wa'].append(val_wa)

    star = " *" if val_wa > best_val_wa else ""
    print(f"Epoch {epoch:3d}/{EPOCHS} | "
          f"Loss {avg_train:.4f}/{avg_val:.4f} | "
          f"AUC {val_auc:.4f} | WA {val_wa:.4f}{star}")

    if val_wa > best_val_wa:
        best_val_wa  = val_wa
        best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        no_improve   = 0
    else:
        no_improve += 1
        if no_improve >= PATIENCE:
            print(f"\nРанняя остановка на эпохе {epoch}. Best WA = {best_val_wa:.4f}")
            break

# Лучшие веса
model.load_state_dict(best_weights)
print(f"\nЛучшая модель загружена. Val WA = {best_val_wa:.4f}")

# Оптимизация порога
model.eval()
y_t, y_p = [], []
with torch.no_grad():
    for batch in val_loader:
        logits = model(batch['mol1'].to(DEVICE), batch['mol2'].to(DEVICE))
        probs  = torch.sigmoid(logits).cpu().numpy()
        # Гарантируем, что метки находятся на CPU перед преобразованием в numpy
        y_t.extend(batch['label'].cpu().numpy())
        y_p.extend(probs)

best_thr, wa_at_thr = find_best_threshold(y_t, y_p)
print(f"Оптимальный порог: {best_thr}  ->  WA = {wa_at_thr:.4f}")

# ╔══════════════════════════════════════════════════════════════════╗
# ║  БЛОК 9: ВИЗУАЛИЗАЦИЯ                                            ║
# ╚══════════════════════════════════════════════════════════════════╝
plt.figure(figsize=(18, 5))

plt.subplot(1, 3, 1)
plt.plot(history['train_loss'], label='Train')
plt.plot(history['val_loss'],   label='Val')
plt.title('Loss'); plt.xlabel('Epoch'); plt.legend()

plt.subplot(1, 3, 2)
plt.plot(history['val_auc'], color='orange', label='Val AUC')
plt.title('ROC-AUC'); plt.xlabel('Epoch'); plt.legend()

plt.subplot(1, 3, 3)
plt.plot(history['val_wa'], color='green', label='Val WA')
plt.axhline(best_val_wa, ls='--', color='red', label=f'Best={best_val_wa:.4f}')
plt.title('Weighted Accuracy (w=0.3/0.7)'); plt.xlabel('Epoch'); plt.legend()

plt.tight_layout()
plt.savefig('training_history.png', dpi=120)
plt.show()

# ╔══════════════════════════════════════════════════════════════════╗
# ║  БЛОК 10: ИНФЕРЕНС И САБМИТ                                      ║
# ╚══════════════════════════════════════════════════════════════════╝
def predict_pair(smiles1, smiles2, model, device, threshold=0.5):
    g1 = smiles_to_graph(smiles1)
    g2 = smiles_to_graph(smiles2)
    if g1 is None or g2 is None:
        return 0.0, 0

    # Устанавливаем индексы батча для одиночных графов и перенесём на device
    g1.batch = torch.zeros(g1.num_nodes, dtype=torch.long)
    g2.batch = torch.zeros(g2.num_nodes, dtype=torch.long)
    g1, g2 = g1.to(device), g2.to(device)

    model.eval()
    with torch.no_grad():
        prob = torch.sigmoid(model(g1, g2)).item()
    return prob, int(prob > threshold)

ids, preds = [], []
for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Predicting"):
    _, pred = predict_pair(row['SMILES1'], row['SMILES2'],
                           model, DEVICE, threshold=best_thr)
    ids.append(row['id'])
    preds.append(pred)

submission = pd.DataFrame({'id': ids, 'result': preds})
submission.to_csv('submission.csv', index=False)
print(f"\nsubmission.csv сохранён")
print(submission['result'].value_counts())
print(submission.head(10))
