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


# Всегда лучше фиксировать random seed для воспроизводимости результатов
def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch_geometric.seed_everything(seed)

seed_everything()

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

    # Молярная рефракция - коррелирует с объемом молекулы
    results += [Descriptors.MolMR(mol)]

    # Площадь поверхности по Лабуту
    results += [Descriptors.LabuteASA(mol)]

    # Количество доноров водородных связей
    results += [Lipinski.NumHDonors(mol)]

    # Количество акцепторов водородных связей
    results += [Lipinski.NumHAcceptors(mol)]

    """ 
    1) Комплементарность доноров: |Donors_A - Acceptors_B| — насколько доноры молекулы A соответствуют акцепторам молекулы B
    2) Комплементарность акцепторов: |Acceptors_A - Donors_B|
    3) Общая комплементарность: |(Donors_A + Donors_B) - (Acceptors_A + Acceptors_B)| (чем меньше, тем лучше сбалансирована система)
    4) ΔpKa — разница в кислотности/основности (если вы можете ее предсказать)
    """

    return torch.tensor(results, dtype=torch.float)


def get_atom_features(atom):
    """
    Собирает признаки атома.
    Мы не только рассматриваем структуру поатомно, но и учитываем химическое окружение.
    Вы можете улучшать эту часть, извлекая больше информации об атомах и связях между ними
    """
    # Тип атома
    # Элемент Unknown нужен для кодирования атомов, которых нет в списке PERMITTED_ATOM_TYPES
    results = one_hot_encoding(atom.GetSymbol(), PERMITTED_ATOM_TYPES)

    # Степень - количество соседей
    results += one_hot_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 'More'])

    # Явный заряд - важно для сокристаллов (соли vs кокристаллы)
    results += one_hot_encoding(atom.GetFormalCharge(), [-1, -2, 1, 2, 0])  # 0 в конце как дефолт

    # Гибридизация (плоская vs объемная молекула)
    results += one_hot_encoding(atom.GetHybridization(),
                                [Chem.rdchem.HybridizationType.SP,
                                 Chem.rdchem.HybridizationType.SP2,
                                 Chem.rdchem.HybridizationType.SP3])

    # Ароматичность - булевый флаг, важно для кристаллов
    results += [int(atom.GetIsAromatic())]
    # Точка роста:
    # Можно добавить сюда и другие характеристики для обогащения информации, на
    # которой вы учите сеть: IsInRing, Mass, Chirality, HydrogenBondDonor/Acceptor и т.д.

    return torch.tensor(results, dtype=torch.float)


# определяем виды связей
def get_bond_features(bond):
    result = []

    bond_type = bond.GetBondType()
    result += [
        int(bond_type == Chem.rdchem.BondType.SINGLE),
        int(bond_type == Chem.rdchem.BondType.DOUBLE),
        int(bond_type == Chem.rdchem.BondType.TRIPLE),
        int(bond_type == Chem.rdchem.BondType.AROMATIC)
    ]
    # циклы?
    result += [int(bond.IsInRing())]
    # сопряженные?
    result += [int(bond.GetIsConjugated())]

    return result


def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Узлы
    atom_features = [get_atom_features(atom) for atom in mol.GetAtoms()]
    x = torch.stack(atom_features)

    mol_features = get_molecule_features(mol)

    # Ребра
    edge_indices = []
    edge_features = []

    # Точка роста: здесь мы учитываем только наличие связи, не ее тип (одинарная, двойная и т.д.)
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        edge_indices.append([i, j])
        edge_indices.append([j, i])

        bond_feat = get_bond_features(bond)
        # одинаковые в оба напрявления поэтому дублируем
        edge_features.append(bond_feat)
        edge_features.append(bond_feat)

    # Формируем граф
    if len(edge_indices) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, len(edge_features[0]) if edge_features else 0), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, global_features=mol_features)


# Проверка размерности признаков - это важно учесть при проектировании архитектуры модели
test_dim = smiles_to_graph('C').x.shape[1]
print(f"Размерность вектора признаков одного атома: {test_dim}")

# Проверим работу функции на одной молекуле
test_smiles = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C" # Кофеин
graph = smiles_to_graph(test_smiles)

print("\n--- Пример графа (Кофеин) ---")
print(f"SMILES: {test_smiles}")
print(f"Количество атомов (nodes): {graph.num_nodes}")
print(f"Количество связей (edges): {graph.num_edges // 2}") # Делим на 2 так как связи двунаправленные
print(f"Размерность признаков узла (node features): {graph.num_node_features}")
print(f"Размерность признаков связи (edge features): {graph.edge_attr.shape[1] if graph.edge_attr is not None else 0}")
print(f"Размерность глобальных признаков молекулы (global features): {graph.global_features.shape[0]}")
print("\nСтруктура объекта Data:")
print(graph)

# Дополнительная информация для понимания
print("\n--- Детали признаков ---")
print(f"Форма тензора признаков узлов x: {graph.x.shape}")
print(f"Форма тензора индексов связей edge_index: {graph.edge_index.shape}")
print(f"Форма тензора признаков связей edge_attr: {graph.edge_attr.shape}")
print(f"Форма тензора глобальных признаков global_features: {graph.global_features.shape}")

# Пример первых нескольких признаков
print("\n--- Пример первых 5 признаков первого атома ---")
print(graph.x[0][:5])

print("\n--- Пример первых 5 признаков первой связи ---")
if graph.edge_attr is not None:
    print(graph.edge_attr[0][:5])

print("\n--- Все глобальные признаки молекулы ---")
print(f"MR Volume: {graph.global_features[0].item():.2f}")
print(f"Labute ASA: {graph.global_features[1].item():.2f}")
print(f"Доноров H-связей: {int(graph.global_features[2].item())}")
print(f"Акцепторов H-связей: {int(graph.global_features[3].item())}")


