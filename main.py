from rdkit import Chem
from rdkit.Chem import Draw
from IPython.display import display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# import torch
import rdkit

from rdkit.Chem import Descriptors, Lipinski, Crippen, rdMolDescriptors, QED
from rdkit.Chem import AllChem
from rdkit.Chem import rdFreeSASA  # для площади поверхности


# ======================== 1. Признаки водородных связей ========================
def get_hbond_features(mol):
    """
    Количество доноров и акцепторов водородной связи.
    Использует стандартные дескрипторы RDKit.
    """
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    return hbd, hba

def get_gasteiger_charges(mol):
    """
    Вычисляет заряды Гайгера (Gasteiger) для всех атомов.
    Возвращает средний, минимальный, максимальный заряд и заряд на определённых типах атомов.
    """
    try:
        AllChem.ComputeGasteigerCharges(mol)
        charges = [float(atom.GetProp('_GasteigerCharge')) for atom in mol.GetAtoms()]
        avg_charge = np.mean(charges)
        min_charge = np.min(charges)
        max_charge = np.max(charges)
        # Дополнительно: заряд на атомах кислорода и азота (важно для H-связей)
        o_charges = []
        n_charges = []
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 8:  # кислород
                o_charges.append(float(atom.GetProp('_GasteigerCharge')))
            elif atom.GetAtomicNum() == 7:  # азот
                n_charges.append(float(atom.GetProp('_GasteigerCharge')))
        avg_o_charge = np.mean(o_charges) if o_charges else 0.0
        avg_n_charge = np.mean(n_charges) if n_charges else 0.0
        return avg_charge, min_charge, max_charge, avg_o_charge, avg_n_charge
    except:
        return 0.0, 0.0, 0.0, 0.0, 0.0

# ======================== 2. Геометрические признаки ========================
def get_geometric_features(mol, conf_id=-1):
    """
    Молекулярный объём, площадь поверхности, наличие ароматических колец.
    Для объёма и площади требуется 3D конформация. Если её нет – возвращаем приближённые значения.
    """
    # Количество ароматических колец
    aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
    # Количество колец всего
    rings = rdMolDescriptors.CalcNumRings(mol)

    # Попытаемся сгенерировать 3D конформацию для более точных расчётов
    try:
        mol3d = Chem.AddHs(mol)  # добавим водороды для 3D
        AllChem.EmbedMolecule(mol3d, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol3d)

        # Объём (ван-дер-ваальсов) через Labute ASA
        mol_volume = rdMolDescriptors.CalcExactMolVol(mol3d)  # или CalcMolVolume?
        # Площадь поверхности (SASA)
        sasa = rdFreeSASA.CalcSASA(mol3d)
        # Labute ASA (ещё одна мера полярной поверхности)
        labute_asa = rdMolDescriptors.CalcLabuteASA(mol3d)
    except:
        # Если не получилось, используем приближения из дескрипторов
        mol_volume = Descriptors.MolMR(mol)  # молярная рефракция как замена объёму
        sasa = 0.0
        labute_asa = 0.0

    return aromatic_rings, rings, mol_volume, sasa, labute_asa

# ======================== 3. Гидрофобность и липофильность ========================
def get_lipophilicity_features(mol):
    """LogP, поляризуемость."""
    logp = Crippen.MolLogP(mol)
    mr = Descriptors.MolMR(mol)  # молярная рефракция (связана с поляризуемостью)
    return logp, mr

# ======================== 4. Признаки функциональных групп ========================
def get_functional_group_flags(mol):
    """
    Определяет наличие ключевых функциональных групп с помощью SMARTS.
    Возвращает бинарные флаги.
    """
    patterns = {
        'carboxyl': '[CX3](=O)[OX2H1]',           # карбоксильная группа -COOH
        'amide': '[CX3](=O)[NX3H2,H1]',            # первичный/вторичный амид
        'alcohol': '[OX2H]',                        # спирт/фенол -OH
        'phenol': '[cX3][OH]',                      # фенольный гидроксил
        'pyridine_N': 'n1ccccc1',                   # пиридиновый азот (ароматический)
        'imidazole_N': 'n1cncc1',                    # азот имидазола
        'primary_amine': '[NX3;H2]',                 # первичный амин
        'secondary_amine': '[NX3;H1]',               # вторичный амин
        'sulfonyl': '[SX4](=O)(=O)',                  # сульфонил
        'nitro': '[NX3](=O)=O',                       # нитрогруппа
    }
    flags = {}
    for name, smarts in patterns.items():
        patt = Chem.MolFromSmarts(smarts)
        flags[name] = 1 if mol.HasSubstructMatch(patt) else 0
    return flags

# ======================== 5. Топологические и структурные признаки ========================
def get_topological_features(mol):
    """Количество вращающихся связей, индекс Виннера, радиус и диаметр."""
    rot_bonds = Descriptors.NumRotatableBonds(mol)
    wiener = rdMolDescriptors.CalcWienerIndex(mol)
    # радиус и диаметр по Баланбану (можно через CalcBalabanIndex)
    balaban = rdMolDescriptors.CalcBalabanIJ(mol)  # индекс Баланбана
    return rot_bonds, wiener, balaban

# ======================== 6. Энергетические признаки (упрощённые) ========================
def get_energetic_features(mol):
    """
    Дипольный момент (требует 3D), HOMO/LUMO (не доступны в RDKit).
    Здесь мы используем приближения: электроотрицательность, электрофильный индекс.
    Для реальных HOMO/LUMO можно использовать xTB или другие внешние инструменты.
    """
    # Дипольный момент (если есть 3D)
    dipole = 0.0
    try:
        mol3d = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol3d, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol3d)
        dipole = rdMolDescriptors.CalcExactMolDipole(mol3d)  # возвращает вектор, берём норму
        if dipole:
            dipole = dipole.Length()
    except:
        pass

    # Максимальный частичный заряд (как грубая оценка акцепторности)
    try:
        AllChem.ComputeGasteigerCharges(mol)
        charges = [float(atom.GetProp('_GasteigerCharge')) for atom in mol.GetAttoms()]
        max_charge = max(charges)
        min_charge = min(charges)
    except:
        max_charge, min_charge = 0.0, 0.0

    return dipole, max_charge, min_charge

# ======================== 7. Молекулярные отпечатки (fingerprints) ========================
def get_fingerprints(mol, fp_type='maccs'):
    """
    Возвращает битовый вектор отпечатка.
    Поддерживаемые типы: 'maccs', 'morgan2', 'morgan3', 'rdkit'.
    """
    if fp_type == 'maccs':
        fp = rdMolDescriptors.GetMACCSKeysFingerprint(mol)
    elif fp_type.startswith('morgan'):
        radius = 2 if fp_type == 'morgan2' else 3
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=2048)
    elif fp_type == 'rdkit':
        fp = Chem.RDKFingerprint(mol, fpSize=2048)
    else:
        raise ValueError("Unknown fingerprint type")
    # Преобразуем в список int
    return list(fp.ToBitString())  # или fp.ToList()

# ======================== Общая функция для одной молекулы ========================
def compute_all_molecule_features(smiles):
    """
    Для заданной SMILES возвращает словарь всех признаков молекулы.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    features = {}

    # 1. H-bond
    hbd, hba = get_hbond_features(mol)
    features['hbd'] = hbd
    features['hba'] = hba

    # 2. Заряды
    avg_c, min_c, max_c, avg_o, avg_n = get_gasteiger_charges(mol)
    features['avg_charge'] = avg_c
    features['min_charge'] = min_c
    features['max_charge'] = max_c
    features['avg_o_charge'] = avg_o
    features['avg_n_charge'] = avg_n

    # 3. Геометрия
    ar_rings, rings, vol, sasa, labute = get_geometric_features(mol)
    features['aromatic_rings'] = ar_rings
    features['total_rings'] = rings
    features['volume'] = vol
    features['sasa'] = sasa
    features['labute_asa'] = labute

    # 4. Липофильность
    logp, mr = get_lipophilicity_features(mol)
    features['logp'] = logp
    features['mr'] = mr

    # 5. Функциональные группы
    fg_flags = get_functional_group_flags(mol)
    for k, v in fg_flags.items():
        features[f'fg_{k}'] = v

    # 6. Топология
    rotb, wiener, balaban = get_topological_features(mol)
    features['rotatable_bonds'] = rotb
    features['wiener'] = wiener
    features['balaban'] = balaban

    # 7. Энергетика (упрощённо)
    dipole, max_c_part, min_c_part = get_energetic_features(mol)
    features['dipole'] = dipole
    features['max_partial_charge'] = max_c_part
    features['min_partial_charge'] = min_c_part

    # 8. Отпечатки (MACCS keys как пример)
    maccs = get_fingerprints(mol, 'maccs')
    for i, bit in enumerate(maccs):
        features[f'maccs_{i}'] = bit

    return features

# ======================== Функция для пары молекул ========================
def compute_pair_features(smiles1, smiles2):
    """
    Вычисляет признаки для пары молекул, включая разности и отношения.
    Возвращает плоский словарь, готовый для модели ML.
    """
    f1 = compute_all_molecule_features(smiles1)
    f2 = compute_all_molecule_features(smiles2)
    if f1 is None or f2 is None:
        return None

    pair_features = {}

    # Просто конкатенируем признаки каждой молекулы с префиксами
    for key, value in f1.items():
        pair_features[f'mol1_{key}'] = value
    for key, value in f2.items():
        pair_features[f'mol2_{key}'] = value

    # Вычисляем разности и отношения для ключевых числовых признаков
    numeric_keys = ['hbd', 'hba', 'logp', 'volume', 'mr', 'rotatable_bonds',
                    'aromatic_rings', 'total_rings', 'dipole', 'avg_charge']
    for key in numeric_keys:
        if key in f1 and key in f2:
            val1 = f1[key]
            val2 = f2[key]
            pair_features[f'diff_{key}'] = val1 - val2
            # избегаем деления на ноль
            if abs(val2) > 1e-6:
                pair_features[f'ratio_{key}'] = val1 / val2
            else:
                pair_features[f'ratio_{key}'] = 0.0 if val1 == 0 else 1e6

    # Специфичные для водородных связей: комплементарность доноров и акцепторов
    pair_features['hbd1_hba2_complement'] = abs(f1['hbd'] - f2['hba'])
    pair_features['hba1_hbd2_complement'] = abs(f1['hba'] - f2['hbd'])

    return pair_features



df = pd.read_csv('test.csv', names=['id', 'smiles1', 'smiles2']) #load the CSV file
df.head()  # Display a few rows

# # GetNumAtoms() method returns a general nubmer of all atoms in a molecule
# df['num_of_atoms'] = df['mol'].apply(lambda x: x.GetNumAtoms())

# # GetNumHeavyAtoms() method returns a nubmer of all atoms in a molecule with molecular weight > 1
# df['num_of_heavy_atoms'] = df['mol'].apply(lambda x: x.GetNumHeavyAtoms())
