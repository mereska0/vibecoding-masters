from rdkit import Chem
import rdkit
from rdkit.Chem import rdMolDescriptors, AllChem

print('rdkit.__version__ =', rdkit.__version__)

mol = Chem.MolFromSmiles('CCO')
try:
    gen = rdMolDescriptors.MorganGenerator(radius=2)
    bv = gen.GetFingerprintAsBitVect(mol, nBits=128)
    try:
        s = bv.ToBitString()
        print('MorganGenerator: ToBitString OK, length=', len(s))
    except Exception as e:
        print('MorganGenerator: ToBitString failed:', e)
        try:
            lst = list(bv)
            print('MorganGenerator: iterable OK, len=', len(lst))
        except Exception as e2:
            print('MorganGenerator: iterable failed:', e2)
except Exception as e:
    print('MorganGenerator not available:', e)
    try:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=128)
        print('Fallback GetMorganFingerprintAsBitVect OK, len=', len(fp))
    except Exception as e3:
        print('Fallback also failed:', e3)
