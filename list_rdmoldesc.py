import rdkit
from rdkit.Chem import rdMolDescriptors
print('rdkit.__version__ =', rdkit.__version__)
names = [n for n in dir(rdMolDescriptors) if 'Morgan' in n or 'morgan' in n]
print('matches:', names)
for n in names:
    try:
        print(n, type(getattr(rdMolDescriptors, n)))
    except Exception as e:
        print(n, 'error', e)
