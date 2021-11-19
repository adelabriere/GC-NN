import molinput.tokenizer as tk
from rdkit.Chem import MolFromSmiles

# Constant used for molecular class
SMB_ATOM_FEATURE = 0
SMB_STANDARD_FEATURE = 1
SMB_MASTER_ATOM = 0
SMB_VIRTUAL_BOND = 0


# The rest of the methods in this module are specific functions for computing
# atom and bond features. New ones can be easily added though, and these are
# passed directly to the Preprocessor class.

def get_ring_size(obj, max_size=12):
    if not obj.IsInRing():
        return 0
    else:
        for i in range(max_size):
            if obj.IsInRingSize(i):
                return i
        else:
            return max_size

class MolecularFeature:
    def __init__(self):
        # Two elements needs to be defined
        # _feature_class is tuple of feature types
        # _feature_builder is the function used to build the features
        self._feature_class = None
        self._feature_builder = None
        self.atom_tokenizer  = tk.Tokenizer()
        self.other_tokenizers = []

    def feature_class(self):
        return self._feature_class

    def num_features(self):
        return self.num_features

    def virtual_feature(self):
        raise Exception("virtual_feature needs to be implemented for every MolecularFeature")

    def num_atom(self):
        return len(self.atom_tokenizer._data)

    def atom_encoding(self,atom):
        return self.atom_tokenizer(atom)

    # def one_hot(self):
    #     raise Exception("one_hot needs to be implementes")

    def raw_call(self,item):
        temp = self._feature_builder(item)
        return tuple([temp[idx] if self._feature_class[idx]!=SMB_ATOM_FEATURE else self.atom_tokenizer(temp[idx]) for idx in range(len(self._feature_class))])

    def __call__(self,item):
        temp = self._feature_builder(item)
        return tuple([self.other_tokenizers[idx](temp[idx]) if self._feature_class[idx]!=SMB_ATOM_FEATURE else self.atom_tokenizer(temp[idx]) for idx in range(len(self._feature_class))])

    def __len__(self):
        return self.num_features

#Util function for lazy thing
def test_mol():
    return MolFromSmiles("CC")


#THis part handle the atom features

def atom_features_simple(atom):
    return (
        atom.GetSymbol(),
        atom.GetDegree(),
        atom.GetTotalNumHs(),
    )

def atom_feature_gilmer2017(atom):
    return (
    atom.GetSymbol(),
    atom.GetAtomicNum(),
    atom.GetIsAromatic(),
    atom.GetHybridization(),
    atom.GetTotalNumHs())


def atom_features_intermediate(atom):
    return (
        atom.GetSymbol(),
        atom.GetDegree(),
        atom.GetTotalNumHs(),
        atom.GetImplicitValence(),
        atom.GetIsAromatic(),
    )

props = ['GetSymbol', 'GetChiralTag', 'GetDegree', 'GetExplicitValence',
         'GetFormalCharge', 'GetHybridization', 'GetImplicitValence',
         'GetIsAromatic', 'GetNoImplicit', 'GetNumExplicitHs',
         'GetNumImplicitHs', 'GetNumRadicalElectrons',
         'GetTotalDegree', 'GetTotalNumHs', 'GetTotalValence']

def atom_features_complete(atom):
    atom_type = [getattr(atom, prop)() for prop in props]
    atom_type += [get_ring_size(atom)]
    return tuple(atom_type)



ATOM_CLASSES = {
    "gilmer":atom_feature_gilmer2017,
    "simple":atom_features_simple,
    "intermediate":atom_features_intermediate,
    "extensive":atom_features_complete
}

class AtomFeatures(MolecularFeature):
    def __init__(self,name):
        if name not in ATOM_CLASSES:
            raise ValueError("Atom features {} is not available. Available values are {}".format(name,",".join(list(ATOM_CLASSES.keys()))))
        self._feature_builder = ATOM_CLASSES[name]
        self.atom_tokenizer  = tk.Tokenizer()
        tmol = test_mol()
        self.num_features = len(self._feature_builder(tmol.GetAtomWithIdx(0)))
        self._feature_class = [SMB_STANDARD_FEATURE]*self.num_features
        self._feature_class[0] = SMB_ATOM_FEATURE
        self.other_tokenizers = [tk.Tokenizer() for _ in range(self.num_features)]


##Bond attributes.


def bond_features_gilmer2017(bond):
    return (bond.GetBondType(),)

def bond_features_simple(bond):
    return (bond.GetBeginAtom().GetSymbol(),
    bond.GetEndAtom().GetSymbol(),
    bond.GetBondType())

# Bond features types
def bond_features_intermediate(bond):
    return (bond.GetBeginAtom().GetSymbol(),
        bond.GetEndAtom().GetSymbol(),
        bond.GetBondType(),
        bond.GetIsConjugated(),
        bond.IsInRing())


def bond_features_extensive(bond):
    temp = sorted([
                bond.GetBeginAtom().GetSymbol(),
                bond.GetEndAtom().GetSymbol()])+[bond.GetBondType(),
        bond.GetIsConjugated(),
        bond.GetStereo(),
        get_ring_size(bond)]
    return tuple(temp)
    

BOND_CLASSES = {
    "gilmer":bond_features_gilmer2017,
    "simple":bond_features_simple,
    "intermediate":bond_features_intermediate,
    "extensive":bond_features_extensive
}

class BondFeatures(MolecularFeature):
    def __init__(self,name):
        if name not in BOND_CLASSES:
            raise ValueError("Bond features {} is not available. Available values are {}".format(name,",".join(list(BOND_CLASSES.keys()))))
        self._feature_builder = BOND_CLASSES[name]
        self.atom_tokenizer  = tk.Tokenizer()
        tmol = test_mol()
        self.num_features = len(self._feature_builder(tmol.GetBondWithIdx(0)))
        self._feature_class = [SMB_STANDARD_FEATURE]*self.num_features
        self._feature_class[0] = SMB_ATOM_FEATURE
        self._feature_class[1] = SMB_ATOM_FEATURE
        self.other_tokenizers = [tk.Tokenizer() for _ in range(self.num_features)]

