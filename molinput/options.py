from molinput.tokenizer import Tokenizer
import rdkit.Chem as Chem

#Could a policy be an arbitrary size vector ?
#Far form ideal, as the output of a NN is anyway a vector of ixed size.




DEFAULT_ATOMS = ["C","O","N","S","P","Br","Se"]
DEFAULT_BONDS = [Chem.rdchem.BondType.SINGLE,Chem.rdchem.BondType.DOUBLE,Chem.rdchem.BondType.TRIPLE,Chem.rdchem.BondType.AROMATIC]
class ConversionOptions:
    def __init__(self,**kwargs)->None:
        self.atom_tokenizer = Tokenizer()
        self.bond_tokenizer = Tokenizer()
        #All the default are added to the tokenizer
        for atom in DEFAULT_ATOMS:
            self.atom_tokenizer(atom)
        for bond in DEFAULT_BONDS:
            self.bond_tokenizer(bond)
        self.explicit_hs = False
        self.max_nodes = 100
        self.canonical = True
        self.chirality = True
        self.max_charge = 3
        self.formal_charge = True
        for arg in kwargs:
            setattr(self, arg, kwargs[arg])
        self._initialize()

    def _initialize(self):
        for atom in DEFAULT_ATOMS:
            _ = self.atom_tokenizer(atom)
        
        

    
