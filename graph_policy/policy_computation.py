from abc import ABC
import networkx as nx
import torch
import torchist
import numpy as np
from typing import List,Optional,Tuple
from molinput.options import ConversionOptions
import rdkit.Chem as Chem
from rdkit.Chem import CanonicalRankAtoms,MolFromSmiles
import rdkit.Chem as Chem
from molinput.options import ConversionOptions #This is the option used to pass form a molecule to any other dataset
import constants #This is the names of the different fileds

#Could a policy be an arbitrary size vector ?
#Far form ideal, as the output of a NN is anyway a vector of ixed size.
chirality_encoding = {
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED:0,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW:1,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW:2,
    Chem.rdchem.ChiralType.CHI_OTHER:3
}

chirality_decoding = dict([x[::-1] for x in chirality_encoding.items()])

"""This script convert a molecule into a graph"""
def demo_molecule():
    return MolFromSmiles("CCC")


"""Convert a molecule into a graph using the conversion options NO TOKENIZATION at this step"""
def _mol_to_graph(mol,options:ConversionOptions) -> nx.Graph():
    #Adding the nodes
    graph = nx.Graph()
    #Options parsing
    canonical = options.canonical
    chirality = options.chirality
    formal_charge = options.formal_charge

    #The name are build directly
    names_attributes = [constants.ATOM]
    if formal_charge:
        names_attributes += [constants.FORMAL_CHARGE]
    if chirality:
        names_attributes += [constants.CHIRALITY]

    graph.graph[constants.DESCRIPTORS] = names_attributes

    def build_attributes(atom):
        lvalues = [atom.GetSymbol()]
        if formal_charge:
            lvalues = lvalues + [atom.GetFormalCharge()]
        if chirality:
            lvalues += [chirality_encoding[atom.GetChiralTag()]]
        return lvalues

    ###We get the canonical ordering if necessaryL
    if canonical:
        new_order = {atom:idx for idx,atom in enumerate(CanonicalRankAtoms(mol))}
    else:
        new_order = {idx:idx for idx,_ in enumerate(mol.GetAtoms())}

    nodes_seq = [(new_order[atom.GetIdx()],dict(zip(names_attributes,build_attributes(atom)))) for atom in mol.GetAtoms()]

    ### We extract the chirality information if necessary
    graph.add_nodes_from(nodes_seq)
    #Adding the edges
    edges_seq = [(new_order[bond.GetBeginAtom().GetIdx()],new_order[bond.GetEndAtom().GetIdx()],{constants.BOND:bond.GetBondType()}) for bond in mol.GetBonds()]
    graph.add_edges_from(edges_seq)
    return graph


#Passing from graph to molecules
def _graph_to_mol(G,options): #options is unused but will maybe be used later.
    # create empty editable mol object
    mol = Chem.RWMol()

    # add atoms to mol and keep track of indexfa
    node_to_idx = {}
    # The idx are added in dfs order for simplicity purpose
    for i in G.nodes():
        #We first create the atom using the correct class of atom 
        node_to_idx[i] = mol.AddAtom(Chem.Atom(G.nodes[i][constants.ATOM]))

    for i,j,infos in G.edges(data=True):
        idx_j = node_to_idx[j]
        idx_i = node_to_idx[i]
        bond_type = infos[constants.BOND]
        print("bond {} {} {}".format(idx_i, idx_j, bond_type))
        mol.AddBond(idx_i, idx_j, bond_type)
    mol = mol.GetMol()
    return mol

class MolecularEncoder:
    """Convert rdkit molecules into graph and reversely"""
    def __init__(self,options:Optional[ConversionOptions] = None) -> None:
        if options is None:
            self.options = ConversionOptions()
        else:
            self.options = options
        self.atom_tokenizer = self.options.atom_tokenizer
        self.bond_tokenizer = self.options.bond_tokenizer


    def mol_to_graph(self,mol)-> nx.Graph:
        return _mol_to_graph(mol,self.options)

    def graph_to_mol(self,G)-> Chem.Mol:
        return _graph_to_mol(G,self.options)

    def n_atoms(self)-> int:
        return len(self.atom_tokenizer)

    def n_bonds(self) -> int:
        return len(self.bond_tokenizer)


#THis is not use at the moment but hsould be used in the best of the case
class APDs:
    apds: List[Tuple[nx.Graph,torch.Tensor,torch.Tensor,torch.Tensor]]


class DecompositionPathBuilder:
    def __init__(self, options:ConversionOptions, max_nodes: int) -> None:
        self.atom_tokenizer = options.atom_tokenizer
        self.bond_tokenizer =options.bond_tokenizer
        self.natoms = len(self.atom_tokenizer)
        self.nbonds = len(self.bond_tokenizer)
        self.max_nodes = max_nodes
        self.max_charge = options.max_charge

    def initialize_nodes_probs(self):
        return torch.zeros((self.max_nodes,self.natoms+1,self.nbonds+1),dtype=torch.float)

    def initialize_edges_probs(self):
        return torch.zeros((self.max_nodes,self.nbonds+1),dtype=torch.float)

    def initialize_starting_prob(self):
        return torch.zeros((1,self.natoms+1),dtype=torch.float)
        
    
    def decompose(self,graph:nx.Graph,starting_node:int = 0):
        """Decompose molecular grpah in ADPs. The first one is always the starting atoms"""
        #The nodes are generated in DFS preorder nodes
        nodes_dfs = list(nx.dfs_preorder_nodes(graph))
        all_nodes = set([])
        ranks = {node:idx for idx,node in enumerate(nodes_dfs)}
        ##We initialize the starting probability
        starting_prob = self.initialize_starting_prob()
        initial_label = self.atom_tokenizer(graph.nodes[nodes_dfs[0]][constants.ATOM])
        all_nodes.add(nodes_dfs[0])
        starting_prob[0,initial_label] = 1.0
        all_decompositions = [starting_prob]
        for idx,node in enumerate(nodes_dfs):
            if idx==0:
                continue
            node_label = self.atom_tokenizer(graph.nodes[node][constants.ATOM])
            #We build the induced subgraph 
            all_nodes.add(node)
            #We build the current subgraph
            subgraph = graph.subgraph(all_nodes)
            apd_nodes = self.initialize_nodes_probs()
            precursors = [x for x in  subgraph[node].keys()]
            #This is a convention and another solution could be founnd
            best_precursor_idx = np.argmin([ranks[x] for x in precursors])
            precursor = precursors[best_precursor_idx]
            bond_type = self.bond_tokenizer(graph[precursor][node][constants.BOND])
            #print("atom_index",precursor,node_label,constants.BOND,bond_type)
            apd_nodes[ranks[precursor],node_label,bond_type] = 1.0
            apd_edges = self.initialize_edges_probs()
            if len(precursors)>1:
                childrens = [prec for prec in precursors if prec != precursor]
                for child in childrens:
                    edge_label = self.bond_tokenizer(graph[node][child][constants.BOND])
                    apd_edges[ranks[child],edge_label] = 1.0
            all_decompositions.append((subgraph,apd_nodes,apd_edges))
        return all_decompositions

    def build_graph(self,starting_apd,apds,bond_threshold = 0.5):
        #We find the non zero
        graph = nx.Graph()
        init_label = torch.argmax(starting_apd)
        vsymbol = self.atom_tokenizer.get_invert(init_label.item())
        supp_dict = {constants.ATOM:vsymbol}
        graph.add_node(0,**supp_dict)
        for idx,(pnodes,pedges) in enumerate(apds):
            #We add the node
            sel_pos,sel_atom,bond_type = get_max_index(pnodes)
            node_attr = {constants.ATOM:self.atom_tokenizer.get_invert(sel_atom.item())}
            graph.add_node(idx+1,**node_attr)
            #We add the edge of the ocrrect bond type that are easier to learn in the data
            edge_attr = {constants.BOND:self.bond_tokenizer.invert[bond_type]}
            graph.add_edge(sel_pos,idx+1,**edge_attr)
            #We add other connection if needed
            sel_pos,sel_bonds = get_above_threshold(pedges,threshold=bond_threshold)
            sel_pos = sel_pos.numpy()
            sel_bonds = sel_bonds.numpy()
            edges_to_add = [(int(pos),idx+1,{constants.BOND:self.bond_tokenizer.invert[bond]})for pos,bond in zip(sel_pos,sel_bonds)]
            graph.add_edges_from(edges_to_add)
        return graph


def get_max_index(apd):
    return torchist.unravel_index(torch.argmax(apd),apd.shape).numpy()

def get_above_threshold(apd,threshold=0.5):
    return torch.where(apd>threshold)


def plot_subgraph(subgraph):
    nx.draw_networkx(subgraph,labels={nn:infos[constants.ATOM] for nn,infos in subgraph.nodes(data=True)})

def plot_graph(subgraph,atokens):
    nx.draw_networkx(subgraph,labels={nn:atokens.invert[infos[constants.ATOM]] for nn,infos in subgraph.nodes(data=True)})


    pass
if __name__=="__main__":
    from rdkit.Chem import MolFromSmiles,MolToInchiKey
    mol = MolFromSmiles('C(C1C(C(C(C(O1)O)O)O)O)O')
    options = ConversionOptions()
    MD = MolecularEncoder(options) 
    G = MD.mol_to_graph(mol)
    dpb = DecompositionPathBuilder(options, max_nodes=20)
    build_apds = dpb.decompose(G)
    const_apds = [(x,y) for _,x,y in  build_apds[1:]]
    re_g = dpb.build_graph(apds = const_apds,starting_apd=build_apds[0])

    re_g.edges(data=True)
    G.edges(data=True)

    mol_g = MD.graph_to_mol(G)
    mol_re = MD.graph_to_mol(re_g)





    re_g.edges(data=True)
    G.edges(data=True)

    ##Let s count the number of edges
    ##We count the edge for the apb
    import networkx as nx
    import matplotlib.pyplot as plt
    plot_graph(G,atokens=MD.atom_tokenizer)
    plt.show()
    plot_subgraph(re_g)
    plt.show()




    mol
    from rdkit.Chem import FindMolChiralCenters
