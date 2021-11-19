
import torch
from torch_geometric.data import (InMemoryDataset, Data,
                                  extract_gz, Dataset)
import shutil
from random import sample
import os.path as osp
import tqdm
import logging
import random
import os


from rdkit import Chem,RDLogger
import networkx as nx

from molinput.features import AtomFeatures,BondFeatures

class MolEncoder:
    def __init__(self,atom_features,bond_features):
        if atom_features is None:
            atom_features = AtomFeatures("simple")
        elif isinstance(atom_features,AtomFeatures):
            atom_features = atom_features
        else:
            atom_features = AtomFeatures(atom_features)
        if bond_features is None:
            bond_features = BondFeatures("simple")
        elif isinstance(bond_features,BondFeatures):
            bond_features = bond_features
        else:
            bond_features = BondFeatures(bond_features)

        self.atom_features = atom_features
        self.bond_features = bond_features

    def __call__(self,mol):
        if mol is None:
            return None
            # raise ValueError("The molecule is not valid")
        xs = []
        for atom in mol.GetAtoms():
            x = self.atom_features(atom)
            xs.append(x)
        x = torch.tensor(xs, dtype=torch.float).view(-1, len(self.atom_features))

        edge_indices, edge_attrs = [], []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            e = self.bond_features(bond)

            edge_indices += [[i, j], [j, i]]
            edge_attrs += [e, e]

        edge_index = torch.tensor(edge_indices)
        edge_index = edge_index.t().to(torch.long).view(2, -1)
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float).view(-1, len(self.bond_features))

        # Sort indices.
        if edge_index.numel() > 0:
            perm = (edge_index[0] * x.size(0) + edge_index[1]).argsort()
            edge_index, edge_attr = edge_index[:, perm], edge_attr[perm]

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        return data





#len and get needs to be implemented the molecule and their representation are written
class BiologicalNetworkQuadruplet(Dataset):

    def __init__(self, root, transform=None, pre_transform=None, atom_features=None, bond_features=None, max_mol = None, max_connection=None,
    num_pos = 2, num_neg = 2, num_neighbours = 2, reset = False):

        self.num_pos = num_pos
        self.num_neg = num_neg
        self.num_neighbours = num_neighbours
        self.mol_encoder = MolEncoder(atom_features,bond_features)
        self.max_connection = max_connection
        self.max_mol = max_mol
        self.reset = reset
        super().__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return ["biological_network.graphml"]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        #Just copy because we are cheaters.
        DIR_NETWORK = "/home/dalexis/Documents/dev/NN/embedding/data/biological_network.graphml"
        # Download to `self.raw_dir`.
        shutil.copy(DIR_NETWORK, self.raw_dir)

    def transform_mol(self,mol):
        return self.mol_encoder(mol) 

    def make_save_path(self,idx):
        return osp.join(self.processed_dir, 'data_{}.pt'.format(idx))

    def save_mol(self,mol,idx):
        path = self.make_save_path(idx)
        if not os.path.isfile(path) or self.reset:
            data = self.transform_mol(mol)
            torch.save(data, path)

    def get_graph(self,idx):
        data = torch.load(self.make_save_path(idx))
        return data

    def process(self):
        net = nx.read_graphml(self.raw_paths[0])

        #We disable the rdkit log
        RDLogger.DisableLog('rdApp.*') 

        ##We remove the nodes with a degree higher than the data
        niter = 0
        while niter==0 or len(to_remove)>0:
            to_remove = []
            if self.max_connection is not None:
                for node in net.nodes():
                    if len(net[node]) > self.max_connection or len(net[node])==0 or 'pubchem_neighbours' not in net.nodes[node]:
                        to_remove.append(node)
            for invalid_node in to_remove:
                net.remove_node(invalid_node)
            niter+=1

        #Mol counter and id cache
        mol_count = 0
        pos_example_count  = 0
        self.inchikey_id = {}
        self.neighbours_cache = {}
        self.positive_cache = {}

        #We select the right number of nodes
        all_nodes = list(net.nodes()) 

        if (self.max_mol is not None) and (self.max_mol<len(all_nodes)):
            all_nodes = random.sample(all_nodes,self.max_mol)

        #We write the correct mol
        logging.info("Writing all the molecules in the network")
        for node in all_nodes:
            infos = net.nodes[node]
            rmdol = Chem.MolFromInchi(infos['inchi'])
            ikey = Chem.MolToInchiKey(rmdol)
            self.inchikey_id[ikey] = mol_count
            self.save_mol(rmdol,mol_count)
            mol_count += 1
            pos_example_count += 1
        

        #We now do exactly the same for every inchi in the network
        logging.info("Processing neighbours data")
        for node in tqdm.tqdm(all_nodes):
            infos = net.nodes[node]
            if 'pubchem_neighbours' not in infos:
                continue
            else:
                neighbours = infos['pubchem_neighbours'].split('||')
                self.neighbours_cache[self.inchikey_id[node]] = []
                for neighbour in neighbours:
                    rdmol = Chem.MolFromInchi(neighbour)
                    if rdmol is None: continue
                    self.save_mol(rdmol,mol_count)
                    #The rest of the dat
                    ikey = Chem.MolToInchiKey(rdmol)
                    if ikey not in self.inchikey_id:
                        self.inchikey_id[ikey] = mol_count
                    self.neighbours_cache[self.inchikey_id[node]].append(mol_count)
                    mol_count += 1
                    

        self.total_data = pos_example_count

        #We store the positive information also in the cache system to avoid any supplementary ifnromations.
        for node in all_nodes:
            infos = net.nodes[node]
            self.positive_cache[self.inchikey_id[node]] = [self.inchikey_id[nn] for nn in net[node] if nn in self.inchikey_id]

        #We enable the log
        RDLogger.EnableLog('rdApp.*')

    def len(self):
        ##Why is it -2 ?
        return self.total_data

    def get(self,idx):
        """This function will vuild a batches with the right numbers of examples"""
        ##In this example the dataset are regenerated each tiem, and each item is sampled
        ref_graph = self.get_graph(idx)

        #Wew sample the true neighbours
        #print("POS",self.positive_cache[idx])
        spos = random.choices(self.positive_cache[idx],k=1)
        pos_graphs = self.get_graph(spos[0])

        #We get all the negative graph sampling form all the spoddible graph 
        #print("NEG")
        #snegs = [random.randint(0,self.total_data-1) for _ in range(self.num_neg)]
        sneg = random.randint(0,self.total_data-1)
        neg_graphs = self.get_graph(sneg)

        #We get the rest of the data in this computationnal
        #print("NEIGH",self.neighbours_cache[idx])
        sneighbour = random.choices(self.neighbours_cache[idx],k=1)
        neigh_graphs = self.get_graph(sneighbour)

        #We returnt the paris with the labels
        # labels_list= [1]*self.num_pos+[0]*self.num_neg+[-1]*self.num_neighbours

        # #We returnt eh whole network as a single batch
        # temp = pos_graphs + neg_graphs + neigh_graphs
        # temp = [(ref_graph,other) for other in temp]

        return ref_graph,pos_graphs,neg_graphs,neigh_graphs

#len and get needs to be implemented
class BiologicalNetworkExamples(Dataset):

    def __init__(self, root, transform=None, pre_transform=None, atom_features=None, bond_features=None, max_mol = None, max_connection=None,
    max_sample=5):

        self.mol_encoder = MolEncoder(atom_features,bond_features)
        self.max_connection = max_connection
        self.max_sample = max_sample
        super().__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return ["biological_network.graphml"]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        #Just copy because we are cheaters.
        DIR_NETWORK = "/home/dalexis/Documents/dev/NN/embedding/data/biological_network.graphml"
        # Download to `self.raw_dir`.
        shutil.copy(DIR_NETWORK, self.raw_dir)

    def transform_mol(self,mol):
        return self.mol_encoder(mol)        

    def process(self):
        # Read data into huge `Data` list.
        net = nx.read_graphml(self.raw_paths[0])

        ##We remove the nodes with a degree higher than the data
        to_remove = []
        if self.max_connection is not None:
            for node in net.nodes():
                if len(net[node]) > self.max_connection:
                    to_remove.append(node)
        for invalid_node in to_remove:
            net.remove_node(invalid_node)

        #We cache all the molecules 
        self.cache_graph = {}
        for node in net.nodes(data=True):
            rmdol = Chem.MolFromInchi(node[1]['inchi'])
            self.cache_graph[node[0]] = self.transform_mol(rmdol)

        total_data = 0
        for node in tqdm.tqdm(net.nodes(data=True)):
            graph = self.cache_graph[node[0]]
            neighbors = net[node[0]]
            #We collect all the neigbours
            neighbors_mol = [self.cache_graph[neighbor] for neighbor in neighbors]
            num_sample = min([len(neighbors_mol),self.max_sample])
            if 'pubchem_neighbours' in node[1]:
                num_sample = min(num_sample,len(node[1]['pubchem_neighbours']))

            datalist = []

            #We first get the positive samples
            sel_neighbors = neighbors_mol
            if num_sample<len(sel_neighbors):
                sel_neighbors = sample(sel_neighbors,num_sample)

            ##If the neighbours are in the same sample 
            sel_neg = sample(net.nodes(),num_sample)
            while len(set(sel_neighbors).intersection(sel_neg))>0:
                sel_neg = sample(net.nodes(),num_sample)
            sel_neg = [self.cache_graph[neighbor] for neighbor in sel_neg]

            ##We get the negative samples form pubchem
            if 'pubchem_neighbours' in node[1]:
                pubchem_neighbours = node[1]['pubchem_neighbours']
                if num_sample<len(pubchem_neighbours):
                    pubchem_neighbours = sample(pubchem_neighbours,num_sample)
                
                sel_pubchem = [self.transform_mol(Chem.MolFromInchi(neighbor)) for neighbor in pubchem_neighbours]
                sel_pubchem = [pp for pp in sel_pubchem if pp is not None]

            pos_label = torch.tensor(1.0, dtype=torch.float)
            neg_label = torch.tensor(-1.0, dtype=torch.float)


            #We add the labels
            pos_data = [(graph.clone(),pos_dat.clone(),pos_label) for pos_dat in sel_neighbors]
            neg_data = [(graph.clone(),neg_dat.clone(),neg_label) for neg_dat in sel_neg]
            if 'pubchem_neighbours' in node[1]:
                neg_data = neg_data + [(graph,neg_dat,neg_label) for neg_dat in sel_pubchem]
            datalist = pos_data + neg_data

            # We write each dataset in a file
            for vdata in datalist:
                #print("path {}".format(osp.join(self.processed_dir, 'data_{}.pt'.format(total_data))))
                torch.save(vdata, osp.join(self.processed_dir, 'data_{}.pt'.format(total_data)))
                total_data += 1
        
        self.total_data = total_data+1

    def len(self):
        ##Why is it -2 ?
        return self.total_data-2

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, 'data_{}.pt'.format(idx)))
        return data
                     
if __name__ == "__main__":
    PROOT = "/home/dalexis/Documents/data/tdat"
    bne = BiologicalNetworkExamples(PROOT)

    SUPP_ROOT = "/home/dalexis/Documents/data/supp_data"
    bne = BiologicalNetworkQuadruplet(SUPP_ROOT,max_mol=None)
    bne.get_graph(10)
    bne.positive_cache[12]
    bne[12]