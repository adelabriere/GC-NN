import os
import os.path as osp
import re

import torch
from torch_geometric.data import (InMemoryDataset, Data,
                                  extract_gz)
import shutil

from molinput.features import AtomFeatures,BondFeatures


class METLINInMemory(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, atom_features=None, bond_features=None, max_mol = None):
        #We first change the molecule paramters
        if atom_features is None:
            self.atom_features = AtomFeatures("simple")
        elif isinstance(atom_features,AtomFeatures):
            self.atom_features = bond_features
        else:
            self.atom_features = AtomFeatures(atom_features)
        self.max_mol = max_mol
        if bond_features is None:
            self.bond_features = BondFeatures("simple")
        elif isinstance(bond_features,BondFeatures):
            self.bond_features = bond_features
        else:
            self.bond_features = BondFeatures(bond_features)

        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["SMRT_dataset.csv"]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        #Just copy because we are cheaters.
        PATH_DATA = "/home/dalexis/Documents/dev/NN/attentive_fp/data/SMRT_dataset.csv"

        # Download to `self.raw_dir`.
        shutil.copy(PATH_DATA, self.raw_dir)

    def process(self):
        # Read data into huge `Data` list.
        from rdkit import Chem
        import pandas as pd

        with open(self.raw_paths[0], 'r') as f:
            dataset = pd.read_csv(self.raw_paths[0],delimiter = ";",header=0,quotechar='"',)

        data_list = []
        for icount,elem in dataset.iterrows():
            if self.max_mol is not None and icount>self.max_mol:
                break
            y = float(elem.rt)
            inchi = str(elem.inchi)
            mol = Chem.inchi.MolFromInchi(inchi)
            if mol is None:
                continue

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

            ty = torch.tensor(y, dtype=torch.float).view(1,1)
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=ty,
                        inchi=inchi)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])