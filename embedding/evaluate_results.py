###This function just evalute the results
from os import POSIX_FADV_NORMAL
import embedding.data_generator as edg
from embedding.model import SiameseAttentiveNet
import torch
import pickle
import rdkit.Chem as Chem
from rdkit.Chem import MolFromInchi
from rdkit import DataStructs

import networkx as nx
from numpy.linalg import norm
from numpy import dot
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

#We first load the encoder of molecule
path_encoder = 'data/mol_encoder.pyt'
with open(path_encoder, 'rb') as f:
    mol_encoder = pickle.load(f)


size_embedding = 24
N_atom_feature = mol_encoder.atom_features.num_features
N_bond_feature = mol_encoder.bond_features.num_features

model = SiameseAttentiveNet(N_atom_feature,N_bond_feature,out_channels=size_embedding,num_layers=1)

#We load the model
path_model = 'data/embbeding_model.pyt'
model.load_state_dict(torch.load(path_model))


class EmbeddingMol:
    def __init__(self,encoder,model) -> None:
        self.encoder = encoder
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @torch.no_grad()
    def __getitem__(self, mol):
        graph = self.encoder(mol)
        batch = torch.tensor([0], dtype=torch.int64).repeat_interleave(graph.x.shape[0]).to(self.device)
        embedding = self.model.afp(graph.x, graph.edge_index, graph.edge_attr, batch)
        print("embedding shape: {}".format(embedding.shape))
        return embedding 
        

#TEst of embbedding
embmol = EmbeddingMol(mol_encoder,model)
#mol = MolFromInchi("InChI=1S/C6H12O6/c7-1-3(9)5(11)6(12)4(10)2-8/h1,3-6,8-12H,2H2/t3-,4+,5+,6+/m0/s1")
#coded = embmol[mol]

#We load the network to get the data
PATH_NETWORK = "embedding/data/biological_network.graphml"
net = nx.read_graphml(PATH_NETWORK)

mol_cache = {x:MolFromInchi(net.nodes[x]['inchi']) for x in net}

# We sample neighbouring values
NSAMPLE = 2000
import random
ref_nodes = random.choices(list([x for x in net.nodes() if (len(net[x])>0 and 'pubchem_neighbours' in net.nodes[x])]),k=NSAMPLE)

pos_nodes = []
neigh_nodes = []
neg_nodes = random.choices(list([x for x in net.nodes() if len(net[x])>0]),k=NSAMPLE)
for x in ref_nodes:
    pos_nodes.append(random.sample(list(net[x]),k=1)[0])
    neighbours = net.nodes[x]['pubchem_neighbours']
    neighbours = neighbours.split("||")
    neigh_nodes.append(random.sample(neighbours,k=1)[0])

##We cache the neighbours
neigh_nodes = [MolFromInchi(inchi) for inchi in neigh_nodes]

#The first is always tanimoto, the second is always the embbeding
pos_sim = []
neg_sim = []
neigh_sim = []
for idx in range(len(neigh_nodes)):

    #Mols
    ref_mol = mol_cache[ref_nodes[idx]]
    neigh_mol = neigh_nodes[idx]
    if neigh_mol is None:
        continue
    neg_mol = mol_cache[neg_nodes[idx]]
    pos_mol = mol_cache[pos_nodes[idx]]

    #Embeddings
    emb_ref = embmol[ref_mol].squeeze().numpy()
    emb_neg = embmol[neg_mol].squeeze().numpy()
    emb_pos = embmol[pos_mol].squeeze().numpy()
    emb_neigh = embmol[neigh_mol].squeeze().numpy()

    #Fingerprints
    ref_fp = Chem.RDKFingerprint(ref_mol)
    neigh_fp = Chem.RDKFingerprint(neigh_mol)
    neg_fp = Chem.RDKFingerprint(neg_mol)
    pos_fp = Chem.RDKFingerprint(pos_mol)
    pos_sim.append((DataStructs.FingerprintSimilarity(ref_fp,pos_fp),norm(emb_ref-emb_pos),dot(emb_ref, emb_pos)/(norm(emb_ref)*norm(emb_pos))))
    neg_sim.append((DataStructs.FingerprintSimilarity(ref_fp,neg_fp),norm(emb_ref-emb_neg),dot(emb_ref, emb_neg)/(norm(emb_ref)*norm(emb_neg))))
    neigh_sim.append((DataStructs.FingerprintSimilarity(ref_fp,neigh_fp),norm(emb_ref-emb_neigh),dot(emb_ref, emb_neigh)/(norm(emb_ref)*norm(emb_neigh))))
    
labels = (["Pos"]*len(pos_sim))+(["Neg"]*len(neg_sim))+(["Pubchem neighbour"]*len(neigh_sim))
tanimoto_sim = [x[0] for x in pos_sim] + [x[0] for x in neg_sim] + [x[0] for x in neigh_sim]
biol_euclidian_sim = [x[1] for x in pos_sim] + [x[1] for x in neg_sim] + [x[1] for x in neigh_sim]
biol_cosine_sim = [x[2] for x in pos_sim] + [x[2] for x in neg_sim] + [x[2] for x in neigh_sim]


dm_biol = pd.DataFrame({"similarity_euclidian":biol_euclidian_sim,"similarity_cosine":biol_cosine_sim,"label":labels})
dm_tanimoto = pd.DataFrame({"similarity":tanimoto_sim,"label":labels})

sns.histplot(dm_tanimoto, x="similarity", hue="label")
plt.show()


sns.histplot(dm_biol, x="similarity", hue="label")
plt.show()