# The model is a siamese network
from math import sqrt
import torch
import embedding.data_generator as edg
from torch_geometric.loader import DataLoader
from torch_geometric.nn.models import AttentiveFP
import argparse
import tqdm
import numpy as np


##Use of quadruplet or triplet loss
class QuadrupletLoss:
    def __init__(self,margin,distance_function=None,wneg=0.7,wneigh=1.0,wpos=1.0) -> None:
        self.margin = margin
        if distance_function is None:
            self.distance_function = lambda x, y: torch.linalg.norm(x - y, ord=2, dim=1)
        else:
            self.distance_function = distance_function
        self.wneg = wneg
        self.wneigh = wneigh
        self.wpos = wpos

    def __call__(self,ref,pos,neg,neighbours):
        #We compute the euclidian norm of ever
        dpos = torch.mean(self.distance_function(ref,pos))
        dneg = torch.mean(self.distance_function(ref,neg))
        dneigh = torch.mean(self.distance_function(ref,neighbours))
        return dpos*self.wpos-dneg*self.wneg-dneigh*self.wneigh


                
class SimpleAttentiveNet(torch.nn.Module):
    def __init__(self, N_atoms,N_bonds, out_channels, hidden_channels=20, 
    num_layers=2, dropout=0.2, num_timesteps=2):
        super(SiameseAttentiveNet, self).__init__()
        self.afp = AttentiveFP(in_channels=N_atoms, hidden_channels=hidden_channels, out_channels=out_channels,
                        edge_dim=N_bonds, num_layers=num_layers, num_timesteps=num_timesteps,
                        dropout=dropout)


    def forward(self, graph):
        repr1 = self.afp(graph.x, graph.edge_index, graph.edge_attr, graph.batch)
        return repr1
        

class SiameseAttentiveNet(torch.nn.Module):
    def __init__(self, N_atoms,N_bonds, out_channels, hidden_channels=20, 
    num_layers=2, dropout=0.2, num_timesteps=2):
        super(SiameseAttentiveNet, self).__init__()
        self.afp = AttentiveFP(in_channels=N_atoms, hidden_channels=hidden_channels, out_channels=out_channels,
                        edge_dim=N_bonds, num_layers=num_layers, num_timesteps=num_timesteps,
                        dropout=dropout)


    def forward(self, graph1, graph2):
        repr1 = self.afp(graph1.x, graph1.edge_index, graph1.edge_attr, graph1.batch)
        repr2 = self.afp(graph2.x, graph2.edge_index, graph2.edge_attr, graph2.batch)
        return repr1, repr2

# This is the train and test for the contrastive loss
def train(loader,optimizer,model,loss,device):
    total_loss = total_examples = 0
    for g1,g2,y in tqdm.tqdm(loader):
        g1 = g1.to(device)
        g2 = g2.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        repr1,repr2 = model(g1,g2)
        vloss = loss(repr1,repr2,torch.squeeze(y))
        vloss.backward()
        optimizer.step()
        total_loss += float(vloss)
        total_examples += 1
    return total_loss / total_examples

@torch.no_grad()
def test(loader,model,loss,device):
    total_loss = total_examples = 0
    for g1,g2,y in loader:
        g1 = g1.to(device)
        g2 = g2.to(device)
        y = y.to(device)
        repr1,repr2 = model(g1,g2)
        vloss = loss(repr1,repr2,torch.squeeze(y))
        total_loss += float(vloss) 
        total_examples += 1
    return total_loss / total_examples


# This is the train and test for the quadruplet loss
def train_quadruplet(loader,optimizer,model,loss,device):
    """specific training using the quadruplet loss"""
    total_loss = total_examples = 0.0
    for ref,pos,neg,neigh in tqdm.tqdm(loader):
        # if any([isinstance(ref,list),isinstance(pos,list),isinstance(neg,list),isinstance(neigh,list)]):
        #     print("ref: {} pos {} neg {} neigh {}".format(ref,pos,neg,neigh))
            # ref = ref[0]
            # pos = pos[0]
            # neg = neg[0]
            # neigh = neigh[0]
        ref = ref.to(device)
        pos = pos.to(device)
        neg = neg.to(device)
        neigh = neigh.to(device)
        emb_ref  = model(ref)
        emb_pos = model(pos)
        emb_neg = model(neg)
        emb_neigh = model(neigh)
        optimizer.zero_grad()
        vloss = loss(emb_ref,emb_pos,emb_neg,emb_neigh)
        vloss.backward()
        optimizer.step()
        total_loss += float(vloss)
        total_examples += 1
    return total_loss / total_examples

@torch.no_grad()
def test_quadruplet(loader,model,loss,device):
    total_loss = total_examples = 0
    for ref,pos,neg,neigh in tqdm.tqdm(loader):
        # if any([isinstance(ref,list),isinstance(pos,list),isinstance(neg,list),isinstance(neigh,list)]):
        #     print("ref: {} pos {} neg {} neigh {}".format(ref,pos,neg,neigh))
        ref = ref.to(device)
        pos = pos.to(device)
        neg = neg.to(device)
        neigh = neigh.to(device)
        emb_ref  = model(ref)
        emb_pos = [model(pp) for pp in pos]
        emb_neg = model(neg)
        emb_neigh = model(neigh)
        vloss = loss(emb_ref,emb_pos,emb_neg,emb_neigh)
        total_loss += float(vloss)
        total_examples += 1
    return total_loss / total_examples


class EarlyStopping:
    def __init__(self,patience,verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
    
    def update(self,val_loss):
        if self.best_score is None:
            self.best_score = val_loss
        elif val_loss < self.best_score:
            self.best_score = val_loss
            self.counter = 0
        else:
            self.counter += 1
        if self.counter >= self.patience:
            self.early_stop = True

    def need_stop(self):
        return self.early_stop


# def ContrastiveLoss(margin=1.0):
#     def loss(repr1,repr2,y):
#         d = torch.pow(repr1 - repr2, 2).sum(1)
#         d = torch.sqrt(d + 1e-6)
#         return torch.mean(torch.where(y==1, d, torch.clamp(margin - d, min=0.0)))
#     return loss

def complete_training_quadruplet(nums,epochs,path_dataset,size_embedding=24,early_stopping=3,bond_features="extensive",atom_features="extensive",
    num_pos=2, num_neg=2,num_neighbours=2,reset=False,**kwargs):
    ##We first parse the dataset and susbset it

    dataset =  edg.BiologicalNetworkQuadruplet(path_dataset,bond_features=bond_features,atom_features=atom_features,max_connection=10,
    num_pos = num_pos, num_neg = num_neg, num_neighbours = num_neighbours, reset = reset)

    #dataset = edg.BiologicalNetworkExamples(path_dataset,bond_features=bond_features,atom_features=atom_features,max_connection=10)
    N = min(len(dataset),nums) // 10
    val_dataset = dataset[:N]
    test_dataset = dataset[N:2 * N]
    train_dataset = dataset[2 * N:]

    N_atom_feature = dataset.mol_encoder.atom_features.num_features
    N_bond_feature = dataset.mol_encoder.bond_features.num_features

    early = EarlyStopping(early_stopping)

    train_loader = DataLoader(train_dataset, batch_size=200, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=200)
    test_loader = DataLoader(test_dataset, batch_size=200)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SiameseAttentiveNet(N_atom_feature,N_bond_feature,out_channels=size_embedding,**kwargs).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=10**-2.5,
                                weight_decay=10**-5)

    # loss = torch.nn.CosineEmbeddingLoss()
    loss = QuadrupletLoss(0.1)


    initial_test_loss = test_quadruplet(test_loader,model,loss,device)
    initial_train_loss = test_quadruplet(train_loader,model,loss,device)
    print(f'Beginning. Loss: {initial_train_loss:.4f}, '
        f'Test: {initial_test_loss:.4f}')

    ##We start by computing a loss befoe any training is Done
    for epoch in range(1, epochs):
        train_rmse = train_quadruplet(train_loader,optimizer,model,loss,device)
        #val_rmse = test(val_loader,model,loss,device)
        test_rmse = test_quadruplet(test_loader,model,loss,device)
        print(f'Epoch: {epoch:03d}, Loss: {train_rmse:.4f}, '
            f'Test: {test_rmse:.4f}')

        early.update(test_rmse)
        if early.need_stop():
            print('Training stopped early after {} epochs'.format(early.counter))
            break
    
    print('Training finished.')
    return model

def complete_training(nums,epochs,path_dataset,size_embedding=24,early_stopping=3,bond_features="extensive",atom_features="extensive",**kwargs):
    ##We first parse the dataset and susbset it
    dataset = edg.BiologicalNetworkExamples(path_dataset,bond_features=bond_features,atom_features=atom_features,max_connection=10)
    N = min(len(dataset),nums) // 10
    val_dataset = dataset[:N]
    test_dataset = dataset[N:2 * N]
    train_dataset = dataset[2 * N:]

    N_atom_feature = dataset.mol_encoder.atom_features.num_features
    N_bond_feature = dataset.mol_encoder.bond_features.num_features

    early = EarlyStopping(early_stopping)

    train_loader = DataLoader(train_dataset, batch_size=200, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=200)
    test_loader = DataLoader(test_dataset, batch_size=200)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SiameseAttentiveNet(N_atom_feature,N_bond_feature,out_channels=size_embedding,**kwargs).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=10**-2.5,
                                weight_decay=10**-5)

    loss = torch.nn.CosineEmbeddingLoss()
    initial_test_loss = test(test_loader,model,loss,device)
    initial_train_loss = test(train_loader,model,loss,device)
    print(f'Beginning. Loss: {initial_train_loss:.4f}, '
        f'Test: {initial_test_loss:.4f}')


    ##We start by computing a loss befoe any training is Done
    for epoch in range(1, epochs):
        train_rmse = train(train_loader,optimizer,model,loss,device)
        #val_rmse = test(val_loader,model,loss,device)
        test_rmse = test(test_loader,model,loss,device)
        print(f'Epoch: {epoch:03d}, Loss: {train_rmse:.4f}, '
            f'Test: {test_rmse:.4f}')

        early.update(test_rmse)
        if early.need_stop():
            print('Training stopped early after {} epochs'.format(early.counter))
            break
    
    print('Training finished.')
    return model


if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Some training options to learn retention time fro the METLIN rt dataset.')
    parser.add_argument('-n', type=int, default=100000,
                    help='The number of training examples which will be used.')
    parser.add_argument('--epochs', type = int,default=20,
                    help='The number of training epochs which will be performed.')
    parser.add_argument('--size_embedding', type = int,default=24,help='The size of the embedding to learn.')
    parser.add_argument('--early_stopping', type = int,default=3,help='The number of epochs to wait before stopping the training.')
    parser.add_argument('--path_model', type = str,default='',help='The path to the model which will be saved.')
    parser.add_argument('--temp_folder', type = str,default='temp',help='The path to the folder where the temporary files will be saved.')
    args = parser.parse_args()
    path_model = args.path_model
    model = complete_training(args.n,args.epochs,early_stopping=args.early_stopping,size_embedding=args.size_embedding,path_dataset=args.temp_folder)
    path_model = 'data/embbeding_model.pyt'
    model = complete_training(100000,3,early_stopping=3,size_embedding=24,path_dataset='temp',num_layers=2,num_timesteps=2)
    model = complete_training_quadruplet(100000,3,early_stopping=3,size_embedding=24,path_dataset='temp',num_layers=2,num_timesteps=2,reset=False)
    torch.save(model.state_dict(),path_model)
    path_encoder = 'data/mol_encoder.pyt'
    mencoder = edg.MolEncoder(bond_features="extensive",atom_features="extensive")

    ##We do a simple test
    from rdkit.Chem import MolFromInchi
    mol = MolFromInchi("InChI=1S/C6H12O6/c7-1-3(9)5(11)6(12)4(10)2-8/h1,3-6,8-12H,2H2/t3-,4+,5+,6+/m0/s1")    
    code = mencoder(mol)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        pred = model.afp(code.x,code.edge_index,code.edge_attr,torch.tensor([1]).repeat_interleave(code.x.shape[0]).to(device))
        print(pred)

    ##Let s try it with a data loader
    DataLoader()
                
    with torch.no_grad():
        pred = model.afp.forward(code.x,code.edge_index,code.edge_attr,torch.tensor([1]).to(device))
        print(pred)
     


    import pickle
    with open(path_encoder,"wb") as f:
        pickle.dump(mencoder,f)
    ##This is to save the encoder
    dataset = edg.BiologicalNetworkExamples(path_dataset,max_connection=10)


    edg.MolEncoder()
    ##We change the encoder