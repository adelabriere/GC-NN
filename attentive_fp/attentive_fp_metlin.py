from math import sqrt
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn.models import AttentiveFP
import attentive_fp.dataset_metlin_rt as drt
import argparse

def train(loader,optimizer,model,device):
    total_loss = total_examples = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        loss = F.mse_loss(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * data.num_graphs
        total_examples += data.num_graphs
    return sqrt(total_loss / total_examples)

@torch.no_grad()
def test(loader,model,device):
    mse = []
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        mse.append(F.mse_loss(out, data.y, reduction='none').cpu())
    return float(torch.cat(mse, dim=0).mean().sqrt())

@torch.no_grad()
def mae(loader,model,device):
    mse = []
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        mse.append(F.l1_loss(out, data.y, reduction='none').cpu())
    return float(torch.cat(mse, dim=0).mean().sqrt())


def complete_training(nums,epochs):
    ##We first parse the dataset and susbset it
    dataset = drt.METLINInMemory("../data/temp",atom_features="extensive",bond_features="extensive",max_mol=nums)

    N = min(len(dataset),nums) // 10
    val_dataset = dataset[:N]
    test_dataset = dataset[N:2 * N]
    train_dataset = dataset[2 * N:]

    N_atom_feature = dataset.atom_features.num_features
    N_bond_feature = dataset.bond_features.num_features

    train_loader = DataLoader(train_dataset, batch_size=200, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=200)
    test_loader = DataLoader(test_dataset, batch_size=200)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AttentiveFP(in_channels=N_atom_feature, hidden_channels=20, out_channels=1,
                        edge_dim=N_bond_feature, num_layers=1, num_timesteps=2,
                        dropout=0.1).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=10**-2.5,
                                weight_decay=10**-5)

    for epoch in range(1, epochs):
        train_rmse = train(train_loader,optimizer,model,device)
        val_rmse = test(val_loader,model,device)
        test_rmse = test(test_loader,model,device)
        tmae = mae(test_loader,model,device)
        trmae = mae(train_loader,model,device)
        print(f'Epoch: {epoch:03d}, Loss: {train_rmse:.4f} Val: {val_rmse:.4f} '
            f'Test: {test_rmse:.4f}, Test MAE: {tmae:.4f}, Train MAE: {trmae:.4f}')
    
    return(model)
    print('Training finished.')


if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Some training options to learn retention time fro the METLIN rt dataset.')
    parser.add_argument('-n', type=int, default=100000,
                    help='The number of training examples which will be used.')
    parser.add_argument('--epochs', type = int,default=3,
                    help='The number of training epochs which will be performed.')
    args = parser.parse_args()
    model = complete_training(args.n,args.epochs)


    dataset = drt.METLINInMemory("../data/temp",atom_features="extensive",bond_features="extensive",max_mol=args.n)
    
    vmod = dataset[0]
    with torch.no_grad():
        val = model(vmod.x, vmod.edge_index, vmod.edge_attr, torch.tensor([0]))
