import pickle
from rdkit.Chem import MolFromInchi,MolToInchiKey,InchiToInchiKey
import networkx as nx

PATH_PUBCHEM = "/home/dalexis/Documents/dev/link_predict/data/processed/neighbours_filtered_pubchem_cluster.pickle"
PATH_NETWORK = "/home/dalexis/Documents/dev/link_predict/data/processed/full_joint.pickle"

with open(PATH_NETWORK,"rb") as f:
    network = pickle.load(f)

with open(PATH_PUBCHEM,"rb") as f:
    pubchem_neighbours = pickle.load(f)


#We just laod all the possible network

PATH_KEGG = "/home/dalexis/Documents/dev/link_predict/data/network/net_kegg.pickle"
PATH_BIOCYC = "/home/dalexis/Documents/dev/link_predict/data/network/net_biocyc.pickle"
with open(PATH_KEGG,"rb") as f:
    kegg_network = pickle.load(f)

with open(PATH_BIOCYC,"rb") as f:
    biocyc_network = pickle.load(f)


full_net = nx.Graph()

##We first add the nodes corresponding to both network using inchi
inchikey_registy = {}
count_kegg = 0
count_biocyc = 0
count_edge_kegg = 0
count_edge_biocyc = 0

for kegg_id,infos in kegg_network.nodes(data=True):
    if "INCHI" in infos:
        try:
            inchikey = MolToInchiKey(MolFromInchi(infos['INCHI']))
            inchikey_registy[kegg_id] = inchikey
            count_kegg += 1
        except Exception as e:
            continue
        full_net.add_node(inchikey,inchi=infos['INCHI'],kegg_id=kegg_id)

for biocyc_id,infos in biocyc_network.nodes(data=True):
    if "INCHI" in infos:
        try:
            inchikey = MolToInchiKey(MolFromInchi(infos['INCHI']))
            inchikey_registy[biocyc_id] = inchikey
            count_biocyc += 1
        except Exception as e:

            continue
        if inchikey in full_net:
            full_net.add_node(inchikey,inchi=infos['INCHI'],biocyc_id=biocyc_id)
print("Found {nkegg} molecules in KEGG and {nbiocyc} in BioCyc".format(nkegg=count_kegg,nbiocyc=count_biocyc))

#We now add all the edhes
for b1,b2,vdat in kegg_network.edges(data=True):
    if b1 in inchikey_registy and b2 in inchikey_registy:
        full_net.add_edge(inchikey_registy[b1],inchikey_registy[b2],**vdat)
        count_edge_kegg += 1

for b1,b2,vdat in biocyc_network.edges(data=True):
    vdat2 = vdat.copy()
    if 'pathway' in vdat2:
        vdat2['pathway'] = vdat2['pathway'].join("||")
    if b1 in inchikey_registy and b2 in inchikey_registy:
        full_net.add_edge(inchikey_registy[b1],inchikey_registy[b2],**vdat2)
        count_edge_biocyc += 1
print("Found {nkegg} edges in KEGG and {nbiocyc} in BioCyc".format(nkegg=count_edge_kegg,nbiocyc=count_edge_biocyc))

#We now add the neighbours from pubchem
for mol,neighbours in zip(network['mols'],pubchem_neighbours):
    ikey = MolToInchiKey(mol)
    if ikey in full_net:
        #We remove all the nwighbours which are in the network
        nneighbours= [nn for nn in neighbours if InchiToInchiKey(nn) not in full_net[ikey]]
        full_net.nodes[ikey]['pubchem_neighbours'] = "||".join(nneighbours)

#We file out the molecules without neighbours nor datasets
correct_neighbours = sum([("inchi" in infos and "pubchem_neighbours" in infos) for x,infos in full_net.nodes(data=True)])

#We save the data in the rest of the data
DIR_NETWORK = "/home/dalexis/Documents/dev/NN/embedding/data/biological_network.graphml"
nx.write_graphml(full_net,DIR_NETWORK)
