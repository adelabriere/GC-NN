from graph_policy.policy_computation import MolecularEncoder,DecompositionPathBuilder
import pandas as pd
from molinput.options import ConversionOptions
from rdkit.Chem import MolFromInchi,MolToInchiKey


##Random molecules taken form KEGG
inchi_list = ['InChI=1S/C6H12O7/c7-1-2(8)3(9)4(10)5(11)6(12)13/h2-5,7-11H,1H2,(H,12,13)/p-1/t2-,3+,4-,5-/m0/s1',
 'InChI=1S/C28H44O/c1-18(2)8-7-9-19(3)22-12-13-24-21-10-11-23-20(4)26(29)15-17-28(23,6)25(21)14-16-27(22,24)5/h8,19-20,22-24H,7,9-17H2,1-6H3/t19-,20?,22-,23+,24+,27-,28+/m1/s1',
 'InChI=1S/C8H9NO2/c9-4-3-6-1-2-7(10)8(11)5-6/h1-2,5H,3-4,9H2', 
 'InChI=1S/C24H39N8O18P3S/c1-24(2,19(37)22(38)28-4-3-14(34)27-5-6-54-15(35)7-13(25)33)9-47-53(44,45)50-52(42,43)46-8-12-18(49-51(39,40)41)17(36)23(48-12)32-11-31-16-20(26)29-10-30-21(16)32/h10-12,17-19,23,36-37H,3-9H2,1-2H3,(H2,25,33)(H,27,34)(H,28,38)(H,42,43)(H,44,45)(H2,26,29,30)(H2,39,40,41)/t12-,17-,18-,19+,23-/m1/s1',
 'InChI=1S/C20H14O5/c1-25-15-4-2-3-12-17(15)19(23)13-6-5-11-7-10(9-21)8-14(22)16(11)18(13)20(12)24/h2-8,21-22H,9H2,1H3',
 'InChI=1S/C20H32N6O12S2Se/c21-9(19(35)36)1-3-13(27)25-11(17(33)23-5-15(29)30)7-39-41-40-8-12(18(34)24-6-16(31)32)26-14(28)4-2-10(22)20(37)38/h9-12H,1-8,21-22H2,(H,23,33)(H,24,34)(H,25,27)(H,26,28)(H,29,30)(H,31,32)(H,35,36)(H,37,38)/t9-,10-,11-,12-/m0/s1',
 'InChI=1S/C5H13O8P/c6-1-3(7)5(9)4(8)2-13-14(10,11)12/h3-9H,1-2H2,(H2,10,11,12)/t3-,4+,5+/m0/s1',
  'InChI=1S/C15H10O6/c16-8-4-11(19)15-12(20)6-13(21-14(15)5-8)7-1-2-9(17)10(18)3-7/h1-6,16-19H',
    'InChI=1S/C7H6O7/c8-4(7(13)14)1-3(6(11)12)2-5(9)10/h2H,1H2,(H,9,10)(H,11,12)(H,13,14)/b3-2+']


def test_policy_generation():
    mols = [MolFromInchi(x) for x in inchi_list]
    options = ConversionOptions()
    encoder = MolecularEncoder(options)
    dpb = DecompositionPathBuilder(options,50)
    for mol in mols:
        G = encoder.mol_to_graph(mol)
        apds = dpb.decompose(G)
        starting_apd = apds[0]
        prob_apd = [x[1:3] for x in apds[1:]]
        eG = dpb.build_graph(starting_apd,prob_apd)
        rdeG = encoder.graph_to_mol(eG)
        rdG = encoder.graph_to_mol(G)
        assert MolToInchiKey(rdG) == MolToInchiKey(rdeG)

if __name__=="__main__":
    test_policy_generation()