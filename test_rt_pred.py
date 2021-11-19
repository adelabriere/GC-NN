from attentive_fp.attentive_fp_metlin import complete_training
import argparse

parser = argparse.ArgumentParser(description='Some training options to learn retention time fro the METLIN rt dataset.')
parser.add_argument('-n', type=int, default=100000,
                help='The number of training examples which will be used.')
parser.add_argument('--epochs', type = int,default=20,
                help='The number of training epochs which will be performed.')
args = parser.parse_args()
complete_training(args.n,args.epochs)