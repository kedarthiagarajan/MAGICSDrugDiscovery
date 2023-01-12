import numpy as np
import pandas
import torch
import argparse
from tdc.single_pred import ADME
from torch_geometric.utils import from_smiles

def convert_dataset(dataset_name):
"""
    Given a dataset, convert the SMILES Drug representations to PyTorch Geometric Representations
"""
	data = ADME(name = dataset_name)
	df = data.get_data()
	new_data = []
	for index, row in df.iterrows():
	    smile_mol = row["Drug"]
	    atom = from_smiles(smile_mol, True)
	    new_data.append((row["Drug_ID"], atom, row["Y"]))
	    
	df_converted = pandas.DataFrame(new_data, columns=["Drug_ID", "Drug", "Y"])
	return df_converted

def main():
	parser = argparse.ArgumentParser(prog='TDCommons Dataset Converter',
									 description='Convert TDCommons Dataset using SMILES strings to PyTorch Geometric Data')
	parser.add_argument('dataset_name', help='Name of the dataset on TDCommons to convert')
	parser.add_argument('--output', help='Filename to store new pandas dataframe in')
	args = parser.parse_args()
	converted_dataset = convert_dataset(args.dataset_name)
	convert_dataset.to_csv(args.output)



