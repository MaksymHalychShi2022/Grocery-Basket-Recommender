import argparse
import os

import pandas as pd
import scipy.io as io
from scipy.sparse import coo_matrix


def main(args: argparse.Namespace) -> None:
	"""
	Construct a user-product interaction sparse matrix and save mappings.

	This function performs the following steps:
	1. Loads the user-product interaction DataFrame.
	2. Maps user_id and product_id to integer indices for matrix rows and columns.
	3. Creates a sparse matrix in COO format and converts it to CSR format.
	4. Saves the sparse matrix to an .mtx file.
	5. Saves user and product mappings to separate CSV files.

	Args:
		args (argparse.Namespace): Command-line arguments including:
			- interaction_path: Path to the user-product interaction CSV file.
			- sparse_matrix_path: Path to save the sparse matrix in .mtx format.
			- user_mapping_path: Path to save the user mapping CSV.
			- product_mapping_path: Path to save the product mapping CSV.
	"""

	# Load the user-product interaction DataFrame
	interaction_df = pd.read_csv(args.interaction_path)

	# Map user_id and product_id to integer indices (for matrix rows and columns)
	user_mapping = {user_id: idx for idx, user_id in enumerate(interaction_df['user_id'].unique())}
	product_mapping = {product_id: idx for idx, product_id in enumerate(interaction_df['product_id'].unique())}

	# Create new columns in the DataFrame to hold the integer indices
	interaction_df['user_idx'] = interaction_df['user_id'].map(user_mapping)
	interaction_df['product_idx'] = interaction_df['product_id'].map(product_mapping)

	# Create the sparse matrix in COO format (interaction values as data, user_idx as rows, product_idx as columns)
	sparse_matrix = coo_matrix(
		(interaction_df['interaction'], (interaction_df['user_idx'], interaction_df['product_idx'])),
		shape=(len(user_mapping), len(product_mapping))
	)

	# Optionally convert to CSR format for more efficient row operations
	sparse_matrix_csr = sparse_matrix.tocsr()

	# Print the shape of the resulting sparse matrix
	print(f"Sparse Matrix shape: {sparse_matrix.shape}")

	# Ensure the output directories exist
	os.makedirs(os.path.dirname(args.sparse_matrix_path), exist_ok=True)
	os.makedirs(os.path.dirname(args.user_mapping_path), exist_ok=True)
	os.makedirs(os.path.dirname(args.product_mapping_path), exist_ok=True)

	# Save sparse matrix to disk
	io.mmwrite(args.sparse_matrix_path, sparse_matrix_csr)
	print(f"Sparse matrix saved to '{args.sparse_matrix_path}'")

	# Save user mapping to CSV
	user_mapping_df = pd.DataFrame(list(user_mapping.items()), columns=['user_id', 'user_idx'])
	user_mapping_df.to_csv(args.user_mapping_path, index=False)
	print(f"User mapping saved to '{args.user_mapping_path}'")

	# Save product mapping to CSV
	product_mapping_df = pd.DataFrame(list(product_mapping.items()), columns=['product_id', 'product_idx'])
	product_mapping_df.to_csv(args.product_mapping_path, index=False)
	print(f"Product mapping saved to '{args.product_mapping_path}'")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="User-Product Interaction Matrix Construction Script")
	parser.add_argument(
		'--interaction_path',
		type=str,
		default='../data/extracted/user_product_interaction.csv',
		help='Path to the user-product interaction CSV file'
	)
	parser.add_argument(
		'--sparse_matrix_path',
		type=str,
		default='../data/extracted/user_product_interaction_sparse.mtx',
		help='Path to save the sparse matrix in .mtx format'
	)
	parser.add_argument(
		'--user_mapping_path',
		type=str,
		default='../data/extracted/user_mapping.csv',
		help='Path to save the user mapping CSV file'
	)
	parser.add_argument(
		'--product_mapping_path',
		type=str,
		default='../data/extracted/product_mapping.csv',
		help='Path to save the product mapping CSV file'
	)

	main(args=parser.parse_args())
