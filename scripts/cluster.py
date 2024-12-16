import argparse
import os
import pandas as pd
from scipy.io import mmread
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD


def main(args: argparse.Namespace) -> None:
	"""
	Perform user clustering based on a sparse user-product interaction matrix.

	This function performs the following steps:
	1. Loads the sparse user-product interaction matrix from an .mtx file.
	2. Loads the user mapping from a CSV file.
	3. Performs dimensionality reduction using Truncated SVD.
	4. Applies KMeans clustering to cluster users into specified clusters.
	5. Maps cluster IDs back to the original user IDs using the user mapping.
	6. Saves the clustering results to a specified CSV file.

	Args:
		args (argparse.Namespace): Command-line arguments including:
			- input_path: Path to the input sparse matrix (.mtx format).
			- user_mapping_path: Path to the user mapping CSV file.
			- output_path: Path to save the clustering results CSV.
			- n_components: Number of components for Truncated SVD (dimensionality reduction).
			- n_clusters: Number of clusters for KMeans.
			- random_state: Random state for reproducibility.
	"""
	# Load the sparse user-product interaction matrix from the .mtx file
	print("Loading the sparse user-product interaction matrix...")
	sparse_matrix = mmread(args.input_path).tocsr()

	# Load user mapping from the CSV file
	print("Loading user mapping...")
	user_mapping_df = pd.read_csv(args.user_mapping_path)
	user_idx_to_id = user_mapping_df.set_index('user_idx')['user_id'].to_dict()

	# Perform dimensionality reduction using Truncated SVD
	print("Performing dimensionality reduction with Truncated SVD...")
	svd = TruncatedSVD(n_components=args.n_components, random_state=args.random_state)
	reduced_matrix = svd.fit_transform(sparse_matrix)

	# Train KMeans clustering model to cluster users
	print(f"Clustering users into {args.n_clusters} clusters...")
	kmeans = KMeans(n_clusters=args.n_clusters, random_state=args.random_state)
	user_clusters = kmeans.fit_predict(reduced_matrix)

	# Map cluster IDs back to original user IDs using the mapping
	user_ids = [user_idx_to_id[idx] for idx in range(sparse_matrix.shape[0])]
	clustering_results = pd.DataFrame({'user_id': user_ids, 'cluster_id': user_clusters})

	# Ensure the output directory exists
	os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

	# Save the clustering results to a CSV file
	print(f"Saving clustering results to {args.output_path}...")
	clustering_results.to_csv(args.output_path, index=False)
	print("Clustering results saved successfully!")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="User Clustering Script")
	parser.add_argument(
		'--input_path',
		type=str,
		default='data/extracted/user_product_interaction_sparse.mtx',
		help="Path to the input sparse matrix (.mtx format)"
	)
	parser.add_argument(
		'--user_mapping_path',
		type=str,
		default='data/extracted/user_mapping.csv',
		help="Path to the user mapping CSV file"
	)
	parser.add_argument(
		'--output_path',
		type=str,
		default='data/extracted/user_clusters.csv',
		help="Path to save the clustering results CSV"
	)
	parser.add_argument(
		'--n_components',
		type=int,
		default=50,
		help="Number of components for Truncated SVD (dimensionality reduction)"
	)
	parser.add_argument(
		'--n_clusters',
		type=int,
		default=1000,
		help="Number of clusters for KMeans"
	)
	parser.add_argument(
		'--random_state',
		type=int,
		default=42,
		help="Random state for reproducibility"
	)

	main(args=parser.parse_args())
