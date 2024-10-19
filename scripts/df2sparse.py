import argparse
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix
import scipy.io as io

def main(args: argparse.Namespace) -> None:
    # Load the user-product interaction DataFrame
    interaction_df = pd.read_csv('../data/extracted/user_product_interaction.csv')

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

    # Save sparse matrix to disk
    io.mmwrite('../data/extracted/user_product_interaction_sparse.mtx', sparse_matrix_csr)

    # Save user mapping to CSV
    user_mapping_df = pd.DataFrame(list(user_mapping.items()), columns=['user_id', 'user_idx'])
    user_mapping_df.to_csv('../data/extracted/user_mapping.csv', index=False)
    print("User mapping saved to '../data/extracted/user_mapping.csv'")

    # Save product mapping to CSV
    product_mapping_df = pd.DataFrame(list(product_mapping.items()), columns=['product_id', 'product_idx'])
    product_mapping_df.to_csv('../data/extracted/product_mapping.csv', index=False)
    print("Product mapping saved to '../data/extracted/product_mapping.csv'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="User-Product Interaction Matrix Construction Script")
    args = parser.parse_args()
    main(args)
