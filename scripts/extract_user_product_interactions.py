import argparse
import os

import pandas as pd


def main(args: argparse.Namespace) -> None:
	"""
	Generate a user-product interaction CSV based on orders data.

	This function performs the following steps:
	1. Loads the orders and order_products datasets.
	2. Merges the orders data with the order_products data on 'order_id'.
	3. Calculates the number of times each user purchased each product.
	4. Ensures that the output directory exists.
	5. Saves the resulting user-product interaction DataFrame to a CSV file.

	Args:
		args (argparse.Namespace): Command-line arguments including:
			- orders_path: Path to the orders.csv file.
			- order_products_path: Path to the order_products.csv file.
			- output_path: Path to save the resulting user-product interaction CSV.
	"""

	# Load orders.csv
	print(f"Loading orders data from '{args.orders_path}'...")
	orders = pd.read_csv(args.orders_path)
	print(f"Loaded {len(orders)} records from orders data.")

	# Load order_products.csv
	print(f"Loading order_products data from '{args.order_products_path}'...")
	order_products = pd.read_csv(args.order_products_path)
	print(f"Loaded {len(order_products)} records from order_products data.")

	# Merge the orders table with order_products on 'order_id'
	print("Merging orders data with order_products data...")
	merged_df = pd.merge(order_products, orders[['order_id', 'user_id']], on='order_id', how='inner')
	print(f"Merged data has {len(merged_df)} records.")

	# Calculate the interaction as the number of times a user purchased a product
	print("Calculating user-product interactions...")
	interaction_df = merged_df.groupby(['user_id', 'product_id']).size().reset_index(name='interaction')
	print(f"Calculated {len(interaction_df)} user-product interactions.")

	# Ensure output directory exists
	output_dir = os.path.dirname(args.output_path)
	if output_dir:
		os.makedirs(output_dir, exist_ok=True)
		print(f"Ensured that output directory '{output_dir}' exists.")

	# Save the result to a new CSV
	print(f"Saving user-product interaction data to '{args.output_path}'...")
	interaction_df.to_csv(args.output_path, index=False)
	print("User-product interaction data saved successfully!")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Generate User-Product Interaction CSV")
	parser.add_argument(
		'--orders_path',
		type=str,
		default='data/raw/orders.csv',
		help='Path to the orders.csv file'
	)
	parser.add_argument(
		'--order_products_path',
		type=str,
		default='data/raw/order_products__prior.csv',
		help='Path to the order_products.csv file'
	)
	parser.add_argument(
		'--output_path',
		type=str,
		default='data/extracted/user_product_interaction.csv',
		help='Path to save the user-product interaction CSV'
	)

	main(args=parser.parse_args())
