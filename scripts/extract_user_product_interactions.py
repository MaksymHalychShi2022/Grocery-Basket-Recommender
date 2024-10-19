import argparse
import pandas as pd


def main(args: argparse.Namespace) -> None:
	# Load orders.csv
	orders = pd.read_csv('../data/raw/orders.csv')

	# Load order_products__prior.csv
	order_products_prior = pd.read_csv('../data/raw/order_products__prior.csv')

	# Filter the 'orders' table to only include the "prior" eval_set
	orders_prior = orders[orders['eval_set'] == 'prior']

	# Merge the orders_prior table with order_products_prior on 'order_id'
	merged_df = pd.merge(order_products_prior, orders_prior[['order_id', 'user_id']], on='order_id', how='inner')

	# Calculate the interaction as the number of times a user purchased a product
	interaction_df = merged_df.groupby(['user_id', 'product_id']).size().reset_index(name='interaction')

	# Save the result to a new CSV (optional)
	interaction_df.to_csv('../data/extracted/user_product_interaction.csv', index=False)

	# Print the first few rows to verify the output
	print(interaction_df.head())


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument()
	main(args=parser.parse_args())
