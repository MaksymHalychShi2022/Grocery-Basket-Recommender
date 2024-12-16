import pandas as pd

# Load the data
user_clusters = pd.read_csv('data/extracted/user_clusters.csv')
user_product_interactions = pd.read_csv('data/extracted/user_product_interaction.csv')

# Merge the data on user_id to replace user_id with cluster_id
merged_data = user_product_interactions.merge(user_clusters, on='user_id', how='left')

# Drop the original user_id column and rename cluster_id to user_id
merged_data = merged_data.drop(columns=['user_id'])

# Group by cluster_id and product_id, summing the interactions
aggregated_data = merged_data.groupby(['cluster_id', 'product_id'], as_index=False).agg({'interaction': 'sum'})

# Save or display the result
print(aggregated_data)
# Optionally, save to a new CSV
aggregated_data.to_csv('data/extracted/cluster_product_interaction.csv', index=False)
