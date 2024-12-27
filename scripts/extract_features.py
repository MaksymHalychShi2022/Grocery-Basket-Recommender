#%%
import os
import pandas as pd

RAW_DATA_PATH = '../data/raw/'
FEATURES_PATH = '../data/features/'

# Ensure the output directory exists
os.makedirs(FEATURES_PATH, exist_ok=True)
#%%
orders = pd.read_csv(os.path.join(RAW_DATA_PATH, 'orders.csv'))
order_products_prior = pd.read_csv(os.path.join(RAW_DATA_PATH, 'order_products__prior.csv'))
order_products_train = pd.read_csv(os.path.join(RAW_DATA_PATH, 'order_products__train.csv'))
#%%
orders['eval_set'] = orders['eval_set'].astype('category')
#%%
orders_prior = orders.merge(order_products_prior, on='order_id', how='inner')
#%%
# user features
u_total_orders = orders_prior.groupby('user_id')['order_number'].max().to_frame('u_total_orders').reset_index()

u_total_orders.to_csv(
	os.path.join(FEATURES_PATH, 'u_total_orders.csv'),
	index=False
)
#%%
# product features
p_total_orders = orders_prior.groupby('product_id')['order_id'].count().to_frame('p_total_orders').reset_index()

p_total_orders.to_csv(
	os.path.join(FEATURES_PATH, 'p_total_orders.csv'),
	index=False
)
#%%
# user-product features
up_total_orders = orders_prior.groupby(['user_id', 'product_id'])['order_id'].count().to_frame(
	'up_total_orders').reset_index()

up_total_orders.to_csv(
	os.path.join(FEATURES_PATH, 'up_total_orders.csv'),
	index=False
)
#%%

