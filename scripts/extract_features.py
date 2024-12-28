#%%
import os
import platform

import numpy as np
import pandas as pd
from scipy import stats

if platform.system() == "Darwin":  # macOS
	RAW_DATA_PATH = 'data/raw/'
	FEATURES_PATH = 'data/features/'
else:
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
#%% md
# # user features
#%%
u_total_orders = orders_prior.groupby('user_id')['order_number'].max().to_frame('u_total_orders').reset_index()

u_total_orders.to_csv(
	os.path.join(FEATURES_PATH, 'u_total_orders.csv'),
	index=False
)
#%%
#1. First getting the total number of products in each order.
total_prd_per_order = orders_prior.groupby(by=['user_id', 'order_id'])['product_id'].aggregate('count').to_frame('total_products_per_order').reset_index()

#2. Getting the average products purchased by each user
u_avg_prd = total_prd_per_order.groupby(by=['user_id'])['total_products_per_order'].mean().to_frame('u_avg_prd').reset_index()

u_avg_prd.to_csv(
	os.path.join(FEATURES_PATH, 'u_avg_prd.csv'),
	index=False
)
#%%
u_dow_mode = orders_prior.groupby(by=['user_id'])['order_dow'].aggregate(lambda x : stats.mode(x)[0]).to_frame('u_dow_mode').reset_index()

u_dow_mode.to_csv(
	os.path.join(FEATURES_PATH, 'u_dow_mode.csv'),
	index=False
)
#%%
u_hod_mode = orders_prior.groupby(by=['user_id'])['order_hour_of_day'].aggregate(lambda x : stats.mode(x)[0]).to_frame('u_hod_mode').reset_index()

u_hod_mode.to_csv(
	os.path.join(FEATURES_PATH, 'u_hod_mode.csv'),
	index=False
)
#%%
u_reorder_ratio = orders_prior.groupby(by='user_id')['reordered'].aggregate('mean').to_frame('u_reorder_ratio').reset_index()
u_reorder_ratio['u_reorder_ratio'] = u_reorder_ratio['u_reorder_ratio'].astype(np.float16)

u_reorder_ratio.to_csv(
	os.path.join(FEATURES_PATH, 'u_reorder_ratio.csv'),
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

