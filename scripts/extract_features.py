# %%
import os

import autorootcwd  # noqa
import numpy as np
import pandas as pd
from scipy import stats

from config import RAW_DATA_PATH, FEATURES_PATH

# Ensure the output directory exists
os.makedirs(FEATURES_PATH, exist_ok=True)
# %%
print("Loading data...")
orders = pd.read_csv(os.path.join(RAW_DATA_PATH, 'orders.csv'))
order_products_prior = pd.read_csv(os.path.join(RAW_DATA_PATH, 'order_products__prior.csv'))
order_products_train = pd.read_csv(os.path.join(RAW_DATA_PATH, 'order_products__train.csv'))
# %%
orders['eval_set'] = orders['eval_set'].astype('category')
# %%
orders_prior = orders.merge(order_products_prior, on='order_id', how='inner')
# %% md
# # user features
# %%
print("Extracting 'u_total_orders'")
u_total_orders = orders_prior.groupby('user_id')['order_number'].max().to_frame('u_total_orders').reset_index()

u_total_orders.to_csv(
	os.path.join(FEATURES_PATH, 'u_total_orders.csv'),
	index=False
)
# %%
print("Extracting 'u_avg_prd'")
# 1. First getting the total number of products in each order.
total_prd_per_order = orders_prior.groupby(by=['user_id', 'order_id'])['product_id'].aggregate('count').to_frame(
	'total_products_per_order').reset_index()

# 2. Getting the average products purchased by each user
u_avg_prd = total_prd_per_order.groupby(by=['user_id'])['total_products_per_order'].mean().to_frame(
	'u_avg_prd').reset_index()

u_avg_prd.to_csv(
	os.path.join(FEATURES_PATH, 'u_avg_prd.csv'),
	index=False
)
# %%
print("Extracting 'u_dow_mode'")
u_dow_mode = orders_prior.groupby(by=['user_id'])['order_dow'].aggregate(lambda x: stats.mode(x)[0]).to_frame(
	'u_dow_mode').reset_index()

u_dow_mode.to_csv(
	os.path.join(FEATURES_PATH, 'u_dow_mode.csv'),
	index=False
)
# %%
print("Extracting 'u_hod_mode'")
u_hod_mode = orders_prior.groupby(by=['user_id'])['order_hour_of_day'].aggregate(lambda x: stats.mode(x)[0]).to_frame(
	'u_hod_mode').reset_index()

u_hod_mode.to_csv(
	os.path.join(FEATURES_PATH, 'u_hod_mode.csv'),
	index=False
)
# %%
print("Extracting 'u_reorder_ratio'")
u_reorder_ratio = orders_prior.groupby(by='user_id')['reordered'].aggregate('mean').to_frame(
	'u_reorder_ratio').reset_index()
u_reorder_ratio['u_reorder_ratio'] = u_reorder_ratio['u_reorder_ratio'].astype(np.float16)

u_reorder_ratio.to_csv(
	os.path.join(FEATURES_PATH, 'u_reorder_ratio.csv'),
	index=False
)
# %% md
# # product features
# %%
print("Extracting 'p_total_orders'")
p_total_orders = orders_prior.groupby('product_id')['order_id'].count().to_frame('p_total_orders').reset_index()

p_total_orders.to_csv(
	os.path.join(FEATURES_PATH, 'p_total_orders.csv'),
	index=False
)
# %%
print("Extracting 'p_reorder_ratio'")
p_reorder_ratio = orders_prior.groupby(by='product_id')['reordered'].mean().to_frame('p_reorder_ratio').reset_index()

p_reorder_ratio.to_csv(
	os.path.join(FEATURES_PATH, 'p_reorder_ratio.csv'),
	index=False
)
# %%
print("Extracting 'p_avg_cart_position'")
p_avg_cart_position = orders_prior.groupby(by='product_id')['add_to_cart_order'].mean().to_frame(
	'p_avg_cart_position').reset_index()

p_avg_cart_position.to_csv(
	os.path.join(FEATURES_PATH, 'p_avg_cart_position.csv'),
	index=False
)
# %% md
# # user-product features
# %%
print("Extracting 'up_total_orders'")
up_total_orders = orders_prior.groupby(['user_id', 'product_id'])['order_id'].count().to_frame(
	'up_total_orders').reset_index()

up_total_orders.to_csv(
	os.path.join(FEATURES_PATH, 'up_total_orders.csv'),
	index=False
)
# %%
print("Extracting 'up_reorder_ratio'")
# Finding when the user has bought a product the first time.
up_reorder_ratio = orders_prior.groupby(by=['user_id', 'product_id'])['order_number'].min().to_frame(
	'up_first_order').reset_index()

# Add u_total_orders
up_reorder_ratio = up_reorder_ratio.merge(u_total_orders, on='user_id', how='left')

# Calculating the order range between first and last.
# The +1 includes in the difference the first order were the product has been purchased
up_reorder_ratio['range'] = up_reorder_ratio.u_total_orders - up_reorder_ratio.up_first_order + 1

# Add u_total_orders
up_reorder_ratio = up_total_orders.merge(up_reorder_ratio, on=['user_id', 'product_id'], how='left')

# Calculating the ratio.
up_reorder_ratio['up_reorder_ratio'] = up_reorder_ratio.up_total_orders / up_reorder_ratio.range
up_reorder_ratio = up_reorder_ratio[["user_id", "product_id", "up_reorder_ratio"]]
up_reorder_ratio.to_csv(
	os.path.join(FEATURES_PATH, 'up_reorder_ratio.csv'),
	index=False
)
# %%
print("Extracting 'up_last_five'")
# Calculate number of order but from bask
orders_prior['order_number_back'] = orders_prior.groupby(by=['user_id'])['order_number'].transform(
	'max') - orders_prior.order_number + 1
# Take only last 5 orders for each user
up_last_five = orders_prior.loc[orders_prior.order_number_back <= 5][
	['order_id', 'user_id', 'product_id', 'order_number_back']]
# Count total in last five
up_last_five = up_last_five.groupby(by=['user_id', 'product_id'])['order_id'].aggregate('count').to_frame(
	'up_last_five').reset_index()

# Normalize
up_last_five.up_last_five /= 5

up_last_five.to_csv(
	os.path.join(FEATURES_PATH, 'up_last_five.csv'),
	index=False
)
# %%
