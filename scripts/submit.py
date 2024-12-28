#%%
import os
import platform
from datetime import datetime

import pandas as pd

from utils import load_features, load_model


if platform.system() == "Darwin":  # macOS
	RAW_DATA_PATH = 'data/raw/'
	SUBMIT_PATH = 'submit/'
else:
	RAW_DATA_PATH = '../data/raw/'
	SUBMIT_PATH = '../submit/'


THRESHOLD = 0.5  # for probability of been reordered

# Ensure the output directory exists
os.makedirs(SUBMIT_PATH, exist_ok=True)
#%%
print("Loading orders...")
orders = pd.read_csv(os.path.join(RAW_DATA_PATH, 'orders.csv'))
orders_test = orders[orders.eval_set == 'test']
#%%
print("Loading features...")
df = load_features()
df = orders_test[['user_id']].merge(
	df,
	on='user_id',
	how='left'
)
df.set_index(['user_id', 'product_id'], inplace=True)
#%%
print("Loading model...")
model = load_model()
#%%
print("Predicting proba...")
y_pred = model.predict_proba(df)
#%%
print("Creating submission file...")
df['reordered_pred'] = (y_pred[:, 1] > THRESHOLD).astype(int)
df = df.reset_index()[['product_id', 'user_id', 'reordered_pred']]
df = df.merge(  # add order_id
	orders_test[["user_id", "order_id"]], on='user_id', how='left'
)
#%%
df = (
	df[df.reordered_pred == 1]
	.groupby('order_id')['product_id']
	.apply(lambda x: ' '.join(map(str, x)))
	.to_frame('products')
	.reset_index()
)
df = orders_test[['order_id']].merge(
	df, on='order_id', how='left'
)
df.fillna("None", inplace=True)
#%%
assert df.shape[0] == 75_000
save_path = os.path.join(SUBMIT_PATH, f"submission_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
df.to_csv(save_path, index=False)
#%%
