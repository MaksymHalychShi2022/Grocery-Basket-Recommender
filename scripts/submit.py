#%%
import os
from datetime import datetime

import joblib
import pandas as pd

os.chdir("..")

RAW_DATA_PATH = 'data/raw/'
FEATURES_PATH = 'data/features/'
MODELS_PATH = 'models/'
SUBMIT_PATH = 'submit/'

# Ensure the output directory exists
os.makedirs(SUBMIT_PATH, exist_ok=True)
#%%
orders = pd.read_csv(os.path.join(RAW_DATA_PATH, 'orders.csv'))

up_total_orders = pd.read_csv(os.path.join(FEATURES_PATH, 'up_total_orders.csv'))
u_total_orders = pd.read_csv(os.path.join(FEATURES_PATH, 'u_total_orders.csv'))
p_total_orders = pd.read_csv(os.path.join(FEATURES_PATH, 'p_total_orders.csv'))
#%%
orders_test = orders[orders.eval_set == 'test']
#%%
df = up_total_orders.merge(
	u_total_orders, on='user_id', how='left'
)
df = df.merge(
	p_total_orders, on='product_id', how='left'
)
df = df.merge(
	orders_test[['user_id']],
	on='user_id',
	how='right'  # only users preset in 'test' set
)
df.set_index(['user_id', 'product_id'], inplace=True)
#%%
model = joblib.load(os.path.join(MODELS_PATH, 'model_20241227_085936.pkl'))
#%%
y_pred = model.predict_proba(df)
#%%
df['reordered_pred'] = (y_pred[:, 1] > 0.5).astype(int)
#%%
final = df.reset_index()[['product_id', 'user_id', 'reordered_pred']]
final = final.merge(  # add order_id
	orders_test[["user_id", "order_id"]], on='user_id', how='left'
)
#%%
submission = (
	final[final.reordered_pred == 1]
	.groupby('order_id')['product_id']
	.apply(lambda x: ' '.join(map(str, x)))
	.to_frame('products')
	.reset_index()
)
submission = orders_test[['order_id']].merge(
	submission, on='order_id', how='left'
)
submission.fillna("None", inplace=True)
#%%
assert submission.shape[0] == 75_000
save_path = os.path.join(SUBMIT_PATH, f"submission_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
submission.to_csv(save_path, index=False)
#%%
