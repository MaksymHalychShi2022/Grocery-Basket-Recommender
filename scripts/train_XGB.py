#%%
import os
import platform
from datetime import datetime

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

##%% Import XGBoost
from xgboost import XGBClassifier

from utils import load_features


if platform.system() == "Darwin":  # macOS
	RAW_DATA_PATH = 'data/raw/'
	MODELS_PATH = 'models/'
else:
	RAW_DATA_PATH = '../data/raw/'
	MODELS_PATH = '../models/'
    

# Ensure the output directory exists
os.makedirs(MODELS_PATH, exist_ok=True)

#%%
print("Loading orders...")
orders = pd.read_csv(os.path.join(RAW_DATA_PATH, 'orders.csv'))
order_products_train = pd.read_csv(os.path.join(RAW_DATA_PATH, 'order_products__train.csv'))

#%%
print("Loading features...")
df = load_features()
df = df.merge(
    orders[orders.eval_set == 'train'][['user_id', 'order_id']],
    on='user_id',
    how='left'
)
df = df.merge(
    order_products_train[['product_id', 'order_id', 'reordered']],
    on=['product_id', 'order_id'],
    how='left'
)
df.set_index(['user_id', 'product_id'], inplace=True)
df.drop(['order_id'], axis=1, inplace=True)
df['reordered'] = df['reordered'].fillna(0)

#%%
X = df.drop(['reordered'], axis=1)
y = df['reordered']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

#%%
print("Training model...")
model = XGBClassifier(
    n_estimators=100,        # Number of trees in the ensemble
    learning_rate=0.1,       # Step size shrinkage used in updates
    max_depth=6,             # Maximum depth of trees
    random_state=42,         # For reproducibility
    use_label_encoder=False,  # Avoid warnings in newer versions
	eval_metric="logloss" 
)

eval_set = [(X_train, y_train), (X_val, y_val)]

model.fit(
    X_train,
    y_train,
    eval_set=eval_set,
    verbose=True  # Set to True for progress updates during training
)

#%%
print("Evaluating model...")
y_pred = model.predict(X_val)

f1 = f1_score(y_val, y_pred)
precision = precision_score(y_val, y_pred)
recall = recall_score(y_val, y_pred)

print(f"F1 Score: {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")

#%%
print("Saving model...")
save_path = os.path.join(MODELS_PATH, f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl")
joblib.dump(model, save_path)
print(f"Model saved as {save_path}.")
