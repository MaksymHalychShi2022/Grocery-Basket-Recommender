# %%
import os
from datetime import datetime

import autorootcwd  # noqa
import joblib
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from config import MODELS_PATH
from scripts.utils import load_train_dataset

MODELS_PATH = os.path.join(MODELS_PATH, 'xgboost')

# Ensure the output directory exists
os.makedirs(MODELS_PATH, exist_ok=True)
# %%
print("Loading dataset...")
df = load_train_dataset()
# %%
X = df.drop(['reordered'], axis=1)
y = df['reordered']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
# %%
print("Training model...")
model = XGBClassifier(
	n_estimators=100,  # Number of trees in the ensemble
	learning_rate=0.1,  # Step size shrinkage used in updates
	max_depth=6,  # Maximum depth of trees
	random_state=42,  # For reproducibility
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

# %%
print("Evaluating model...")
y_pred = model.predict(X_val)

f1 = f1_score(y_val, y_pred)
precision = precision_score(y_val, y_pred)
recall = recall_score(y_val, y_pred)

print(f"F1 Score: {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")

# %%
print("Saving model...")
save_path = os.path.join(MODELS_PATH, f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl")
joblib.dump(model, save_path)
print(f"Model saved as {save_path}.")
