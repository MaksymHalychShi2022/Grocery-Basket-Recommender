# %%
import os
from datetime import datetime

import autorootcwd  # noqa
import joblib
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from config import MODELS_PATH
from scripts.utils import load_train_dataset

MODELS_PATH = os.path.join(MODELS_PATH, 'lightgbm')

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
model = LGBMClassifier(
	n_estimators=100,  # Number of boosting iterations
	learning_rate=0.1,  # Step size shrinkage
	max_depth=6,  # Maximum tree depth
	random_state=42,  # Seed for reproducibility
	class_weight='balanced'  # Handle class imbalance
)

model.fit(
	X_train, y_train,
	eval_set=[(X_val, y_val)],  # Validation set for monitoring
	eval_metric='logloss',  # Metric to evaluate during training
	# early_stopping_rounds=10,  # Stop early if validation score doesn't improve
	# verbose=True               # Print progress during training
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
save_path = os.path.join(MODELS_PATH, "", f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl")
joblib.dump(model, save_path)
print(f"Model saved as {save_path}.")
