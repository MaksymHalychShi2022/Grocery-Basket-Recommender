# %%
import os
from datetime import datetime

import autorootcwd  # noqa
import joblib
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, GridSearchCV

from config import MODELS_PATH
from scripts.utils import load_train_dataset

MODELS_PATH = os.path.join(MODELS_PATH, 'lightgbm')

# Ensure the output directory exists
os.makedirs(MODELS_PATH, exist_ok=True)
# %%
print("Loading dataset...")
df = load_train_dataset()

for col in df.select_dtypes(include=['float64']).columns:
    df[col] = df[col].astype('float32')

for col in df.select_dtypes(include=['int64']).columns:
    df[col] = df[col].astype('int32')

print(f"DataFrame size in memory: {df.memory_usage(deep=True).sum() / (1024 ** 2):.2f} MB")


# %%
X = df.drop(['reordered'], axis=1)
y = df['reordered']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
# %%
print("Training model...")

param_grid = {
    'n_estimators': [50, 100],
    'learning_rate': [0.05, 0.1],
    'max_depth': [4, 6, 8],
    'class_weight': ['balanced'],
}

base_model = LGBMClassifier(random_state=42)

grid_search = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    scoring='f1',  # Optimize for F1 score
    cv=5,
    verbose=1, 
    n_jobs=4 # Use all available cores
)

# %%
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_
print(f"Best Hyperparameters: {best_params}")

print("Evaluating best model...")
y_pred = best_model.predict(X_val)

f1 = f1_score(y_val, y_pred)
precision = precision_score(y_val, y_pred)
recall = recall_score(y_val, y_pred)

print(f"F1 Score: {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")

# %%
print("Saving model...")
save_path = os.path.join(MODELS_PATH, "", f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl")
joblib.dump(best_model, save_path)
print(f"Model saved as {save_path}.")
