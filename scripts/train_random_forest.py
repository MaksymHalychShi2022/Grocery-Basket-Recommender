# %%
import os
from datetime import datetime

import autorootcwd  # noqa
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, GridSearchCV


from config import MODELS_PATH
from scripts.utils import load_train_dataset

MODELS_PATH = os.path.join(MODELS_PATH, 'random_forest')

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

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)  # %%

print("Training model...")

param_grid = {
    'n_estimators': [50, 100],  
    'max_depth': [5, 7],       
    'class_weight': ['balanced'],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2, 5]
}

base_model = RandomForestClassifier(random_state=42)

grid_search = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    scoring='f1', 
    cv=2,
    verbose=3, 
    n_jobs=4 # -1 to use all available cores
)

# %%
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_
print(f"Best Hyperparameters: {best_params}")

print("Evaluating model...")
y_pred = best_model.predict(X_val)

f1 = f1_score(y_val, y_pred)
precision = precision_score(y_val, y_pred)
recall = recall_score(y_val, y_pred)

print(f"F1 Score: {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
# %%
print("Saving model...")
save_path = os.path.join(MODELS_PATH, f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl")
joblib.dump(best_model, save_path)
print(f"Model saved as {save_path}.")
# %%


# # 1. Load the saved model
# saved_model_path = "models/random_forest/model_20250103_221724.pkl"  # Replace with your actual path
# loaded_model = joblib.load(saved_model_path)

# # 2. Generate predictions
# y_pred_loaded = loaded_model.predict(X_val)

# # 3. Calculate metrics
# f1_loaded = f1_score(y_val, y_pred_loaded)
# precision_loaded = precision_score(y_val, y_pred_loaded)
# recall_loaded = recall_score(y_val, y_pred_loaded)

# print(f"F1 Score (Loaded Model): {f1_loaded:.4f}")
# print(f"Precision (Loaded Model): {precision_loaded:.4f}")
# print(f"Recall (Loaded Model): {recall_loaded:.4f}")