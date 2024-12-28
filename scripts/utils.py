import os
from typing import List, Union, Optional

import joblib
import pandas as pd

from config import MODELS_PATH, FEATURES_PATH


def load_model(model_name: Optional[str] = None):
	"""
	Load a model from MODELS_PATH. If model_name is None, load the latest model.

	Args:
		model_name (Optional[str]): Name of the model file to load. If None, load the latest model.

	Returns:
		Loaded model object.
	"""

	if model_name:
		# Load the specified model
		model_path = os.path.join(MODELS_PATH, model_name)
	else:
		# Get all model files in the directory
		model_files = [
			f.replace("model_", "").replace(".pkl", "")
			for f in os.listdir(MODELS_PATH)
			if f.startswith("model_") and f.endswith(".pkl")
		]

		if not model_files:
			raise FileNotFoundError("No models found in the specified directory.")

		# Pick the latest model
		model_path = os.path.join(MODELS_PATH, f"model_{sorted(model_files)[-1]}.pkl")

	# Load the model
	return joblib.load(model_path)


def load_features_with_prefix(prefix: str, merge_on: Union[List[str], str]) -> pd.DataFrame:
	"""
	Loads and merges all feature files with a given prefix.

	Parameters:
		prefix (str): The prefix of the files to load (e.g., 'up', 'u', 'p').
		merge_on (Union[List[str], str]): The column(s) to merge on.

	Returns:
		pd.DataFrame: A merged DataFrame for the given prefix.
	"""
	merged_df = None
	for file in os.listdir(FEATURES_PATH):
		if file.startswith(prefix) and file.endswith('.csv'):
			file_path = os.path.join(FEATURES_PATH, file)
			feature_df = pd.read_csv(file_path)

			if merged_df is None:
				merged_df = feature_df
			else:
				merged_df = merged_df.merge(feature_df, on=merge_on, how='left')

	return merged_df


def load_features() -> pd.DataFrame:
	"""
	Loads and merges all features (user-product, user, and product) into a single DataFrame.

	Returns:
		pd.DataFrame: A merged DataFrame containing all features.
	"""
	# Load user-product features
	up_features = load_features_with_prefix('up_', merge_on=['user_id', 'product_id'])
	# Load user features
	u_features = load_features_with_prefix('u_', merge_on='user_id')
	# Load product features
	p_features = load_features_with_prefix('p_', merge_on='product_id')

	# Merge user and product features into user-product features
	features = up_features.merge(
		u_features, on='user_id', how='left'
	).merge(
		p_features, on='product_id', how='left'
	)

	return features
