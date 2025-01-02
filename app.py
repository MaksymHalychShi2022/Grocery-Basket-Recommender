import os.path

import gradio as gr
import joblib
import pandas as pd
from config import RAW_DATA_PATH

from scripts.utils import load_features_with_prefix

up_features = load_features_with_prefix("up_", merge_on=['user_id', 'product_id'])
up_features.fillna(0, inplace=True)
u_features = load_features_with_prefix('u_', merge_on='user_id')
p_features = load_features_with_prefix('p_', merge_on='product_id')

products = pd.read_csv(os.path.join(RAW_DATA_PATH, "products.csv"))

current_user_id = None
current_user_features = None

current_model_path = None
current_model = None


# Define the mock recommendation function
def recommend(user_id, model_path, probability_threshold):
	global current_user_id, current_user_features, current_model_path, current_model

	if int(user_id) != current_user_id:
		current_user_id = int(user_id)
		current_user_features = up_features[up_features.user_id == current_user_id].merge(
			u_features[u_features.user_id == current_user_id], on='user_id', how='left'
		).merge(
			p_features, on='product_id', how='left'
		).drop(
			'user_id', axis=1
		).set_index(
			'product_id'
		)
		current_user_features.fillna(0, inplace=True)

	if model_path != current_model_path:
		current_model_path = model_path
		current_model = joblib.load(model_path)

	predictions = pd.DataFrame()
	predictions["product_id"] = current_user_features.index
	predictions = predictions.merge(products[["product_id", "product_name"]], on="product_id", how="left")
	predictions["probability"] = current_model.predict_proba(current_user_features)[:, 1].round(2)

	return predictions[predictions.probability >= probability_threshold]


# Gradio Interface
input_description = "Enter 'user_id':"
output_description = "Recommended Products:"
model_file_description = "Upload model file:"
probability_threshold_description = "Set probability threshold (0.0 to 1.0):"

interface = gr.Interface(
	fn=recommend,
	inputs=[
		gr.Textbox(label=input_description, placeholder="e.g., 123"),
		gr.File(label=model_file_description, type="filepath"),
		gr.Slider(label=probability_threshold_description, minimum=0.0, maximum=1.0, step=0.01, value=0.5),
	],
	outputs=gr.Dataframe(label=output_description),
	title="Instacart Product Recommendation Demo",
	description=(
		"Provide user ID to get personalized recommendations. Upload a model file and set a probability threshold."
	)
)

# Run the Gradio app
if __name__ == "__main__":
	interface.launch()
