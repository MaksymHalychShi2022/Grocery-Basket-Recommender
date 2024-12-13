import gradio as gr

from src.mock_grocery_basket_recommender import MockGroceryBasketRecommender

# Instantiate the recommender
recommender = MockGroceryBasketRecommender()

# Gradio interface
with gr.Blocks() as app:
	gr.Markdown("# Grocery Basket Recommender")
	gr.Markdown("Search for items to add to your basket. Get a single recommendation based on the items.")

	basket = gr.State([])  # To maintain the list of items in the basket

	with gr.Row():
		with gr.Column():
			search_box = gr.Textbox(label="Search Items", placeholder="Type to search...")
			dropdown = gr.Dropdown(choices=recommender.available_items[:10], label="Suggestions", interactive=True)
			add_button = gr.Button("Add to Basket")
			basket_list = gr.Textbox(label="Your Basket", interactive=False)
		with gr.Column():
			recommend_button = gr.Button("Recommend Item")
			recommendation = gr.Textbox(label="Recommended Item", interactive=False)


	@search_box.change(inputs=search_box, outputs=dropdown)
	def filter_items(query):
		query = query.lower()
		filtered_items = [item for item in recommender.available_items if query in item]

		return gr.update(value=None, choices=filtered_items[:10])


	# Add selected item from dropdown to the basket
	@add_button.click(inputs=[dropdown, basket], outputs=[basket_list, search_box])
	def add_item_to_basket(item, basket):
		if item and item.lower() in recommender.available_items and item not in basket:
			basket.append(item)
		return basket, ""


	# Generate a single recommendation
	recommend_button.click(recommender.recommend, inputs=basket, outputs=recommendation)

app.launch()
