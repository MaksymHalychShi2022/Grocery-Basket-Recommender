import random
from typing import List

import pandas as pd

from src.grocery_basket_recommender import GroceryBasketRecommender


class MockGroceryBasketRecommender(GroceryBasketRecommender):
	def __init__(self, products_csv_path="data/raw/products.csv"):
		self.products_df = pd.read_csv(products_csv_path)
		self.available_items = [item.lower() for item in self.products_df['product_name']]

	def recommend(self, basket: List[str]) -> str:
		return random.choice(self.available_items)
