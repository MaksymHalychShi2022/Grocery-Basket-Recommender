from abc import ABC, abstractmethod
from typing import List


class GroceryBasketRecommender(ABC):

	@abstractmethod
	def recommend(self, basket: List[str]) -> str:
		"""
		Recommend a single item based on the user's basket.
		"""
		pass
