# %%
import pandas as pd
from surprise import Dataset, Reader

# Завантажуємо ваші дані
df = pd.read_csv('/Users/ivanna/Documents/University/Grocery-Basket-Recommender/data/extracted/user_product_interaction.csv')

# Визначаємо діапазон рейтингів
reader = Reader(rating_scale=(df['interaction'].min(), df['interaction'].max()))

# Створюємо датасет для Surprise
data = Dataset.load_from_df(df[['user_id', 'product_id', 'interaction']], reader)


# %%
from surprise.model_selection import train_test_split

trainset, testset = train_test_split(data, test_size=0.25)

# %%
from surprise import SVD

algo = SVD(n_factors=100, n_epochs=20, biased=True)

# %%
algo.fit(trainset)

# %%
from surprise import accuracy

predictions = algo.test(testset)
rmse = accuracy.rmse(predictions)

# %% [markdown]
# Що означає RMSE: 3.4567?
# 
# Величина помилки: Значення RMSE 3.4567 означає, що в середньому ваші прогнозовані рейтинги відрізняються від реальних на приблизно 3.4567 одиниць.
# Порівняння з діапазоном рейтингів: Важливо порівняти це значення з діапазоном ваших рейтингів. Якщо ваші рейтинги знаходяться в діапазоні від 1 до 5, то помилка в 3.4567 є досить великою, що вказує на низьку точність моделі. Якщо ж діапазон рейтингів більший (наприклад, від 1 до 100), то така помилка може бути прийнятною.

# %%
user_id = '1'
item_id = '101'
prediction = algo.predict(user_id, item_id)
print(prediction.est)


