from surprise import Dataset, Reader, SVD
import pandas as pd

# Завантажте згенеровані дані взаємодії
interaction_df = pd.read_csv('data/extracted/user_product_interaction.csv')

# Задайте параметри для Reader, вказуючи мінімальне і максимальне значення
reader = Reader(rating_scale=(1, interaction_df['interaction'].max()))

# Створіть датасет із взаємодій
data = Dataset.load_from_df(interaction_df[['user_id', 'product_id', 'interaction']], reader)

# Ініціалізуйте та навчіть модель SVD
svd = SVD()
trainset = data.build_full_trainset()
svd.fit(trainset)

# Приклад рекомендації для певного користувача
user_id = 1  # Задайте ID користувача
product_id = 10  # Задайте ID продукту

# Отримайте прогноз для конкретного користувача і продукту
prediction = svd.predict(user_id, product_id)
print(f"Очікувана оцінка для продукту {product_id} від користувача {user_id}: {prediction.est}")
