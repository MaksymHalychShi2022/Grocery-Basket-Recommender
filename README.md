# Grocery-Basket-Recommender

Analysis (Visualization, Forecasting) of Sales in Supermarkets

## Clustering based on user-product interactions

Place the original csv files into `data/raw/`

Run:

```bash
# run from `scripts/` dir, because of relative paths
cd scripts

python extract_user_product_interactions.py
python df2sparse.py
python cluster.py
```

# Reference

- https://github.com/ashleve/lightning-hydra-template