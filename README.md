# Grocery-Basket-Recommender

Analysis (Visualization, Forecasting) of Sales in Supermarkets

## Clustering based on user-product interactions

Place the original csv files into `data/raw/`

Run (for Windows):

```bash
# run from `scripts/` dir, because of relative paths
cd scripts

python extract_user_product_interactions.py
python df2sparse.py
python cluster.py
```

Run (for macOS):
```bash
# run from root dir, because of relative paths

python scripts/extract_user_product_interactions.py
python scripts/df2sparse.py
python scripts/cluster.py
```

## Train model for competition

For Windows:

```bash
cd scripts
 
# extract features from prior data
python extract_features.py

# train model on extracted features
python train.py

# generate submission file
python submit.py
```

For macOS:

```bash
 
# extract features from prior data
python scripts/extract_features.py

# train model on extracted features
python scripts/train.py

# generate submission file
python scripts/submit.py
```

# Reference

- https://github.com/ashleve/lightning-hydra-template