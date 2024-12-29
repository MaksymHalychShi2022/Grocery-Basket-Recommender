# Grocery-Basket-Recommender

Analysis (Visualization, Forecasting) of Sales in Supermarkets

## Train model for competition

```bash
 
# extract features from prior data
python scripts/extract_features.py

# train model on extracted features
python python scripts/train_lightGBM.py 

# generate submission file
python scripts/submit.py
```

# Reference

- https://github.com/ashleve/lightning-hydra-template