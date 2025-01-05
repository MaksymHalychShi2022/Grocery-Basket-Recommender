# Grocery-Basket-Recommender

This project involves developing a recommendation model for supermarket shopping using the open dataset provided by
Instacart. The goal is to predict the products a customer is likely to purchase during their next shopping trip.

## Setup Environment

### Using Docker Dev-Container (Recommended)

The Docker development container supports multiplatform builds (Mac/Linux) and includes both Quarto and Python
dependencies.

#### Build the Image:

```bash
docker build -t gbr-dev -f Dockerfile_dev .
```

#### Run the Dev-Container:

- `-it` flag to run the container in interactive mode.
- `--rm` flag to automatically remove the container after you close the bash session.
- `-v` flag to mount the current directory, allowing you to edit files in the container from your IDE.
- `--net` flag to enable network traffic between the host and container (useful for `preview` mode and the
  `Gradio` app).

```bash
docker run -it --rm -v $(pwd):/Grocery-Basket-Recommender --net=host gbr-dev
```

### Using a Python Virtual Environment

If you prefer not to use the Docker container, you can run the Python code in a `venv`, `conda`, or `mamba` environment.

#### Create a `venv`:

```bash
python3 -m venv gbr
source gbr/bin/activate
```

#### Or Create a `conda` Environment:

```bash
conda create -n gbr python=3.10
conda activate gbr
```

#### Install Python Dependencies:

```bash
pip install -r requirements.txt
```

## Extract Features

1. Download the dataset from
   the [Instacart Market Basket Analysis](https://www.kaggle.com/competitions/instacart-market-basket-analysis) Kaggle
   competition.
2. Place the CSV files in the `data/raw` directory.

Run the following command to extract features:

```bash
python scripts/extract_features.py
```

This script will generate new CSV files with features:

- **User-level features**: Prefixed with `u_`, containing two columns: `user_id` and the feature itself.
- **Product-level features**: Prefixed with `p_`, containing columns `product_id` and the feature.
- **User-product interaction features**: Prefixed with `up_`, containing columns `user_id`, `product_id`, and the
  feature value.

## Train Models

You can either download pre-trained models from the release page or train them yourself using the following scripts:

```bash
python scripts/train_random_forest.py
python scripts/train_lightgbm.py
python scripts/train_xgboost.py
```

## Run the Gradio App

```bash
python app.py
```

Start the application using this command (may take up to a minute):Access the app in your browser
at: [http://127.0.0.1:7860](http://127.0.0.1:7860/)

## Generate Quarto Report

You can download the generated report from the release page or generate it yourself.

#### Generate Report:

```bash
cd quarto
quarto render report.qmd
```

If Quarto is not installed, you can use the Docker dev-container to generate the report.

## Generate Sweetviz Report

Download the pre-generated report from the release page or generate it yourself using:

```bash
python scripts/sweetviz_report.py
```
