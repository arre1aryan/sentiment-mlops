import os
import pandas as pd
from datasets import load_dataset
DATA_DIR = os.path.join(os.path.dirname(__file__), "..","data")

def load_imdb():
    print("Downloading IMDB dataset...")
    dataset = load_dataset("imdb")

    train_df = pd.DataFrame(dataset["train"])
    test_df = pd.DataFrame(dataset["test"])

    print(f"Train size: {len(train_df)}")
    print(f"Test size: {len(test_df)}")

    return train_df, test_df

def save_splits(train_df, test_df):
    os.makedirs(DATA_DIR, exist_ok=True)
    train_df.to_csv(os.path.join(DATA_DIR, "train.csv"), index=False)
    test_df.to_csv(os.path.join(DATA_DIR, "test.csv"), index=False)
    print("Saved train.csv and test.csv to data/")

if __name__ == "__main__":
    train_df, test_df = load_imdb()
    save_splits(train_df, test_df)