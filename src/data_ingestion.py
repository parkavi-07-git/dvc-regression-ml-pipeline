
import os
import pandas as pd
from sklearn.datasets import load_diabetes


def load_data():

    dataset = load_diabetes(as_frame=True)

    return dataset.frame


def save_data(df):

    os.makedirs("data/raw", exist_ok=True)

    output_path = "data/raw/data.csv"

    df.to_csv(output_path, index=False)

    print("===================================")
    print("      DATA INGESTION COMPLETED")
    print("===================================")
    print(f"Saved to : {output_path}")
    print(f"Shape    : {df.shape}")
    print("===================================")


def main():

    df = load_data()

    save_data(df)


if __name__ == "__main__":
    main()
