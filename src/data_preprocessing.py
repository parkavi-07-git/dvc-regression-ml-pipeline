
import os
import pandas as pd


def load_data():

    path = "data/raw/data.csv"

    df = pd.read_csv(path)

    print(f"Raw data shape: {df.shape}")

    return df


def preprocess_data(df):

    # Clean column names
    df.columns = [
        column.strip().replace(" ", "_")
        for column in df.columns
    ]

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Handle missing numerical values
    numeric_columns = df.select_dtypes(
        include="number"
    ).columns

    for column in numeric_columns:

        df[column] = df[column].fillna(
            df[column].median()
        )

    return df


def save_data(df):

    os.makedirs(
        "data/processed",
        exist_ok=True
    )

    output_path = "data/processed/data.csv"

    df.to_csv(
        output_path,
        index=False
    )

    print(f"Processed data saved to: {output_path}")
    print(f"Processed shape: {df.shape}")


def main():

    df = load_data()

    df = preprocess_data(df)

    save_data(df)


if __name__ == "__main__":
    main()
