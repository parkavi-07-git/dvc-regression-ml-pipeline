
import os
import yaml
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_params():

    with open("params.yaml", "r") as file:

        return yaml.safe_load(file)


def load_data():

    return pd.read_csv(
        "data/processed/data.csv"
    )


def create_features(
    df,
    test_size,
    random_state
):

    X = df.drop(
        columns=["target"]
    )

    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=test_size,

        random_state=random_state
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    X_train_scaled = pd.DataFrame(
        X_train_scaled,
        columns=X_train.columns
    )

    X_test_scaled = pd.DataFrame(
        X_test_scaled,
        columns=X_test.columns
    )

    X_train_scaled["target"] = y_train.values

    X_test_scaled["target"] = y_test.values

    return (
        X_train_scaled,
        X_test_scaled,
        scaler
    )


def save_features(
    train_df,
    test_df,
    scaler
):

    output_dir = "data/features"

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    train_df.to_csv(
        f"{output_dir}/train.csv",
        index=False
    )

    test_df.to_csv(
        f"{output_dir}/test.csv",
        index=False
    )

    joblib.dump(
        scaler,
        f"{output_dir}/scaler.pkl"
    )

    print("===================================")
    print("   FEATURE ENGINEERING COMPLETED")
    print("===================================")
    print("✅ train.csv")
    print("✅ test.csv")
    print("✅ scaler.pkl")
    print("===================================")


def main():

    params = load_params()

    df = load_data()

    train_df, test_df, scaler = create_features(

        df,

        params["feature_engineering"]["test_size"],

        params["feature_engineering"]["random_state"]
    )

    save_features(
        train_df,
        test_df,
        scaler
    )


if __name__ == "__main__":
    main()
