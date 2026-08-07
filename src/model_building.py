
import yaml
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestRegressor


def load_params():

    with open("params.yaml", "r") as file:

        return yaml.safe_load(file)


def load_training_data():

    return pd.read_csv(
        "data/features/train.csv"
    )


def train_model(
    df,
    n_estimators,
    max_depth,
    random_state
):

    X_train = df.drop(
        columns=["target"]
    )

    y_train = df["target"]

    model = RandomForestRegressor(

        n_estimators=n_estimators,

        max_depth=max_depth,

        random_state=random_state
    )

    model.fit(
        X_train,
        y_train
    )

    print("===================================")
    print("   MODEL TRAINING COMPLETED")
    print("===================================")
    print("Model: Random Forest Regressor")
    print(f"Estimators: {n_estimators}")
    print(f"Max Depth: {max_depth}")
    print("===================================")

    return model


def save_model(model):

    path = "model.pkl"

    joblib.dump(
        model,
        path
    )

    print(f"✅ Model saved to: {path}")


def main():

    params = load_params()

    df = load_training_data()

    model_params = params["model_building"]

    model = train_model(

        df,

        model_params["n_estimators"],

        model_params["max_depth"],

        model_params["random_state"]
    )

    save_model(model)


if __name__ == "__main__":
    main()
