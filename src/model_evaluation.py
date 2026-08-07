
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


def load_model():

    return joblib.load(
        "model.pkl"
    )


def load_test_data():

    return pd.read_csv(
        "data/features/test.csv"
    )


def evaluate_model(
    model,
    df
):

    X_test = df.drop(
        columns=["target"]
    )

    y_test = df["target"]

    y_pred = model.predict(
        X_test
    )

    mae = mean_absolute_error(
        y_test,
        y_pred
    )

    mse = mean_squared_error(
        y_test,
        y_pred
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        y_test,
        y_pred
    )

    metrics = {

        "MAE": mae,

        "MSE": mse,

        "RMSE": rmse,

        "R2": r2
    }

    return metrics


def save_metrics(metrics):

    path = "metrics.json"

    with open(
        path,
        "w"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4
        )

    print("\n===================================")
    print("      REGRESSION EVALUATION")
    print("===================================")

    for name, value in metrics.items():

        print(
            f"{name:10s}: {value:.4f}"
        )

    print("===================================")
    print(f"✅ Metrics saved to: {path}")


def main():

    model = load_model()

    test_data = load_test_data()

    metrics = evaluate_model(
        model,
        test_data
    )

    save_metrics(metrics)


if __name__ == "__main__":
    main()
