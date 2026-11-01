import argparse
import warnings

import mlflow
import mlflow.pyfunc

from train import load_data  # memakai pemrosesan data yang sama dengan pelatihan

warnings.filterwarnings("ignore", category=FutureWarning)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tracking_uri", default="sqlite:///mlflow.db")
    p.add_argument("--model_name", default="pm25-xgboost")
    p.add_argument("--data_path", default="data/processed/features.csv")
    p.add_argument("--n_rows", type=int, default=3)
    args = p.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    X, y = load_data(args.data_path)
    X_new, y_true = X.tail(args.n_rows), y.tail(args.n_rows)

    uris = {
        "stage Production": f"models:/{args.model_name}/Production",
        "alias @production": f"models:/{args.model_name}@production",
    }
    for label, uri in uris.items():
        model = mlflow.pyfunc.load_model(uri)
        preds = model.predict(X_new)
        info = model.metadata
        print(f"\n[{label}] {uri}")
        print(f"  run_id={info.run_id} | jumlah input={X_new.shape[1]}")
        for i, (pr, tr) in enumerate(zip(preds, y_true), 1):
            print(f"  baris {i}: prediksi PM2.5 (+3 jam) = {pr:.2f} | aktual = {tr:.2f}")
    print("\n[OK] Model Production berhasil dimuat dan dipakai untuk inferensi.")


if __name__ == "__main__":
    main()
