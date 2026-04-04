import os
import json
import argparse

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

DATA_DIR          = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_DIR         = os.path.join(os.path.dirname(__file__), "..", "models")
MLFLOW_EXPERIMENT = "imdb-sentiment"

def load_data(sample=None):
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test_df  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

    if sample:
        train_df = train_df.sample(n=sample, random_state=42).reset_index(drop=True)
        test_df  = test_df.sample(n=sample // 5, random_state=42).reset_index(drop=True)

    X_train = train_df["text"].tolist()
    y_train = train_df["label"].tolist()
    X_test  = test_df["text"].tolist()
    y_test  = test_df["label"].tolist()

    return X_train, y_train, X_test, y_test

def build_pipeline(max_features, C, max_iter):
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            sublinear_tf=True,
            stop_words="english",
        )),
        ("clf", LogisticRegression(
            C=C,
            max_iter=max_iter,
            solver="lbfgs",
            random_state=42,
        )),
    ])

def evaluate(pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)

    metrics = {
        "accuracy"  : round(accuracy_score(y_test, y_pred), 4),
        "f1"        : round(f1_score(y_test, y_pred), 4),
        "precision" : round(precision_score(y_test, y_pred), 4),
        "recall"    : round(recall_score(y_test, y_pred), 4),
    }

    print(classification_report(y_test, y_pred, target_names=["negative", "positive"]))
    return metrics

def train(args):
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    print("Loading data...")
    X_train, y_train, X_test, y_test = load_data(sample=args.sample)

    params = {
        "max_features" : args.max_features,
        "C"            : args.C,
        "max_iter"     : args.max_iter,
        "train_size"   : len(X_train),
        "test_size"    : len(X_test),
    }

    pipeline = build_pipeline(
        max_features=args.max_features,
        C=args.C,
        max_iter=args.max_iter,
    )

    with mlflow.start_run() as run:
        pipeline.fit(X_train, y_train)
        metrics = evaluate(pipeline, X_test, y_test)

        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            registered_model_name="imdb-sentiment-classifier",
        )

        os.makedirs(MODEL_DIR, exist_ok=True)
        metrics_path = os.path.join(MODEL_DIR, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump({**metrics, "run_id": run.info.run_id}, f, indent=2)

        print(f"Run ID     : {run.info.run_id}")
        print(f"Accuracy   : {metrics['accuracy']}")
        print(f"F1         : {metrics['f1']}")

    return metrics

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--max-features", type=int,   default=50000)
    p.add_argument("--C",            type=float, default=1.0)
    p.add_argument("--max-iter",     type=int,   default=100)
    p.add_argument("--sample",       type=int,   default=None)
    return p.parse_args()

if __name__ == "__main__":
    train(parse_args())