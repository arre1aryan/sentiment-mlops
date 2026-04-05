import mlflow
import pickle
import os

mlflow.set_tracking_uri("./mlruns")

model = mlflow.sklearn.load_model(
    "./mlruns/265397782536417658/40fc770893bb4fee9fd875036fec7b24/artifacts/model"
)

os.makedirs("models", exist_ok=True)

with open("models/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved to models/model.pkl")