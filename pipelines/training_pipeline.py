from zenml import pipeline

from steps.cleaning_data import clean_data
from steps.evaluation import evaluate_model
from steps.ingest_data import ingest_data
from steps.model_train import train_model

@pipeline(enable_cache=True)
def train_pipeline():
    df = ingest_data()
    X_train, X_test, y_train, y_test = clean_data(df)
    model = train_model(X_train, X_test , y_train ,y_test)
    r2_score, rmse = evaluate_model(model, X_test, y_test)
