from zenml import pipeline

from steps.cleaning_data import clean_df
from steps.evaluation import evaluate_model
from steps.ingest_data import ingest_data
from steps.model_train import train_model

@pipeline
def train_pipeline(data_path: str):
    df = ingest_data()
    clean_df(df)
    train_model(df)
    evaluate_model(df)