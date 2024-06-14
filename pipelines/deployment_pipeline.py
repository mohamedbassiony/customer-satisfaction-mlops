import json 

# from .utils import get_data_for_test
import os

import numpy as np
import pandas as pd
from materializer.custom_materializer import cs_materializer
from steps.cleaning_data import clean_data
from steps.evaluation import evaluate_model
from steps.ingest_data import ingest_data
from steps.model_train import train_model
