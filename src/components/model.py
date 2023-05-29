import os 
import sys 
from exception import CustomException
from logger import logging
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
from xgboost import XGBClassifier
from xgboost import plot_importance
from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    model_obj_file_path = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_transforamtion_config = ModelTrainerConfig()

    def initiate_model(self, train_data_array, test_data_array):
        pass