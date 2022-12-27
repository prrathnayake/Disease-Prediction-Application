import pandas as pd
from sklearn import tree
import joblib

testing_data = pd.read_csv('Testing.csv')
training_data = pd.read_csv('Training.csv')

training_data.drop('Unnamed: 133', axis=1, inplace=True)

X_train = training_data.drop(columns=['prognosis'])
y_train = training_data['prognosis']

model = joblib.load('disease-prediction.joblib')

tree.export_graphviz(model, out_file='disease-prediction.dot', feature_names=X_train.columns,
                     class_names=sorted(y_train.unique()), label='all')
