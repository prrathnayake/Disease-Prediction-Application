import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib

testing_data = pd.read_csv('Testing.csv')
training_data = pd.read_csv('Training.csv')

# training_data.drop('Unnamed: 133', axis=1, inplace=True)

X_train = training_data.drop(columns=['prognosis'])
y_train = training_data['prognosis']

X_test = testing_data.drop(columns=['prognosis'])
y_test = testing_data['prognosis']

model = DecisionTreeClassifier()
model.fit(X_train.values, y_train)

joblib.dump(model, 'disease-prediction.joblib')


