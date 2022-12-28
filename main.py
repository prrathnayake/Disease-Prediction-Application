import pandas as pd
from sklearn import tree
import joblib

def returnColumns():
    training_data = pd.read_csv('Training.csv')
    X_train = training_data.drop(columns=['prognosis'])

    return X_train.columns.values.tolist()

def createArray(symptoms):
    columns = returnColumns()
    list = []

    for symptom in columns:
        bool = symptom in symptoms
        if bool == True:
            list.append(1)
        else:
            list.append(0)
    
    return list

def predictDisease(symptoms):
    array = createArray(symptoms)
    model = joblib.load('disease-prediction.joblib')

    prediction = model.predict([array])
    print(prediction)

predictDisease(['itching', 'chills'])