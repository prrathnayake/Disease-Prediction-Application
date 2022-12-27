import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

testing_data = pd.read_csv('Testing.csv')
training_data = pd.read_csv('Training.csv')

training_data.drop('Unnamed: 133', axis=1, inplace=True)

X_train = training_data.drop(columns=['prognosis'])
y_train = training_data['prognosis']

X_test = testing_data.drop(columns=['prognosis'])
y_test = testing_data['prognosis']

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

prediction = model.predict(X_test)

score = accuracy_score(y_test,prediction)

print(score)


