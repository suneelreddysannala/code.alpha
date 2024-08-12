import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression

file_path = 'C:/Users/sunee/Desktop/scoring/Project-1/a_Dataset_CreditScoring.xlsx'
with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
    data = file.read()

dataset = pd.read_excel('C:/Users/sunee/Desktop/scoring/Project-1/a_Dataset_CreditScoring.xlsx')

dataset.shape
dataset.head()
dataset=dataset.drop('ID',axis=1)
dataset.shape
dataset.isna().sum()
dataset=dataset.fillna(dataset.mean())
dataset.isna().sum()
y = dataset.iloc[:, 0].values
X = dataset.iloc[:, 1:29].values
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=0,
                                                    stratify=y)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
import joblib
joblib.dump(sc,'C:/Users/sunee/Desktop/scoring/Project-1/f2_Normalisation_CreditScoring')
classifier =  LogisticRegression()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
joblib.dump(classifier,'C:/Users/sunee/Desktop/scoring/Project-1/f1_Classifier_CreditScoring')
print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test,y_pred))
predictions = classifier.predict_proba(X_test)
predictions
f_prediction_prob = pd.DataFrame(predictions, columns = ['prob_0', 'prob_1'])
df_prediction_target = pd.DataFrame(classifier.predict(X_test), columns = ['predicted_TARGET'])
df_test_dataset = pd.DataFrame(y_test,columns= ['Actual Outcome'])

dfx=pd.concat([df_test_dataset, f_prediction_prob, df_prediction_target], axis=1)

# Save the DataFrame to an Excel file without the 'encoding' argument
dfx.to_excel("C:/Users/sunee/Desktop/scoring/Project-1/c1_Model_Prediction.xlsx", index=False)


dfx.head()
