import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
admissions=pd.read_csv("admissions.csv")
admissions.head(5)
admissions.shape[0]
admissions.isnull().sum()
admissions.dropna(subset=('gre','gpa','ranking'),inplace=True)
admissions.boxplot('gpa',by='admit')
admissions.boxplot('gre',by='admit')

kf = KFold(len(admissions), 3, shuffle=True, random_state=8)
lr = LogisticRegression()
accuracies = cross_val_score(lr,admissions[['gpa','gre','ranking']], admissions["admit"], scoring="accuracy", cv=kf)
accuracies.mean()
lr.fit(admissions[['gpa','gre','ranking']],admissions['admit'])
prediction=lr.predict(admissions[['gpa','gre','ranking']])
admissions['predicted_labels']=prediction
matches=admissions['admit']==admissions['predicted_labels']
correct_predictions=admissions[matches]
accuracy=len(correct_predictions)/admissions.shape[0]




