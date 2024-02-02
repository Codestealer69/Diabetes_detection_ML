import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score


# To read the csv file whice is stored in the same file location named Sayan.py
data = pd.read_csv("diabetes.csv")
# To display top 5 rows of the data set
print(data.head(5))
# To know the shape of our dataset we use shape attribute of pandas
print(data.shape)
print("Total no. of Rows",data.shape[0])
print("Total no. of columns",data.shape[1])
# To get detailed info about dataset we use info method of pandas detaframe
print(data.info())
# To check if there is any null value in my dataset
print(data.isnull().sum())
# To get overall statistics of our dataset we use describe method
print(data.describe())
# Now we are checking how many '0' and '1' are there in Outcome column
print(data['Outcome'].value_counts())
# By the Outcome column diffentiate other columns with their mean value
print(data.groupby('Outcome').mean())
# Now make independent variable in x axis and dependent variable in y axis
x=data.drop('Outcome',axis=1)
y=data['Outcome']
print(x)
print(y)
scaler=StandardScaler()
scaler.fit(x)
Standardized_data=scaler.transform(x)
print(Standardized_data)
x=Standardized_data
y=data['Outcome']
print(x)
print(y)

# Splitting our dataset in training anf testing dataset
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,stratify=y,random_state=2)
print(x.shape,x_train.shape,x_test.shape)
# Training the model
classifier = svm.SVC(kernel='linear')
classifier.fit(x_train,y_train)
# Accuracy score
x_train_pred=classifier.predict(x_train)
training_data_accuracy=accuracy_score(x_train_pred,y_train)
print("Accuracy score of the training data : ",training_data_accuracy )
# making a predictive system
input_data=(5,166,72,19,175,25.8,0.587,51)
input_data_as_np=np.asarray(input_data)
input_data_reshaped=input_data_as_np.reshape(1,-1)
std_data=scaler.transform(input_data_reshaped)
print(std_data)
prediction = classifier.predict(std_data)
print(prediction)
if(prediction[0]==0):
    print("The person is not diabetec")
else:
    print("The person is diabetic")
