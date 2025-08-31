# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required packages and print the present data.

2.Print the placement data and salary data. 

3.Find the null and duplicate values.

4.Using logistic regression find the predicted values of accuracy , confusion matrices. 

5.Display the results.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: SHIVANI M
RegisterNumber: 212224040313
*/
```
```
import pandas as pd
data=pd.read_csv(r"C:\Users\admin\Desktop\Placement_Data.csv")
data.head()
data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()
data1.isnull().sum()
data1.duplicated().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1
x=data1.iloc[:,:-1]
x
y=data1["status"]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
print(y_pred)
print()
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
print(accuracy)
print()
from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
print(confusion)
print()
from sklearn.metrics import classification_report
classification_report1=classification_report(y_test,y_pred)
print(classification_report1)
print("NAME : SHIVANI M")
print("Register No: 212224040313")
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])


```

## Output:
### Head
<img width="681" height="154" alt="image" src="https://github.com/user-attachments/assets/7abb369c-83a8-4f43-9810-dc4b53b63538" />

### Copy
<img width="668" height="136" alt="image" src="https://github.com/user-attachments/assets/6dde5fb4-338e-45d2-92e1-1e9c6ed23021" />

### Fit Transform
<img width="637" height="288" alt="image" src="https://github.com/user-attachments/assets/33c37a73-788b-45c2-af8b-d28883fce4a3" />

### X
<img width="604" height="283" alt="image" src="https://github.com/user-attachments/assets/3d32cf97-d77a-4021-a1d7-8358cde34b27" />

### Y
<img width="294" height="173" alt="image" src="https://github.com/user-attachments/assets/a5a97b5b-aaa7-4679-828c-4bea3dda2092" />

### Logistic Regression
<img width="486" height="37" alt="image" src="https://github.com/user-attachments/assets/96718922-2651-4828-ab73-7c0ec5ef12f7" />

### Accuracy
<img width="219" height="25" alt="image" src="https://github.com/user-attachments/assets/2d36dcbc-e524-4ad5-9893-20022c22fbb0" />

### Confusion
<img width="214" height="38" alt="image" src="https://github.com/user-attachments/assets/2b13a920-ebf9-431f-9bd4-0e6925562c87" />

### Classification Report
<img width="686" height="263" alt="Screenshot 2025-08-31 130048" src="https://github.com/user-attachments/assets/e844d3e4-31a8-4e37-8ec8-08c15f7a1c81" />

### Prediction 

<img width="668" height="109" alt="image" src="https://github.com/user-attachments/assets/9dbd3780-3146-414a-88fe-39f5a47c6dda" />









## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
