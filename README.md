# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages and print the present data.
2. Print the placement data and salary data.
3. Find the null and duplicate values.
4. Using logistic regression find the predicted values of accuracy , confusion matrices.

## Program and Output:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: NIKSHITHA G
RegisterNumber:  212223110031
*/
```
```
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

data=pd.read_csv("/content/Placement_Data.csv") 
data.head()
```
![Screenshot 2025-04-10 092948](https://github.com/user-attachments/assets/b1c2a378-6483-4335-8845-e134727f7aeb)

```
data1=data.copy() 
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()
```
![Screenshot 2025-04-10 093047](https://github.com/user-attachments/assets/f1cd4aea-f490-475d-9ee0-709e833b9dc1)

```
data1.isnull()
```
![Screenshot 2025-04-10 093146](https://github.com/user-attachments/assets/1465be90-73f0-4fe3-b109-2b749e928768)

```
data1.duplicated().sum()
```
![Screenshot 2025-04-10 093153](https://github.com/user-attachments/assets/db4985dc-f2d2-4039-b1e3-58561e795df8)

```
le = LabelEncoder()
cols = ["gender", "ssc_b", "hsc_b", "hsc_s", "degree_t", "workex", "specialisation", "status"]
for col in cols:
    data1[col] = le.fit_transform(data1[col])
data1
```
![Screenshot 2025-04-10 093221](https://github.com/user-attachments/assets/0621305c-498a-4b80-9073-cd118f071e5e)

```
x = data1.iloc[:, :-1]
x
y = data1["status"]
y
```
![Screenshot 2025-04-10 093348](https://github.com/user-attachments/assets/c79e2235-e526-4d3a-9fb0-28de323eacbd)

```
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
lr = LogisticRegression(solver="liblinear")
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)
y_pred
```
![Screenshot 2025-04-10 093405](https://github.com/user-attachments/assets/72291306-0315-400b-9e4e-8377c2fe3309)

```
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
```
![Screenshot 2025-04-10 093419](https://github.com/user-attachments/assets/9765d6c4-3c97-453d-9441-8c125af8ed32)

```
classification_report1 = classification_report(y_test, y_pred)
print(classification_report1)
```
![Screenshot 2025-04-10 093438](https://github.com/user-attachments/assets/4cc75bca-4f2a-4e4f-82e2-10701d12b1fb)

```
lr.predict([[1, 80, 1, 90, 1, 1, 90, 1, 0, 85, 1, 85]])
```
![Screenshot 2025-04-10 093501](https://github.com/user-attachments/assets/a1b83c6a-7cd9-4d79-b868-6865626d4ad6)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
