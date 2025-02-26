# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.
## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
#### 1.Import the standard Libraries. 
#### 2.Set variables for assigning dataset values. 
#### 3.Import linear regression from sklearn. 
#### 4.Assign the points for representing in the graph. 
#### 5.Predict the regression for marks by using the representation of the graph. 
#### 6.Compare the graphs and hence we obtained the linear regression for the given datas.
## Program:
### RANJANI A(212223230170)
```py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```
## Output:
### Dataset
![Screenshot 2025-02-26 092959](https://github.com/user-attachments/assets/c6778f5e-0a5d-47a7-85f6-ecbf7ccecf6d)

### Head Values
![Screenshot 2025-02-26 093009](https://github.com/user-attachments/assets/883e4eff-bb64-4dbd-b166-142e04e26ccf)

### Tail Values
![Screenshot 2025-02-26 093015](https://github.com/user-attachments/assets/abb2fc2d-7138-4c2e-a462-b211b9ac0fb5)

### X and Y values
![Screenshot 2025-02-26 093043](https://github.com/user-attachments/assets/5802b9a1-4e8e-4fb0-a6bd-95342d6a0db6)

### Predication values of X and Y
![Screenshot 2025-02-26 093104](https://github.com/user-attachments/assets/f73f92d8-3b0d-452d-9751-55fd82e6abbc)

### MSE,MAE and RMSE
![Screenshot 2025-02-26 093111](https://github.com/user-attachments/assets/fc649d0e-a6dc-4a08-a4a5-e0b94c958e94)

### Training Set
![Screenshot 2025-02-26 093123](https://github.com/user-attachments/assets/c62b3949-4c35-43ea-bcbe-4a9c730a0f4b)

### Testing Set

![Screenshot 2025-02-26 093130](https://github.com/user-attachments/assets/dba19b98-2961-4615-a180-f64c953982bf)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
