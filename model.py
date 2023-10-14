import pandas as pd
from sklearn.linear_model import LinearRegression

data = pd.read_csv(r"C:\Users\aarus\coding\kaggle\Udacity_SVM\data.csv")

X = data[['YearsExperience']]
y = data.Salary

model = LinearRegression()
model.fit(X,y)

print(model.predict(8.5))



