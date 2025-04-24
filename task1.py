import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

df = pd.read_csv(r"C:\Users\AKSHAYA\Downloads\archive(1)\Titanic-Dataset.csv")


print(df.info())  
print(df.describe())  
print(df.isnull().sum()) 

df['Age'] = df['Age'].fillna(df['Age'].median())  
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])  

df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)
scaler = StandardScaler()
num_cols = ['Age', 'Fare', 'SibSp', 'Parch']
df[num_cols] = scaler.fit_transform(df[num_cols])

for col in num_cols:
    plt.figure(figsize=(6, 3))
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot of {col}')
    plt.show()

print(df.head())  
print(df.isnull().sum())
print(df.info())  
print(df.describe())  
print(df.isnull().sum()) 

df['Age'] = df['Age'].fillna(df['Age'].median())  
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])  

df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)
scaler = StandardScaler()
num_cols = ['Age', 'Fare', 'SibSp', 'Parch']
df[num_cols] = scaler.fit_transform(df[num_cols])

for col in num_cols:
    plt.figure(figsize=(6, 3))
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot of {col}')
    plt.show()

print(df.head())  
print(df.isnull().sum())