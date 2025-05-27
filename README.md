# Titanic_eda
# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os
print("Current working directory:", os.getcwd())

# Load data
df = pd.read_csv('C:/Users/katta/OneDrive/Python-ai/Titanic.csv')

# Show first few rows
df.head()

# Basic info and nulls
print(df.info())
print(df.isnull().sum())

# Summary statistics
print(df.describe())

# Fill missing Age with median
df['Age'].fillna(df['Age'].median(), inplace=True)

# Fill Embarked with mode
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Drop Cabin (too many missing values)
df.drop(columns='Cabin', inplace=True)

# Histogram for numeric features
df.hist(figsize=(12, 10))
plt.tight_layout()
plt.show()

# Boxplots
plt.figure(figsize=(8, 6))
sns.boxplot(data=df[['Age', 'Fare']])
plt.title("Boxplots for Age and Fare")
plt.show()

# Correlation matrix
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# Pairplot
sns.pairplot(df[['Survived', 'Pclass', 'Age', 'Fare']])
plt.show()

# Countplots
sns.countplot(data=df, x='Survived', hue='Sex')
plt.title("Survival Count by Sex")
plt.show()

# Pie chart using Plotly
fig = px.pie(df, names='Pclass', title='Passenger Class Distribution')
fig.show()
