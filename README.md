![2 6](https://github.com/user-attachments/assets/8456c251-304e-46c4-bb44-b0c7491d32ae)
![2 5](https://github.com/user-attachments/assets/5ccea793-4360-41ae-8948-cc9d54012569)
![2 4](https://github.com/user-attachments/assets/33bce231-ee67-4293-92ee-b57e5afabe9e)
![2 3](https://github.com/user-attachments/assets/9baeb541-9a73-49a3-9981-e8dc329cc88c)
![2 2](https://github.com/user-attachments/assets/fe8efd4f-9a91-4209-9d47-4b1b18638845)
![2 1](https://github.com/user-attachments/assets/3ea2c6ae-7ced-4587-91a2-71ca8b62d087)
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
