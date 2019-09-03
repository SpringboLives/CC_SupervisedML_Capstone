import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from collections import Counter


df = pd.read_csv("profiles.csv")
pd.set_option('display.max_columns', None)


plt.figure(figsize=(6, 9), dpi=72)
grid = plt.GridSpec(3, 2, wspace=0.4, hspace=0.6)

#Range of ages of users
data = df.age
ax = plt.subplot(grid[0, :2])
plt.hist(data, bins=20, range=(16, 85))
plt.xlabel('Age in Years')
plt.ylabel('Number of Users')
ax.set_xticks(range(10, 90, 10))
plt.title('Ages of OKCupid Users')

#Percentage of Male vs Female Users
labels = df['sex'].value_counts().keys().tolist()
values = df['sex'].value_counts().tolist()
df.dropna(subset=['sex'], inplace=True)
plt.subplot(grid[1, :2])
plt.pie(values, labels=labels, autopct='%0.2f%%')
plt.title("Percent of Male vs. Female users")

#Range of Incomes of Users
data = df.income
ax = plt.subplot(grid[2, :2])
plt.hist(data, bins=10, range=(20000, 105000)) #Trying to remove outliers
plt.xlabel("Income in USD")
plt.ylabel("Number of Users")
plt.show()

print(df.body_type.value_counts()['average'] / len(df.body_type))
