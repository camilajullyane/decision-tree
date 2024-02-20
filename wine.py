import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("WineQT.csv")

print(df)

features = ["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol","Id"]
variables = df[features]
response = df["quality"]
dtree = DecisionTreeClassifier()
dtree = dtree.fit(variables, response)
tree.plot_tree(dtree, feature_names=features)

plt.show()