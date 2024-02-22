# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn import tree
# from sklearn.tree import DecisionTreeClassifier

# df = pd.read_csv("WineQT.csv")

# print(df)

# features = ["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol","Id"]
# variables = df[features]
# response = df["quality"]
# dtree = DecisionTreeClassifier(max_depth=5)
# dtree = dtree.fit(variables, response)
# tree.plot_tree(dtree, feature_names=features)

# plt.show()
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

df = pd.read_csv("WineQT.csv")

features = ["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol","Id"]
variables = df[features]
response = df["quality"]

dtree = DecisionTreeClassifier(max_depth=5)

# Aplicando a validação cruzada de 10-folds
scores = cross_val_score(dtree, variables, response, cv=10)

print("Scores de cada execução da validação cruzada: ", scores)
print("Média dos scores: ", scores.mean())

dtree = dtree.fit(variables, response)
tree.plot_tree(dtree, feature_names=features)

plt.show()