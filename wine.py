import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

def changeField(quality):
  if quality <= 4:
    return 0
  if quality > 4 and quality <= 7:
    return 1
  if quality >= 8:
    return 2

def CreatTree():
  df = pd.read_csv("WineQT.csv")
  
  df['quality'] = df['quality'].apply(changeField)
  
  features = ["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","density","pH","sulphates","alcohol"]
  variables = df[features]
  response = df["quality"]
  dtree = DecisionTreeClassifier(max_depth=3)
  scores = cross_val_score(dtree, variables, response, cv=10)
  dtree = dtree.fit(variables, response)
  tree.plot_tree(dtree, feature_names=features, filled=True)

  return dtree, scores

def validationInput(txt):
  while True:
    try:
      user_input = float(input(txt))
    except:
      print('Erro de entrada. ', end='')
    else:
      return user_input

def menu():
  print("0 - Testar modelo")
  print("1 - Exibir gráfico de árvore de decisão")
  print("2 - Mostrar a precisão do modelo")
  print("3 - Sair")

dtree, scores = CreatTree()
while True:
  menu()
  input_user = validationInput("Digite a opção desejada: ")
  match(input_user):
    case 0:
        print('Digite os valores das variáveis para teste: ')
        fixed_acidity = validationInput("")
        volatile_acidity = validationInput("")
        citric_acid = validationInput("")
        residual_sugar = validationInput("")
        chlorides = validationInput("")
        density = validationInput("")
        pH = validationInput("")
        sulphates = validationInput("")
        alcohol = validationInput("")
        prediction = dtree.predict([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, density, pH, sulphates, alcohol]])
    case 1:
      plt.show()
    case 2:
      print("Scores de cada execução da validação cruzada: ", scores)
      print("Média dos scores: ", scores.mean())
    case 3:
      break
