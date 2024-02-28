import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import graphviz

def changeField(quality):
  if quality <= 4:
    return 0
  if quality > 4 and quality <= 7:
    return 1
  if quality >= 8:
    return 2

def showScore(scores):
  for c, score in enumerate(scores):
    print(f'{c + 1}º teste: {(score * 100):.2f}%')
  print(f'Média: {(scores.mean() * 100):.2f}%')

def convertResult(result):
  if result[0] == 0:
    return 'ruim'
  if result[0] == 1:
    return 'boa'
  if result[0] == 2:
    return 'ótima'
  
def create_tree_pdf(dtree, features):
    dot_data = tree.export_graphviz(dtree, out_file=None, 
    feature_names=features, filled=True, rounded=True,  
    special_characters=True)  
    graph = graphviz.Source(dot_data)  
    graph.render("decision_tree", format='pdf') 

def CreatTree():
  df = pd.read_csv("WineQT.csv")
  
  df['quality'] = df['quality'].apply(changeField)
  
  features = ["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","density","pH","sulphates","alcohol"]
  variables = df[features]
  response = df["quality"]
  dtree = DecisionTreeClassifier(max_depth=5, min_samples_leaf=57)
  scores = cross_val_score(dtree, variables, response, cv=10)
  dtree = dtree.fit(variables, response)
  tree.plot_tree(dtree, feature_names=features, filled=True)

  return dtree, scores, features

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

dtree, scores, features = CreatTree()
while True:
  menu()
  input_user = validationInput("Digite a opção desejada: ")
  match(input_user):
    case 0:
        print('Digite os valores das variáveis para teste: ')
        fixed_acidity = validationInput("Acidez fixa: ")
        volatile_acidity = validationInput("Acidez volátil: ")
        citric_acid = validationInput("Ácido cítrico: ")
        residual_sugar = validationInput("Açúcar residual: ")
        chlorides = validationInput("Cloretos: ")
        density = validationInput("Densidade: ")
        pH = validationInput("Ph: ")
        sulphates = validationInput("Sulfatos: ")
        alcohol = validationInput("Álcool: ")
        prediction = dtree.predict([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, density, pH, sulphates, alcohol]])
        print(f'A qualidade do vinho é {convertResult(prediction)}')
    case 1:
      create_tree_pdf(dtree, features)
    case 2:
      showScore(scores)
    case 3:
      break
