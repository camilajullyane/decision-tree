import pandas
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
# Lendo e tratando o arquivo CSV
df = pandas.read_csv("train.csv")
d = {'male': 1, 'female': 0}
df['Sex'] = df['Sex'].map(d)

def deadOrNot(pred): 
    return 'viver' if pred[0] == 0 else 'morrer'

def madeDecisionTree():
    features = ["Pclass", "Sex", "Age"]
    variables = df[features]
    response = df["Survived"]
    dtree = DecisionTreeClassifier()
    dtree = dtree.fit(variables, response)
    tree.plot_tree(dtree, feature_names=features)

    return dtree

def showResults():
    if len(dict) > 0:
        for k, v in dict.items():
            print(f'{k} irá {v}')
    else:
        print('Não houveram simulações')

def menu():
    print('1 - Começar simulação\n2 - Ver resultador anteriores\n3 - Sair')

dict = {}
dtree = madeDecisionTree()
print('Seja Bem Vindo ao simulador do Titanic.\nVocê sobreveria?')
while True:
    menu()
    choice = int(input('Digite sua escolha: '))
    match choice:
        case 1:
            nome = str(input('Nome: '))
            age = int(input("Idade: "))
            pclass = int(input('Classe Social: '))
            sex = int(input('Sexo: '))
            predictions = deadOrNot(dtree.predict([[pclass, sex, age]]))
            dict[nome] = predictions
        case 2:
            showResults()
        case 3:
            print('Obrigado pela participação. Espero que você tenha sobrevivido!')
            break
