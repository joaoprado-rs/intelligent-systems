import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt


### Explorando e preparando os dados

dataset = pd.read_csv('dataset/credit.csv')

variavel_classificadora = dataset['default']
dataset = dataset.drop(columns=['default'])

#A classe DecisionTree do sklearn não aceita strings, e várias colunas são categóricas em texto (checking_balance, savings_balance, purpose, etc).
#O get_dummies transforma cada categoria em uma coluna, ex: o registro com savings_balance = > 1000 DM fica com savings_balance_> 1000 DM = true e savings_balance_< 100 DM = false
dataset = pd.get_dummies(dataset)


### Criando e treinando o modelo

#O parametro 'random_state' automaticamente randomiza o dataset com base nessa seed. Nesse caso, estamos usando 42. Precisamos necessariamente sortar as linhas para evitar a possibilidade de
# treinar e testar o modelo com amostras muito diferentes entre si.

#Holdout: 90% para treino e 10% para teste, como sugerido pelo Lantz. O train_test_split embaralha as linhas antes de dividir, para evitar treino e teste muito
#diferentes entre si; o random_state é a seed desse embaralhamento (42) e o stratify mantém a proporção nos dois conjuntos.
x_train, x_test, y_train, y_test = train_test_split(
dataset, variavel_classificadora, test_size=0.1, random_state=42, stratify=variavel_classificadora
)

#K-Fold (pedido no enunciado do lab): divide o dataset em 10 partes e treina/testa 10 vezes, cada vez usando uma parte como teste e as outras 9 como treino.
#n_splits é o número de partes (folds), shuffle embaralha antes de dividir e random_state é a seed. O Stratified mantém a proporção das classes em cada fold.
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)


model = DecisionTreeClassifier(
criterion='entropy',
max_depth=6, # para evitar overfitting
min_samples_split=10,
random_state=42 )

#KFold. Não chamamos o fit: o cross_val_score e o cross_val_predict fazem o fit internamente em cada fold, numa cópia do modelo.
scores = cross_val_score(model, dataset, variavel_classificadora, cv=kfold, scoring='accuracy')
pred_cv = cross_val_predict(model, dataset, variavel_classificadora, cv=kfold)

#Treinando o modelo com Holdout. Mesmo método usado no KNN último lab.
model.fit(x_train, y_train)
pred = model.predict(x_test)

#Resultado do Holdout
print(classification_report(y_test, pred))
print(confusion_matrix(y_test, pred))

#Resultado do KFold
#print(classification_report(variavel_classificadora, pred_cv))
#print(confusion_matrix(variavel_classificadora, pred_cv))


plt.figure(figsize=(24, 10))
plot_tree(model, feature_names=dataset.columns, class_names=model.classes_.astype(str),
          filled=True, max_depth=3, fontsize=8)
plt.savefig('arvore.png', dpi=200, bbox_inches='tight')
plt.show()