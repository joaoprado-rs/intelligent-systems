import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report


### Explorando e preparando os dados

dataset = pd.read_csv('dataset/football.csv')
dataset = dataset[['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']]

print(dataset['FTR'].value_counts())

#Gols do vencedor (W) e do perdedor (L). Equivalente às funções my_f1 e my_f2 do livro.
dataset['W'] = dataset[['FTHG', 'FTAG']].max(axis=1)
dataset['L'] = dataset[['FTHG', 'FTAG']].min(axis=1)

#Cada jogo vira dois registros: os gols do vencedor com alvo 1 (vitória) e os do perdedor com alvo 0 (derrota).
#O livro usa dtype=np.int, que foi removido do NumPy na versão 1.24.
gols = pd.concat([dataset['W'], dataset['L']], ignore_index=True).to_frame('gols')
variavel_alvo = pd.Series(np.concatenate([
np.ones(len(dataset), dtype=int), np.zeros(len(dataset), dtype=int)
]))

#Quantidade de vitórias e derrotas para cada número de gols
print(pd.crosstab(gols['gols'], variavel_alvo))


### Criando e treinando o modelo

#Holdout: 90% para treino e 10% para teste. O alvo voltou a ser uma classe, então o stratify mantém a proporção de vitórias e derrotas.
x_train, x_test, y_train, y_test = train_test_split(
gols, variavel_alvo, test_size=0.1, stratify=variavel_alvo, random_state=42
)

#StratifiedKFold: 10 folds, cada um com a mesma proporção de vitórias e derrotas.
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

model = LogisticRegression()

#KFold. O cross_val_predict faz o fit internamente em cada fold, numa cópia do modelo.
# pred_cv = cross_val_predict(model, gols, variavel_alvo, cv=kfold)

#Treinando o modelo com Holdout
model.fit(x_train, y_train)
pred = model.predict(x_test)

#Resultado do Holdout
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))

#Resultado do KFold
# print(confusion_matrix(variavel_alvo, pred_cv))
# print(classification_report(variavel_alvo, pred_cv))

#Coeficiente, intercepto e ponto de corte (número de gols com 50% de probabilidade de vitória)
print(model.coef_, model.intercept_)
print(-model.intercept_ / model.coef_[0])

#Probabilidade e previsão para 0 a 6 gols, no lugar do gráfico do livro
novos = pd.DataFrame({'gols': range(7)})
print(model.predict_proba(novos)[:, 1])
print(model.predict(novos))


ikf