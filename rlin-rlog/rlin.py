import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, KFold, cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error


### Explorando e preparando os dados

dataset = pd.read_csv('dataset/insurance.csv')

variavel_alvo = dataset['charges']
dataset = dataset.drop(columns=['charges'])

#A LinearRegression do sklearn não aceita strings. O get_dummies transforma cada categoria em uma coluna 0/1.
dataset = pd.get_dummies(dataset, drop_first=True)

#Melhorias sugeridas pelo Lantz. A idade ao quadrado, indicador de obesidade (IMC >= 30) e interação obesidade x fumante.
#Esse trecho será comentado pois o Lantz divide em modelos diferentes.
dataset['age2'] = dataset['age'] ** 2
dataset['bmi30'] = (dataset['bmi'] >= 30).astype(int)
dataset['bmi30_smoker'] = dataset['bmi30'] * dataset['smoker_yes']


### Criando e treinando o modelo

#Holdout: 90% para treino e 10% para teste. Sem stratify, pois o alvo (charges) é contínuo e não tem classes.
x_train, x_test, y_train, y_test = train_test_split(
dataset, variavel_alvo, test_size=0.1, random_state=42
)

#K-Fold: divide o dataset em 10 partes e treina/testa 10 vezes, cada vez usando uma parte como teste.
kfold = KFold(n_splits=10, shuffle=True, random_state=42)

model = LinearRegression()

#KFold. Não chamamos o fit: o cross_val_predict faz o fit internamente em cada fold, numa cópia do modelo.
# pred_cv = cross_val_predict(model, dataset, variavel_alvo, cv=kfold)

#Treinando o modelo com Holdout
model.fit(x_train, y_train)
pred = model.predict(x_test)

#Resultado do Holdout. Como é regressão, usamos R², MAE e RMSE no lugar da matriz de confusão.
print(r2_score(y_test, pred))
print(mean_absolute_error(y_test, pred))
print(root_mean_squared_error(y_test, pred))

#Resultado do KFold
# print(r2_score(variavel_alvo, pred_cv))
# print(mean_absolute_error(variavel_alvo, pred_cv))
# print(root_mean_squared_error(variavel_alvo, pred_cv))

#Coeficientes, equivalente ao summary() do Lantz (só funciona com o Holdout, pois o KFold treina cópias do modelo)
print(pd.Series(model.coef_, index=dataset.columns))