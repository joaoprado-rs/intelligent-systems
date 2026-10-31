import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score

dataset = pd.read_csv('dataset/insurance.csv')

variavel_classificadora = dataset['charges']
dataset = dataset.drop(columns=['charges'])

#Modelos de regressao necessitam de valores numericos e features como region e sex tem uma string comp conteudo.
#O método 'get_dummies' do pandas serve para categorizar essas variaveis de forma binária.
dataset = pd.get_dummies(dataset, drop_first=True)

dataset["age2"] = dataset["age"] ** 2
dataset["bmi30"] = (dataset["bmi"] >= 30).astype(int)
dataset["bmi30_smoker"] = dataset["bmi30"] * dataset["smoker_yes"]

x_train,x_test,y_train,y_test= train_test_split(
 dataset,variavel_classificadora,test_size=0.1,random_state=42,
)

modelo = LinearRegression().fit(x_train, y_train)

scores = cross_val_score(LinearRegression(), dataset, variavel_classificadora,
                         cv=10, scoring="r2")

print(scores.mean(), scores.std())

