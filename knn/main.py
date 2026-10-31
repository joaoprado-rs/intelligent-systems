import pandas as pd               # limpeza/análise de dados
import numpy as np                # computação científica/análise de dados
import matplotlib.pyplot as plt   # visualização de dados
import seaborn as sns             # gráficos
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def normalize(x):
    return (x - x.min()) / (x.max() - x.min())


# O dataset não tem o nome das colunas, é necessário definir manualmente.
colunas = [
    "ID", "Diagnosis",
    "radius1", "texture1", "perimeter1", "area1", "smoothness1",
    "compactness1", "concavity1", "concave_points1", "symmetry1", "fractal_dimension1",
    "radius2", "texture2", "perimeter2", "area2", "smoothness2",
    "compactness2", "concavity2", "concave_points2", "symmetry2", "fractal_dimension2",
    "radius3", "texture3", "perimeter3", "area3", "smoothness3",
    "compactness3", "concavity3", "concave_points3", "symmetry3", "fractal_dimension3",
]

# Carregando o dataset
dataset = pd.read_csv('dataset/wdbc.data', header=None, names=colunas)

# Removendo a coluna ID
dataset = dataset.drop(columns=['ID'])

# Trocando os rótulos da coluna de diagnóstico de B/M para Benigno/Maligno
dataset["Diagnosis"] = dataset["Diagnosis"].replace({"M": "Maligno", "B": "Benigno"})

dataset_normalizado = dataset.iloc[:, 1:32].apply(normalize)

# Alternativa: padronização z-score
# z_score = StandardScaler()
# dataset_normalizado = z_score.fit_transform(dataset.iloc[:, 1:32])

### Preparação dos dados ###

# Dividindo o dataset em treino e teste na proporção 80/20
x = dataset_normalizado
y = dataset["Diagnosis"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

### Treinando o modelo ###

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)
pred = knn.predict(x_test)

print(classification_report(y_test, pred))
print(confusion_matrix(y_test, pred))
print(accuracy_score(y_test, pred))