# Importamos librerias 
# Regresion Logistica
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
#Matriz de confusión
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import seaborn as sns


###Importar el dataset
dataset = pd.read_csv("Datos_encuestas.csv")
print(dataset.head())
#Promedio de quien aprendio
print(np.average(dataset.Promedio))
#Definir las variables x & y
x = dataset.iloc[:,[1,9]].values
y = dataset.iloc[:,10].values

###Separación de train y test
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 0)

###Escalado de variables
standard_x = StandardScaler()
x_train = standard_x.fit_transform(x_train)
x_test = standard_x.fit_transform(x_test)

###Entrenar el modelo de regresión logística
#Ajuste del modelo
reg = LogisticRegression(random_state = 0)
reg.fit(x_train,y_train)
#Predicción
pred = reg.predict(x_test)

###Evaluación
#Matriz de confusión
labels = [1, 0]
c = confusion_matrix(y_test, pred, labels = labels)
panda = pd.DataFrame(c, index = labels, columns = labels)
print(pd.DataFrame(c, index = labels, columns = labels))

#Grafica
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))
sns.heatmap(panda, annot=True, ax = ax1)
ax1.set_title('Matriz de Confusion')
ax1.set(xlabel = 'Verdaderos', ylabel = 'Predicción')


75/len(y_test)
X_set, y_set = x_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, reg.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('blue', 'black'))(i), label = j)
plt.title('Training')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
