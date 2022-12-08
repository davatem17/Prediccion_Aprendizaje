import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
#
import joblib as jb

dataset = pd.read_csv("General.csv")

print(dataset.head())
print(np.average(dataset.Resultados))

#Definir las Variables X & Y
x=(dataset[["intervalo1","intervalo2","intervalo3","intervalo4","intervalo5"]])
y=(dataset["Resultados"])

#Entrenamiento
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=0)

reg = LogisticRegression(random_state=0)
reg.fit(X_train,y_train)

#Probar el Modelo
dataNew = {'intervalo1': [1,1,0],
            'intervalo2': [0,0,1],
            'intervalo3': [0,1,1],
            'intervalo4': [0,0,1],
            'intervalo5': [0,1,1]
}

#participantesCharlasNew = pd.DataFrame(dataNew, columns=['intervalo1','intervalo2','intervalo3','intervalo4','intervalo5'])
#prediccion=reg.predict(participantesCharlasNew)
#print(participantesCharlasNew)
#print(prediccion)

prediccion=reg.predict(X_test)
print(X_test)
print(prediccion)


#Matriz de Confusion
confm = confusion_matrix(y_test,prediccion)
print(confm)

columnas = ['clase']