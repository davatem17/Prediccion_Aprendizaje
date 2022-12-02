import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
#
import joblib as jb

#Extraccion de Datos
#leemos el cvs 
datos = pd.read_csv("Hora.csv")
dataframe = pd.DataFrame(datos)
print(datos)
X=(dataframe[["intervalo1","intervalo2","intervalo3","intervalo4","intervalo5","intervalo6"]])
y=(dataframe["Resultados"])

#Entrenamiento
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)
model = LogisticRegression()
model.fit(X_train,y_train)

#Ver q tambien aprendio el algoritmo
print(model.score(X_test,y_test))
#Guardar Modelo
jb.dump(model,'modelo_Hora.pkl')

