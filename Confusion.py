import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np
import joblib as jb

#Extraccion de Datos
#leemos el cvs 
datos = pd.read_csv("Libro2.csv")
dataframe = pd.DataFrame(datos)
print(datos)
X=(dataframe[["intervalo1","intervalo2","intervalo3","intervalo4","intervalo5","intervalo6"]])
y=(dataframe["Resultados"])

#Entrenamiento
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.05,random_state=0)
model = LogisticRegression()
model.fit(X_train,y_train)

#Ver q tambien aprendio el algoritmo
print(model.score(X_test,y_test))

#Guardar Modelo
jb.dump(model,'modelo_Hora.pkl')

# Confusion Matrix
#from sklearn.metrics import confusion_matrix
#confusion_matrix(y_test, prediccion)
#print(confusion_matrix(y_test, prediccion))
# Accuracy
#from sklearn.metrics import accuracy_score
#accuracy_score(y_true, y_pred)
# Recall
#from sklearn.metrics import recall_score
#recall_score(y_true, y_pred, average=None)
# Precision
#from sklearn.metrics import precision_score
#precision_score(y_true, y_pred, average=None)

# Method 1: sklearn
#from sklearn.metrics import f1_score
#f1_score(y_test, prediccion, average=None)
# Method 2: Manual Calculation
#F1 = 2 * (precision * recall) / (precision + recall)
# Method 3: Classification report [BONUS]
#from sklearn.metrics import classification_report
#print(classification_report(y_true, y_pred, target_names=target_names))

#tn, fp, fn, tp = confusion_matrix(pred, y_test).ravel()

#c = confusion_matrix(y_test, pred)
#np.flip(c.T)
#print(c)
#print(np.flip(c.T))

pred = model.predict(X_test)

confusion_matrix(y_test, pred, labels = [1, 0])
print(confusion_matrix(y_test, pred, labels = [1, 0]))