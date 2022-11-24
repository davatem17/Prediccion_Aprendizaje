import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import linear_model

#leemos el csv
datos = pd.read_csv("DataParticipantesCharla.csv")
dataframe = pd.DataFrame(datos)
print(datos)
X=(dataframe[["intervalo1","intervalo2","intervalo3","intervalo4","intervalo5","intervalo6"]])
y=(dataframe["Resultados"])

#Entrenamiento
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)
model = LogisticRegression()
model.fit(X_train,y_train)
dataNew = {'intervalo1': [1,1,0],
            'intervalo2': [0,0,1],
            'intervalo3': [0,1,1],
            'intervalo4': [0,0,1],
            'intervalo5': [0,1,1],
            'intervalo6': [0,1,1]
}

participantesCharlasNew = pd.DataFrame(dataNew, columns=['intervalo1','intervalo2','intervalo3','intervalo4','intervalo5','intervalo6'])
prediccion=model.predict(participantesCharlasNew)
print(participantesCharlasNew)
print(prediccion)

