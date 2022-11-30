import pandas as pd
import joblib as jb

#Cargar el Modelo
model = jb.load('modelo_entrenado.pkl')

#Probar el Modelo
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

