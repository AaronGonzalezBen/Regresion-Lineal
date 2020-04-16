# Regresión lineal usando módulos de la librería Keras

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD

# Lectura y visualización del set de datos

# se lee el dataset, separado por comas, se saltan las primeras 32 filas (encabezado) y se usan solo las columnas 2 y 3 
datos = pd.read_csv('dataset.csv', sep = ",", skiprows = 32, usecols = [2,3])
print(datos)

# Graficamos los datos
datos.plot.scatter(x = 'Age', y = 'Systolic blood pressure')
plt.xlabel('Edad (años)')
plt.ylabel('Presión sistólica (mm Hg)')
plt.show()

x = datos['Age'].values
y = datos['Systolic blood pressure'].values

# Construcción del modelo de regresión en Keras

# - Capa de entrada: 1 dato (cada dato "x" correspondiente a la edad)
# - Capa de salida: 1 dato (cada dato "y" correspondiente a la regresión lineal)
# - Activación: 'linear' (referente a implementar una regresión lineal)

np.random.seed(2)   # Con esto se tiene una referencia respecto al tutorial

input_dim = 1
output_dim = 1

# Sequential crea un objeto (contenedor) vacío en el cual se pueden ingresar operaciones y 
# ejecutarse de forma secuencial
modelo = Sequential()

# Con .add empezamos a anadir elementos, en este caso la función que permite ejecutar una
# regresión lineal: Dense
modelo.add(Dense(output_dim, input_dim = input_dim, activation = 'linear'))

# Definición del método de optimización (gradiente descendente), con una
# tasa de aprendizaje de 0.0004 y una pérdida igual al error cuadrático medio

sgd = SGD(lr = 0.0004)

# Se agrega el optimizador al modelo y la función de costo (Error Cuadrático Medio)
modelo.compile(loss = 'mse', optimizer = sgd)

# Imprimir en pantalla la información del modelo
modelo.summary()

# Entrenamiento: realizar la regresión lineal

# 40000 iteraciones y todos los datos de entrenamiento (29) se usarán en cada iteración
# batch_size = 29

num_epochs = 40000
batch_size = x.shape[0]     # Cantidad de datos que se usarán en cada iteración, shape[0] = # de filas

# Aplicamos las iteraciones al modelo (entrenamiento)
# verbose = 0 - no imprime cada una de las iteraciones
history = modelo.fit(x,y, epochs = num_epochs, batch_size = batch_size, verbose = 0)

# Visualizar resultados del entrenamiento

# Imprimir coeficientes "w" y "b"
capas = modelo.layers[0]    # Layers extrae las capas del modelo, en este caso nos traemos la unica capa del modelo
w, b = capas.get_weights()  # Extraemos el peso y el bias
print('Parámetros: w = {:.1f}'.format(w[0][0], b[0]))

# Graficar el error vs epochs y el resultado de la regresión
# Superpuesto a los datos originales
plt.subplot(1,2,1)
plt.plot(history.history['loss'])
plt.xlabel('epoch')
plt.ylabel('ECM')
plt.title('ECM vs epochs')

y_regr = modelo.predict(x)
plt.subplot(1,2,2)
plt.scatter(x,y)    # Grafico de dispersión de los datos
plt.plot(x,y_regr,'r')  # Recta de regresión generada
plt.xlabel('x')
plt.ylabel('y')
plt.title('Datos originales y Regresion lineal')
plt.show()

# Predicción
x_pred = np.array([90])
y_pred = modelo.predict(x_pred)
print("La presión sanguinea será de {:.1f} mm Hg".format(y_pred[0][0]), " para una persona de {} años".format(x_pred[0]))