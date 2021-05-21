import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

dados = pd.read_csv('dataset.csv')


X = dados['Height'].values
Y = dados['Weight'].values

plt.scatter(X,Y, label="peso(altura) | polega & libras")
plt.xlabel(X)
plt.ylabel(Y)
plt.legend()

modelo = LinearRegression()


X = X.reshape(-1,1)


modelo.fit(X,Y)

acuracia = modelo.score(X,Y)

print("Precisão: {:0.2f}".format(acuracia))

# Ax+B = y(x)
#calculo dos coeficientes
A = modelo.coef_[0]
B = modelo.intercept_

print("Coenficiente Angular : {:0.2f}".format(A))
print("Coeficiente Linear: {:0.2f}".format(B))

reta = A*X+B

plt.scatter(X,Y, label="peso(altura) | polega & libras")
plt.plot(X,reta, label="Ajuste linear", color="red")
plt.xlabel(X)
plt.ylabel(Y)
plt.legend()
plt.show()

alturaPolegada = input('Sua altura em polegadas: ')
pesoPresumido  = A*float(alturaPolegada) +B
print("Peso presumido" , pesoPresumido)






