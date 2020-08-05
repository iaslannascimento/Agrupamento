#Autor Iaslan Nascimento
#Data/04/08/20
#Código para fazer o agrupamento de vinhos 
#Mapa autoorganizaveis 
from minisom import MiniSom
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

base  = pd.read_csv('wines.csv')
#pegando todos os atributos
X = base.iloc[:,1:14].values
#pegando o atributo de classe 
Y = base.iloc[:,0].values

#normalização de dados 

normalizador = MinMaxScaler(feature_range=(0,1))
X = normalizador.fit_transform(X)
print("Ok")
print(X)

som = MiniSom(x = 8, y = 8, input_len = 13,  sigma = 1.0, learning_rate= 0.5, random_seed= 2 )
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100)

som._weights
som._activation_map
q = som.activation_response(X)
print(q)
#plot
from pylab import pcolor, colorbar
pcolor(som.distance_map().T)
colorbar()

#melhor registro
w = som.winner(X[1])
#marcadores das classes
markers = ['o', 's','D']
#cores para as classes 
color = ['r', 'g', 'b']

#Y[Y == 1] = 0
#Y[Y == 2] = 1
#Y[Y == 3] = 2

for i,x in enumerate(X):
    #print(i)
    #print(x)
    w = som.winner(x)
    plot(w[0] +0.5,w[1]+0.5, markers[y[i]],
         markerfacecolor = 'None', markersize = 10,
         markeredgecolor = color[y[i]], markeredgewidth = 2)
    
