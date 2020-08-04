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