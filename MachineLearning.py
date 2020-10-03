# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 13:32:01 2020

@author: MARIA
"""
'''====================================================================='''
'''numpy'''
'''====================================================================='''
import numpy as np

#Arreglos Unidimensionales 
lista = [1, 2, 3, 4 , 5]
a = np.array(lista) #a raíz de una lista

b = np.array([1, 2, 3, 4, 5]) 

print(a) #[1 2 3 4 5]

#tipo de datos
print(a.dtype) #int32

a_complejo = np.array(lista, dtype=complex)
print(a_complejo) #[1.+0.j 2.+0.j 3.+0.j 4.+0.j 5.+0.j]

#Arreglos multidimensionales
c = np.array([[0, 1, 2], [3, 4, 5]]) # arreglo 2 x 3
print(c)
print(c.ndim) #dimensiones
print(c.shape) #(2, 3)

#Creacion con funciones
d = np.arange(10) 
print(d) # [0 1 2 3 4 5 6 7 8 9]
x = np.arange(3,10,2) # argumentos: inicio, fin, paso
print(x) #[3 5 7 9]

f = np.zeros((3,3))
print(f)
#[[0. 0. 0.]
# [0. 0. 0.]
# [0. 0. 0.]]
g = np.ones((2,3))
print(g)

y = np.linspace(0, 10, 25) # incio, fin, n° puntos intermedios
print(y)

z = np.logspace(0, 10, 10, base=np.e)
print(z)

e = d[:].copy() # fuerzo la copia (sino apuntan al mismo objeto)

d[2:9] # [2, 3, 4, 5, 6, 7, 8]
d[2:9:3] # [2, 5, 8]
d[3::2]  #[3, 5, 7, 9]

#números aleatorios
a = np.random.rand(5, 3)
print(a)

# números aleatorios normalmente distribuidos
b = np.random.randn(5,5) 
print(b)

#operaciones numéricas
a = np.array([1, 2, 3, 4])
print(a+1) #[2, 3, 4, 5]

b = np.array([5, 5, 5, 5])
c = np.ones(4)
print(b - a) #[4 3 2 1]
print(c - a) #[ 0. -1. -2. -3.]
print(a*b) #[ 5 10 15 20]
print(a.sum()) #10
print(a.mean()) #2.5
print(a.min(), a.max()) #1 4

#multiplicación matrices
c = np.ones((4,4))
print(c.dot(b))
print(c.T)

#Broadcasting
a = np.array([[0.0,0.0,0.0],[10.0,10.0,10.0],[20.0,20.0,20.0],[30.0,30.0,30.0]]) 
b = np.array([1.0,2.0,3.0])  
   
print('broadcasting',a + b)

'''====================================================================='''
'''pandas'''
'''====================================================================='''
import pandas as pd

#Series
edad = pd.Series([10, 20, 14, 11])
print(edad)

bacteria = pd.Series([10, 20, 14, 11], index=['a', 'b', 'c', 'd'])
print(bacteria)
print(bacteria.values, bacteria.index)
print(bacteria.describe(percentiles=[0.25, 0.5, 0.75]))

bacteria_dict = {'a': 10, 'b': 20, 'c': 14, 'd': 11}
print(pd.Series(bacteria_dict))

#DataFrame
data = np.array([['','Col1','Col2'], ['Fila1',11,22], ['Fila2',33,44]])      
dataFrame = pd.DataFrame(data = data[1:,1:], 
                         index = data[1:,0], 
                         columns = data[0,1:])
print(dataFrame)

print('Forma del DataFrame:')
print(dataFrame.shape)

print('Estadísticas del DataFrame:')
print(dataFrame.describe())

print('Media de las columnas DataFrame:')
print(dataFrame.mean())

print('Conteo de datos del DataFrame:')
print(dataFrame.count())

print('Valor más alto de la columna del DataFrame:')
print(dataFrame.max())

print('Datos nulos en el DataFrame:')
print(dataFrame.isnull())

#pd.read_csv('train.csv') #lee archivo

'''====================================================================='''
'''matplotlib'''
'''====================================================================='''

import matplotlib.pyplot as plt

#Definir los datos
x1 = [3, 4, 5, 6]
y1 = [5, 6, 3, 4]
x2 = [2, 5, 8]
y2 = [3, 4, 3]

#Configurar las características del gráfico
plt.plot(x1, y1, label = 'Línea 1', linewidth = 4, color = 'blue')
plt.plot(x2, y2, label = 'Línea 2', linewidth = 4, color = 'green')

#Definir título y nombres de ejes
plt.title('Diagrama de Líneas')
plt.ylabel('Eje Y')
plt.xlabel('Eje X')

#Mostrar leyenda, cuadrícula y figura
plt.legend()
plt.grid()
plt.show()

a = [3, 4, 5, 6]
b = [5, 6, 3, 4]
plt.plot(a, b)
plt.show()

'''====================================================================='''
'''Regresión Lineal Simple'''
'''====================================================================='''
from sklearn import linear_model
from sklearn.metrics import r2_score

data = pd.read_csv('Fuel.csv')

plt.scatter(data['ENGINESIZE'],data['CO2EMISSIONS'], color='blue')
plt.xlabel("ENGINESIZE")
plt.ylabel("CO2EMISSIONS")
plt.show()

size = int(len(data) * 0.1) 
train_set, test_set = data[size:], data[:size]

train_x = np.array(train_set[['ENGINESIZE']])
train_y = np.array(train_set[['CO2EMISSIONS']])

modelo = linear_model.LinearRegression()
modelo.fit(train_x, train_y)

print('Coeficiente:',modelo.coef_)
print('Interceptor:',modelo.intercept_)

plt.scatter(train_set['ENGINESIZE'],train_set['CO2EMISSIONS'], color='blue')
plt.plot(train_x,modelo.coef_*train_x + modelo.intercept_, 'r')
plt.xlabel("Tamaño")
plt.ylabel("EmisionesCO2")
plt.show()

def get_prediccion(dato, coeficiente, intercepto):
    valor = dato*coeficiente + intercepto
    return valor
    
my_size = 3.5
estimado = get_prediccion(my_size,modelo.coef_,modelo.intercept_)

test_x = np.array(test_set[['ENGINESIZE']])
test_y = np.array(test_set[['CO2EMISSIONS']])
test_y_ = modelo.predict(test_x)
print('Mean absolute error: %.2f' % np.mean(np.absolute(test_y_-test_y)))
print('Mean sum of squares (MSE): %.2f' % np.mean((test_y_ - test_y) ** 2))
print('R2-score: %.2f' % r2_score(test_y_ , test_y) )

'''====================================================================='''
'''Clasificacion K nearest neighbor'''
'''====================================================================='''
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

dataframe = pd.read_csv(r"reviews_sentiment.csv",sep=';')
print(dataframe.head(10))
print(dataframe.describe())
dataframe.hist()
print(dataframe.groupby('Star Rating').size())

X = dataframe[['wordcount','sentimentValue']].values
y = dataframe['Star Rating'].values
 
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

n_neighbors = 7
knn = KNeighborsClassifier(n_neighbors)
knn.fit(X_train, y_train)
print('Accuracy of K-NN classifier on training set: {:.2f}'
     .format(knn.score(X_train, y_train)))
print('Accuracy of K-NN classifier on test set: {:.2f}'
     .format(knn.score(X_test, y_test)))
	
pred = knn.predict(X_test)
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))

# we create an instance of Neighbours Classifier and fit the data.
clf = KNeighborsClassifier(n_neighbors, weights='distance')
clf.fit(X, y)
print(clf.predict([[5, 1.0]])) 	
print(clf.predict_proba([[20, 0.0]]))

h = .02  # step size in the mesh
 # Create color maps
cmap_light = ListedColormap(['#FFAAAA', '#ffcc99', '#ffffb3','#b3ffff','#c2f0c2'])
cmap_bold = ListedColormap(['#FF0000', '#ff9933','#FFFF00','#00ffff','#00FF00'])
                            
# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
 
# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
 
# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
                edgecolor='k', s=20)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
    
patch0 = mpatches.Patch(color='#FF0000', label='1')
patch1 = mpatches.Patch(color='#ff9933', label='2')
patch2 = mpatches.Patch(color='#FFFF00', label='3')
patch3 = mpatches.Patch(color='#00ffff', label='4')
patch4 = mpatches.Patch(color='#00FF00', label='5')
plt.legend(handles=[patch0, patch1, patch2, patch3,patch4])
plt.title("5-Class classification (k = %i, weights = '%s')"
              % (n_neighbors, 'distance'))
plt.show()


'''====================================================================='''
'''Clasificacion Decision Tree'''
'''====================================================================='''

from sklearn.datasets import load_iris 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
from pydotplus import graph_from_dot_data

iris = load_iris()
print(iris.DESCR) 

df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

#Conjunto de entrenamiento y test
X_train, X_test, Y_train, Y_test = train_test_split(
        df[iris.feature_names], df['target'], random_state=0)

# árbol de profundidad 2
tree = DecisionTreeClassifier(max_depth=2, random_state=42) 
tree.fit(X_train, Y_train) 

#Predicciones
print(tree.predict(X_test.iloc[0].values.reshape(1, -1))) #[2]
print(tree.predict(X_test[0:10])) #[2 1 0 2 0 2 0 1 1 1]

# Accuracy
score = tree.score(X_test, Y_test)
print(score) #0.8947368421052632

#Visualizar el arbol
dot_data = export_graphviz(tree, feature_names=iris.feature_names)
graph = graph_from_dot_data(dot_data)
graph.write_png('tree.png')

'''====================================================================='''
'''Clustering'''
'''====================================================================='''

from sklearn.cluster import KMeans

#ingreso anual en miles y puntuación del cliente
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values 

#Representación gráfica de los datos.
x1 = dataset.iloc[:, [3]].values 
y1 = dataset.iloc[:, [4]].values 
plt.xlabel('Ingreso Anual (k$)')
plt.ylabel('Puntuación del cliente (1-100)')
plt.title('Ingreso vs Puntuacion')
plt.plot(x1,y1,'o',markersize=1)
#plt.plot(x1,y1)
plt.show()

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Grafica de la suma de las distancias
plt.plot(range(1, 11), wcss)
plt.title('El método Elbow')
plt.xlabel('Número de Clusters')
plt.ylabel('WCSS')
plt.show()

# Creando el k-Means para los 5 grupos encontrados
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)

centroids = kmeans.cluster_centers_
print(centroids)

# Visualizacion grafica de los clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, 
            c = 'yellow', label = 'Centroids')

plt.title('Clusters de Clientes')
plt.xlabel('Ingreso Anual (k$)')
plt.ylabel('Puntuación del cliente (1-100)')
plt.legend()
plt.show()
