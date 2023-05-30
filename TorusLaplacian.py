import numpy as np
from numpy.linalg import eig
import math
import matplotlib.pyplot as plt
import networkx as nx
from itertools import combinations
from google.colab import drive
import pandas as pd

def density_gauss(epsilon, x):
    """Gaussian density function.
    Parameters:
    -------------------------------------------------------
    G: float
       epsilon
    x: numpy array
       data point   

    Return:
    --------------------------------------------------------
    density: numpy array
             desinty estimation
    """
    density = np.exp(((-1)*(x**2))/(epsilon))
    return density

def cover_circles_r2(minx, miny, maxx, maxy, r):
  """Function to obtain the cover in R^2 with circles of radius R with centers 
     located from circles of radius r.
    Parameters:
    -------------------------------------------------------
    minx, miny, maxx, maxy: coordinates of the filter range.
    r: radius of the circle.

    Return:
    --------------------------------------------------------
    numpy array
    Array with the coordinates of the centers of the cover elements (circles).
  """
  sigma = r*math.sin(math.pi/3)
  covered = []
  covered.append([minx, miny])
  x = minx
  y = miny + 2*sigma
  num = 1
  while x < maxx:
    while y < maxy:
      covered.append([x, y])
      y = y + 2*sigma
    covered.append([x, y])
    x = x + 3*r/2
    num = num + 1
    if num % 2 == 0:
      y = miny + sigma
    else:
      y = miny
  while y < maxy:
      covered.append([x, y])
      y = y + 2*sigma
  covered.append([x, y])

  return np.array(covered)

def rotation_matrix_n(n):
  """Function to obtain a rotation matrix in R^n at a random angle.
    Parameters:
    -------------------------------------------------------
    n: int
       dimension

    Return:
    --------------------------------------------------------
    R: numpy array
       Rotation matrix.
  """
  R = np.eye(n)
  for i in range(n):
    for j in range(i+1, n):
      R_ij = np.eye(n)
      theta = np.random.rand() * 2 * np.pi
      c = np.cos(theta)
      s = np.sin(theta)
      R_ij[i, i] = c
      R_ij[j, j] = c
      R_ij[i, j] = -s
      R_ij[j, i] = s
      R = np.matmul(R, R_ij)
  return R

def rotate_figure(figure_rn, n):
  """Function to rotate the point cloud data at a random angle.
    Parameters:
    -------------------------------------------------------
    figure_rn: numpy array
               point cloud data
    n: int
       dimension

    Return:
    --------------------------------------------------------
    figure_rn: numpy array
               point cloud data rotated.
  """
  R = rotation_matrix_n(n)
  for i in range(figure_rn.shape[0]):
    figure_rn[i, :] = np.matmul(R, figure_rn[i, :])

  return figure_rn
  
def adj_matrix_by_alpha(distance_matrix, alpha):
    """function to create the adjacency matrix from the distance matrix and
    the maximum length alpha.
    Parameters:
    -------------------------------------------------------
    distance_matrix : numpy array
                      distance matrix
    alpha: float
           maximum lenght to consider two points adjacent

    Return:
    --------------------------------------------------------
    numpy array
    adjacency matrix
    """

    A = np.where(distance_matrix <= alpha, 1, 0)
    A = A - np.diag(np.diag(A))
    return A

def adj_matrix_to_graph(A):
    """Function to obtain the nodes and edges of a graph from an adjacency matrix.
    Parameters:
    -------------------------------------------------------
    A: numpy array
       adjacency matrix

    Return:
    --------------------------------------------------------
    list
    nodes

    numpy array
    edges 
    """
    n = A.shape[0]
    nodes = list(range(n))
    edges = np.where(A==1)
    aristas = np.zeros((edges[0].shape[0], 2), dtype= int)
    aristas[:, 0] = edges[0]
    aristas[:, 1] = edges[1]
    return nodes, aristas

def distance_matrix_from_data(data):
    """Function to obtain the distance matrix from a data cloud.
    Parameters:
    -------------------------------------------------------
    data: numpy array
          data cloud

    Return:
    --------------------------------------------------------
    numpy array
    distance matrix
    """
    n = data.shape[0]
    distance_matrix = np.zeros((n,n))
    for i in range(n):
        distance_matrix[i,:] = np.sqrt( ((data - data[i, :])**2).sum(axis=1) )
    return distance_matrix

def graph_alpha(distance_matrix, alpha):
    """Function to obtain a graph from the distance matrix and the maximum
       length alpha.
    Parameters:
    -------------------------------------------------------
    distance_matrix: numpy array
                     distance matrix
    alpha: float
           maximum length to consider two points adjacent

    Return:
    --------------------------------------------------------
    graph object
    G
    """
    A = adj_matrix_by_alpha(distance_matrix, alpha)
    nodes, edges = adj_matrix_to_graph(A)
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    return G

def preimage_f_v2(projected_data, cubierta, R):
  """Function to obtain the pre-image of the cover elements.
    Parameters:
    -------------------------------------------------------
    projected_data: numpy array
                    data projected under the filter function
    cubierta: numpy array
              cover
    R: float
       radius of the circles in the cover

    Return:
    --------------------------------------------------------
    dict
    pre-image of each cover element (circle). 
  """
  V = {}
  n = projected_data.shape[0]
  i=0
  for cubierta_i in cubierta:
      pre_imagen_index = np.where(np.linalg.norm(cubierta_i-projected_data, axis=1) <= R)[0].tolist()
      V[i]= pre_imagen_index
      i+=1
  return V

def components_preimage(G, V, i):
    """Function to obtain the connected components of the pre-image of a cover
    element, its size and its label.
    Parameters:
    -------------------------------------------------------
    G: graph object
       data graph
    V: list
       data in the pre-image of a cover element
    i: int
       index of the cover element

    Return:
    --------------------------------------------------------
    tuple (object)
    C: list
       each element found in C represents a connected component of the induced 
       graph over V
    labels: dictionary({graph object:string})
        labels to identify each node in the graph
    sizes: list
        list of the sizes of each component
    nodes: list
        each list found in 'nodes' is the data in each connected 
        component
    """
    C = []
    labels = dict()
    nodes = []
    sizes = []

    H = G.subgraph(V).copy()
    nodes_components = list(nx.connected_components(H))
    j = 0
    for nodes_component in nodes_components:
        component = H.subgraph(nodes_component).copy()
        C.append(component)
        labels[component] = f'C[{i},{j}]'
        nodes.append(list(component.nodes))
        sizes.append(component.order())
        j += 1
    return C, labels, sizes, nodes

def mapper_v2(projected_data, G, U, n):
    """Mapper function.
    Parameters:
    -------------------------------------------------------
    projected_data: numpy array
                    data projected under the filter function
    G: graph object
       data graph
    U: numpy array
       cover
    n: int
       lenght of the cover
    

    Return:
    --------------------------------------------------------
    list
    components of data (clusters)
    
    list
    labels of the nodes

    list
    sizes of each component

    list
    nodes of the mapper
    """

    R = 0.015
    labels = dict()
    nodes = []
    sizes = []
    components = []
    V = preimage_f_v2(projected_data, U, R)
    for i in range(n):
        components_i, labels_i, sizes_i, nodes_i= components_preimage(G, V[i], i)
        components.extend(components_i)
        labels.update(labels_i)
        nodes.extend(nodes_i)
        sizes.extend(sizes_i)
    lista_nodos_num = [i for i in range(len(nodes))]
    # print(labels)
    # print(nodes)
    # print(sizes)
    # print(lista_nodos_num)
    return components, labels, sizes, nodes



# BASE DE DATOS DEL TORO CONSTRUIDA
theta = math.pi/25
a = 2.1
c = 6
x = c - a
y = 0
z = 0
d = c - a
data_torus = []
theta1 = theta
alpha = math.pi/10
alpha1 = alpha
data_torus.append([x, y, z])
while alpha1 <= math.pi + alpha:
  while theta1 < 2*math.pi:
    if d == c - a or d >= c + a:
      x = math.cos(theta1)*d
      y = math.sin(theta1)*d
      z = 0
      data_torus.append([x, y, z])
    else:
      x = math.cos(theta1)*d
      y = math.sin(theta1)*d
      z1 = math.sqrt(a**2 - (c - math.sqrt(x**2 + y**2))**2)
      z2 = -1*z1
      data_torus.append([x, y, z1])
      data_torus.append([x, y, z2])
    theta1 = theta1 + theta
  d = c - a*math.cos(alpha1)
  alpha1 = alpha1 + alpha
  theta1 = 0
data_torus = np.around(np.asarray(data_torus), decimals=5)
# print(data_torus.shape)

fig, ax = plt.subplots(1, 1, subplot_kw={'projection': '3d'})
for row in data_torus:
   ax.scatter(row[0], row[1], row[2])
plt.show()


# Meterlo en ambient + 3 dimensiones
ambient = 27
add = np.zeros((data_torus.shape[0], ambient))
data_torus = np.append(data_torus, add, axis=1)


# Rotación
data = rotate_figure(data_torus, ambient + 3)

fig, ax = plt.subplots(1, 1, subplot_kw={'projection': '3d'})
for row in data_torus:
   ax.scatter(row[0], row[1], row[2])
plt.show()


# MATRIZ DE PESOS
B = np.zeros((len(data_torus), len(data_torus)))
j = 0
for i in range(B.shape[0]):
    B[i, :] = density_gauss(4, np.linalg.norm(data_torus[i, :]-data_torus,axis=1))

# LAPLACIANO
L = np.zeros((len(data_torus), len(data_torus)))
j = 0
s = np.sum(B,axis=1)
for i in range(len(data_torus)):
    L[i,:] = B[i,:]/(np.sum(B[i, :])*s)

# DATOS FILTRADOS
w, v = eig(L)
flat = w.flatten()
flat.sort()
mayores = np.where(w == flat[-2])
filtered = np.column_stack((v[:, 1], v[:, 2]))

plt.scatter(filtered[:, [0]], filtered[:, [1]], label= "stars", color= "green",
            marker= "*", s=30)
plt.show()

# CUBIERTA Y MAPPER
min = np.amin(filtered, axis=0)
max = np.amax(filtered, axis=0)
# print(min, max)
r = 0.01
U = cover_circles_r2(min[0], min[1], max[0], max[1], r)
distance_matrix = distance_matrix_from_data(data_torus)
G = graph_alpha(distance_matrix, 1.5)
components, labels, sizes, nodes= mapper_v2(filtered, G, U, len(U))

componentes = []
for component in components:
    componentes.append(set(component))
print("componentes: ", len(componentes))
nodos = len(componentes)
graph_mapper = nx.Graph()
graph_mapper.add_nodes_from(components)
edges_components = []
aristas = []
aristas_num = []
for component1 in components:
    for component2 in components:
        if set(component1).isdisjoint(component2):
            continue
        elif component1 != component2:
            edges_components.append((component1, component2))
            aristas.append((set(component1), set(component2)))
            num1 = componentes.index(set(component1))
            num2 = componentes.index(set(component2))
            if aristas_num.__contains__([num2, num1]) is False:
              aristas_num.append([num1, num2])

# print("Aristas: ", aristas_num)
print("Cantidad aristas: ", len(aristas_num))

# TRIÁNGULOS
triangles_components = []
triangles_num = []
comb = combinations(componentes, 3)
for i in list(comb):
    if len(set.intersection(i[0], i[1], i[2])) == 0:
        continue
    else:
        triangles_components.append((i[0], i[1], i[2]))
        num1 = componentes.index(i[0])
        num2 = componentes.index(i[1])
        num3 = componentes.index(i[2])
        triangles_num.append([num1, num2, num3])

print("triangulos: ", len(triangles_num))
# print(triangles_num)

# TETRAEDROS
four_components = []
comb = combinations(componentes, 4)
for i in list(comb):
    if len(set.intersection(i[0], i[1], i[2], i[3])) == 0:
        continue
    else:
        four_components.append((i[0], i[1], i[2], i[3]))

print("tetraedros: ", len(four_components))

# MATRIZ PARA ESPACIO NULIDAD
N = np.zeros((len(aristas), len(triangles_components)))
num = 0
for triangle in triangles_components:
  comb = list(combinations(triangle, 2))
  for i in comb:
    indice = aristas.index((i[0], i[1]))
    N[indice][num] = 1
  num = num + 1

N = Matrix(N)
null_space = N.nullspace()
print('Espacio nulidad: ', len(null_space))

graph_mapper.add_edges_from(edges_components)
print(graph_mapper)
sizes = np.array(sizes)
plt.figure(figsize=(8,8), dpi=128)
nx.draw(graph_mapper, with_labels=True, labels=labels, cmap = 'coolwarm')
print('triangulos en el grafo: ', sum(nx.triangles(graph_mapper).values()) / 3)
plt.show()
