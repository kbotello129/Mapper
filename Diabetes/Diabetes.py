import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import networkx as nx
from google.colab import drive
import pandas as pd
import seaborn as sns


def v2_cover(a, b, n, p):
    """Function to obtain a cover of n intervals covering an interval [a,b] with
       percentage intersection p between each pair of intervals.
    Parameters:
    -------------------------------------------------------
    a: float
       minimum value
    b: float
       maximum value
    n: int
       number of intervals
    p: float (between 0 and 1)
       traslape percentaje between the intervals

    Return:
    --------------------------------------------------------
    numpy array
    array with the elements of the cover (intervals)
    """
    min, max, num = a, b, n
    g = p
    l = (max - min)/(num - (num-1)*g)
    U = np.zeros((n, 2))
    rango = np.arange(num)
    a_i = rango*l*(1-g) + min
    b_i = a_i + l
    U[:, 0] = a_i
    U[:, 1] = b_i
    return U

def v2_preimage_f(projected_data, cubierta):
    """Function to obtain the pre-image of the cover elements.
    Parameters:
    -------------------------------------------------------
    projected_data: numpy array
                    data projected under the filter function
    cubierta: numpy array
              cover

    Return:
    --------------------------------------------------------
    dict
    pre-image of each cover element (interval).
    """
    V = {}
    n = projected_data.shape[0]
    for dato_i in range(n):
        pre_imagen_index = np.where((projected_data[dato_i]>=cubierta[:,0])&(projected_data[dato_i]<=cubierta[:,1]))
        V[dato_i]= pre_imagen_index[0].tolist()
    m = cubierta.shape[0]
    pre_imagen_index = [[] for _ in range(m)]
    for k in V.keys():
        for i in V[k]:
            pre_imagen_index[i].append(k)
    return pre_imagen_index

def gaussian_lens(data, h):
    """Function to obtain filtered data using the Gaussian lens.
    Parameters:
    -------------------------------------------------------
    data: numpy array
          data cloud
    h: float
       bandwith of the Gaussian function

    Return:
    --------------------------------------------------------
    numpy array
    filtered data
    """
    N = data.shape[0]
    d = data.shape[1]
    cte = (1 / (N * (h ** d))) * ((2 * np.pi) ** (-d / 2))
    projected_data = None
    for i in range(N):
        square_distances_from_i = ((data - data[i, :]) ** 2).sum(axis=1)
        gaussian_i = np.sum(cte * np.exp((-0.5) * square_distances_from_i / (h**2)))
        if i == 0:
            projected_data = np.array([gaussian_i])
        else:
            projected_data = np.vstack([projected_data, [gaussian_i]])
    return projected_data

def v2_distance_matrix_from_data(data):
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
    edges = list(map(tuple,edges))
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    return G

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
        labels[component] = f'C[{i},{j}]={component.order()}'
        nodes.append(list(component.nodes))
        sizes.append(component.order())
        j += 1
    return C, labels, sizes, nodes


def mapper(projected_data, G, U):
    """Mapper function.
    Parameters:
    -------------------------------------------------------
    projected_data: numpy array
                    data projected under the filter function
    G: graph object
       data graph
    U: numpy array
       cover

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
    labels = dict()
    nodes = []
    sizes = []
    components = []
    V = v2_preimage_f(projected_data, U)
    n = len(U)
    for i in range(n):
        components_i, labels_i, sizes_i, nodes_i = components_preimage(G, V[i], i)
        components.extend(components_i)
        labels.update(labels_i)
        nodes.extend(nodes_i)
        sizes.extend(sizes_i)
    #print(labels)
    #print(nodes)
    #print(sizes)

    colors = []
    for data_points in nodes:
        colors.append(np.mean(data_diab_df.iloc[data_points, 5]))
    return components, labels, sizes, nodes, colors


# DIABETES DATABASE 
data_diab_df = pd.read_csv('/content/gdrive/MyDrive/Data-Diabetes.csv')
data_diab = data_diab_df.values

# DATA NORMALIZATION
scaler = MinMaxScaler()
normdiab = scaler.fit_transform(data_diab)

# DISTANCE MATRIX
distance_matrix = v2_distance_matrix_from_data(normdiab)

# AGRUPATION
G = graph_alpha(distance_matrix, 0.58) # 0.55 for Low Dimension mapper

# FILTER / LENS
projected_data = gaussian_lens(normdiab, 0.58) # 0.57 for Low Dimension mapper

# COVER
min = np.min(projected_data)
max = np.max(projected_data)
n = 4 # 3 for Low Dimension mapper
p = 0.499
U = v2_cover(min, max, n, p)
print("cubierta: ", U)

# RESULTING MAPPER CONSTRUCTION 
components, labels, sizes, nodes, colors = mapper(projected_data, G, U)
graph_mapper = nx.Graph()
graph_mapper.add_nodes_from(components)
edges_components = []
for component1 in components:
    for component2 in components:
        if set(component1).isdisjoint(component2):
            continue
        elif component1 != component2:
            edges_components.append((component1, component2))

graph_mapper.add_edges_from(edges_components)
print(graph_mapper)
sizes = np.array(sizes)
colors = np.array(colors)
plt.figure(figsize=(5,5), dpi=128)
nx.draw(graph_mapper, with_labels=True, labels=labels, node_size=sizes * 20,
        node_color = colors * 90, cmap = 'coolwarm')
plt.show()


# CLASIFICATION REPORT
def make_report(labels, df):
  report = []
  for component in labels.keys():
    row = {'Component': labels[component]}
    for i in range(3):
      perc = 100*len(df.iloc[component][df.iloc[component, 5] == i+1])/len(component.nodes)
      row[f'% Tipo {i+1}'] = perc
    report.append(row)

  return pd.DataFrame(report)

def visual_report(labels, df):
  report = make_report(labels, df)
  df_graph = pd.melt(report, id_vars="Component", var_name="Diagnostic", value_name="Percentage people")
  sns.catplot(x='Percentage people', y='Component', hue='Diagnostic', data=df_graph, kind='bar', orient='h')

visual_report(labels, data_diab_df)
