import sklearn
import numpy as np
import pandas as pd
import kmapper as km
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tabulate as tb
import matplotlib.pyplot as plt
import math
from random import randint


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

def eccentricity(data, p):
    N = data.shape[0]
    cte = 1/N
    projected_data = None
    for i in range(N):
        p_distances_from_i = ((data - data[i, :]) ** 2).sum(axis=1)
        p_distances_from_i = np.array([math.sqrt(i) for i in p_distances_from_i])
        p_distances_from_i = (p_distances_from_i)**p
        ecc_i = (cte * np.sum(p_distances_from_i))**(1/p)
        if i == 0:
            projected_data = np.array([ecc_i])
        else:
            projected_data = np.vstack([projected_data, [ecc_i]])
    return projected_data

parametros = ["NO2","O3","PM10", "WSP"]
def similitud(cluster,X):
    distance = {'parametros':parametros}

    nodes = list(cluster["nodes"].keys())
    for node in nodes:
        values = cluster["nodes"][node]
        distance[node]=np.sum(np.diff(X[values],axis=0),axis=0)
    return distance


def count_label(graph,y_label)->None:
    labels = ['nodes', 'bueno', 'aceptable', 'malo', 'muy mala', 'extremadamente', 'outliers']
    df_graph = {k:[] for k in labels}
    nodes = list(graph["nodes"].keys())
    df_graph['nodes']=nodes
    for node in nodes:
        Y_ = y_label[graph['nodes'][node]]
        for k in range(1,len(labels)):
            df_graph[labels[k]].append(np.where(Y_==k)[0].size)
    return df_graph


def create_fig(df,title)->None:
    df = pd.DataFrame(df_graph)
    df.set_index('nodes')
    df.plot(kind = 'barh',x='nodes')
    plt.title(title)
    plt.savefig(str(title)+'.png')



#data_aire_df = pd.read_csv("level_pollution.csv") # DATABASE FROM 6 AM TO 10 PM
data_aire_df = pd.read_csv("data_level_pollution_O3.csv") # CRITIC DATABASE FROM 6 AM TO 10 AM
# REMOVE COMMENT TO OBTAIN MAPPER FOR EACH STATION
#data_aire_df = data_aire_df[data_aire_df.id_station_id=="TLA"]
#data_aire_df = data_aire_df[data_aire_df.id_station_id=="XAL"]
#data_aire_df = data_aire_df[data_aire_df.id_station_id=="MGH"]
#data_aire_df = data_aire_df[data_aire_df.id_station_id=="CUA"]
#data_aire_df = data_aire_df[data_aire_df.id_station_id=="MER"]
#data_aire_df = pd.read_csv("level_pollution_2019.csv")
print("size: ", data_aire_df.shape)

# TIMES OF THE DATA
data_aire_df.dateUTCShiftedDown = pd.to_datetime(data_aire_df.dateUTCShiftedDown)
hours = data_aire_df.dateUTCShiftedDown.dt.hour.unique()
hours_complete = np.array(data_aire_df.dateUTCShiftedDown.dt.hour)
print("horas registradas ", hours)

X = data_aire_df.iloc[:, 2:6].values
times = data_aire_df.iloc[:, 0].values
Y = data_aire_df.iloc[:, 8].values #8 for data_level_pollution and 6 for level_pollution

# SHOW MONTHS REGISTERED IN DATABASE
data_aire_df.dateUTCShiftedDown = pd.to_datetime(data_aire_df.dateUTCShiftedDown)
months = data_aire_df.dateUTCShiftedDown.dt.month.unique()
months_complete = np.array(data_aire_df.dateUTCShiftedDown.dt.month)
print("Meses registrados ", months)

scaler = MinMaxScaler()
norm_aire = scaler.fit_transform(X)
print("Normalized: ", norm_aire.shape)

# GAUSSIAN LENS
projected_data = gaussian_lens(data=norm_aire,h=0.2)
print("filtered: ", projected_data.shape)

mapper = km.KeplerMapper(verbose=2)
clusterer = sklearn.cluster.DBSCAN(eps=0.129).fit(norm_aire) # 0.129 for data_level_pollution_o3, if not 0.12
graph = mapper.map(
    projected_data,
    norm_aire,
    cover=km.Cover(n_cubes=20, perc_overlap=0.15), # 20 and 0.15 for data_level_pollution_o3, if not 30
    clusterer=sklearn.cluster.DBSCAN(eps=0.129),
)
#print(graph)

# SHOW CLUSTERED DATABASE
fig, ax = plt.subplots(1, 1, subplot_kw={'projection': '3d'})
x = data_aire_df.iloc[:, 2].values
y = data_aire_df.iloc[:, 3].values
z = data_aire_df.iloc[:, 4].values
n = np.max(clusterer.labels_)+1
print(n)
colores = []
for i in range(n):
    colores.append('#%06X' % randint(0, 0xFFFFFF))
colores.append('#000000')

print(np.where(clusterer.labels_==-1)[0].shape, np.where(clusterer.labels_==0)[0].shape, 
      np.where(clusterer.labels_==1)[0].shape, np.where(clusterer.labels_==2)[0].shape)
ax.scatter(x, y, z, c=np.take(colores, clusterer.labels_))
ax. set_xlabel('NO2')
ax. set_ylabel('O3')
ax. set_zlabel('PM10')
plt.show()
mapper.visualize(
    graph,
    path_html="Mapper_Aire_Critica_cla.html",
    title="Aire",
    custom_tooltips=Y,
    color_values=Y,
    color_function_name=["Clasification"])


nodes = list(graph["nodes"].keys())
for node in nodes:
    # SAVE REPORTS PER NODE
    data_aire_df.iloc[graph["nodes"][node], :][["NO2", "PM10", "O3"]].plot(kind="hist", bins=50)
    plt.title(node)
    plt.savefig(node+".png")

    # SAVE HOUR REPORTS PER NODE
    x = hours_complete[graph["nodes"][node]]
    _, plot = plt.subplots()
    plt.hist(x, bins=50)
    plot.set_xlabel('Hours')
    plot.set_ylabel('Frecuency')
    plot.set_title(str(node)+' hour record')
    #plt.show()
    plt.savefig(str(node)+" hour record.png")

    # SAVE MONTH REPORTS PER NODE
    x = months_complete[graph["nodes"][node]]
    _, plot = plt.subplots()
    plt.hist(x, bins=50)
    plot.set_xlabel('Months')
    plot.set_ylabel('Frecuency')
    plot.set_title(str(node)+' month record')
    plt.savefig(str(node)+" month record.png")
    #plt.show()


df_graph = count_label(graph=graph, y_label=Y)
print()
title = "tabla"
print(tb.tabulate(df_graph,headers=df_graph.keys(),tablefmt="github"))
print()
#create_fig(df_graph,title=title)
diff = similitud(graph,X)
