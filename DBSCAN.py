import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from algorithms import DBScan

# df_purse_seines = pd.read_csv('Data/drifting_longlines.csv')

# df_use = df_purse_seines[['vessel_num','lat','lon','timestamp']]
# x = df_use[['lat','lon']]
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df_purse_seines = pd.read_csv('Data/drifting_longlines.csv')

df_use = df_purse_seines[['vessel_num','lat','lon','timestamp']]

from collections import Counter
# y = Counter(df_use['vessel_num'])
# print(y)
first_vessel=df_use.loc[df_use['vessel_num']==1.263960e+13]
first_vessel = first_vessel.sample(n=10000,random_state=1)

second_vessel = df_use.loc[df_use['vessel_num']==23770800000000.0]
second_vessel = second_vessel.sample(n=15000,random_state=1)

third_vessel = df_use.loc[df_use['vessel_num']==118486000000000.0]
third_vessel = third_vessel.sample(n=15000,random_state=1)

fourth_vessel = df_use.loc[df_use['vessel_num']==51394400000000.0]
fourth_vessel = fourth_vessel.sample(n=15000,random_state=1)

fifth_vessel = df_use.loc[df_use['vessel_num']==81264000000000.0]
fifth_vessel = fifth_vessel.sample(n=10000,random_state=1)

sixth_vessel = df_use.loc[df_use['vessel_num']==215151000000000.0]
sixth_vessel = sixth_vessel.sample(n=10000,random_state=1)

seventh_vessel = df_use.loc[df_use['vessel_num']==77182400000000.0]
seventh_vessel = seventh_vessel.sample(n=10000,random_state=1)

eighth_vessel = df_use.loc[df_use['vessel_num']==103576000000000.0]
eighth_vessel = eighth_vessel.sample(n=10000,random_state=1)

nineth_vessel = df_use.loc[df_use['vessel_num']==257110000000000.0]
nineth_vessel = nineth_vessel.sample(n=10000,random_state=1)

tenth_vessel = df_use.loc[df_use['vessel_num']==281060000000000.0]
tenth_vessel = tenth_vessel.sample(n=10000,random_state=1)

eleventh_vessel = df_use.loc[df_use['vessel_num']==168016000000000.0]
eleventh_vessel = eleventh_vessel.sample(n=10000,random_state=1)


df_new = pd.concat([first_vessel,second_vessel,third_vessel,fourth_vessel,fifth_vessel,sixth_vessel,seventh_vessel,eighth_vessel,nineth_vessel,tenth_vessel,eleventh_vessel])
x = df_new[['lat','lon']]

x = x.to_numpy()


def Dbscan(X):
    # first test parameter：eps=0.1, min_samples=10
    print('Start clustering...')
    cluster = DBSCAN(eps=0.3, min_samples=10)
    y = cluster.fit_predict(X=X)
    return y


def jiangzao (labels):

    # Number of clusters in the tag, ignore noise (if present)
    clusters = len(set(labels)) - (1 if -1 in labels else 0)
    return clusters

def draw(db):

    coreSamplesMask = np.zeros_like(db.labels_, dtype=bool)
    coreSamplesMask[db.core_sample_indices_] = True
    labels = db.labels_
    nclusters = jiangzao(labels)

    # plot result 
    # The black was removed and marked as noise
    unique_labels = set(labels)
    colors = plt.cm.get_cmap('Spectral')(np.linspace(0, 1, len(unique_labels)))
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # black for noice
            col = 'k'

        classMemberMask = (labels == k)

        # plot the set of classification points
        xy = [classMemberMask & coreSamplesMask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=6)

        # plot the set of noise points
        xy = [classMemberMask & ~coreSamplesMask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=3)
    # add title, show the numbers of clusters
    plt.title('Estimated number of clusters: %d' % nclusters)
    plt.show()

db = DBScan(x, eps=7, min_samples=1000)

db.draw()

# plt.figure()
# plt.title('DBSCAN For Trajectories ')
# plt.plot(df_use['lat'], df_use['lon'])
# plt.legend() # 显示图例
  
# plt.xlabel('lat')
# plt.ylabel('lon')
# plt.show




