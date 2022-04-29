import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fontTools.merge import cmap
from sklearn.cluster import KMeans
from sklearn import metrics
from matplotlib.animation import FuncAnimation
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter  
import matplotlib.ticker as mticker 

df_data = pd.read_csv('Data/drifting_longlines.csv')
df_use = df_data[['lat', 'lon']]

X = df_use.to_numpy()

k = 5

# Formalize the model
model1 = KMeans(n_clusters=k)
# run the model
model1.fit(X)
# need to know what parameters each category has
C_i = model1.predict(X)
# need to know the coordinates of the cluster center
Muk = model1.cluster_centers_



fig = plt.figure(figsize=(16, 10))  

ax = plt.axes(projection=ccrs.PlateCarree())
ax.stock_img()

extent = [-180, 180, -90, 90]

dx = 60
dy = 30
offset = min(dx, dy)

xticks = np.arange(extent[0], extent[1] + offset, dx)  
yticks = np.arange(extent[2], extent[3] + offset, dy)  

ax.set_xticks(xticks, crs=ccrs.PlateCarree())
ax.set_yticks(yticks, crs=ccrs.PlateCarree())

ax.tick_params(labelsize=15)

ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
ax.yaxis.set_major_formatter(LatitudeFormatter())

ax.set_extent(extent)

plt.scatter(X[:,1],X[:,0],
            s = 0.4,c=C_i,
            cmap=plt.cm.inferno,
            transform=ccrs.Geodetic(),
            )


plt.scatter(Muk[:,1],Muk[:,0],
            marker='*',s=80,color = 'k',
            transform=ccrs.Geodetic(),
            )

for i in range(k):
    plt.annotate('center'+str(i + 1),(Muk[i,1],Muk[i,0]))
plt.show()



# plt.scatter(X[:,0],X[:,1],c=C_i,cmap=plt.cm.Paired, s = 0.5 )
#
# # plot the cluster center
# plt.scatter(Muk[:,0],Muk[:,1],marker='*',s=60)
# for i in range(k):
#     plt.annotate('center'+str(i + 1),(Muk[i,0],Muk[i,1]))
# plt.show()