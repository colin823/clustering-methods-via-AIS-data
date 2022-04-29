import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics

df_purse_seines = pd.read_csv('Data/drifting_longlines.csv')
df_use = df_purse_seines[['vessel_num','lat','lon','timestamp']]

X = df_use[['lat','lon']]
X = X.to_numpy()

k = 5

# Formalize the model
model1 = KMeans(n_clusters=k)
# run the model
model1.fit(X)
# need to know what parameters each category has
C_i = model1.predict(X)
# need to know the coordinates of the cluster center
Muk = model1.cluster_centers_

print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, C_i))

# plot
plt.figure(figsize=(10,7))
plt.title('Estimated number of clusters:5')
plt.scatter(X[:,0],X[:,1],c=C_i,cmap=plt.cm.Paired, s = 0.5 )

# plot the cluster center
plt.scatter(Muk[:,0],Muk[:,1],marker='*',s=60)
for i in range(k):
    plt.annotate('center'+str(i + 1),(Muk[i,0],Muk[i,1]))
plt.show()





