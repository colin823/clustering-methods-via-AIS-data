import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn import metrics

df_purse_seines = pd.read_csv('Data/drifting_longlines.csv')
df_use = df_purse_seines[['vessel_num','lat','lon','timestamp']]

x = df_use[['lat','lon']]
x = x.to_numpy()

gmm = GaussianMixture(n_components=7)
gmm.fit(x)
labels = gmm.predict(x)

plt.figure(figsize=(10,7))
plt.title('Estimated number of clusters:7')
plt.scatter(x[:, 0], x[:, 1], c=labels, cmap='viridis', s = 0.5)
plt.show()
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(x, labels))


      