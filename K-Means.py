import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# df_purse_seines = pd.read_csv('Data/drifting_longlines.csv')

# df_use = df_purse_seines[['vessel_num','lat','lon','timestamp']]
# x = df_use[['lat','lon']]


df_purse_seines = pd.read_csv('Data/drifting_longlines.csv')

df_use = df_purse_seines[['vessel_num','lat','lon','timestamp']]

# from collections import Counter
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
X = df_new[['lat','lon']]

X = X.to_numpy()


# Determine the values of k
k = np.arange(1,11)
jarr = []
for i in k:
    model = KMeans(n_clusters=i)
    model.fit(X)
    jarr.append(model.inertia_)
    # Label these points
    plt.annotate(str(i),(i,model.inertia_))
plt.plot(k,jarr)
plt.show()
# according to the drawing，k=5
k = 5

# Formalize the model
model1 = KMeans(n_clusters=k)
# run the model
model1.fit(X)
# need to know what parameters each category has
C_i = model1.predict(X)
# need to know the coordinates of the cluster center
Muk = model1.cluster_centers_


# plot
plt.scatter(X[:,0],X[:,1],c=C_i,cmap=plt.cm.Paired)
# plot the cluster center
plt.scatter(Muk[:,0],Muk[:,1],marker='*',s=60)
for i in range(k):
    plt.annotate('center'+str(i + 1),(Muk[i,0],Muk[i,1]))
plt.show()
