import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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
# y2 = Counter(df_new['vessel_num'])
# print(y2)
plt.title('Result Analysis')
plt.plot(first_vessel['lat'], first_vessel['lon'],'o', color='green', label='first_vessel')
plt.plot(second_vessel['lat'], second_vessel['lon'],'o', color='green', label='second_vessel')
plt.plot(third_vessel['lat'], third_vessel['lon'], 'o', color='red', label='third_vessel')
plt.plot(fourth_vessel['lat'], fourth_vessel['lon'],'o',  color='skyblue', label='fourth_vessel')
plt.plot(fifth_vessel['lat'], fifth_vessel['lon'], 'o', color='blue', label='fifth_vessel')
plt.plot(sixth_vessel['lat'], sixth_vessel['lon'], 'o', color='grey', label='sixth_vessel')
plt.plot(seventh_vessel['lat'], seventh_vessel['lon'], 'o', color='pink', label='seventh_vessel')
plt.plot(eighth_vessel['lat'], eighth_vessel['lon'], 'o', color='purple', label='eighth_vessel')
plt.plot(nineth_vessel['lat'], nineth_vessel['lon'], 'o', color='tomato', label='nineth_vessel')
plt.plot(tenth_vessel['lat'], tenth_vessel['lon'], 'o', color='brown', label='tenth_vessel')
plt.plot(eleventh_vessel['lat'], eleventh_vessel['lon'], 'o', color='navy', label='eleventh_vessel')


plt.legend() # 显示图例
  
plt.xlabel('lat')
plt.ylabel('lon')
plt.show()


