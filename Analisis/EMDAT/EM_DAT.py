# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 17:36:05 2022

@author: Sebastian Barroso
"""

# Script para realizar el análisis de la base de la EMDAT.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Se carga el archivo csv a un objeto DataFrame de pandas.
df = pd.read_csv(r'G:\Mi unidad\Tesis (Avances)\BASES\emdat_public_2022_01_15.csv')

# Se observa que varias columnas tienen valores nulos, por lo que removeremos algunas columnas
df.info() 

cols = pd.DataFrame(df.isna().sum()/len(df)).reset_index() # Identificamos el % de renglones con na con un Df
cols.columns = ['Column', 'Porcentaje'] # Cambiamos el nombre de las columnas
cols = cols[cols.Porcentaje < 0.7] # Seleccionamos todas aquellas que tengan menos del 70% de valores nulos
cols = cols.Column.values # Seleccionamos el nombre de las columnas
df = df.loc[:, cols] # Filtramos a df con las columnas anteriores.

# ANALISIS A NIVEL GENERAL

# Total de Eventos por década
t1 = df.groupby(['Year'])['Dis No'].count().reset_index() # Agrupamos por año para contar el núm de eventos.
bucket_decade = [1900, 1910, 1920, 1930, 1940, 1950, 1960, 1970, 1980, 1990, 2000, 2010, 2020, 2030] # Bucket de decadas
decade = [] # Lista para llenarla con dataframes y agregar como columna

for d in range(len(bucket_decade) - 1):
    temp = t1[(t1.Year >= bucket_decade[d]) & (t1.Year < bucket_decade[d + 1])]
    temp['Decada'] = [bucket_decade[d]]*len(temp)
    decade.append(temp)
    
decade = pd.concat(decade) # Concatenamos c/u de los dataframes generados

decade = decade.groupby(['Decada'])['Dis No'].sum().reset_index() # Agrupamos por década para contar los eventos
decade['porcentaje_total'] = decade['Dis No'] / sum(decade['Dis No']) # Calculamos el % del total
decade['Accumulative'] = decade['porcentaje_total'].values.cumsum() # % Acumulado por década

# Figura 1: % Del total de eventos por Década
y_ticks = np.round(np.linspace(min(decade['porcentaje_total'].values), max(decade['porcentaje_total'].values), 10),2)

plt.figure(dpi = 150, figsize = (10,8))
plt.title('% Del total de Eventos por Década', fontsize = 14)
ax = sns.barplot(x = 'Decada', y = 'porcentaje_total', color = '#c81025', alpha = 0.8, data = decade)
ax.bar_label(ax.containers[0], fmt = '%.2f')
plt.xlabel('')
plt.ylabel('% del Total')
plt.grid(color = '#191a1a', linestyle='--', linewidth = 0.1, alpha = 0.5)
plt.xticks(size = 8)
plt.yticks(ticks = y_ticks, size = 8)
plt.show()

# Figura 2: % Del Total de Eventos Acumulado por Década
y_ticks = np.round(np.linspace(min(decade['Accumulative'].values), max(decade['Accumulative'].values), 10),2)

plt.figure(dpi = 150, figsize = (10,8))
plt.title('% Del Total de Eventos Acumulado por Década', fontsize = 14)
ax = sns.barplot(x = 'Decada', y = 'Accumulative', color = '#c81025', alpha = 0.8, data = decade)
ax.bar_label(ax.containers[0], fmt = '%.2f')

plt.xlabel('')
plt.ylabel('% del Total ')
plt.grid(color = '#191a1a', linestyle='--', linewidth = 0.1, alpha = 0.5)
plt.xticks(size = 8)
plt.yticks(ticks = y_ticks, size = 8)
plt.show()

# Se decide remover la información previa a 1970 para evitar sesgo en los datos
# También se remueve el evento de subgrupo 'extra terrestrial' ya que solo hay un evento.
df = df[(df.Year >= 1970) & (~df['Disaster Subgroup'].isin(['Extra-terrestrial']))]

# Descripciones generales
cc = df.loc[:, ['Year', 'Total Deaths', 'Total Affected', 'Total Damages (\'000 US$)']].describe()
cc.columns = cc.columns.str.replace(' ', '_').str.upper()
cc['TOTAL_AFFECTED'] = cc['TOTAL_AFFECTED'].astype('int64')
cc["TOTAL_DAMAGES_(\'000_US$)"] = cc["TOTAL_DAMAGES_(\'000_US$)"].astype('int64')
cc.columns = ['AÑOS', 'TOTAL DE MUERTES', ' TOTAL DE AFECTADOS', ' TOTAL DE DAÑO']

var = df.describe()
con_count = df.groupby(['Continent'])[['Dis No']].count().reset_index()
con_count.sort_values(['Dis No'], ascending = [False], inplace = True)

avg_ev = np.round(len(df)/(121 - 70),0) # Promedio de número de eventos catastróficos

dec1 = df.sort_values(['Total Damages (\'000 US$)'], ascending = [False]).head(50) # Peores dessastres en términos $$

# Figura ?: Daño vs Total de muertos
plt.figure(dpi = 150, figsize = (10,8))
plt.title('Daño total vs Total de Muertes', fontsize = 14)
plt.scatter(x = dec1['Total Damages, Adjusted (\'000 US$)'].values, 
            y = dec1['Total Deaths'].values, alpha = 0.7)
plt.grid(color = '#191a1a', linestyle='--', linewidth = 0.1, alpha = 0.5)
plt.show()

dec2 = df.sort_values(['Total Deaths'], ascending = [False]).head(50) # Peores desastres por total de muertes

# Figura ?: Daño vs Total de muertos
plt.figure(dpi = 150, figsize = (10,8))
plt.title('Daño total vs Total de Muertes', fontsize = 14)
plt.scatter(x = dec2['Total Damages, Adjusted (\'000 US$)'].values, 
            y = dec2['Total Deaths'].values, alpha = 0.7)
plt.grid(color = '#191a1a', linestyle='--', linewidth = 0.1, alpha = 0.5)
plt.show()

t1 = df.groupby(['Year'])['Dis No'].count().reset_index() # Conteo de eventos por año

# Figura 3: Total de eventos por año
plt.figure(dpi = 150, figsize = (10,8))
plt.title('Total de Desastres por Año', fontsize = 14)
plt.plot(t1.Year.values, t1['Dis No'].values, linewidth = 3, color = '#c81025')
plt.ylabel('Número de Eventos')
plt.grid(color = '#191a1a', linestyle='--', linewidth = 0.1, alpha = 0.5)
plt.show()

t2 = df.groupby(['Year', 'Continent'])['Dis No'].count().reset_index() # Total de Eventos por  Año y Continente

# Figura 4: Total de Eventos por Continente
colors = {'Americas': '#ee2d21', 'Asia': '#f9f02d', 'Europe': '#3895e0', 'Africa': '#000000', 'Oceania': '#48f814'}
plt.figure(dpi = 150, figsize = (10,8))
plt.title('Total de Eventos por Continente por Año', fontsize = 12)
for value in list(set(t2.Continent.values)):
    temp = t2[t2.Continent == value]
    plt.plot(temp.Year.values, temp['Dis No'].values, linewidth = 1.5, color = colors[value], label = value)
    
plt.ylabel('Número de Eventos')
plt.legend()
plt.grid(color = '#191a1a', linestyle='--', linewidth = 0.1, alpha = 0.5)
plt.show()


t3 = df.groupby( ['Disaster Subgroup'])[['Dis No']].count().reset_index() # Conteo por Sub Tipo de Desastre
t3['porcentaje_del_total'] = t3['Dis No']/sum(t3['Dis No']) # Obtenemos % del total
t3.sort_values(['Disaster Subgroup'], inplace = True) # Ordenamos por Subgrupo de desastre

# Colores de las categorias
colors = ['#21e50d', '#ea16d4', '#b03a2e', '#16adea', '#ead016']

# Total de Muertes por subtipo de desastre
numeric_var = ['Total Deaths', 'Total Affected', 'Total Damages (\'000 US$)']
t4 = df.groupby(['Disaster Subgroup'])[numeric_var].sum().reset_index()
t4.sort_values(['Disaster Subgroup'], inplace = True)

a = t4.iloc[:,[1,2,3]]
X = a.values
s = X.sum(axis = 0).T
X = X/s

t5 = pd.DataFrame(X, index = t4.iloc[:,0].values, columns = a.columns).reset_index()
t5.columns = ['index', 'Muertes', 'Afectados', 'Daño en 000 USD']
t5['Número de Eventos'] = t3['porcentaje_del_total'].values
t5 = t5[['index', 'Número de Eventos', 'Afectados', 'Muertes', 'Daño en 000 USD']]

ran = t5.iloc[:,[1,2,3,4]]
ran = ran.values
min_ran = np.min(ran)
max_ran = np.max(ran)

y_ticks = np.round(np.linspace(0,max_ran, 10),2)

etiquetas = pd.DataFrame({'index': ['Biological', 'Climatological', 'Geophysical', 'Hydrological','Meteorological'],
                          'Index': ['Bio', 'Clima', 'Geo', 'Hidro', 'Meteor']})

t5 = pd.merge(t5, etiquetas, how = 'left', left_on = ['index'], right_on = ['index'])
t5 = t5[['Index', 'Número de Eventos', 'Afectados', 'Muertes', 'Daño en 000 USD']]

# Figura 6 - 8: % Del total de x por Subtipo de Desastre
for col in list(t5.columns)[1:]:
    
    plt.figure(dpi = 250, figsize = (10,10))
    plt.title('% ' + str(col.capitalize()), fontsize = 14)
    plt.bar(t5['Index'], t5[col], color = colors, edgecolor = 'black')
    plt.grid(color = '#191a1a', linestyle='--', linewidth = 0.1, alpha = 0.5)
    plt.xticks(size = 14)
    if col == 'Número de Eventos':
        plt.yticks(ticks = y_ticks, size = 14)
    else:
        plt.yticks([])
    # plt.axis('off')
    plt.show()


test = pd.pivot_table(df[~df['Disaster Subgroup'].isin(['Extra-terrestrial'])], values = 'Dis No', index = ['ISO'], 
                      columns = ['Disaster Subgroup'], aggfunc = 'count') # pivot de los datos para el heatmap
test = test.fillna(0) # Rellenamos con 0s los valores nulos (no hubo eventos)
test['Total'] = test.values.sum(axis = 1) # Sumamos por renglones
test.sort_values(['Total'], ascending = [False], inplace = True) # Ordenamos por el total de eventos de mayor a menor

# Figure 6: Mapa de Calor
top = 10
plt.figure(dpi = 250, figsize = (10,10))
plt.title(f'''Mapa de Calor: Top {top} del total de Eventos por País''', fontsize = 16)
ax = sns.heatmap(test.head(top), linewidths = 0.1, annot = True, fmt = '.0f', 
                 cbar = False, cmap = 'crest', annot_kws = {'fontsize':14})
plt.ylabel('', size =14)
plt.xlabel('', size = 14)
plt.show()

#Figura 8: Grid Por Categoría
t6 = df.groupby( ['Disaster Subgroup', 'Continent'])[['Dis No']].count().reset_index() # Conteo por Sub Tipo de Desastre
continents = t6.Continent.value_counts().reset_index()['index'].values # Continentes

fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize = (25, 6), sharey = False)
fig.suptitle('% Del Total de Eventos por Continente y Subgrupo de Desastre')
axs = [ax1, ax2, ax3, ax4, ax5]

for i in range(len(axs)):
    
    temp = t6[t6['Continent'] == continents[i]]
    x = temp['Dis No'].values/sum(temp['Dis No'].values)
    axs[i].bar(temp['Disaster Subgroup'].values, x, color = colors, edgecolor = 'black')
    axs[i].set(ylabel = '% Del Total')
    axs[i].set_title(str(continents[i]), fontsize = 11)
    axs[i].grid(color = '#191a1a', linestyle='--', linewidth = 0.1, alpha = 0.5)
    axs[i].tick_params(labelsize = 6, axis='both', which='major')
    

# ANALISIS SOLO MEXICO
mex = df[df.ISO=='MEX']

# Seleccionamos las columnas con las que estaremos trabajando.
mex = mex[['Dis No', 'Year', 'Disaster Subgroup', 'Disaster Type', 'Total Deaths','Total Affected', 
           'Total Damages (\'000 US$)', 'Total Damages, Adjusted (\'000 US$)']]

# Renombramos las columnas por comidad
mex.columns = ['ID', 'YEAR', 'SUBGROUP', 'TYPE', 'DEATHS', 'AFFECTED', 'DAMAGES', 'DAMAGES_ADJUSTED']

# Total de Muertes por subtipo de desastre MEX

m0 = mex.groupby(['SUBGROUP'])[['ID']].count().reset_index() # Conteo por Sub Tipo de Desastre
m0['porcentaje_del_total'] = m0['ID']/sum(m0['ID']) # Obtenemos % del total
m0.sort_values(['SUBGROUP'], inplace = True) # Ordenamos por Subgrupo de desastre

numeric_var = ['DEATHS', 'AFFECTED', 'DAMAGES']
m1 = mex.groupby(['SUBGROUP'])[numeric_var].sum().reset_index()
m1.sort_values(['SUBGROUP'], inplace = True)

a = m1.iloc[:,[1,2,3]]
X = a.values
s = X.sum(axis = 0).T
X = X/s

m2 = pd.DataFrame(X, index = m1.iloc[:,0].values, columns = a.columns).reset_index()
m2.columns = ['index', 'Muertes', 'Afectados', 'Daño en 000 USD']

m3 = m2
m3['Número de Eventos'] = m0['porcentaje_del_total'].values
m3 = m3[['index', 'Número de Eventos', 'Afectados', 'Muertes', 'Daño en 000 USD']]

m3 = pd.merge(m3, etiquetas, how = 'left', left_on = ['index'], right_on = ['index'])
m3 = m3[['Index', 'Número de Eventos', 'Afectados', 'Muertes', 'Daño en 000 USD']]

# Figura 6 - 8: % Del total de x por Subtipo de Desastre
for col in list(m3.columns)[1:]:
    
    plt.figure(dpi = 250, figsize = (10,10))
    plt.title('% ' + str(col.capitalize()), fontsize = 14)
    plt.bar(m3['Index'], m3[col], color = colors, edgecolor = 'black')
    plt.grid(color = '#191a1a', linestyle='--', linewidth = 0.1, alpha = 0.5)
    plt.xticks(size = 14)
    if col == 'Número de Eventos':
        plt.yticks(ticks = y_ticks, size = 14)
    else:
        plt.yticks([])
    # plt.axis('off')
    plt.show()