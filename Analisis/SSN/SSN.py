# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 22:31:50 2021

@author: Sebastian Barroso
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

# Paqueteria plotly para las graficas
import plotly.io as pio
import plotly.express as px

# sns.set_context(rc={"axes.labelsize":10})

file = r'G:\Mi unidad\Tesis (Avances)\BASES\SSNMX_catalogo_19000101_20210913.csv'
ssn = pd.read_csv(file, encoding = 'utf-8', engine = 'python')

# Descripcion de los datos
ssn.info()
print('Total de Registros: ' + str(len(ssn)))

ssn.Fecha = pd.to_datetime(ssn['Fecha'].values + ' ' + ssn['Hora'].values, dayfirst = True)
ssn.sort_values( by = ['Fecha'], ascending = [True], inplace = True)
ssn.drop(['Hora'], axis = 1, inplace = True)

ssn.isna().sum()[ssn.isna().sum() > 0]
print('Valores nulos: ' + str(sum(ssn.isna().sum().values)))

# 17,649 valores nulos para la variable Magnitud.
# 66 Valores Nulos para la variable profundidad.

# Imputamos los valores nulos para las variables que observamos, haciendo uso de la mediana
ssn.Magnitud.fillna(round(np.nanmedian(ssn.Magnitud.values),2), inplace = True)
ssn.Profundidad.fillna(round(np.nanmedian(ssn.Profundidad.values),2), inplace = True)

# Creacion de nuevas variables
estados = []
for estado in ssn['Referencia de localizacion'].values:
    estados.append(estado.split(',')[1].strip())

ssn['epicentro'] = estados
ssn.drop(['Referencia de localizacion'], axis = 1, inplace = True)

ssn['year'] = ssn['Fecha'].dt.year # Agregamos el anio
ssn['quarter'] = ssn['Fecha'].dt.quarter # Agregamos el trimestre
ssn['month'] = ssn['Fecha'].dt.month # Agregamos el mes
ssn['day'] = ssn['Fecha'].dt.day # Agregamos el dia
ssn['hour'] = ssn['Fecha'].dt.hour # Agregamos la hora
ssn['minute'] = ssn['Fecha'].dt.minute # Agregamos el minuto

# Creacion de la categoria
cat = []
color = []
for magnitud in ssn.Magnitud.values:
    
    if magnitud < 3.5:
        cat.append('Daños Mínimos')
        color.append('#83b02e')
        
    elif magnitud >= 3.5 and magnitud < 5.4: 
        cat.append('Daños Menores')
        color.append('#FFC30F')
        
    elif magnitud >= 5.5 and magnitud < 6: 
        cat.append('Daños Ligeros')
        color.append('#FF5733')
        
    elif magnitud >= 6.1 and magnitud < 6.9: 
        cat.append('Daños Severos')
        color.append('#C70039')
        
    elif magnitud >= 7 and magnitud < 7.9: 
        cat.append('Daños Graves')
        color.append('#900C3F')
        
    else:
        cat.append('Gran Terremoto')
        color.append('#581845')
        
ssn['categoria'] = cat
ssn['color'] = color

# Creando Dataframe para Fechas
q = pd.DataFrame({'quarter':[1,2,3,4], 'Trimestre': ['Q1','Q2','Q3','Q4']})
ssn = pd.merge(ssn, q, how = 'left', left_on = ['quarter'], right_on = ['quarter'])

mes = pd.DataFrame({'month': [1,2,3,4,5,6,7,8,9,10,11,12],
                    'Mes': ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 
                            'Nov', 'Dic']})
ssn = pd.merge(ssn, mes, how = 'left', left_on = ['month'], right_on = ['month'])

ssn.reset_index(inplace = True)

resumen = ssn.describe()

df_prom = ssn[ssn.year == 2020]
print('Promedio mensual de sismos en México: ' + str(int(len(df_prom)/12)))
print('Promedio Magnitud: ' + str(np.mean(df_prom.Magnitud.values)))

# FUNCION PARA LAS GRAFICAS
def metricas(x, flag = True, round_n = 2):
    
    count = len(x)
    mean = np.round(np.mean(x),round_n)
    std = np.round(np.std(x),round_n)
    min_x = np.round(np.min(x),round_n)
    max_x = np.round(np.max(x),round_n)
    q  = np.percentile(a = x, q = [25,50,75])
    q_25 = q[0]
    mediana = q[1]
    q_75 = q[2]
    
    m = {'count': count,
         'mean': mean,
         'std': std,
         'min': min_x,
         'max': max_x,
         'q_25': q_25,
         'mediana': mediana,
         'q_75': q_75}
    
    str_values = 'Media: ' + str(mean) + ' | ' + \
                 'Mediana: ' + str(mediana) + ' | ' + \
                 'Std: ' + str(std) + ' | ' + \
                 'Min: ' + str(min_x) + ' | ' + \
                 'Max: ' + str(max_x) + ' | ' + \
                 '$q_{25}$: ' + str(q_25) + ' | ' + \
                 '$q_{75}$: ' + str(q_75) 
    
    if flag:
        
        print(str_values)
        
    return m, str_values


# HISTOGRAMAS
# Histograma de la variable Magnitud
fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (10,8), dpi = 250)
ax.set_title('Histograma de la variable Magnitud', fontsize = 22)
sns.histplot(data = ssn, x = 'Magnitud', color = '#c81025',
             bins = 40, stat = 'probability', label = 'Magnitud', ax = ax)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.yaxis.set_tick_params(labelsize = 18)
ax.xaxis.set_tick_params(labelsize = 18)
ax.legend(loc = 'upper left', shadow = True, ncol = 1, fontsize = 22)
ax.set_xlabel(metricas(ssn.Magnitud.values)[1], fontsize = 16)
ax.set_ylabel('Probabilidad', fontsize = 18)
plt.grid(color = '#191a1a', linestyle='--', linewidth = 0.1, alpha = 0.5)

# Histograma de la variable Profundidad
fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (10,8), dpi = 250)
ax.set_title('Histograma de la variable Profundidad', fontsize = 22)
sns.histplot(data = ssn, x = 'Profundidad', color = '#c81025',
             bins = 40, stat = 'probability', label = 'Profundidad', ax = ax)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.yaxis.set_tick_params(labelsize = 18)
ax.xaxis.set_tick_params(labelsize = 18)
ax.legend(loc = 'upper right', shadow = True, ncol = 1, fontsize = 22)
ax.set_xlabel(metricas(ssn.Profundidad.values)[1], fontsize = 16)
ax.set_ylabel('Probabilidad', fontsize = 18)
plt.grid(color = '#191a1a', linestyle='--', linewidth = 0.1, alpha = 0.5)

# Matriz de correlaciones
corr_matrix = ssn[['Magnitud', 'Latitud', 'Longitud', 'Profundidad','year', 'month', 'day', 'quarter']].corr()

plt.figure(dpi = 150, figsize = (12,12))
plt.title('Matriz de Correlación', fontsize = 12)
ax = sns.heatmap(corr_matrix, linewidths = 0.1, annot = True, cmap = 'YlGnBu')
plt.show()

test = pd.pivot_table(ssn, values = 'index', index = ['day'], 
                      columns = ['month'], aggfunc = 'count')

# avg = int(np.round(np.nanmean(test),0))

plt.figure(dpi = 150, figsize = (12,12))
plt.title('Heatmap: Conteo de Sismos por Mes/Día', fontsize = 12)
ax = sns.heatmap(test, linewidths = 0.1, annot = True, fmt = '.0f', cmap = 'YlGnBu')
# plt.xlabel('Mes \n Promedio de Sismos Mensual: ' + str(avg))
plt.xlabel('Mes')
plt.ylabel('Día')
plt.show()

# Sin el 29 de Feb
ssn_temp = ssn[(ssn.month != 2) | (ssn.day != 29)]

test = pd.pivot_table(ssn_temp, values = 'index', index = ['day'], 
                      columns = ['month'], aggfunc = 'count')

# avg = int(np.round(np.nanmean(test),0))

plt.figure(dpi = 250, figsize = (12,12))
plt.title('Heatmap: Conteo de Sismos por Mes/Día', fontsize = 20)
ax = sns.heatmap(test, linewidths = 0.1, annot = True, fmt = '.0f', cmap = 'YlGnBu', annot_kws = {'fontsize':14})
plt.xlabel('')
plt.ylabel('')
plt.xticks(size = 15)
plt.yticks(size = 15)
plt.yticks(rotation = 0)
plt.show()

# Heatmap Hour vs Day
test = pd.pivot_table(ssn, values = 'index', index = ['day'], 
                      columns = ['hour'], aggfunc = 'count')
plt.figure(dpi = 250, figsize = (12,14))
plt.title('Heatmap: Conteo de Sismos por Día/Hora', fontsize = 12)
ax = sns.heatmap(test, linewidths = 0.1, annot = True, fmt = '.0f', 
                 cmap = 'YlGnBu', annot_kws={'size': 7})
plt.xlabel('Hora')
plt.ylabel('Día')
plt.show()

test = pd.pivot_table(ssn[ssn.month == 9], values = 'index', index = ['day'], 
                      columns = ['hour'], aggfunc = 'count')

# avg = int(np.round(np.nanmean(test),0))

plt.figure(dpi = 250, figsize = (12,14))
plt.title('Heatmap: Conteo de Sismos por Día/Hora', fontsize = 12)
ax = sns.heatmap(test, linewidths = 0.1, annot = True, fmt = '.0f', 
                 cmap = 'YlGnBu', annot_kws={'size': 7})

plt.xlabel('Hora')
plt.ylabel('Día')
plt.show()


# SISMOS Mayores
test = pd.pivot_table(ssn[ssn.Magnitud >= 5], values = 'index', index = ['day'], 
                      columns = ['month'], aggfunc = 'count')

# avg = int(np.round(np.nanmean(test),0))

plt.figure(dpi = 250, figsize = (12,14))
plt.title('Heatmap: Conteo de Sismos por Día/Hora', fontsize = 12)
ax = sns.heatmap(test, linewidths = 0.1, annot = True, fmt = '.0f', 
                 cmap = 'YlGnBu', annot_kws={'size': 7})
plt.xlabel('Mes')
plt.ylabel('Día')
plt.show()

# Conteo de estados
c1 = ssn.groupby(by = ['epicentro'])[['Magnitud']].count().reset_index()
c1.Magnitud = c1.Magnitud/sum(c1['Magnitud'].values)
c1.sort_values(['Magnitud'], ascending = [False], inplace = True)
c1.reset_index(inplace = True, drop = True)
c1.columns = ['epicentro', 'Total']

# Conteo de sismos por Epicentro (TOP 10)
plt.figure(dpi = 100, figsize = (10,8))
plt.title('Top 10 de Estados: % Del Total de Sismos', fontsize = 15)
ax = sns.barplot(x = 'epicentro', y = 'Total', palette= 'vlag', data = c1.head(10))
ax.bar_label(ax.containers[0], fmt = '%.2f')
# plt.legend()

# Scatter Plot
temp = ssn[['Magnitud', 'Latitud', 'Longitud', 'Profundidad']]

g = sns.pairplot(temp, diag_kind = 'hist', diag_kws = {'alpha':0.99, 'bins':40})
g.fig.subplots_adjust(top = 0.95)
g.fig.suptitle('Resumen General de las Variables Numéricas', fontsize = 20)
g.fig.dpi = 200
g.fig.set_size_inches(10,10)
# sns.set_context("paper", rc={"axes.labelsize":30})

# fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (15,15), dpi = 200)
# ax.set_title('Resumen General de las Variables Numéricas', fontsize = 22)
# g = sns.pairplot(temp, diag_kind = 'hist', diag_kws = {'alpha':0.99, 'bins':40})
# ax.yaxis.set_tick_params(labelsize = 14)
# ax.xaxis.set_tick_params(labelsize = 14)


def stats(df, g, t):
    
    grupos = list(set(df[g].values))
    grupos.sort()
    
    d = {}
    
    for grupo in grupos:
        
        temp = df[(df[g] == grupo)]
        x = temp[t].values
        
        td = {'Mean': np.nanmean(x),
              'Median': np.nanmedian(x),
              'Min': np.nanmin(x),
              'Max': np.nanmax(x)}
        
        d[grupo] = td
        
    return d
    
# Catplot

count_q = ssn.groupby( by = ['Trimestre','month','day'])['epicentro'].count().reset_index()
count_q.columns = ['Trimestre', 'Mes', 'Dia', 'Total']

g = sns.catplot(data = count_q, x = 'Trimestre', y = 'Total',
                kind = 'box', order = ['Q1','Q2','Q3','Q4'])
g.fig.subplots_adjust(top = 0.95)
g.fig.suptitle('Diagrama de Caja: Total de Sismos Trimestral')
g.fig.dpi = 100
g.fig.set_size_inches(10,8)

# SIN 29 DE FEB
count_q = ssn_temp.groupby( by = ['Trimestre','month','day'])['epicentro'].count().reset_index()
count_q.columns = ['Trimestre', 'Mes', 'Dia', 'Total']

fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (10,8), dpi = 250)
ax.set_title('Diagrama de Caja: Total de Sismos Trimestral', fontsize = 22)
sns.boxplot(data = count_q, x = 'Trimestre', y = 'Total', order = ['Q1','Q2','Q3','Q4'])
ax.yaxis.set_tick_params(labelsize = 18)
ax.xaxis.set_tick_params(labelsize = 18)
ax.set_xlabel('', fontsize = 18)
ax.set_ylabel('', fontsize = 18)
plt.grid(color = '#191a1a', linestyle='--', linewidth = 0.1, alpha = 0.5)

# Conteo de sismos por Mes y Día
count_md = ssn.groupby( by = ['month','day'])['epicentro'].count().reset_index()
count_md.columns = ['Mes', 'dia', 'Total']
meses = pd.DataFrame({ 'Mes': [1,2,3,4,5,6,7,8,9,10,11,12],
                       'Mes_N': ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun',
                                 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']})
count_md = pd.merge(count_md, meses, how = 'left', left_on = ['Mes'], right_on = ['Mes'])
count_md.drop(['Mes'], axis = 1, inplace = True)
count_md.columns = ['dia', 'Total', 'Mes']
count_md = count_md[(count_md.Mes != 'Feb') | (count_md.dia != 29)]


fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (10,8), dpi = 250)
ax.set_title('Diagrama de Caja: Total de Sismos Mensual', fontsize = 22)
sns.boxplot(data = count_md, x = 'Mes', y = 'Total', order = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun',
                                                              'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic'])
ax.yaxis.set_tick_params(labelsize = 18)
ax.xaxis.set_tick_params(labelsize = 18)
ax.set_xlabel('', fontsize = 18)
ax.set_ylabel('', fontsize = 18)
plt.grid(color = '#191a1a', linestyle='--', linewidth = 0.1, alpha = 0.5)


# Boxplot
count_estado = ssn.groupby( by = ['epicentro','month', 'day'])['Magnitud'].count().reset_index()

plt.figure(dpi = 150, figsize = (10,8))
plt.title('Diagrama de Caja: Total de Sismos Top 10 de Estados', fontsize = 15)
sns.boxplot(x = "epicentro", y = "Magnitud",
            data = count_estado[count_estado.epicentro.isin(list(c1.head(10)['epicentro'].values))],
            order = c1.head(10)['epicentro'].values)
sns.despine(offset=10, trim=True)


count_estado = ssn_temp.groupby( by = ['epicentro','month', 'day'])['Magnitud'].count().reset_index()

plt.figure(dpi = 150, figsize = (10,8))
plt.title('Diagrama de Caja: Total de Sismos Top 10 de Estados', fontsize = 15)
sns.boxplot(x = "epicentro", y = "Magnitud",
            data = count_estado[count_estado.epicentro.isin(list(c1.head(10)['epicentro'].values))],
            order = c1.head(10)['epicentro'].values)
sns.despine(offset=10, trim=True)

# Facet Grid Por Categoría
g = sns.FacetGrid(ssn, col = 'categoria', sharex = True, sharey = True, hue = 'Trimestre',
                  col_order = ['Daños Mínimos', 'Daños Menores', 'Daños Ligeros',
                                                   'Daños Severos', 'Daños Graves', 'Gran Terremoto'])
g.map_dataframe(sns.scatterplot, x = 'Magnitud', y = 'Profundidad')
g.fig.dpi = 150
g.fig.set_size_inches(15,4)

for ax in g.axes.ravel():
    ax.legend()
    
# Facet Grid Por Categoría
g = sns.FacetGrid(ssn, col = 'categoria', sharex = True, sharey = True, hue = 'Trimestre',
                  col_order = ['Daños Mínimos', 'Daños Menores', 'Daños Ligeros',
                                                   'Daños Severos', 'Daños Graves', 'Gran Terremoto'])
g.map_dataframe(sns.scatterplot, x = 'Latitud', y = 'Longitud')
g.fig.dpi = 200
g.fig.set_size_inches(15,4)

for ax in g.axes.ravel():
    ax.legend()
    
    
# MAPA
pio.renderers.default = 'browser'
token = 'pk.eyJ1Ijoic2ViYXN0aWFuYnAyNiIsImEiOiJja3o4Zmdmd3Ixa2duMnZuZmw1MXhkbGVxIn0.kCIdJxC5h6PbfGBvkKwO-A'

px.set_mapbox_access_token(token)

mag = 7
temp = ssn[ssn.Magnitud >= mag]

# EXPRESS
fig = px.scatter_mapbox(temp, 
                        lat = 'Latitud', 
                        lon = 'Longitud', 
                        color = 'categoria', 
                        size = np.exp(temp['Magnitud'].values)*10,
                        template = 'simple_white',
                        category_orders = {'categoria':['Gran Terremoto','Daños Graves','Daños Severos','Daños Ligeros','Daños Menores']},
                        color_discrete_sequence   = ['#581845', '#900C3F', '#C70039', '#FF5733', '#FFC30F', '#83b02e'],
                        zoom = 4,
                        opacity = 0.7,
                        center = dict(lat = 22.76843, 
                                      lon = -102.58141),
                        title = 'Sismos en México de Magnitud mayor o igual a '+ str(mag))
fig.show()

