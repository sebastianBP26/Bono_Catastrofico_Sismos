# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 22:17:30 2022

@author: Sebastian Barroso
"""

import pandas as pd
import numpy as np
from scipy.stats import poisson
from catbond_module import PPH
import matplotlib.pyplot as plt

def get_lambda(path, mw):
    
    '''
    Permite obtener el estimador para el parametro del Proceso de Poisson Homogeneo, el cual es 
    la media muestral:
            Total de eventos (Mayores o iguales a la mw elegida) / total de tiempo transcurrido. 
        path: Es la ubicacion de la base del SSN limpiada.
        mw: Magnitud minima para filtrar los datos
        
    Output: 
        lambda_m: Parametro lambda (ms) estimado.
        size: Total de eventos.
        total: Rango del periodo.
        df: Siniestros.
    '''
    
    df = pd.read_csv(path, encoding = 'unicode_escape', engine = 'python') # Leemos el csv
    df = df[df.Magnitud >= mw] # Se filtra la base
    df = df[['Fecha', 'Magnitud', 'Profundidad', 'Hora']] # Elegimos las columnas necesarias
    df.Fecha = pd.to_datetime(df['Fecha'].values + ' ' + df['Hora'].values, dayfirst = True)
    df.sort_values( by = ['Fecha'], ascending = [True], inplace = True) 
    df.drop(['Hora'], axis = 1, inplace = True)
    df['year'] = df['Fecha'].dt.year # Agregamos el anio
    
    years = list(set(df.year.values)) # lista de valores unicos de los anios
    total = int(max(years)) - int(min(years))  # Rango del periodo
    size = len(df) # Total de eventos mayores o iguales a mw
    
    lambda_m = np.round(size / total, 4) # Parametro estimado
    
    return lambda_m, size, total, df

def get_accumulatted(data, rango, mu, print_plot = True):
    
    '''
    Permite realizar la grafica del total de eventos acumulados vs total de eventos estimados
    haciendo uso del Proceso de Poisson Homogeneo.
        data: DataFrame con los datos filtrados.
        rango: Hasta que periodo estaremos simulando.
        mu: Tasa del PPH
        
    Output:
        main: pandas DataFrame con los datos simulados y datos reales.
    '''
    
    df = data
    df = df.groupby(['year'])['Magnitud'].count().reset_index()
    df['cummulative'] = df['Magnitud'].cumsum()
    df['key'] = df['year'].values - 1900
    df = df[['key', 'cummulative']]
    df.columns = ['key', 'cummulative_df']
    
    p = PPH(rango, mu)
    p = p.reset_index()
    p = p.groupby(['Value'])['index'].count().reset_index()
    p['cummulative'] = p['index'].cumsum()
    p = p[['Value', 'cummulative']]
    p.columns = ['key', 'cummulative_est']
    
    # par = np.linspace(0, pph.year, pph.year + 1)
    main = pd.DataFrame({'key': np.linspace(0, rango, rango + 1)})
    main = pd.merge(main, df, how = 'left', left_on = ['key'], right_on = ['key'])
    main = pd.merge(main, p, how = 'left', left_on = ['key'], right_on = ['key'])
    main['cummulative_df'] = main['cummulative_df'].fillna(method = 'pad')
    main['cummulative_est'] = main['cummulative_est'].fillna(method = 'pad')
    main.columns = ['tiempo', 'real', 'estimado']
    
    # Agregar las etiquetas
    clr = {'real': '#2eb094', 'estimado': '#b02e61'}
    plt.figure(dpi = 150, figsize = (10,8))
    plt.title('Eventos Acumulados vs Eventos Estimados | ms = ' + str(np.round(mu,4)), fontsize = 15)
    
    for col in list(main.columns[1:]):    
        plt.plot(main.tiempo.values,main[col].values, linewidth = 2, label = col, color = clr[col])
    
    plt.ylabel('Total de Eventos')
    plt.legend()
    plt.grid(color = '#191a1a', linestyle='--', linewidth = 0.1, alpha = 0.5)
    plt.show()
    
    return main

class HomogeneousPoissonProcess():
    
    def __init__(self):
        
        self.data = None # SSN Database which will be filter with the events.
        self.magnitud = None # Float: Value to be filter (at least >=)
        self.mu = 0.0 # Float: PPH Lambda Estimated
        self.size = 0.0 # Int: Total events
        self.year_max = 0.0
        self.model = 0.0 # scipy.poisson: Poisson Distribution Model
        self.mean = 0.0 # Media del modelo Poisson
        self.var = 0.0 # Varianza del modelo Poisson
        self.skew = 0.0 # Skewness del modelo Poisson
        self.kurtosis = 0.0 # Kurtosis del modelo Poisson
        
        self.summary = pd.DataFrame() # Resumen que queremos obtener
        self.nbr_rnd = 4
        
        self.accumulatted = pd.DataFrame()
        
    def fit(self, path, mw):
        
        self.data = path
        self.magnitud = mw
        
        # Obtenemos el parametro lambda y el total de sismos para esa lambda
        self.mu, self.size, self.year_max, self.data = get_lambda(self.data,self.magnitud)
                
        # Creamos el modelo, el cual utiliza la clase stats.poisson
        self.model = poisson(self.mu)
        
        # Medidas estadisticas: Media, Varianza, Kurtosis y Skweness
        self.mean, self.var, self.skew, self.kurtosis = self.model.stats('mvks')
        
        # Redondeamos los valores
        self.mean = round(float(self.mean), self.nbr_rnd)
        self.var = round(float(self.var), self.nbr_rnd)
        self.skew = round(float(self.skew), self.nbr_rnd)
        self.kurtosis = round(float(self.kurtosis), self.nbr_rnd)
        self.desv = round(np.sqrt(float(self.var)), self.nbr_rnd) #Desviacion estandar
        
        # Calculo de probabilidades
        probabilities = []
        cum_p = []
        sup_p = []
        
        values = [0,1,2,3,4,5,6,7,8]
        for w in values:
            probabilities.append(round(self.model.pmf(w),4))
            cum_p.append(round(self.model.cdf(w),4))
            sup_p.append(round(self.model.sf(w),4))
        
        # Asignamos los valores en un Data Frame de Pandas.
        self.summary['x'] = values
        self.summary['P(X=x)'] = probabilities
        self.summary['P(X<=x)'] = cum_p
        self.summary['P(X>=x)'] = sup_p
        
        self.mu = round(self.mu,4)
        
    def simulate_process(self):
    
        # Get the Accumulated Events vs Estimated
        self.accumulatted = get_accumulatted(self.data, self.year_max, self.mu)
        print(f'''Total de Eventos Acumulados vs Estimados: \n {self.accumulatted.iloc[-1]}''')
