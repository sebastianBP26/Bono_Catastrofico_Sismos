# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 21:52:34 2022

@author: Sebastian Barroso
"""

import pandas as pd
import numpy as np
import catbond_module as cm

class HypothesisTesting():
    
    def __init__(self):
        
        # La clase HypothesisTesting nos permite realizar la prueba de hipótesis para nuestro vector X.
        
        self.damages = None # Observaciones del vector X, en nuestro caso serán los daños provocados.
        self.plot_hist = None # objeto boolean que nos permite imprimir el histograma de nuestros datos
        self.plot_bins = None # Número de bins del histograma
        
        # Atributos de la clase:
        self.best_dist = '' # String: Mejor distribución con base en el máximo p-value.
        self.best_pvalue = 0.0 # Float: p-value Máximo obtenido
        self.best_dist_attributes = {} # Dictionary: Atributos de la mejor distribución elegida.
        self.dist_attributes = {} # Dict: Atributos, modelo para todas las distribuciones probadas
        self.summary_ht = pd.DataFrame() # pd.DataFrame: Resumen de la Prueba de Kolmogorov (Dist | D | p-value)
        self.distributions_names = [] # List: Almacena el nombre de cada una de las distribuciones probadas.
        
    def __str__(self):
        # Nos permite imprimir los "valores" de la clase
        str_self = 'Best Distribution: ' + self.best_dist + ' | p-value ' + str(np.round(self.best_pvalue,4)) 
        return str_self
    
    def fit(self, x, params = {'plot_hist': False, 'bins': 30}):
        # El método fit nos permitirá realizar la graficación y obtención de resultados de las prueba.
    
        self.damages = x # Daños del evento aleatorio.
        self.plot_hist = params['plot_hist'] # Revisamos si se desea o no imprimir el histograma
        self.plot_bins = params['bins'] # Número de bins
        
        # Llamamos a la función hypothesis_testing que nos permite realizar todo el proceso.
        self.best_dist, self.best_pvalue, self.best_dist_attributes, self.dist_attributes, self.summary_ht, self.distributions_names\
            = cm.hypothesis_testing(x = self.damages, print_hist= self.plot_hist, bn = self.plot_bins)
            
        self.dist = self.best_dist_attributes['model']
                