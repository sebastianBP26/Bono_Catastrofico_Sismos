# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 21:52:34 2022

@author: Sebastian Barroso
"""

# Modulos
# Pre - instalacion
import pandas as pd
import numpy as np
# Propios
import catbond_module as cm

class HypothesisTesting():
    
    def __init__(self):
        
        # La clase HypothesisTesting nos permite realizar la prueba de hipotesis para nuestro vector X.
        
        self.damages = None # Observaciones del vector X, en nuestro caso seran los danios provocados.
        self.plot_hist = None # objeto boolean que nos permite imprimir el histograma de nuestros datos
        self.plot_bins = None # Numero de bins del histograma
        
        # Atributos de la clase:
        self.best_dist = '' # String: Mejor distribucion con base en el maximo p-value.
        self.best_pvalue = 0.0 # Float: p-value Maximo obtenido
        self.best_dist_attributes = {} # Dictionary: Atributos de la mejor distribucion elegida.
        self.dist_attributes = {} # Dict: Atributos, modelo para todas las distribuciones probadas
        self.summary_ht = pd.DataFrame() # pd.DataFrame: Resumen de la Prueba de Kolmogorov (Dist | D | p-value)
        self.distributions_names = [] # List: Almacena el nombre de cada una de las distribuciones probadas.
        
    def __str__(self):
        # Nos permite imprimir los "valores" de la clase
        str_self = 'Best Distribution: ' + self.best_dist + ' | p-value ' + str(np.round(self.best_pvalue,4)) 
        return str_self
    
    def fit(self, x, params = {'plot_hist': False, 'bins': 30}):
        # El metodo fit nos permitira realizar la graficacion y obtencion de resultados de las prueba.
    
        self.damages = x # Danios del evento aleatorio.
        self.plot_hist = params['plot_hist'] # Revisamos si se desea o no imprimir el histograma
        self.plot_bins = params['bins'] # Numero de bins
        
        # Llamamos a la funcion hypothesis_testing que nos permite realizar todo el proceso.
        self.best_dist, self.best_pvalue, self.best_dist_attributes, self.dist_attributes, self.summary_ht, self.distributions_names\
            = cm.hypothesis_testing(x = self.damages, print_hist= self.plot_hist, bn = self.plot_bins)
            
        self.dist = self.best_dist_attributes['model']
                