# -*- coding: utf-8 -*-
"""
Created on Sat Feb 12 22:02:24 2022

@author: Sebastian Barroso
"""
import pandas as pd
import numpy as np
from catbond_class import CatBond

# Cargamos los datos de la frecuencia (total de sismos)
path = r'G:\Mi unidad\Tesis (Avances)\BASES\SSNMX_catalogo_19000101_20210913.csv'
mw = 6.5

# Cargamos los datos de severidad de sismos
severidad = pd.read_csv(r'G:\Mi unidad\Tesis (Avances)\BASES\damaged.csv', engine = 'python', encoding = 'unicode_escape')
x = severidad.Damaged.values
              
# Creamos una instancia de la clase  
cb = CatBond()

# Hacemos fit para ajustar las distribuciones de perdidas.
cb.hyp_test.fit(x, {'plot_hist': True, 'bins':10})
out_dist = cb.hyp_test.summary_ht # Resultados del ajuste (Estadisticas y p-values)
att = cb.hyp_test.dist_attributes # Parametros estimados
print(cb.hyp_test.summary_ht) # Imprimimos los resultados

# Estimamos la tasa del proceso de poisson
cb.hom_poisson.fit(path, mw)
# Simulamos el proceso con la tasa estimada.
cb.hom_poisson.simulate_process()

# main = pd.DataFrame()
# mws = np.arange(6.5, 7.6, .1)

# for mw in mws:
#     # Estimamos la tasa del proceso de poisson
#     cb.hom_poisson.fit(path, mw)
#     # Simulamos el proceso con la tasa estimada.
#     # cb.hom_poisson.simulate_process()
    
#     temp = pd.DataFrame({'Total': [cb.hom_poisson.size],
#                           'mu': [cb.hom_poisson.mu],
#                          'P(X=0)': [cb.hom_poisson.summary['P(X=x)'].values[0]],
#                          'P(X > 0)': [1 - cb.hom_poisson.summary['P(X=x)'].values[0]]})
    
#     main = pd.concat([main, temp])

# Distribuciones elegidas
distributions = cb.hyp_test.dist_attributes

# Especificamos los parametros para la simulacion del sismo
params = {'simulaciones': 10000, # n (Numero de caminatas de Lt a generar)
          'maduracion': 365*4, # Tiempo de maduracion del Bono.
          'delta_pph': cb.hom_poisson.mu/360, # Tasa del proceso de Poisson (estimada)
          'valor_facial': 1, # Valor Facial del Bono.
          'tasa': -np.log(1/(1 + 0.05)), # Tasa de descuento/interes del bono.
          'print_sheet': True, # Permite imprimir en pantalla la sabana.
          'plot': False} # Permite imprimir las simulaciones de Lt}

sheets = [] # Lista para guardar las sabanas
lts = [] # Guardamos las simulaciones de las Lt
for distribution in distributions.keys():

    cb.hyp_test.dist = cb.hyp_test.dist_attributes[distribution]['model']
    cb.get_sheet(params)
    sheets.append(cb.sheet)
    lts.append(cb.Lt)
    # cb.interactive_surface()
    
    print(f''' Se realizo el ejercicio para la distribucion: {distribution}''')
