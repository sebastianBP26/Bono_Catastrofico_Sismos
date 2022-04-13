# -*- coding: utf-8 -*-
"""
Created on Sat Feb 12 22:02:24 2022

@author: Sebastian Barroso
"""
# Archivo principal donde se realizan las simulaciones de los precios sugeridos.
# Carga de modulos

# Pre - instalacion
import pandas as pd
import numpy as np

# Propios
from catbond_class import CatBond

# Carga de datos para la frecuencia (Base del Servicio Sismologico Nacional: http://www2.ssn.unam.mx:8080/catalogo/)
path = r'G:\Mi unidad\Tesis (Avances)\BASES\SSNMX_catalogo_19000101_20210913.csv'
mw = 6.5 # Magnitud de sismos que utilizaremos como "filtro", es decir, sismos mayores o iguales a mw

# Carga de datos de severidad de sismos (Recoleccion de datos de la EM-DAT, NOAA y CENAPRED)
severidad = pd.read_csv(r'G:\Mi unidad\Tesis (Avances)\BASES\damaged.csv', engine = 'python', encoding = 'unicode_escape')
x = severidad.Damaged.values # array x con los valores de la severidad
              
cb = CatBond() # Creamos una instancia de la clase CatBond

cb.hyp_test.fit(x, {'plot_hist': True, 'bins':10}) # Se llama el metodo fit para ajustar las distribuciones de perdidas.
out_dist = cb.hyp_test.summary_ht # Resultados del ajuste (Estadisticas y p-values)
att = cb.hyp_test.dist_attributes # Parametros estimados
print(cb.hyp_test.summary_ht) # Imprimimos los resultados

cb.hom_poisson.fit(path, mw) # Estimamos la tasa del proceso de poisson
cb.hom_poisson.simulate_process() # Simulamos el proceso con la tasa estimada.

distributions = cb.hyp_test.dist_attributes # Valores de las distribuciones ajustadas

# Parametros para la simulacion
params = {'simulaciones': 10000, # n (Numero de caminatas de Lt a generar)
          'maduracion': 365*4, # Tiempo de maduracion del Bono.
          'delta_pph': cb.hom_poisson.mu/360, # Tasa del proceso de Poisson (Tasa estimada con la clase Homogeneous Poisson Process)
          'valor_facial': 1, # Valor Facial del Bono.
          'tasa': -np.log(1/(1 + 0.05)), # Tasa de descuento/interes del bono (Se transforma a una tasa continua).
          'print_sheet': True, # Permite imprimir en pantalla la sabana.
          'plot': False} # Permite imprimir las simulaciones de Lt (No se recomienda cuando n es granda, ya que demora un poco mas).

sheets = [] # Lista para guardar las sabanas
lts = [] # Lista para guardar las simulaciones del Proceso de Perdida Agregado Lt_{i}

# Se utiliza un ciclo for para realizar la simulacion para cada una de las distribuciones
for distribution in distributions.keys():

    cb.hyp_test.dist = cb.hyp_test.dist_attributes[distribution]['model'] # Se selecciona la distribucion x
    cb.get_sheet(params) # Se calcula la sabana con el etodod
    sheets.append(cb.sheet) # Se hace un append del dataframe con los precios
    lts.append(cb.Lt) # Se hace un append del dataframe con las simulaciones de Lt
    # cb.interactive_surface() # Si desea imprimir una superficie interactiva en su navegador web.
    
    print(f''' Se realizo el ejercicio para la distribucion: {distribution}''')
