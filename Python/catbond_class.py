# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 22:30:20 2022

@author: Sebastian Barroso
"""
# Clase para modelar el precio del CatBond para Sismos en Mexico.

# MODULOS 
import time # Modulos de Python
# Pre-instalacion
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
# Propios
import catbond_module as cm
from class_hypothesis_testing import HypothesisTesting
from class_HomogeneousPoissonProcess import HomogeneousPoissonProcess

class CatBond():
    
    def __init__(self):
        
        '''
        La clase tiene como objetivo poder realizar el analisis del precio del bono (CatBond) para
        sismos en Mexico de una manera "directa". Concretamente, nos referimos a que si se cuenta con una
        base (archivo csv o vector) con los datos de la severidad y de la frecuencia (Se supone la base del SSN)
        se pueda realizar la valuacion del bono con la metodologia elegida. 
        
        Para esto, la clase CatBond tendra diversos atributos para hacer referencia a aquellos que
        afectan al precio del bono (Distribucion, tiempo de maduracion, tasa, etc.). Sin embargo,
        tendra dos atributos que a su vez son dos clases y que nos permitiran ajustar las distribuciones elegidas
        ('lognorm', 'gamma', 'pareto', 'burr12') para la Severidad y ajustar el PPH(m) para la severidad. 
        
            HypothesisTesting: La clase permite realizar la prueba Kolmogorov-Smirnov y Cramer para las distribuciones
                               mencionadas. 
                               
            HomogeneousPoissonProcess: Permite estimar la lambda del proceso de Poisson Homogeneo con la informacion
                                      proporcionada por el Servicio Sismologico Nacional (SSN).
                                      
            Para mas informacion acerca de los atributos y metodos de estas dos clases, favor de acceder a su archivo,
            donde se encuentran comentados.
            
        Una vez que contamos con la distribucion y la tasa del proceso de Poisson, se podran usar estos atributos como
        parametros del metodo get_price() o get_sheet() para calcular los precios del bono.
        
        * Cuando se llame a los metodos para obtener precios, solicitaran un diccionario con los parametros del precio del bono,
        los cuales son:
            n: Numero de simulaciones (caminatas aleatorias) de Lt.
            maturity_time: Tiempo de maduracion del bono.
            delt_pph: Delta del PPH().
            face_value: Valor facial del Bono.
            rate: Tasa de interes/descuento del bono.
            *umbral: threshold.
            *tiempo: Tiempo al cual se valua el precio del bono.
            plot: Boolean que permite imprimir la grafica de las caminatas aleatorias.
            print_sheet: Boolean que permite imprimir la surface de los precios.
            
        Los campos marcados con * solo son para el metodo get_price() pues es para un solo valor y se debe especificar
        el threshold y el tiempo.
        
        Observaciones:
            Por como esta construida la clase, no es necesario ejecutar self.hom_poisson.fit(), ya que podemos asignar
            "manualmente" la tasa, es decir, la clase se encarga de estimar lambda, pero si deseara valuar con otra lambda,
            se podria asignar este valor al atributo de la clase.
        
        '''
        # Atributos de la clase.
        
        # Variables de entrada para inicializar la clase: Datos de la severidad.
        self.hyp_test = HypothesisTesting() # Para mayor detalle de la clase, ir al archivo.
        
        # Variables de entrada para inicializar la clase: Datos de la frecuencia.
        self.hom_poisson = HomogeneousPoissonProcess() # Para mayor detalle de la clase, ir al archivo.
        
        # Atributos: Metodo  get_price: para calcular el precio del bono a tiempo t.
        self.n = 0 # int: Numero de simulaciones a realizar.
        self.Lt = pd.DataFrame() # Lt: Proceso de Perdida Agregado
        self.maturity_time = 0 # int: Tiempo de Maduracion.
        self.face_value = 0.0 # Float: Valor Faciald del Bono
        self.rate = 0.0 # Float: Tasa del Proceso de Poisson.
        self.umbral = 0.0 # Float: Umbral (Threshold)
        self.price = 0.0 # Float: Precio del Bono.
        
        # Atributos del metodo get_sheet: Permite calcular el precio del bono para todo t y D.
        self.time_range = [] # Lista que tiene los tiempos en los que haremos la simulacion (particion del intervalo).
        self.umbrales = [] # Lista que tiene los umbrales en los que haremos la simulacion (particion del intervalo).
        self.sheet = pd.DataFrame() # pd.DF que tendra los precios del bono para cada tiempo y umbral.
        self.time_spend = 0.0 # Variable que nos indicara el tiempo en segundos que tardo el proceso en encontrar los precios.
        self.print_sheet = False # Bool: Permite imprimir la sabana de precios (Estatica).
        

    def get_price(self, params):
        
        self.n = params['simulaciones']
        self.maturity_time = params['maduracion']
        self.delta_pph = params['delta_pph']
        self.face_value = params['valor_facial']
        self.rate = params['tasa']
        self.umbral = params['umbral']
        self.time = params['tiempo']
        
        re = cm.generate_process(n = self.n, T = self.maturity_time, m = self.delta_pph, 
                                 dist = self.hyp_test.dist, plot = params['plot'])
        self.price = cm.price_bond(lt = re, zt = self.face_value, r = self.rate, t = self.time, 
                                   T = self.maturity_time, D = self.umbral)
        
        print(f'''El precio del bono es: {self.price} a tiempo {self.time}.''')
        
    def get_sheet(self, params):
        
        # Parametros de entrada
        self.n = params['simulaciones']
        self.maturity_time = params['maduracion']
        self.delta_pph = params['delta_pph']
        self.face_value = params['valor_facial']
        self.rate = params['tasa']
        
        # Se crean dos listas para tener los tiempos y los umbrales
        
        # Saltos de tiempo de 5 en 5 dias
        self.time_range = list(range(0,params['maduracion'],5)) 
        # Umbral de cuantil de 0.7 a 0.9 de la distribucion elegida
        self.umbrales = list(np.linspace(self.hyp_test.dist.ppf(0.7), self.hyp_test.dist.ppf(0.9), 30))
        
        # ''' INICIO DEL PROCESO '''

        # La variable start_time nos permitira visualizar el tiempo en segundos para realizar todo el proceso
        start_time = time.time()
        
        print('Inicio del proceso: ')
        
        # Se genera el proceso con los parametros
        self.Lt = cm.generate_process(n = self.n, T = self.maturity_time, m = self.delta_pph, 
                                      dist = self.hyp_test.dist, plot = params['plot'])
        
        # Se crea una matriz de n (tiempos) x m (umbrales) de 0s
        ta = np.zeros([len(self.time_range), len(self.umbrales)])
        
        # El ciclo for nos poermitira ir llenando la matriz ta con los precios estimados, para cada umbral y cada tiempo
        for i in range(len(self.umbrales)):
            for j in range(len(self.time_range)):
                
                ta[j][i] = cm.price_bond(lt = self.Lt, zt = self.face_value, r = self.rate, 
                                         t = self.time_range[j], T = self.maturity_time , 
                                         D = self.umbrales[i])
                
            print('Umbral: ' + str(i))
            
        print("--- %s seconds ---" % (time.time() - start_time))
        
        # Al tiempo en el que estemos se le resta start_time = tiempo utilizado
        self.time_spend = time.time() - start_time
        
        # Se asigna ta al atributo sheet
        self.sheet = ta 
        # Se transforma en un pandas DataFrame
        self.sheet = pd.DataFrame(self.sheet)
        
        # Se asignan los nombres de los renglones y columnas de la sabana
        self.sheet.columns = self.umbrales
        self.sheet.index = self.time_range
        
        # Se obtiene el valor para ver si se quiere imprimir la sabana
        self.print_sheet = params['print_sheet']
        
        if self.print_sheet:
            
            # Se obtienen las variables X: tiempo, Y: umbrales y Z: Precios
            X, Y = np.meshgrid(self.time_range, self.umbrales)
            Z = np.array(self.sheet).T
            
            # INICIO DE LA FIGURA
            plt.figure(dpi = 200, figsize = (12,12))
            ax = plt.axes(projection = '3d')
            ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                            cmap='plasma', edgecolor='none') # ['viridis', 'plasma', 'inferno', 'magma', 'cividis']
            ax.set_xlabel('Tiempo', fontsize = 13, labelpad = 6) # Se agrega nombre eje X
            ax.set_ylabel('Umbral', fontsize = 13, labelpad = 10) # Se agrega nombre eje Y
            ax.set_zlabel('Precio', fontsize = 13, labelpad = 6) # Se agrega nombre eje Z
            ax.xaxis.set_tick_params(labelsize = 13) # size de los ticks X
            ax.yaxis.set_tick_params(labelsize = 12) # size de los ticks Y
            ax.zaxis.set_tick_params(labelsize = 13) # size de los ticks Y
            ax.set_title(f'''| Duracion: {str(round(self.time_spend/60,2))} | $F(x)$: {self.hyp_test.dist.dist.name.capitalize()} | \n | T: {str(self.maturity_time)} | Simulaciones: {str(self.n)} | $\delta$: {str(round(self.delta_pph,5))}''',
                         fontsize = 22)
                         
            
    def interactive_surface(self):
        
        pio.renderers.default = 'browser' # Se elige esta opcion para poder imprimir la grafica interactiva en el navegador web
        
        fig = go.Figure(data = [go.Surface(z = self.sheet.values)])
        text_title = f'''F(x): {self.hyp_test.dist.dist.name.capitalize()} | T: {str(self.maturity_time)} | Simulaciones: {str(self.n)} | mu: {str(round(self.delta_pph,5))}'''
        title = {'text': text_title}
        
        fig.update_layout(title = title, autosize=False, width = 800, height = 750,
                          scene = dict(xaxis_title='Tiempo',
                                       yaxis_title='Umbral',
                                       zaxis_title='Precio'
                                       ),
                          xaxis_autorange = 'reversed',
                          yaxis_autorange = 'reversed')
        fig.show()