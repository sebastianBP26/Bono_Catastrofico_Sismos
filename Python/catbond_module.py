# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 21:04:42 2021

@author: Sebastian Barroso
"""

# Modules
from scipy import stats
import pandas as pd
from scipy.stats import poisson, uniform
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
    
def cdf(x):
    
    '''Permite obtener la función de distribución empírica para un vector x.
    Salida: Regresa un pd DataFarme que contiene dos columnas: x | P(X <= x)'''
    
    tr = pd.DataFrame(x, columns = ['damaged']) # Construimos un dataFrame con el vector de entrada.
    cdf = [] # List: Será llenada en el ciclo for con cada una de las probabilidades estimadas.
    
    a = list(set(x)) # Valores únicos del vector x.
    a.sort() # Se ordena la lista.
    
    for value in a:
        temp = tr[tr.damaged <= value] # Filtramos el df con valores <= x.
        cdf.append(np.round(len(temp)/len(tr),4)) # P(X <= x): count(temp) / len(df)
        
    cdf = pd.DataFrame({'x':a, 'cdf':cdf}) # salida.

    return cdf

def hypothesis_testing(x, distributions = ['lognorm', 'gamma', 'pareto', 'burr12'], print_hist = False, bn = 50):
    
    '''
    La funcion hypothesis testing tiene como objetivo dar una respuesta al usuario de si sus datos
    de severidad de cierto evento aleatorio (en este caso sismos), se ajusta a alguna de las funciones
    de distribución elegidas. Para esta función, se tienen dos parámetros:
        
        1. x: Vector de observaciones del evento: Array.
        2. distributions: Lista de distribuciones que se ajustarán, con base en la documentación de scipy. 
            Por defecto, se tienen las distribuciones lognormal, gamma, pareto y burr12.
        3. print_hist: Parámetro booleano que nos permite indicar si queremos imprimir o no el histograma
        del vector x.
        4. bn: número de bins del histograma en caso de imprimir el histograma.
        
    El ajuste tiene dos faces:
        
        1. Se ajustan los parámetros con Máxima Verosimilitud.
        2. Se aplica la prueba de Kolmogorov - Smirnov. Se pide un p-value > 0.05 para 
        aceptar la Hipótesis Nula de que los datos siguen la distribución elegida.
        3. Se elige la distribución mayor P-Value.
        
    Finalmente, la función regresa los siguientes objetos:
        1. best_dist: Nombre de la mejor distribución ajustada con base al p-value.
        2. best_p: P-value de la mejor distribución ajustada.
        3. values[best_dist]: Parámetros y distribución ya inicializada con los parámetros.
        4. values: Parámetros y distribuciones ajustadas.
        5. df: Resumen de las pruebas realizadas |distribucion| D | P-VALUE
        6. distributions: Nombres de las distribuciones que hemos probado.
        
        OBS: El hecho de que se seleccione una distribución haciendo uso del máximo p-value, 
        no implica que la distribución ajusta a los datos. Para esto, solicitaríamos un p-value > 0.05
        
    Para más información sobre las distribuciones: atributos, métodos y parametrización, puede revisar las siguientes ligas:
        gamma: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gamma.html#scipy.stats.gamma
        lognorm:https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.lognorm.html#scipy.stats.lognorm
        pareto: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pareto.html#scipy.stats.pareto
        burr12: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.burr12.html#scipy.stats.burr12
    
    El ajuste y las pruebas por default se realizan con la lista de distribuciones que viene en los
    parámetros de la función. Si desea agregar una nueva distribución (o una nueva lista), deberá
    agregar los parámetros de las mismas como se muestra al final de la función, para poder obtener
    los parámetros. 
    
    '''
    try:
            
        values = {} # Dic: Tendrá parámetros y distribuciones
        dist_results = [] # List: Será llenado con tuplas para crear un dataframe
        p = [] # List: Metricas Calculadas para cada distribución.
        
        # El ciclo for nos permite realizar el ajuste de parámetros y la prueba para cada distribución.
        for k in distributions:
            
            dist = getattr(stats, k) # Inicializamos la distribución.
            param = dist.fit(x, floc = 0) # Ajustamos los parámetros por máxima verosimilitud.
            
            # D y p-value de la prueba Kolmogorov - Smirnov.
            D, p_value = stats.kstest(x, k, args = param) 
            # D y p-value de la prueba Cramer Von Mises.
            cr = stats.cramervonmises(x, k, args = param)
        
            dist_results.append((k, p_value)) # Agregamos los resultados.
            
            test = {} # Llenaremos este diccionario temporal con los parámetros, estadística y p-value.
            test['Parametros'] = param # Parámetros
            test['D_Kolmogorov'] = D # Estadística D
            test['p_value'] = p_value # P-value.
            test['W_Cramer'] = cr.statistic # Estadística Cramer
            test['w_p_value'] = cr.pvalue # p-value Cramer
            
            values[k] = test # Agregamos al diccionario el diccionario "temporal" creado
            p.append((k,D,p_value, cr.statistic, cr.pvalue)) # Misma idea, pero con los valores de nombre de la dist, D y p-value.
            
        # Obtenmos la "mejor" distribución con base en el p-value más grande.
        best_dist, best_p = (max(dist_results, key = lambda item: item[1]))
        
        # Creamos un pandas DataFrame con la lista que tiene tuplas.
        df = pd.DataFrame(p)
        df.columns = ['distribucion','D','p_value', 'W', 'W_p_value'] # Cambiamos los nombres de las columnas
        df.reset_index(drop = True, inplace = True) # reset index
        df.sort_values( by = ['p_value'], ascending = [False], inplace = True) # Ordenamos descendentemente por el p-value.
        df['d_mod'] = df['D'].values * (np.sqrt(len(x)) + 0.12 + (0.11/np.sqrt(len(x))))
        df['w_mod'] = (df['W'].values + (0.4/len(x)) + (0.6/(len(x)**2))) * (1 + (1/ len(x)))
        df = df.loc[:, ['distribucion', 'D', 'd_mod', 'p_value', 'W', 'w_mod', 'W_p_value']]
        
        # Inicializamos las distribuciones, aquí es donde debería agregar una nueva distribución, en caso de que lo necesite.
        
        lognorm_p = values['lognorm']['Parametros'] 
        gamma_p = values['gamma']['Parametros']
        pareto_p = values['pareto']['Parametros']
        burr_p = values['burr12']['Parametros']
        
        # Indicamos los parámetros de la distribución, parámetros ajustados previamente.
        values['lognorm']['model'] = stats.lognorm(s = lognorm_p[0], loc = lognorm_p[1], scale = lognorm_p[2])
        values['gamma']['model'] = stats.gamma(a = gamma_p[0], loc = gamma_p[1], scale = gamma_p[2])
        values['pareto']['model'] = stats.pareto(b = pareto_p[0], loc = pareto_p[1], scale = pareto_p[2])
        values['burr12']['model'] = stats.burr12(c = burr_p[0], d = burr_p[1], loc = burr_p[2], scale = burr_p[3])
        
        # Imprimimos en consola la mejor distribución y su correspondiente p-value. 
        print(f'''La mejor distribución para sus datos es: {best_dist} con un P-value de {np.round(best_p,4)}.''')
        
        if print_hist:
            
            # st: string que nos permite agregar los títulos a los gráficos.
            st = 'Best Distribution : ' + str(best_dist) + '. P-value: ' + str(np.round(best_p,4))
            
            # colors: diccionario con los colores para cada distribución.
            colors = {'lognorm': '#273746', 'gamma':'#432746', 'pareto':'#17a589', 'burr12':'#581845'}
            linestyles = {'lognorm': 'solid', 'gamma':'dashed', 'pareto':'dashdot', 'burr12':'dotted'}

            # Histrograma del vector x (sin fx(X))
            xt = np.linspace(min(x), max(x), 10)
            plt.figure(dpi = 150, figsize = (10,8))
            plt.title('Histogram' + '\n' + st, fontsize = 12)
            sns.histplot(data = pd.DataFrame(x, columns = ['Siniestros']), x = 'Siniestros', color = '#c81025',
                          bins = bn, stat = 'density', label = 'Siniestros')
            plt.ylabel('Probabilidad')
            plt.xticks(xt)
            plt.legend()
            plt.grid(color = '#191a1a', linestyle='--', linewidth = 0.1, alpha = 0.5)
            plt.show()
            
            # Histograma de x y la mejor distribución.
            xt = np.linspace(min(x), max(x), 10)
            plt.figure(dpi = 150, figsize = (10,8))
            plt.title('Histogram' + '\n' + st, fontsize = 12)
            sns.histplot(data = pd.DataFrame(x, columns = ['Siniestros']), x = 'Siniestros', color = '#c81025',
                          bins = bn, stat = 'density', label = 'Siniestros')
            
            dist_hist = values[best_dist]['model']
            rango = np.linspace(min(x), max(x), 100)
            fx = dist_hist.pdf(rango)
            plt.plot(rango,fx, linewidth = 2, label = dist_hist.dist.name, color = colors[best_dist])
            plt.ylabel('Probabilidad')
            plt.xticks(xt)
            plt.legend()
            plt.grid(color = '#191a1a', linestyle='--', linewidth = 0.1, alpha = 0.5)
            plt.show()
            
            cdf_function = cdf(x)
            rango = np.linspace(min(x), max(x), 100)
            plt.figure(dpi = 150, figsize = (10,8))
            plt.title('Funciones de Distribución vs Función De Distribución Empírica', fontsize = 14)
            for dist in distributions:
                dist_hist = values[dist]['model']
                Fx = dist_hist.cdf(rango)
                plt.plot(rango,Fx, linewidth = 1.5, label = dist_hist.dist.name, color = colors[dist],
                         linestyle = linestyles[dist])
                
            plt.plot(cdf_function.x.values, cdf_function.cdf.values, drawstyle = 'steps-post', marker = 'o', 
                     color = '#c81025', label = 'cdf')
            
            plt.xlabel('Damaged')
            plt.ylabel(r'$P(X \leq x)$')
            plt.legend()
            plt.grid(color = '#191a1a', linestyle='--', linewidth = 0.1, alpha = 0.5)
            plt.show()
    
        return best_dist, best_p, values[best_dist], values, df, distributions
    
    except ValueError:
        
        print('La muestra debe tener al menos un valor. Por favor, intente de nuevo.')
    
    except TypeError:
        
        print('Por favor, agregue un valor numérico.')    
        

def PPH(t,tasa):
    
    '''
    Permite simular un Proceso De Poisson Homogéneo de parámetro Lambda a tiempo t.
    Parámetros:
        * t: Tiempo a estimar.
        * tasa: Estimador M.V. de Lambda = Media Muestral
    
    Salida: Regresa un objeto pandas DataFrame con los tiempos donde un evento ocurrió.
    
    '''
    proceso = poisson(t*tasa) # Se genera una clase de poisson con parámetro t*tasa.
    u = uniform(loc = 0, scale = t) # Se genera una clase de uniform de 0 a t.
    xt = proceso.rvs() # Generamos un número aleatorio de la distribución Poisson.
    t = np.sort(np.round(u.rvs(xt))) # Redondeamos los valores de las v.a. Unif para tener num de días enteros.
    # t = np.sort(u.rvs(xt)) # Se generan xt números aleatorios de la dist uniforme y se ordenan.
    t = t.tolist() # Cambiamos el tipo de dato de t a lista.
    t.insert(0,0) # Agregamos un 0 al inicio de t (No = 0).
    seq = list(range(0,xt)) # Hacemos un rango de valores de 0 a xt.
    seq.insert(0, 0) # Agregamos un 0 al inicio de la secuencia t = 0.
    df = pd.DataFrame(t,seq, columns = ['Value']) # Creamos un data frame con las listas seq y t.
    
    return df

def generate_process(n, T, m, dist, umbral = [0.8], plot = False,  dpi = 150, figsize = (10,8)):
    
    ''' Función para generar las simulaciones del Lt. 
    
    El objetivo de esta función es el obtener un objeto pandas dataframe en donde se almacenen todas 
    las simulaciones de tiempo 0 hasta T, de el número de sismos usando un PPH(Lambda). 
    
    Recordando que Lt es la suma de los daños provocados hasta Nt.
    
    Para esto, la función tiene 5 Parámetros:
        
        1. n: Número de caminatas aleatorias que se generan.
        2. T: Temporalidad máxima (tiempo de maduración del CatBond). 
        3. m: Tasa del Proceso de Poisson Homogéneo. 
        4.dist: Función de Distribución elegida.
        5. plot: Permite imprimir la gráfica de las caminatas aleatorias, por default es False. 
                 Esto debido a que entre mayor sea n, mayor el tiempo que toma en imprimir la gráfica.
        6. umbral: Nivel de Threshold, por default se utiliza el quantil 0.8 de la distribución elegida.
      
     Output: La función regresa un objeto DataFrame, donde se almacenan cada una de las simulaciones
     generadas, además les genera una etiqueta para poder diferenciarlas. 
     
     '''    
    simulaciones = {} # Diccionario donde almacenaremos los datos
    concat = pd.DataFrame() # Dataframe donde concatenaremos cada uno de los dfs
    
    for i in range(n):
        
        # Se genera una caminata aleatoria del PPH(Lambda) para simular el número de sismos 
        # En cada iteración se estima el número de sismos entre tiempo [0,T]
        # Se le da como parámetros la temporalidad T y la lambda (ms) del proceso.
        
        pph = PPH(T, m) # Se genera una caminata aleatoria del PPH de parámetro m a tiempo T.
        
        # Dado que el proceso inicia en cero, se elimina un elemento de la longitud del DataFrame
        num = len(pph) - 1 
        
        # Se obtiene el Proceso de pérdidas agregado sumando las Xi simuladas.
        # Simulamos num (Que es el número de eventos que se estiman) v.a de la Fx(X) elegida
        # haciendo uso del método .rvs(num) de la distribución elegida.
        
        damaged = list(dist.rvs(num)) # random_state = 42 si queremos fijar
        damaged.insert(0,0) # Se agrega un 0 al inicio porque comienza en 0 el PPH
        
        # El DataFrame t va acumulando a D
        t = pd.DataFrame( {'pph':pph.Value.values, 'damaged':np.cumsum(damaged)})
        # Agregamos una etiqueta para saber qué caminata aleatoria representa.
        t['tipo'] = 'data_' + str(i)
        
        # Se agrega la simulación i en el diccionario simulaciones.
        simulaciones[i] = t
        
        # Concatenamos el DF concat (que inició vacío) con cada simulación que vamos generando.
        concat = pd.concat([concat,t])
        
    # Si el usuario desea ver la gráfica de todas las caminatas aleatorias
    if plot:
        
        # Generamos las paletas de colores 1 para cada simulación.
        colors = np.random.rand(len(simulaciones),3) 
        
        # Se calcula el umbral (D), éste puede se un array o un escalar.
        umbral_test = dist.ppf(umbral)
        
        # Generamos la figura
        plt.figure(dpi = dpi, figsize = figsize)
        
        # Agregamos el título al gráfico
        plt.title('Proceso de Pérdidas Agregadas $L_{t} = \sum_{i = 1}^{M_{t}} X_{i}$', fontsize = 16)
        
        # Ciclo for para graficar cada una de las caminatas generadas.
        for key in simulaciones.keys():
            plt.plot(simulaciones[key].pph.values,simulaciones[key].damaged.values, 
                     color = colors[key], linewidth = 0.6, marker = 'o',
                     drawstyle = 'steps-post')
        
        # Graficamos cada uno de los umbrales.
        for u in umbral_test: 
            plt.axhline(y = u, color = '#000000', linestyle = '-')
        
        # Etiquetas de los ejes, grid y mostramos el gráfico.            
        plt.xlabel('Días')
        plt.ylabel(r'$L_{t}$')
        plt.grid()
        plt.show()
        
    return concat

def price_bond(lt,zt,r,t,T,D):
    
    '''
    Esta función calcula el precio del CatBond. Los parámetros que se solicitan son:
        
        lt: DataFrame que contiene el proceso de Pérdida agregada. Idealmente, este objeto
        es el que generamos con la función 'generate_process'.
        zt: Valor Facial del bono.
        r: Tasa de interés / descuento.
        t: Tiempo al que queremos valuar el precio del bono.
        T: Tiempo de maduración del bono.
        D: Umbral.
        
        Salida: Regresa el precio del bono (float)
        
    '''
    
    df = pd.DataFrame()
    df['pph'] = lt['pph'].values # Tiempos donde ocurren los eventos
    df['damaged'] = lt['damaged'].values # Pérdidas agregadas
    df['tipo'] = lt['tipo'].values # Etiqueta de la caminata aleatoria
    
    # Haciendo uso del valor D, verificamos para cada caminta cuál ha sobrepasado el valor de D.
    # Para aquellos que hayan rebasado, se les asigna el valor 1 y si no tienen un valor 0.
    df['umbral'] = (df.damaged > D).astype('int32') 
    
    # Filtramos el dataframe con base en el tiempo t, que es donde queremos valuar el precio del bono.
    df = df[df.pph >= t]
    
    # rt es la agrupación por caminata aleatoria, donde calculamos el máximo de la columna 'umbral'.
    # Previamente creada, por construcción, una caminata aleatoria puede tener múltiples "unos", pero
    # al caclular el máximo podemos saber si en esa caminata se sobrepaso o no a D.
    rt = df.groupby(['tipo'])['umbral'].max().reset_index() 
    
    # Verificamos que haya al menos una simulación para calcular la proporción:
    # Número de simulaciones que rebasaron el umbral / total de simulaciones.
    
    if len(rt) > 0:
        # Porporción de simulaciones que rebasan el umbral
        p = sum(rt['umbral'])/len(rt)     
    else:
        p = 0
    
    # Calculamos el precio del bono, que es el valor presente multiplicado por 1 - proporción.
    a = np.exp(-r*(T-t)/360)*zt
    
    # Finalmente, redondeamos el precio a 5 dígitos.
    precio = np.round(a*(1-p),5)
    
    return precio