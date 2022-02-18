# -*- coding: utf-8 -*-
"""
Created on Sat Feb 12 22:02:24 2022

@author: Sebastian Barroso
"""
import pandas as pd
import numpy as np

from catbond_class_refactored import CatBond

# Frecuencia
path = r'G:\Mi unidad\Tesis (Avances)\BASES\SSNMX_catalogo_19000101_20210913.csv'
mw = 6.5

severidad = pd.read_csv(r'G:\Mi unidad\Tesis (Avances)\BASES\damaged.csv', engine = 'python', encoding = 'unicode_escape')
x = severidad.Damaged.values
                
cb = CatBond()
cb.hyp_test.fit(x, {'plot_hist': True, 'bins':10})
print(cb.hyp_test.summary_ht)

cb.hom_poisson.fit(path, mw)
cb.hom_poisson.simulate_process()

distributions = cb.hyp_test.dist_attributes

params = {'simulaciones': 10000,
          'maduracion': 365*4,
          'delta_pph': cb.hom_poisson.mu/360, #214/(120*360)
          'valor_facial': 1,
          'tasa': -np.log(1/(1 + 0.0512)),
          'print_sheet': True,
          'plot': True}

sheets = []
lts = []
for distribution in cb.hyp_test.distributions_names:

    cb.hyp_test.dist = cb.hyp_test.dist_attributes[distribution]['model']
    cb.get_sheet(params)
    sheets.append(cb.sheet)
    lts.append(cb.Lt)
    cb.interactive_surface()
    
    print(f''' Se realizó el ejercicio para la distribución: {distribution}''')
 
# t = []
# t.append(tuple(params.values()))
# t = pd.DataFrame(t, columns=list(params.keys()))