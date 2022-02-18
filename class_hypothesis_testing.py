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
        
        # The HypothesisTesting Classe help us to make the Hypothesis Testing for the X array of damages.
        
        self.damages = None # Damages of the random event.
        self.plot_hist = None # It let us able to print the histogram of the data.
        self.plot_bins = None # Number of bins of the Histogram.
        
        # Class attributes
        self.best_dist = '' # String: The best distribution based on the p-value (maximum p-value).
        self.best_pvalue = 0.0 # Float: Max of the p-value for all the distributions.
        self.best_dist_attributes = {} # Dictionary: Attributes of the best distributions (Model, p-value, etc)
        self.dist_attributes = {} # Dic: Attributes for all the distributions.
        self.summary_ht = pd.DataFrame() # pd.DataFrame: Summary of the HT (Distribution | D | p-value)
        self.distributions_names = [] # List: Store the distributions we have choosen.
        
    def __str__(self):
        str_self = 'Best Distribution: ' + self.best_dist + ' | p-value ' + str(np.round(self.best_pvalue,4)) 
        return str_self
    
    def fit(self, x, params = {'plot_hist': False, 'bins': 30}):
    
        self.damages = x # Damages of the random event.
        self.plot_hist = params['plot_hist'] # It let us able to print the histogram of the data.
        self.plot_bins = params['bins'] # Number of bins of the Histogram.
        
        # We call the function hypothesis_testing from the module which was made for this:
        self.best_dist, self.best_pvalue, self.best_dist_attributes, self.dist_attributes, self.summary_ht, self.distributions_names\
            = cm.hypothesis_testing(x = self.damages, print_hist= self.plot_hist, bn = self.plot_bins)
            
        self.dist = self.best_dist_attributes['model']
                