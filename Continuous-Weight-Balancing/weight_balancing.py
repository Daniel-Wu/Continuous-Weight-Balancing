"""
Function for continuous weighting
"""

from scipy import stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

# Two ways to call this function: 

def continuous_weight(trait, target, addl_trait = None, clipping = None, verbose = True):
    """
    Continuously reweights the trait to approximate the target
    params:
    trait: numpy array of the weight trait
    target: any subclass of scipy.stats.rv_continuous; some target distribution
    addl_trait: if not None, a list of additional trait arrays. 
                weights will be calculated on these traits, but the trait data will not be used in determining the transfer function.
                Useful for val or test sets.
    clipping: if not None, is a tuple (lower, upper) giving the bounds at which to clip the weights. Individual clipping thresholds may be None.
    verbose: If true, prints intermediate outputs and graphs.
    """