"""
Function for continuous weighting
"""

from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

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
    
    Returns
    weights, if addl_trait is None
    weights, [addl_weights, ...], if addl_trait is not None
    """
    
    #Approximate the trait distribution
    source_kde = stats.gaussian_kde(trait)
    estimate = source_kde.evaluate(trait)
    
    #Get weights
    weights = target.pdf(trait)/estimate
    
    #Clip weights
    if clipping != None:
        weights = weights.clip(*clipping)

    #Get weights for the additional traits
    if addl_trait != None:
        addl_weight = []
        for addl in addl_trait:
            addl_weight.append(target.pdf(addl) / source_kde.evaluate(addl))
    
    if verbose:
        #Plot source approximation
        plt.figure(figsize=(8, 6))
        plt.title("Fake Weight Trait Density Approximation")
        plt.xlabel("Trait Value")
        plt.ylabel("Probability Density")
        plt.scatter(trait, estimate)
        plt.show()
        
        #Plot target distribution
        plt.figure(figsize=(8, 6))
        plt.title("Target Distribution")
        x = np.linspace(target.ppf(0.01),
                        target.ppf(0.99), 100)
        plt.plot(x, target.pdf(x), "r")
        plt.xlabel("Trait Value")
        plt.ylabel("Target Probability Density")
        plt.show()
        
        #Plot weights
        plt.figure(figsize=(8, 6))
        plt.title("Transfer Weights")
        plt.scatter(trait, weights)
        plt.xlabel("Trait Value")
        plt.ylabel("Weight Value")
        plt.show()
        
        #Plot weighted source
        weighted_kde = stats.gaussian_kde(trait, weights = weights)
        weighted_estimate = weighted_kde.evaluate(trait)

        plt.figure(figsize=(8, 6))
        plt.title("Weighted Trait Distribution")
        plt.scatter(trait, weighted_estimate, label="Reweighted Data")
        plt.plot(x, target.pdf(x), "r", label = "Target Distribution")
        plt.xlabel("Trait Value")
        plt.ylabel("Weighted Probability Density")
        plt.legend()
        plt.show()
        
    if addl_trait != None:
        return weights, addl_weight
    else:
        return weights