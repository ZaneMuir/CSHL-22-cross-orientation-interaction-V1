import numpy as np
def rectification(x):
	if isinstance(x, np.ndarray):
		y = np.array(x) # make a copy
		y[y < 0] = 0
	else:
		y = 0 if x < 0 else x
	return y

def relu(x):
    """rectified linear unit"""
    if isinstance(x, np.ndarray):
        y = np.array(x)
        y[y < 0] = 0
        return y
    else: # it should be a builtin number type (int or float)
        return 0 if x < 0 else x
	
	
## contrast response function
# reference: https://journals.physiology.org/doi/full/10.1152/jn.00505.2015
# - linear
# - logarithmic
# - power
# - hyperbolic

def contrast_response_linear(c, a=1, b=0):
	return a * c + b

def contrast_response_logarithmic(c, a=1, b=0):
	return a * np.log10(c) + b

def contrast_response_power(c, a=1, b=0, alpha=1):
	return a * c ** alpha + b

def contrast_response_hyperbolic(c, a=1, b=0, c50=0.5, alpha=1):
	return a * c ** alpha / (c ** alpha + c50 ** alpha) + b