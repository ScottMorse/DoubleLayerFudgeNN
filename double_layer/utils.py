import numpy as np
from scipy.stats import truncnorm

@np.vectorize
def sigmoid(x,y_range=1,y_shift=0,x_shift=0):
    return (y_range / (1 + np.exp(-x + x_shift))) + y_shift

def transposed_vector(arr):
    return np.array(arr).T

# gives clean normal distribution
def truncated_normal(mean=0,sd=1,low=0,upper=10):
    return truncnorm( 
        (low - mean) / sd, 
        (upper - mean) / sd, 
        loc=mean, scale=sd
    )

# random weights from normal distribution
def weight_matrix(n_input_nodes,n_output_nodes):
    _rad = 1 / (np.sqrt(n_input_nodes))
    dist = truncated_normal(mean=2,sd=1,low=-_rad,upper=_rad)
    return dist.rvs((n_output_nodes,n_input_nodes))
