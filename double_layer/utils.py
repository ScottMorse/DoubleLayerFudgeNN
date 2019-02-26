import numpy as np
from scipy.stats import truncnorm

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