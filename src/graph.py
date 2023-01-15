import matplotlib.pyplot as plt
import numpy as np

def conf_interv(mcmc_array):
    mean = np.mean(mcmc_array, axis = 1)
    interv = np.array(np.quantile(mcmc.array,[])
