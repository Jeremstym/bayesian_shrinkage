import matplotlib.pyplot as plt
import numpy as np

def comp_conf_interv(mcmc_array, quantile = [0.025,0.975], legend = ["PG", "IG", "EH"], color = ["red", "blue", "green"]):
    dic_mean = {i : [] for i in range(len(mcmc_array))}
    dic_interv = {i : [] for i in range(len(mcmc_array))}
    for method in range(len(mcmc_array)) :
        dic_mean[method] = np.mean(mcmc_array[method], axis = 1)
        dic_interv[method] = [np.quantile(mcmc_array[method][:,col], quantile) for col in range(mcmc_array[method].shape[1])]
    fig, ax = plt.subplots()
    for indiv in range(mcmc_array[0].shape[1]):
        for method in range(len(mcmc_array)):
            if indiv == 0 : 
                ax.scatter(indiv + (method-len(mcmc_array)/2)/(5*len(mcmc_array)), dic_mean[method][indiv], color = color[method], label = legend[method])
            else : 
                ax.scatter(indiv + (method-len(mcmc_array)/2)/(5*len(mcmc_array)), dic_mean[method][indiv], color = color[method])
            plt.vlines(indiv + (method-len(mcmc_array)/2)/(5*len(mcmc_array)), ymin = dic_interv[method][indiv][0], ymax = dic_interv[method][indiv][1], color = color[method])
    ax.set_ylabel("Adjusted risk factor")
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()
    return
    
    
