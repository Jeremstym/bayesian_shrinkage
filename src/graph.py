"""This module to make some visualisation about the main results and to compare the three possible models"""

import matplotlib.pyplot as plt
import numpy as np


def comp_conf_interv(
    mcmc_array, 
    quantile=[0.025,0.975], 
    legend=["PG", "IG", "EH"], 
    color=["red", "blue", "green"], 
    nom_graph='Comparaison_individu_avec_intervalle_de_confiance', 
    title=''
):
    
    
    dic_mean = {i : [] for i in range(len(mcmc_array))}
    dic_interv = {i : [] for i in range(len(mcmc_array))}

    for method in range(len(mcmc_array)) :
        dic_mean[method] = np.mean(mcmc_array[method], axis = 0)
        dic_interv[method] = [np.quantile(mcmc_array[method][:,col], quantile) for col in range(mcmc_array[method].shape[1])]
    fig, ax = plt.subplots()

    for indiv in range(mcmc_array[0].shape[1]):
        for method in range(len(mcmc_array)):
            if indiv == 0 : 
                ax.scatter(indiv + (method-len(mcmc_array)/2)/(5*len(mcmc_array)), dic_mean[method][indiv], s = 5, color = color[method], label = legend[method])
            else : 
                ax.scatter(indiv + (method-len(mcmc_array)/2)/(5*len(mcmc_array)), dic_mean[method][indiv], s = 5, color = color[method])
            
            plt.vlines(indiv + (method-len(mcmc_array)/2)/(5*len(mcmc_array)), ymin = dic_interv[method][indiv][0], ymax = dic_interv[method][indiv][1], color = color[method])
    
    ax.set_ylabel("Adjusted risk factor")
    ax.set_title(title, fontsize = 10)
    ax.legend(
        loc='upper center', 
        bbox_to_anchor=(0.5, -0.05),
        fancybox=True, shadow=True, 
        ncol=mcmc_array[0].shape[1]
    )
    # plt.savefig('../Images/'+nom_graph)
    plt.show()
    
    return None
    
def comp_scatter_plot(
    mcmc_array, legend=["PG vs IG", "PG vs EH"], 
    color=["red", "blue"], 
    combinaison=[[0,1],[0,2]], 
    title="Comparison between PG model and IG and EH model", 
    nom_graph="plot_meanvsmean"
):
    fig, ax = plt.subplots()
    dic_mean = {i : [] for i in range(len(mcmc_array))}
    
    count_color = 0
    for method in range(len(mcmc_array)) :
        dic_mean[method] = np.mean(mcmc_array[method], axis = 0)
    for combi in combinaison :
        ax.scatter(x = dic_mean[combi[0]], y = dic_mean[combi[1]], s = 5, color = color[count_color], label = legend[count_color])
        count_color += 1

    ax.plot([0,np.max(mcmc_array[2])],[0, np.max(mcmc_array[2])])
    ax.set_ylabel("IG or EH estimate")
    ax.set_xlabel("PG estimate")
    ax.set_title(title, fontsize = 1)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
                        ncol=3, fancybox=True, shadow=True)
    # plt.savefig('../Images/'+nom_graph)
    plt.show()
    return None