# This to compute the model of the paper

# @authors: Jérémie Stym-Popper, Adrien Majka
# date: 12/01/2023

import numpy as np
import numpy.random as nprd
import scipy.stats as spst
import scipy.special as spsp
from scipy.optimize import fsolve



## GLSP given adjusment term

# Y: vector of response counts
# eta: vector of adjustment factors
# prior: IG, EH or PG 
# HP: hyperparameters of gamma prior

def GLSP_count(Y, eta=None, prior="EH", mc=3000, burn=500, HP=[1,1]):
    m = len(Y)
    MC = mc + burn

    if eta == None:
        eta = np.ones(m)

    #MCMC sample box
    lam_pos = np.full((MC, m), np.nan)
    u_pos = np.full((MC, m), np.nan)
    beta_pos = alpha_pos = gam_pos = np.full(MC, np.nan)
    

    # Initial values
    lam = nu = Y + 0.1
    u = np.ones(m)
    if prior == 'EH':
        V = W = np.ones(m)
        
    beta = alpha = gam = 1

    # MCMC iterations

    for r in range(MC):
        lam = nprd.gamma(Y+alpha, 1/(eta+beta/u), m)
        lam_pos[r,:] = lam

        # U and Gam (EH prior)
        if prior == 'EH':
            V = nprd.gamma(1+gam, 1+np.log(1+u), m)
            W = nprd.gamma(1+V, 1+u, m)
            
            for i in range(m):
                u[i] = spst.geninvgauss(p=1-alpha, b=2*beta*lam[i]+2*W[i]).rvs()
            
            u_pos[r,:] = u
            ss = np.sum(np.log(1+np.log(1+u)))
            gam = nprd.gamma(HP[0]+m, HP[1]+ss, 1)
            gam_pos[r] = gam
        
        ## u and gam (IG prior)
        if prior == "IG":
            # u
            u = spst.invgamma(a=alpha+gam).rvs(scale=beta*gam+lam, size=m)
            # gam
            bb = 0.1
            gam_new = min(max(gam+bb*nprd.normal(), 0.001), 150)
            L1 = m*gam*np.log(gam) - m*np.log(spsp.gamma(gam)) - gam*np.sum(np.log(u)) - gam*np.sum(1/u)
            L2 = m*gam_new*np.log(gam_new) - m*np.log(spsp.gamma(gam_new)) - gam_new*np.sum(np.log(u)) - gam_new*np.sum(1/u)
            pp = min(np.exp(L2-L1),1)
            gam = gam + nprd.binomial(1, pp)*(gam_new-gam)
            gam_pos[r] = gam
        
        # beta
        beta = nprd.gamma(HP[0]+m*alpha, HP[1]+np.sum(lam/u))
        beta_pos[r] = beta

        # alpha
        alpha = nprd.gamma(HP[0]+np.sum(nu), HP[1]+np.sum(np.log(1+eta*u/beta)))
        alpha_pos[r] = alpha

        # nu
        for i in range(m):
            if Y[i] == 0:
                nu[i] = 0
            elif Y[i] > 0:
                pp = alpha/(np.arange(1,Y[i]+1)-1+alpha)
                nu[i] = np.sum(nprd.binomial(1, pp, len(pp)))
    
    # Summary omission burn : TODO 

    # om = np.arange(1, burn+1)
    # lam_pos = lam_pos[np.setdiff1d(np.arange(lam_pos.shape[0]), om),:]

    if prior == 'PG':
        return lam_pos, beta_pos, alpha_pos
    else:    
        return lam_pos, u_pos, beta_pos, alpha_pos, gam_pos 
    

#GLSP with regression : 

class GlspRegModel : 
    def __init__(self, timedata, covariables, ):
        self.timedata = timedata
        self.covariables = covariables
        self.lamb = nprd.gamma(covariables.shape[0])

    def equation_deltahat(self, delta):
        root_to_find = sum([(self.timedata.loc[county].T - self.lamb[county] * np.exp(delta.T @ self.covariables.loc[county,:]))*self.covariables.loc[county] for county in self.covariables.index])
        return root_to_find

    def second_derivative_deltahat(self,delta):
        index_init = self.covariables.index[0]
        county = -self.lamb[index_init]*np.exp(self.covariables[index_init]@delta)*self.covariables[index_init]@self.covariables[index_init].T
        for county in self.covariables.index[1:]:
            second_derivative -= self.lamb[county]*np.exp(self.covariables[county]@delta)*self.covariables[county]@self.covariables[county].T
        return second_derivative

    def make_proposal(self, previous_delta):
        sol_deltahat, = fsolve(self.equation_deltahat, previous_delta, fprime = self.second_derivative_deltahat)
        precision = self.second_derivative_deltahat(sol_deltahat)
        proposal = (nprd.normal(sol_deltahat.shape) + sol_deltahat)@np.sqrt(precision)
        return proposal

    def dens_delta(self, delta):
        value = 0
        for county in self.covariables.index:
            value += (np.log(self.lamb[county]) + self.covariables.loc[county]@delta)*self.timedata.loc[county] - self.lamb[county]*np.exp(self.covariables.loc[county]@delta) - sum(i for i in range(self.covariables.loc[county]))
        return value

    def step(self, original):
        proposal = self.make_proposal(original)
        if np.log(nprd.uniform()) < self.dens(proposal) - self.dens(original):
            return proposal
        else :
            return original

    def sample(self) :

        return None
