#!/usr/bin/env python
# coding: utf-8

import importlib
from GMM_likelihood import GMM_Generator, Transformer
import bilby
import numpy as np
from scipy import stats
from tqdm import tqdm
import h5py
import glob
import dill
import matplotlib.pyplot as plt
import pandas as pd
G_c2_km_SM = 1.477

def inv_compactness(M, R):
    return G_c2_km_SM**-1 * R/M

def compactness(M, R):
    return G_c2_km_SM * M / R
def dinvC_dR(M):
    return G_c2_km_SM**-1 / M

massgrid = np.linspace(2.08-0.4, 2.08+0.4, 2500)
def effective_prior_J0740(m):
    return stats.norm.pdf(m , loc= 2.08, scale= 0.09)/stats.norm.pdf(m , loc= 2.08, scale= 0.07)

J0437_df = pd.read_csv('./data/Choudhury_J0437_flat_compactness.csv')
J0437_post = pd.DataFrame()

J0437_post['mass'] = J0437_df['M']
J0437_post['compactness'] = compactness(M =J0437_df['M'], R = J0437_df['R'])
J0437priors = bilby.core.prior.PriorDict()
J0437priors['mass'] = bilby.core.prior.Uniform(minimum=min(J0437_df['M'])-0.01, maximum = max(J0437_df['M'])+0.01)
J0437priors['compactness'] = bilby.core.prior.Uniform(minimum=min(J0437_df['compactness'])-0.01, maximum = max(J0437_df['compactness'])+0.01)
J0437 = bilby.core.result.Result(label = 'J0437', posterior=J0437_post, priors = J0437priors,
                                 search_parameter_keys=list(J0437priors.keys()))

transformer_J0437 = Transformer(result = J0437, parameters=['mass', 'compactness'])
GMM_J0437 = GMM_Generator(transformer=transformer_J0437, outdir = './NICER_fits', label='J0437_radiomass_GMM')
GMM_J0437.generate_gmm()
GMM_J0437.plot_corner()


J0740_df = pd.read_csv('./data/NICER+XMM_J0740_RM.txt', sep = " ", comment='#', header=None, 
                       names = ['radius', 'mass'], index_col=False, usecols=[0,1])
J0740_post = pd.DataFrame()
J0740_post['mass'] = J0740_df['mass']
J0740_post['compactness'] = compactness(M =J0740_df['mass'], R = J0740_df['radius'])
J0740_post = J0740_post[(J0740_post['mass'] > min(massgrid)) & (J0740_post['mass'] < max(massgrid))]
J0740priors = bilby.core.prior.PriorDict()

J0740priors['mass'] = bilby.core.prior.Interped(xx = massgrid, yy = effective_prior_J0740(massgrid))
J0740priors['compactness'] = bilby.core.prior.PowerLaw(alpha = -2, minimum=1/8, maximum=1/3.2)
J0740 = bilby.core.result.Result(label = 'J0740', posterior=J0740_post, priors= J0740priors,
                                 search_parameter_keys=list(J0740priors.keys()))

transformer_J0740 = Transformer(result = J0740, parameters=['mass', 'compactness'])

GMM_J0740 = GMM_Generator(transformer=transformer_J0740, outdir = './NICER_fits', label='J0740_radiomass_GMM')
GMM_J0740.generate_gmm()
GMM_J0740.plot_corner()


J0030_df = pd.read_csv('./data/J0030_3spot_RM.txt', sep = " ", comment='#', header=None, 
                       names = ['radius', 'mass'], index_col=False, usecols=[0,1])
J0030_post = pd.DataFrame()
J0030_post['mass'] = J0030_df['mass']
J0030_post['compactness'] = compactness(M =J0030_df['mass'], R = J0030_df['radius'])
J0030priors = bilby.core.prior.PriorDict()
J0030priors['mass'] =  bilby.core.prior.Uniform(minimum=1, maximum=2.4)
J0030priors['compactness'] = bilby.core.prior.Uniform(minimum=0.125, maximum=0.3125)
J0030 = bilby.core.result.Result(label = 'J0030', posterior=J0030_post, priors= J0030priors,
                                 search_parameter_keys=list(J0030priors.keys()))

transformer_J0030 = Transformer(result = J0030, parameters=['mass', 'compactness'])

GMM_J0030 = GMM_Generator(transformer=transformer_J0030, outdir = './NICER_fits', label='J0030_GMMs')
GMM_J0030.generate_gmm()
GMM_J0030.plot_corner()
