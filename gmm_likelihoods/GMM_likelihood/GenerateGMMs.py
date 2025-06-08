import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import corner 
import dill
import pandas as pd
import os

class GMM_Generator(object):
    
    def __init__(
            self, transformer=None, n_components=None,
            outdir='.', label='gmms', max_samples= 50000, fit_kwargs=dict(reg_covar=1e-10, tol=1e-8, max_iter=1000)
    ):
      
        if not transformer:            
            print('No transformation function, fitting untransformed samples.')
            self.transformation_function = lambda x: x
        else:
            self.transformer = transformer
            self.transformation_function = self.transformer.transformation_function
            self.parameters = transformer.parameters

        self.result = transformer.result
        self.label = label
        self.outdir = outdir
        self.fit_kwargs = fit_kwargs
        
        self.collect_posterior(max_samples)
        self.set_n_components(n_components)
        
    def collect_posterior(self, max_samples):    
        postdict = dict()
        n_samples = min(len(self.result.posterior), max_samples)
        postdict = self.result.posterior.sample(n_samples)
        self.posterior = {key: np.array(postdict[key]) for key in self.parameters}

    def set_n_components(self, n_components):
        
        if not n_components:
            self.n_components = self.fit_components_bic()
        else:
            self.n_components = int(n_components)

                
    def _fit_gmm(self, transformed_samples, n_components, reg_covar, tol, max_iter):
        
        if reg_covar is None:
            reg_covar = self.reg_covar
        if tol is None:
            tol = self.tol
        if max_iter is None:
            max_iter = self.max_iter
            
        return GaussianMixture(n_components=n_components, reg_covar=reg_covar, tol=tol, max_iter=max_iter).fit(transformed_samples)
    
    def generate_gmm(self, overwrite=False):
        self.gmm_event()
        self.save_gmm(overwrite=overwrite)
        
    def gmm_event(self):
        data_t = self.transformation_function(self.posterior)
        if not isinstance(data_t, np.ndarray):
            data_t = data_t.get()
        data_t = np.transpose(data_t)
        self.gmm = self._fit_gmm(data_t, n_components=int(self.n_components), **self.fit_kwargs)
            
    def save_gmm(self, overwrite=False):
        outfile = os.path.join(self.outdir, self.label + ".pkl")
        if not overwrite and os.path.exists(outfile):
            self.set_new_label()
            outfile = os.path.join(self.outdir, self.label + ".pkl")
        self.outfile = outfile
        print(f'Saving GMM to {self.outfile}')
        self.transformer.xp = 'numpy'
        with open(self.outfile, 'wb') as gg:
            dill.dump(self, gg)
            
    def set_new_label(self):
        outfile = os.path.join(self.outdir, self.label + ".pkl")
        for letter in range(ord('A'), ord('Z') + 1):
            newlabel = self.label + chr(letter)
            outfile = os.path.join(self.outdir, newlabel + ".pkl")
            if not os.path.exists(outfile):
                break
        self.label = newlabel
        print(f"Setting label to {self.label}")
    
    def run_components(self, training_data, eval_data, components):

        for comp in components:
            model = self._fit_gmm(training_data, n_components = comp)
            score = model.score(eval_data)
            self.scores.append(score)
            print(f'done {comp} components')
            
    def fit_components_bic(self):
        
        print(f'determining optimal components for {self.label}')
        
        data_t = self.transformation_function(self.posterior)

        if not isinstance(data_t, np.ndarray):
            data_t = data_t.get()
        data_t = np.transpose(data_t)
        
        idx = np.random.choice(list(range(len(data_t))), replace=False, size= int(0.8*len(data_t)))
        train_dat = data_t[idx]
        eval_dat = data_t[~idx]
        
        optimal, bics = self.loop_components_bic(train_dat, eval_dat)
            
        components = np.arange(1,len(bics)+1)
        print(f'Using {optimal} components for event {self.label}')
        plt.scatter(optimal, min(bics), label='optimal', color='r', zorder=2.5)
        plt.plot(components, bics)
        plt.xlabel('Number of Components')
        plt.ylabel('BIC')
        plt.legend()
        plt.savefig(self.outdir + f'/{self.label}_components.png')
        plt.show()
        
        return optimal
    
    def _fit_components_bic_wrapped(self):
        return self.fit_components_bic()
        
    def loop_components_bic(self, training_data, eval_data):
        
        bics = []
        components = 1
        while True:
            model = self._fit_gmm(training_data, n_components = components, **self.fit_kwargs)
            bic = model.bic(eval_data)
            bics.append(bic)
            print(f'done {components} components')
            if components > 2 and bics[-1] > bics[-2] and bics[-2] > bics[-3]:
                comp_opt = np.argmin(bics) + 1
                break
            else:
                components += 1
                continue
                
        return comp_opt, bics
    
    def plot_corner(self, plot_injection=False):
                
        data_t = self.transformation_function(self.posterior)

        if not isinstance(data_t, np.ndarray):
            data_t = data_t.get()
        data_t = data_t.T
        data_t = np.squeeze(data_t)
        
        samples = self.gmm.sample(len(data_t))[0]
        
        if plot_injection:
            truths = {key: self.result.injection_parameters[key] for key in self.parameters}
            transformed_truths_list = self.transformation_function({key: np.array([truths[key]]) for key in truths.keys()})
            transformed_truths = dict()
            for column, key in enumerate(self.parameters):
                transformed_truths[key] = transformed_truths_list[column][0]
            transformed_truths =list(transformed_truths.values())
            truths = list(truths.values())
        else:
            truths = None
            transformed_truths = None
                                                              
        
        fig = corner.corner(samples, labels = self.parameters, hist_kwargs=dict(density=True, color='b'), color= 'b', truths = transformed_truths, truth_color='k')
        corner.corner(data_t, fig=fig, color='r', hist_kwargs=dict(density=True))
        plt.savefig(self.outdir + f'/{self.label}_transformed_corner.png')

        samples_t = dict()
        for column, key in enumerate(self.parameters):
            samples_t[key] = samples[:,column]
            
        physical = np.squeeze(pd.DataFrame.from_dict(self.transformer.transformation_back(samples_t)))
        original = np.stack([self.posterior[key] for key in self.parameters]).T

        fig_physical = corner.corner(physical, labels = self.parameters, hist_kwargs=dict(density=True, color='b'), color= 'b', truths = truths, truth_color='k')
        corner.corner(original, fig=fig_physical, color='r', hist_kwargs=dict(density=True))
        plt.savefig(self.outdir + f'/{self.label}_corner.png')
        return fig_physical, fig