import numpy as np
from bilby.core.prior import Uniform, Interped, ConditionalUniform, Gaussian, PowerLaw
from bilby.gw.prior import UniformInComponentsMassRatio, UniformInComponentsChirpMass, UniformSourceFrame
from astropy.cosmology import Planck15, z_at_value, FlatLambdaCDM
from astropy import units as u
from .CDFs import uniformCDF, interpedCDF, powerlawCDF, gaussianCDF
from scipy.stats import norm
from functools import partial

CDF_mapping = {'Uniform': uniformCDF, 'Interped': interpedCDF, 'ConditionalUniform': uniformCDF, 'PowerLaw': powerlawCDF,
              'Gaussian': gaussianCDF}

class Transformer(object):
    
    def __init__(self, result, parameters, generate_source_frame=None):

        self.parameters = sorted(parameters)
        self.result = result
        self.CUPY_LOADED = False
        self.xp = 'numpy'
        if generate_source_frame:
            for det_frame_param in generate_source_frame:
                if not det_frame_param + '_source' in self.parameters:
                    print(f'Source-frame posterior for parameter {det_frame_param} not in specified parameter set.')
                    break
                if not det_frame_param + '_source' in self.result.priors.keys():
                    self.result.priors[det_frame_param + '_source'] = self.generate_source_frame_prior(self.result, det_frame_param)
        
        self.sampling_priors = []
        self.posteriors = []
      
        priordict = dict()
        postdict = dict()

        for key in self.parameters:

            postdict[key] = np.array(self.result.posterior[key])
            priordict[key] = self.result.priors[key]

        self.sampling_priors= priordict
        self.posteriors = postdict

        self.priors = priordict
        self.dims = len(parameters)
        
        for key in self.parameters:
            prior_here = self.priors[key]

            if 'conditional' in str(prior_here):
                prior_here.conditional = True
            else:
                prior_here.conditional = False
                
            if type(prior_here) == Uniform:
                print(f'setting {prior_here} to Uniform')
                prior_here.cdf_func = CDF_mapping['Uniform']
                prior_here.cdf_parameters = dict(minimum= prior_here.minimum, maximum = prior_here.maximum)
                
            elif type(prior_here) == Interped:
                print(f'setting {prior_here} to Interped')
                prior_here.cdf_func = CDF_mapping['Interped']
                xx = np.linspace(prior_here.minimum, prior_here.maximum, 10000)
                yy = prior_here.cdf(xx)
                prior_here.cdf_parameters = dict(xps = xx, fps = yy)
                
            elif type(prior_here) == ConditionalUniform:
                print(f'setting {prior_here} to CU')
                prior_here.cdf = CDF_mapping['Uniform']
                
            elif type(prior_here) == UniformInComponentsMassRatio:
                print(f'setting {prior_here} to UCMR')
                xx = np.linspace(prior_here.minimum, prior_here.maximum, 10000)
                yy = prior_here.cdf(xx)
                prior_here.cdf_func = CDF_mapping['Interped']
                prior_here.cdf_parameters = dict(xps = xx, fps = yy)
                
            elif type(prior_here) == UniformInComponentsChirpMass:
                print(f'setting {prior_here} to UCCM')
                prior_here.cdf_func = CDF_mapping['PowerLaw']
                prior_here.cdf_parameters = dict(alpha = prior_here.alpha, minimum = prior_here.minimum, maximum = prior_here.maximum)
                
            elif type(prior_here) == PowerLaw:
                print(f'setting {prior_here} to PowerLaw')
                prior_here.cdf_func = CDF_mapping['PowerLaw']
                prior_here.cdf_parameters = dict(alpha = prior_here.alpha, minimum = prior_here.minimum, maximum = prior_here.maximum)

            elif type(prior_here) == Gaussian:
                print(f'setting {prior_here} to Gaussian')
                prior_here.cdf_func = CDF_mapping['Gaussian']
                prior_here.cdf_parameters = dict(mu = prior_here.mu, sigma = prior_here.sigma)
            elif type(prior_here) == UniformSourceFrame:
                print(f'setting {prior_here} to Interped')
                xx = np.linspace(prior_here.minimum, prior_here.maximum, 10000)
                yy = prior_here.cdf(xx)
                prior_here.cdf_func = CDF_mapping['Interped']
                prior_here.cdf_parameters = dict(xps = xx, fps = yy)               
            else:
                print(f'Prior type for {key} not supported!')
                
            prior_here.cdf = self.get_cdf_function(prior_here.cdf_func, prior_here.cdf_parameters)

    def get_cdf_function(self, cdf_func, cdf_parameters):
        return partial(cdf_func, **cdf_parameters) 

    def cdf(self, samples, priors=None): 
        
        n_samples = samples[self.parameters[0]].shape
        
        cdfs = self.xp.zeros((self.dims, *n_samples))
        
        for i, key in enumerate(self.parameters):
            cdfs[i] = self.cdf_single_dimension(samples, key)
        return cdfs

    def cdf_single_dimension(self, samples, key):
        prior = self.priors[key]
        print(prior.cdf)
        if not prior.conditional:
            cdfs_here = prior.cdf(samples[key])
        else:
            required_variables = prior.required_variables
            reference_params = prior.reference_params
            bounds = prior.condition_func(reference_params, **{req_key: samples[req_key] for req_key in required_variables})
            cdfs_here = prior.cdf(samples[key])
        return cdfs_here
            
    def transformation_function(self, samples):
        
        n_samples = len(samples[self.parameters[0]])
        
        cdf = self.cdf(samples)
        posteriors_transformed = self.xp.nan_to_num(self.xp.sqrt(2)*self.erfinv(2*cdf - 1))

        return posteriors_transformed
    
    def transformation_back(self, transformed_samples):
        
        cdf = {key: norm.cdf(transformed_samples[key]) for key in self.parameters}
        
        physical_posteriors = dict()
        for key in self.parameters:
            physical_posteriors[key] = self.priors[key].rescale(cdf[key])
        return physical_posteriors
    
    def generate_source_frame_prior(self, result_object, detector_frame_parameter):
        
        detector_frame_prior = result_object.priors[detector_frame_parameter]
        
        if 'redshift' not in result_object.priors.keys():
            try:
                redshift_prior = result_object.priors['luminosity_distance'].get_corresponding_prior('redshift')
            except AttributeError:
                domain = np.linspace(result_object.priors['luminosity_distance'].minimum, result_object.priors['luminosity_distance'].maximum, 10000)
                redshift_prior = DL2Redshift(xx = domain, yy = result_object.priors['luminosity_distance'].prob(domain), name='luminosity_distance')
        
        elif 'redshift' in result_object.priors.keys():
            redshift_prior = result_object.priors['redshift']
            
        redshifts = redshift_prior.rescale(np.linspace(0, 1, 10000))
        
        try:
            new_max = detector_frame_prior.maximum / (1 + redshifts[0])
            new_min = detector_frame_prior.minimum / (1 + redshifts[-1])
        
        except:
            
            detector_frame_masses = detector_frame_prior.rescale(np.linspace(0, 1, 10000))
            new_max = detector_frame_masses[-1] / (1 + redshifts[0])
            new_min = detector_frame_masses[0] / (1 + redshifts[-1])
            
        xx = np.linspace(new_min, new_max, len(result_object.posterior)+1)
        xx = np.sort(np.concatenate((xx, np.array(result_object.posterior[detector_frame_parameter]))))
        yy = np.array([np.trapz(np.nan_to_num(
            detector_frame_prior.prob(x * (1 + redshifts)) * (1 + redshifts) *
            redshift_prior.prob(redshifts)), redshifts) for x in xx])
        new_prior = Interped(xx, yy)
        
        return new_prior

    @property
    def xp(self):
        """
        This should *not* be set by the user! Should only be set by the likelihood object  
        """
        return self._xp

    @xp.setter
    def xp(self, value):
        if value.lower() == 'cupy':
            try:
                import cupy as cp
                from cupyx.scipy.special import erfinv
                self._xp = cp
                self.erfinv = erfinv
                self.set_funcs_to_cupy()
            except Exception as ee:
                print("Cannot import cupy, using numpy instead")
                self._xp = np
                from scipy.special import erfinv
                self.erfinv = erfinv
        else:
            if self.CUPY_LOADED:
                """
                If the object was previous configured to use cupy and now the user wants numpy,
                we must convert the internal arrays back from cupy to numpy array
                """
                for key in self.priors.keys():
                    if hasattr(self.priors[key], 'cdf_parameters'):
                        cdf_parameters = dict()
                        for element in self.priors[key].cdf_parameters:
                            if isinstance(self.priors[key].cdf_parameters[element],  cp.ndarray):
                                cdf_parameters[element] = np.array(self.priors[key].cdf_parameters[element].get())
                        cdf_parameters['cupy'] = False
                        self.priors[key].cdf_parameters.update(cdf_parameters)
                        self.priors.cdf = partial(self.priors[key].cdf_func, **self.priors[key].cdf_parameters)
            self._xp = np
            from scipy.special import erfinv
            self.erfinv = erfinv

    def set_funcs_to_cupy(self):
        self.CUPY_LOADED = True
        """
        Set all the CDF functions and array arguments to use cupy
        """
        for key in self.priors.keys():
            if hasattr(self.priors[key], 'cdf_parameters'):
                cdf_parameters = dict()
                for element in self.priors[key].cdf_parameters:
                    if isinstance(self.priors[key].cdf_parameters[element],  np.ndarray):
                        cdf_parameters[element] = self._xp.array(self.priors[key].cdf_parameters[element])
                cdf_parameters['cupy'] = True
                self.priors[key].cdf_parameters.update(cdf_parameters)
                self.priors[key].cdf = partial(self.priors[key].cdf_func, **self.priors[key].cdf_parameters)
    
class DL2Redshift(Interped):
    def __init__(self, xx, yy, name):
        if name == 'luminosity_distance':
            zz = [z_at_value(Planck15.luminosity_distance, x * u.Mpc) for x in xx]
            yy *= np.gradient(xx, zz)
        else:
            print('Must specify luminosity distance as the name of the prior')
        super(DL2Redshift, self).__init__(xx=zz, yy=yy)