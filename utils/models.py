import jax 
import numpy as np
import scipy
import ILoveQ_utils
def set_backend(backend):
    global xp, erf, norm
    if backend == "numpy":
        xp = np
        erf = scipy.special.erf
        norm = scipy.stats.norm
        ILoveQ_utils.set_backend("numpy")
    elif backend == "jax":
        xp = jax.numpy
        erf = jax.scipy.special.erf
        norm = jax.scipy.stats.norm
        ILoveQ_utils.set_backend("jax")


def truncnorm(xx, mu, sigma, low, high):
    norm = (erf((high - mu) / 2 ** 0.5 / sigma) - erf(
            (low - mu) / 2 ** 0.5 / sigma)) / 2

    val = xp.where((xx < high) & (xx >low), xp.exp(-(mu - xx) ** 2 / (2 * sigma ** 2)) / (2 * np.pi) ** 0.5 \
            / sigma / norm, 0)
    return val

def draw_fiducial_pulsar_samples(mmin, mmax, mu, sigma, gaussian_frac, N=5000, seed=None):
    if seed is None:
        seed = np.random.randint(0,1000)
    key = jax.random.PRNGKey(seed)
    n_norm = int(N * gaussian_frac)
    normal_samples = jax.random.normal(key=key, shape=(n_norm,)) * sigma + mu
    n_unif = int(N * (1 - gaussian_frac))
    
    uniform_samples = jax.random.uniform(key=key,shape=(n_unif,), minval = mmin, maxval=mmax)
    samples = dict()
    samples['mass'] = xp.concatenate((uniform_samples, normal_samples))
    samples['p_draw'] = gaussian_frac * norm.pdf(samples['mass'], loc=mu, scale=sigma) + (1-gaussian_frac) * 1 / (mmax - mmin)

    return samples


def get_pulsar_mass_weights(fiducial_masses, fiducial_pdraws, mmin, mpop, mu1, mu2, sigma1, sigma2, frac1):

    probs = frac1 * truncnorm(fiducial_masses,low=mmin, high=mpop, mu=mu1, sigma=sigma1) + (1-frac1) * truncnorm(fiducial_masses,low=mmin, high=mpop, mu=mu2, sigma=sigma2)
    weights = probs / fiducial_pdraws
    return weights

def weight_pulsar_population(N, parameters, proposal_pulsars):
    samples = dict()
    mass_weights = get_pulsar_mass_weights(proposal_pulsars['mass'], proposal_pulsars['p_draw'], mmin=parameters['mmin'], mpop=parameters['mpop'], mu1 = parameters['mu1'],
                                          mu2=parameters['mu2'], sigma1=parameters['sigma1'], sigma2=parameters['sigma2'], frac1=parameters['frac1'])
    samples['weights'] = mass_weights
    samples['mass'] = proposal_pulsars['mass']
    samples['lambda'] =  ILoveQ_utils.Lambda_of_m(samples['mass'], Im_coeffs=[parameters['m'], parameters['b']])
    samples['compactness']= ILoveQ_utils.C_of_Lambda(samples['lambda'])
    tov = ILoveQ_utils.MTOV([parameters['m'], parameters['b']])
    samples['weights'] *= (parameters['mpop'] < tov)
    return samples

def truncnorm_integral(mu, sigma, low, high):
    norm = (erf((high - mu) / 2 ** 0.5 / sigma) - erf(
            (low - mu) / 2 ** 0.5 / sigma)) / 2
    return norm

def get_pulsar_mass_weights_gap(fiducial_masses, fiducial_pdraws, mmin, mpop, mu1, mu2, sigma1, sigma2, frac1, gap_loc, gap_width, gap_strength):

    probs = frac1 * truncnorm(fiducial_masses,low=mmin, high=mpop, mu=mu1, sigma=sigma1) + (1-frac1) * truncnorm(fiducial_masses,low=mmin, high=mpop, mu=mu2, sigma=sigma2)
    
    gap_start = gap_loc - (gap_width/2)
    gap_end = gap_loc + (gap_width/2)
    integral = frac1*truncnorm_integral(mu1, sigma1, gap_start, gap_end) + (1-frac1)*truncnorm_integral(mu2, sigma2, gap_start, gap_end)
    norm = 1 - gap_strength * integral
    in_gap = (fiducial_masses > gap_start) & (fiducial_masses < gap_end)
    
    scale_fac = in_gap * gap_strength
    probs /= norm
    
    probs *= (1-scale_fac)
    weights = probs / fiducial_pdraws
    return weights
 