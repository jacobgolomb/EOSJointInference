import bilby
def conditional_mu_2(reference_parameters, mu1):
    return dict(minimum = mu1, maximum = reference_parameters['maximum'])

prior = bilby.core.prior.ConditionalPriorDict()
prior['mpop'] = bilby.core.prior.Uniform(minimum=1.8, maximum=3, name='mpop', latex_label=r'$m_{\rm pop}$', unit=None, boundary=None)
prior['mmin']=1
prior['frac1'] = bilby.core.prior.Uniform(minimum = 0, maximum = 1)
prior['mu1'] = bilby.core.prior.Uniform(minimum=1, maximum=2)
prior['sigma1'] = bilby.core.prior.Uniform(minimum = 0.05, maximum=1)
prior['sigma2'] = bilby.core.prior.Uniform(minimum = 0.05, maximum = 1)
prior['mu2'] = bilby.core.prior.ConditionalUniform(minimum = 1, maximum = 2.5, condition_func = conditional_mu_2)
prior['m'] = bilby.core.prior.Uniform(name='m', minimum = -2.1, maximum=-1)
prior['b'] = bilby.core.prior.Uniform(name='b', minimum = 2, maximum= 4)