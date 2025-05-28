import numpy as np
xp = np

def set_backend(backend):
    global xp
    if backend == "numpy":
        xp = np
    elif backend == "jax":
        import jax.numpy as jnp
        xp = jnp


ai = 1.47
bi = 0.0817
ci = 0.0149
spectral_params = [1.9392, -96.4366, -69.8059, -191.0251, -360.1569, 152.5207, -1702.2789]
gp_params =[0.0833, -529.6368, 666.1701, -1119.5632,
                                          -84.2438, 144.0589, -2.7723]
tuned_params = [2.5, -96.4366, -69.8059, -191.0251, -360.1569, 152.5207, -1702.2789]

def Lambda_of_logI_quadratic(logI, coeffs):
    c, b, a = coeffs
    yy = b**2 - 4*a*(c - logI)
    yy = xp.where(yy < 0, xp.inf, yy)
    x = (-b + xp.sqrt(yy))/(2*a)
    return xp.exp(x)

def logI_of_m(m, coeffs):
    slope, intercept = coeffs
    logI = slope * xp.log(m) + intercept
    return logI

def MTOV(coeffs, threshold = 1.89):
    slope, intercept = coeffs
    return xp.exp((threshold - intercept) /  slope)

def Lambda_of_m(m, Im_coeffs, LambdaI_coeffs=[ai,bi,ci]):
    logI = logI_of_m(m, Im_coeffs)
    Lambda = Lambda_of_logI_quadratic(logI, LambdaI_coeffs)
    return Lambda


def C_of_Lambda(Lambda, coeffs=spectral_params):
    K_C_lambda = coeffs[0]
    a = xp.array(coeffs[1:4]).T
    b = xp.array(coeffs[4:]).T
    
    prefactor = K_C_lambda * Lambda**(-1/5)
    L15 = Lambda**(-1/5)
    Lvec = xp.power(L15, xp.arange(1, 4)[:,None])
    avec = xp.matmul(a, Lvec)
    bvec = xp.matmul(b, Lvec)

    return  prefactor* (
        (1 + avec) /
        (1 + bvec)
    )

def R_of_M(m, Im_coeffs, coeffs_lambda_c=spectral_params,
        coeffs_I_lambda=(1.47, 0.0817, 0.0149)):
    C = C_of_Lambda(Lambda_of_m(m, Im_coeffs), coeffs_lambda_c)
    return m / C * 1.477 # Units of kilometer