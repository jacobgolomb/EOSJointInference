import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate

import astropy.units as u

import astropy.constants as constants
import scipy.integrate as integrate

import get_macro_from_eos
import branched_interpolator
#import compute_fit
#import eval_eos_from_json as spectral_stitch

G = constants.G
c = constants.c


# code units are in solar masses
rhonuc = 2.8e14  # cgs
M_solar_geom = G * u.M_sun / c**2

def cgs_to_inverse_solar_masses_density(rho):
    return (rho * u.g/u.cm**3 *
            constants.G / constants.c**2 *
            M_solar_geom**2 ).decompose().value

def geometric_to_cgs(rho):
    return rho/cgs_to_inverse_solar_masses_density(1.0)


def internal_energy_to_energy_density(rho, eps):
    return (rho + eps*rho)


def rho_from_x(x, reference_rho):
    return reference_rho * np.exp(x)


class spectral_interpolator:
    def __init__(self, reference_density, reference_pressure, coefficients,
                 upper_density):
        self.p_ref = reference_pressure
        self.rho_ref = reference_density
        self.spectre_coeffs = coefficients
        self.upper_density = upper_density
        # tabulate epsilon
        eps0 = self.eps_helper(self.p_ref, coefficients, 0.0, self.rho_ref)
        deps_dx = lambda  xi : self.pressure_from_rest_mass_density(self.rho_ref * np.exp(xi))/self.rho_ref * np.exp(-xi)
        x_max = np.log(upper_density / reference_density)
        xis = np.linspace(0, x_max, 1000)
        eps_array = integrate.cumtrapz(deps_dx(xis), xis, initial=0.0)
        # interpolate the internal energy
        self.eps_interp = interpolate.interp1d(xis, eps_array + eps0, bounds_error=False)
        
        
        
    def rho_from_x(self, x, reference_rho) : return reference_rho * np.exp(x)
    
    def eps_helper (self, p0, gammas, x, reference_rho):
        rho = self.rho_from_x(x, reference_rho)
        return self.pressure_from_rest_mass_density(rho)/(rho * (gammas[0] - 1))
    def pressure_from_rest_mass_density(self, rho):
        x = np.log(rho / self.rho_ref)
        return np.where(x <= 0, self.p_ref * np.exp(self.spectre_coeffs[0] * x),
                        self.p_ref * np.exp(np.sum([gamma * x**(i+1)/(i+1)  for i, gamma in enumerate(self.spectre_coeffs)], axis=0))
        )
    def internal_energy_from_rest_mass_desity(self, rho):
        x = np.log(rho / self.rho_ref)
        return np.where(x<=0, self.eps_helper(self.p_ref, self.spectre_coeffs, x, self.rho_ref), self.eps_interp(x))
        
    def energy_density_from_rest_mass_density(self, rho):
        return  rho * self.internal_energy_from_rest_mass_desity(rho) + rho
    def pressure_derivative_with_respect_to_log_rest_mass_density(self, rho):
        x = np.log(rho / self.rho_ref)
        return np.where(x <= 0,  self.spectre_coeffs[0] * self.pressure_from_rest_mass_density(rho),
                 self.pressure_from_rest_mass_density(rho) * np.sum(np.array([gamma*x**(i) for i, gamma  in enumerate(self.spectre_coeffs)]), axis=0)
        )
        
    def enthalpy_from_rest_mass_density(self, rho):
        return 1 + self.internal_energy_from_rest_mass_desity(rho) + self.pressure_from_rest_mass_density(rho)/rho
    
    def check_stability_and_causality(self, rho_range):
        return {
            "stable": np.all(
                0 <=
                self.pressure_derivative_with_respect_to_log_rest_mass_density(
                    rho_range)),
            "causal" : np.all(
                self.pressure_derivative_with_respect_to_log_rest_mass_density(rho_range) <
                rho_range * self.enthalpy_from_rest_mass_density(rho_range))
        }
    def get_macro(self, properties = ["R", "Lambda"], *args, **kwargs):
        self.central_pressurec2, self.macro, self.baryon_density = get_macro_from_eos.eos_to_tov_solution(self, *args, **kwargs)
        self.macro_branches = branched_interpolator.get_branches(self.macro, properties = properties)
        self.interpolators = branched_interpolator.get_macro_interpolators(self.macro_branches, properties)
        self.macro_of_M = lambda m : branched_interpolator.choose_macro_per_m(
            m, self.macro, black_hole_values={"Lambda":lambda m : np.zeros_like(m), "R": lambda m : 2 * 1.477 * m},
            choice_function=None, branches=self.macro_branches, interpolators=self.interpolators)
        
    def Lambda_of_M(self, m):
        return self.macro_of_M(m)["Lambda"]
    def R_of_M(self, m):
        return self.macro_of_M(m)["R"]
    

    
if __name__ == "__main__":
    print("Testing...")
    gammas = [1.4, 0,0,0]
    p_ref = 1e12
    rho_ref = 1e14
    def micro_diagnostics():
        print(f"gamma = {gammas}")
        print(f"p_ref = {p_ref}")
        print(f"rho_ref = {rho_ref}")
        print(f"upper_density={2e15}")
        spectral_eos_simple = spectral_interpolator(reference_density=rho_ref, reference_pressure=p_ref,
                                             coefficients=gammas, upper_density=2e15)
        computed_pressure_at_saturation = spectral_eos_simple.pressure_from_rest_mass_density(rhonuc)
        expected_pressure_at_saturation = p_ref  * np.exp(1.4 / 1 * np.log(rhonuc/rho_ref))
        print("computed_pressure_at_saturation:", computed_pressure_at_saturation)
        print("expected_pressure_at_saturation:", expected_pressure_at_saturation)
        new_gammas = np.array([1.4, .5, -.3,.002])
        spectral_eos = spectral_interpolator(reference_density=rho_ref, reference_pressure=p_ref,
                                             coefficients=new_gammas, upper_density=2e15)
        rhos  = np.geomspace(1e11, 1e15, 200)
        print("rhos", rhos)
        print("pressure", spectral_eos.pressure_from_rest_mass_density(rhos))
        print("energy_density", spectral_eos.energy_density_from_rest_mass_density(rhos))
        print("enthalpy",spectral_eos.enthalpy_from_rest_mass_density(rhos))
        print("is it consistent with definition? Error is...", (spectral_eos.pressure_from_rest_mass_density(rhos) + spectral_eos.energy_density_from_rest_mass_density(rhos))/rhos - spectral_eos.enthalpy_from_rest_mass_density(rhos) )


        es = spectral_eos.energy_density_from_rest_mass_density(rhos)
        print("first law of thermodynamics is approximately true? de/drho - h is" , np.gradient(es, rhos) - spectral_eos.enthalpy_from_rest_mass_density(rhos))
        ps = spectral_eos.pressure_from_rest_mass_density(rhos)
        print("adiabatic index is", 1/ps*spectral_eos.pressure_derivative_with_respect_to_log_rest_mass_density(rhos))
        print("should be ",np.where(np.log(rhos/rho_ref ) <= 0, 1.4 , np.polynomial.polynomial.Polynomial(new_gammas)(np.log(rhos / rho_ref))))
        print("dlog(p)/dlog(rho) is approximately equal to the analytic value? Difference is",  (1/ps * spectral_eos.pressure_derivative_with_respect_to_log_rest_mass_density(rhos) -np.gradient(np.log(ps), np.log(rhos))))
        print("hopefully this is not better than the one above", 1/ps * spectral_eos.pressure_derivative_with_respect_to_log_rest_mass_density(rhos) - 1.4)
    def macro_diagnostics():
        new_gammas = np.array([2.0, 0.0, 0.0,0.0])
        p_ref = 1.35e13
        rho_ref = 2.4e14
        spectral_eos = spectral_interpolator(reference_density=rho_ref, reference_pressure=p_ref,
                                             coefficients=new_gammas, upper_density=2.8e15)
        spectral_eos.get_macro()
        print(spectral_eos.macro)
        print(spectral_eos.Lambda_of_M(np.linspace(1.0, 2.3, 30)))
    macro_diagnostics()
