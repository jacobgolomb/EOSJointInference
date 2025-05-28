import pandas as pd
import numpy as np
import temperance as tmpy
import temperance.core.result as result
from temperance.core.result import EoSPosterior
import json
import temperance.plotting.corner as tmcorner 
import temperance.plotting.get_quantiles as get_quantiles
import temperance.plotting.envelope as envelope
import h5py
import matplotlib.pyplot as plt
import matplotlib as mpl
envelope.get_defaults(mpl, fontsize=20)
import temperance.sampling.branched_interpolator as b_interp
from scipy.stats import gaussian_kde as skde
import scipy.interpolate as interpolate
isaac_path="/home/isaac.legred/EoSPopulation/Reweighting/EoS/AllNicer/"
def check_containment(array, other_array):
    return [array[i] in other_array for i, elt in enumerate(array)]

def compute_quantile(vals, weights, quantile):
    order = np.argsort(np.array(vals))
    cdf = np.cumsum(weights[order])/np.sum(weights)
    return interpolate.griddata(cdf, np.array(vals)[order], quantile)
def compute_quantiles(quantity, post, weight_columns, quantiles=[0.05, 0.5, 0.95], decimals=2, scale_factor=1.0):
    def process_quantiles_to_latex(quantiles):

        median = np.round(quantiles[1], decimals=decimals)
        delta_up = np.round(quantiles[2]-quantiles[1], decimals=decimals)
        delta_low = np.round(quantiles[1]-quantiles[0], decimals=decimals)
        return rf"\result{{${{{median}}}^{{+{delta_up}}}_{{-{delta_low}}}$}}"
    values = post.samples[quantity] * scale_factor
    weights = post.get_total_weight(weight_columns)["total_weight"]
    results= []
    for quantile in quantiles:
        results.append(compute_quantile(values, weights, quantile))
    return process_quantiles_to_latex(results)
if __name__ == "__main__":
    print("running main")
    astro_post = EoSPosterior.from_csv(isaac_path+"/collated_np_all_post.csv", label="astro")
    eos_set = h5py.File(isaac_path+"./gp_mrgagn.h5")
    eoss_used_in_inference = eos_set["id"]
    astro_post = EoSPosterior(astro_post.samples.loc[check_containment(
        astro_post.samples["eos"],  np.array(eoss_used_in_inference))], label="prior")

    weights_data = pd.read_pickle("/home/jacob.golomb/GWs_and_pulsars/combined/highspin_final_weights.pkl")["log_likelihood_eos"]

    #indexed_weights = pd.DataFrame({"eos":eos_set["id"],
    #                                "weight_posterior": np.array([em_astro_weights)_weoweights_soweights_data[f"eos_{loc}"]/1e15 for loc in range(20000)])})
    order = np.argsort([f"eos_{i}" for i in range(20000)])
    order_2 = np.argsort(order)

    #indexed_weights = pd.DataFrame({"eos":eos_set["id"],
    #                                "weight_posterior": np.array([em_astro_weights)_weoweights_soweights_data[f"eos_{loc}"]/1e15 for loc in range(20000)])})
    indexed_weights = pd.DataFrame({"eos": eos_set["id"],
                                    "weight_posterior": np.exp([weights_data[order_2[i]] - 40 for i in range(20000)])})
    print(np.where(np.isnan(indexed_weights["weight_posterior"] )))
    print(indexed_weights)
    posterior_weight = result.WeightColumn("weight_posterior",is_log=False)
    astro_post.add_weight_column(result.WeightColumn("weight_posterior", is_log=False), indexed_weights)   
    print(astro_post.samples)
    #prior_p_of_rho_quantiles = get_quantiles.get_p_of_rho_quantiles(astro_post, weight_columns=[])
    #prior_p_of_rho_quantiles.to_csv("Quantiles/prior_p_of_rho_quantiles.csv", index=False)
    #prior_r_of_m_quantiles = get_quantiles.get_r_of_m_quantiles(astro_post, weight_columns=[])
    #prior_r_of_m_quantiles.to_csv("Quantiles/prior_r_of_m_quantiles.csv", index=False)
    #prior_cs2_of_rho_quantiles = get_quantiles.get_cs2_of_rho_quantiles(astro_post, weight_columns=[])
    #prior_cs2_of_rho_quantiles.to_csv("Quantiles/prior_cs2_of_rho_quantiles.csv", index=False)

    def annotate_nuclear_density(y_level=1.01):
        plt.axvline(2.8e14, color="k", lw=2.0)
        plt.text(2.8e14, y_level, r"$\rho_{\rm nuc}$",{"fontsize":16} )
        plt.axvline(2*2.8e14, color="k", lw=2.0)
        plt.text(2 * 2.8e14, y_level, r"$2\rho_{\rm nuc}$", {"fontsize":16})
        plt.axvline(4 * 2.8e14, color="k", lw=2.0)
        plt.text(4 * 2.8e14, y_level, r"$4\rho_{\rm nuc}$", {"fontsize":16})
    def plot_cs2_quantiles():
        cs2_quantiles = pd.read_csv(isaac_path+"./Quantiles/cs2_of_rho_quantiles.csv")
        prior_cs2_of_rho_quantiles = pd.read_csv(isaac_path+"./Quantiles/prior_cs2_of_rho_quantiles.csv")
        legred_cs2_of_rho_quantiles = pd.read_csv("/home/isaac.legred/New_NICER/NSTOVMaxAnalysis/Quantiles/cs2_rho/all_miller_quantiles_cs2.csv")
        cs2_plottable_quantiles = envelope.PlottableQuantiles(label="posterior", quantiles=cs2_quantiles,posterior=astro_post,color="fuchsia")
        prior_cs2_plottable_quantiles = envelope.PlottableQuantiles(label="prior", quantiles=prior_cs2_of_rho_quantiles,posterior=astro_post, color="black")
        #legred_plottable_cs2_of_rho_quantiles = envelope.PlottableQuantiles(label="Legred+21", quantiles=legred_cs2_of_rho_quantiles, posterior=astro_post, color="orange")
        envelope.plot_envelope([cs2_plottable_quantiles, prior_cs2_plottable_quantiles])#, legred_plottable_cs2_of_rho_quantiles])
        plt.xlim(9.4e13, 2e15)
        plt.xscale("log")
        plt.ylim(1e-3, 1.0)
        #plt.yscaale("log")
        plt.xlabel(r"$\rho\ [\rm{g}/\rm{cm}^3]$")
        plt.ylabel(r"$c_s^2 $")
        plt.legend(loc="upper left")
        annotate_nuclear_density()
        plt.savefig("./plots_sparkler/cs2_of_rho_quantiles.pdf", bbox_inches="tight")
    #plot_cs2_quantiles()
    plt.clf()
    def plot_mr_quantiles():
        mr_quantiles = pd.read_csv(isaac_path+"./Quantiles/r_of_m_quantiles.csv")
        prior_mr_quantiles = pd.read_csv(isaac_path+"./Quantiles/prior_r_of_m_quantiles.csv")
        legred_mr_quantiles = pd.read_csv("/home/isaac.legred/New_NICER/NSTOVMaxAnalysis/Quantiles/m_r/all_miller_mr_quantiles.csv")
        mr_plottable_quantiles = envelope.PlottableQuantiles(label="posterior", quantiles=mr_quantiles,posterior=astro_post,color="deepskyblue", flip_axes=True)
        prior_mr_plottable_quantiles = envelope.PlottableQuantiles(label="prior", quantiles=prior_mr_quantiles,posterior=astro_post, color="black", flip_axes=True)
        #legred_plottable_mr_quantiles = envelope.PlottableQuantiles(label="legred+", quantiles=legred_mr_quantiles, posterior=astro_post, color="orange", flip_axes=True)
        envelope.plot_envelope([mr_plottable_quantiles, prior_mr_plottable_quantiles])#, legred_plottable_mr_quantiles])
        plt.xlim(8, 15)
        plt.ylim(.8, 2.1)

 
        #plt.yscale("log")
        plt.xlabel(r"$R\ [\rm{km}]$")
        plt.ylabel(r"$M\ [M_{\odot}]$")
        plt.legend(loc="upper left")
        #annotate_nuclear_density()
        plt.savefig("./plots_sparkler/r_of_m_quantiles.pdf", bbox_inches="tight")
        
    #plot_mr_quantiles()
    plt.clf()
    def plot_cs2_max(**kwargs):
        bins=50
        #plt.hist(astro_post.samples["cs2c2max"], weights=astro_post.get_total_weight(weight_columns_to_use=[result.WeightColumn("logweight_total")])["total_weight"], color="orange",
                 #histtype="step", density=True, bins=50, **kwargs)
        plt.hist(astro_post.samples["cs2c2max"], weights=astro_post.get_total_weight(weight_columns_to_use=[posterior_weight])["total_weight"],
                 color="deepskyblue", histtype="step", density=True, bins=50, **kwargs)
        plt.hist(astro_post.samples["cs2c2max"], color="black", histtype="step", density=True, bins=50, **kwargs)
        plt.xlabel(r"$\max(c_s^2)$")
        plt.ylabel(r"$p(\max(c_s^2)$")
        plt.legend(["posterior", "prior"])
        plt.savefig("plots_sparkler/max_cs2.pdf", bbox_inches="tight")

    #plot_cs2_max()
    plt.clf()
    def plot_m1p4_quantities(var="R(M=1.4)", **kwargs):
        ###
        # Legred+
        ###

        lambda_bins = np.linspace(40, 1500, 80)
        r_bins = np.linspace(10, 15, 70)

        ###
        # New Results
        ###

        bins = r_bins if var == "R(M=1.4)" else lambda_bins
        print("samples" + var, astro_post.samples[var])
        #plt.hist(
       #     astro_post.samples[var],
          #  weights=astro_post.get_total_weight(
          #      weight_columns_to_use=[
           #         result.WeightColumn("logweight_total")])["total_weight"],
          #  color="orange", histtype="step", density=True, bins=bins, **kwargs)
        plt.hist(
            astro_post.samples[var],
            weights=astro_post.get_total_weight(
                weight_columns_to_use=[posterior_weight])["total_weight"],
            color="deepskyblue",
            histtype="step", density=True, bins=bins, **kwargs)
        plt.hist(astro_post.samples[var], color="black",
                 histtype="step", density=True,bins=bins, **kwargs, fill=True, alpha=.2)
        plt.legend(["full posterior", "gp prior"])
        varlabels ={"R(M=1.4)":r"R_{{1.4}}",
                    "Lambda(M=1.4)":r"\Lambda_{{1.4}}"}
        varlabel = varlabels[var]
        plt.xlabel(rf"${varlabel}$")
        plt.ylabel(rf"$p({varlabel})$")
        savetag = {"R(M=1.4)":"r1p4",
                   "Lambda(M=1.4)":"lambda1p4"}
        if var == "Lambda(M=1.4)":
            plt.xlim(40, 1500)
            plt.ylim(0, .0045)
        plt.savefig(f"plots_sparkler/comparison_{savetag[var]}.pdf", bbox_inches="tight" )
    def plot_m1p4_quantities_corner(**kwargs):
        ###
        # Legred+
        ###

        ###
        # New Results
        ###
        plottable_columns={}
        plottable_columns["Mmax"] = tmcorner.PlottableColumn(
            name="Mmax",
            label=r"$M_{\rm TOV}\ [M_{\odot}]$",
            plot_range=(1.9, 2.9),
            bandwidth=.05)
        plottable_columns["R1p4"] = tmcorner.PlottableColumn(
            name="R(M=1.4)",
            label=tmcorner.get_default_label("R(M=1.4)"),
            plot_range=(10.8, 15.0),
            bandwidth=.2)

        plottable_columns["Lambda1p4"] = tmcorner.PlottableColumn(
            name="Lambda(M=1.4)",
            label=r"$\Lambda_{1.4}$",
            plot_range=(1.1e2,9.1e2),
            bandwidth=40)
        
        posterior_samples = tmcorner.PlottableEoSSamples(
            label="full posterior", posterior=astro_post,
            weight_columns_to_use=[posterior_weight],
            color="deepskyblue",  additional_properties=astro_post.samples)
        prior_samples = tmcorner.PlottableEoSSamples(
            label="gp prior", posterior=astro_post,
            weight_columns_to_use=[],
            color="black", additional_properties=astro_post.samples)
        
        tmcorner.corner_eos([posterior_samples, prior_samples], use_universality=True,
                            columns_to_plot=plottable_columns.values(), **kwargs)

        plt.savefig("plots_sparkler/mtov_vs_r_lambda_corner.pdf", bbox_inches="tight")
    #plot_m1p4_quantities_corner()
    #plot_m1p4_quantities(var="R(M=1.4)", lw=2.0)
    #plot_m1p4_quantities(var="Lambda(M=1.4)", lw=2.0)
    def plot_mass_radius_of_j0030():
        samples=pd.read_csv(isaac_path+
            "../../../New_NICER/NSTOVMaxAnalysis/NewCalcSamples/Miller_J0030_three-spot_post.csv")
        print(samples)
        samples=samples.sample(20000, weights=result.get_total_weight(samples, weight_columns=[
            result.WeightColumn("logweight")])["total_weight"])
        samples=pd.merge(samples, astro_post.samples[["eos", "logweight_total",
                                              "logweight_Miller_J0030_threespot"]])
        # Don't use logweight cause we already sampled for it
        informed_samples = tmcorner.PlottableSamples(
            label="Astro", samples=samples,
            weight_columns_to_use=[
                result.WeightColumn("logweight_total"),
                result.WeightColumn("logweight_Miller_J0030_threespot", is_inverted=True)],
            color="navy")
        original_samples = tmcorner.PlottableSamples(
            label="Original", samples=samples,
            weight_columns_to_use=[], color="red")
        plottable_columns = [
            tmcorner.PlottableColumn(name="R", label=r"$R\ [\rm{km}]$", plot_range=(8, 16)),
            tmcorner.PlottableColumn(name="m", label=r"$M\ [\rm{km}]$", plot_range=(1.0, 2.0))]
        tmcorner.corner_samples([original_samples, informed_samples], 
                                use_universality=True, columns_to_plot=plottable_columns)
        
        plt.legend()
        plt.xlabel(r"$M\ [M_{\odot}]$")
        plt.ylabel(r"$R\ [\rm{km}$")
        plt.savefig("plots_sparkler/J0030_m_r_astro_informed.pdf", bbox_inches="tight")
    #plot_mass_radius_of_j0030()
    def estimate_evidence_of_conformal_violations():
        conformal_weights = astro_post.samples[["eos", "cs2c2max"]].copy()
        conformal_weights["weight_violates_conformal_limit"] = np.array(
            conformal_weights["cs2c2max"] > 1/3, dtype=float)
        conformal_weights.pop("cs2c2max")
        conformal_weight_column = result.WeightColumn("weight_violates_conformal_limit",
                                is_log=False)
        astro_post.add_weight_column(
            conformal_weight_column, conformal_weights)
        weight_to_use = posterior_weight
        weight_to_use = result.WeightColumn("logweight_total")
        Z_conf_violated = astro_post.estimate_evidence(
            weight_columns_to_use=[weight_to_use], prior_weight_columns= [conformal_weight_column])
        Z_conf_obeyed = astro_post.estimate_evidence(
            weight_columns_to_use=[weight_to_use],prior_weight_columns=
                                   [conformal_weight_column.get_inverse()])
        p_violate = np.sum(conformal_weights["weight_violates_conformal_limit"])/len(conformal_weights["weight_violates_conformal_limit"])
        print(p_violate/(1-p_violate))
        return (Z_conf_violated[0]/ Z_conf_obeyed[0], Z_conf_violated[1], Z_conf_obeyed[1])
    print("Bayes Factor for violation", estimate_evidence_of_conformal_violations())
    def plot_m_of_rhoc_quantiles():
        m_of_rhoc_quantiles  = pd.read_csv(isaac_path + "./Quantiles/M_of_rhoc_quantiles.csv")
        prior_m_of_rhoc_quantiles = pd.read_csv(isaac_path + "../Quantiles/prior_M_of_rhoc_quantiles.csv")
        legred_m_of_rhoc_quantiles = pd.read_csv(isaac_path + "./Quantiles/legred_M_of_rhoc_quantiles.csv")
        plottable_m_of_rhoc_quantiles = envelope.PlottableQuantiles(label="posterior", quantiles = m_of_rhoc_quantiles, posterior=astro_post, color="fuchsia")
        plottable_prior_m_of_rhoc_quantiles = envelope.PlottableQuantiles(label="prior", quantiles = prior_m_of_rhoc_quantiles, posterior=astro_post, color="black")
        #plottable_legred_m_of_rhoc_quantiles = envelope.PlottableQuantiles(
            #label="Legred21", quantiles = legred_m_of_rhoc_quantiles, posterior=astro_post, color="orange")
        envelope.plot_envelope([plottable_m_of_rhoc_quantiles, plottable_prior_m_of_rhoc_quantiles])
        # Plot inferred maximum-mass rhoc contours
        rhoc_range = np.linspace(2.8e14, 7*2.8e14, 30)
        M_range = np.linspace(0.5, 2.8, 30)
        X, Y = np.meshgrid(rhoc_range, M_range)
        density_estimate = skde(
            np.transpose(np.array(astro_post.samples[["rhoc(M@Mmax)", "Mmax"]])),
            weights = astro_post.get_total_weight([posterior_weight])["total_weight"] )
        #legred_density_estimate = skde(
          #  np.transpose(np.array(astro_post.samples[["rhoc(M@Mmax)", "Mmax"]])),
          #  weights = astro_post.get_total_weight([lwp_legred_logweight])["total_weight"] )
        pdf = density_estimate(np.vstack([X.flatten(),Y.flatten()])).reshape(X.shape[0], X.shape[1])
        #pdf_legred = legred_density_estimate(np.vstack([X.flatten(),Y.flatten()])).reshape(X.shape[0], X.shape[1])
        e = np.exp(1)
        plt.contour(X, Y, pdf, colors=["limegreen", "green"], levels=[np.max(pdf)/e**2, np.max(pdf)/e**(.5)], linewidth=3.0, label=r"$M_{\rm TOV}$")
        #plt.contour(X, Y, pdf_legred, colors=["orange", "darkgoldenrod"], levels=[np.max(pdf)/e**2, np.max(pdf)/e**(.5)], linewidth=3.0)
        print("rho(cs2@cs2max) is inferred to be", compute_quantiles(
            quantity="baryon_density(cs2c2@cs2c2max)", post=astro_post,
            weight_columns=[posterior_weight], quantiles=[0.05, 0.5, 0.95], decimals=2, scale_factor=1/2.8e14))
        print("rhoc(Mmax) is inferred to be", compute_quantiles(
            quantity="rhoc(M@Mmax)", post=astro_post,
            weight_columns=[posterior_weight], quantiles=[0.05, 0.5, 0.95], decimals=2, scale_factor=1/2.8e14))

        #print("rhoc(Mmax) for lwp-legred211 is inferred to be", compute_quantiles(
            #quantity="rhoc(M@Mmax)", post=astro_post,
            #weight_columns=[lwp_legred_logweight], quantiles=[0.05, 0.5, 0.95], decimals=2, scale_factor=1/2.8e14))

        plt.xlim(min(rhoc_range), max(rhoc_range))
        plt.ylim(min(M_range), max(M_range))
        plt.xlabel(r"$\rho_{c}\ [\rm{g}/\rm{cm}^3]$")
        plt.ylabel(r"$m\ [M_{\odot}] $")
        plt.legend(loc="lower right")
        annotate_nuclear_density(y_level=2.86)
        plt.savefig("./plots_sparkler/m_of_rhoc_quantiles.pdf", bbox_inches="tight")
    plot_m_of_rhoc_quantiles()

def get_pkl_weights_data_per_eos(paths, marginalization_path,  eos_set, styles, marginalization_style):
    likelihood_datas = [pd.read_pickle(path) for path in paths]
    marginalization_prior_data = pd.read_pickle(marginalization_path)
    if marginalization_style == "GW_new":
        marginalization_logweights = np.transpose(marginalization_prior_data["mass_samples"]["log_likelihood"][:])
    elif marginalization_style=="Nicer_old":
        marginalization_logweights = np.transpose(marginalization_prior_data["mass_population_list"]["log_likelihood"][:])
    elif marginalization_style=="Pulsar_old":
        marginalization_logweights = np.transpose(marginalization_prior_data["mass_population_list"]["log_likelihood"][:])
    def get_likelihoods_from_data_and_style(likelihood_data, style="GW_new"):
        if style=="GW_new":
            log_likelihood = np.array([likelihood_data[f"eos_{i}"] for i in range(20000)])

        elif style=="Nicer_old":
            log_likelihood = likelihood_data["log_likelihoods_sorted"]
        elif style=="Pulsar_old":
            order = np.argsort([f"eos_{i}" for i in range(20000)])
            log_likelihood = np.array([likelihood_data["full"][i,:] for i in order])
        else:
            raise ValueError(f"can't find style {style}")
        print("log likelihood shape", log_likelihood.shape)
        return log_likelihood
    log_likelihood_total = None
    for n, likelihood_data in enumerate(likelihood_datas):
        if log_likelihood_total is None:
            log_likelihood_total = get_likelihoods_from_data_and_style(likelihood_data, styles[n])
        else:
            log_likelihood_total += get_likelihoods_from_data_and_style(likelihood_data, styles[n])
            log_likelihood_total -= np.max(log_likelihood_total)

    eos_weights_marginalized_over_hyperparameters = np.sum(np.exp(log_likelihood_total - marginalization_logweights), axis=1)
    return pd.DataFrame({"eos": eos_set["id"],"weight_posterior": eos_weights_marginalized_over_hyperparameters  } )


