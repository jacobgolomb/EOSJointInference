import bilby
from GMM_likelihood import Transformer, GMM_Generator

results = dict()
#results['GW170817'] = bilby.result.read_in_result("GW170817/GW170817_HLVlowspin_data0_1187008882-43_analysis_H1L1V1_merge_result.hdf5")
#results["GW190425"] = bilby.result.read_in_result("GW190425/LVK_highspin/GW190425_Pv2NRTidalv2_HighSpin.hdf5")
#results['GW170817'].priors['luminosity_distance'] = bilby.gw.prior.UniformSourceFrame(name='luminosity_distance', minimum = 1, maximum=100)

for result in results:
    transformer = Transformer(result = results[result], parameters = ["chirp_mass_source", "mass_ratio", "lambda_1", "lambda_2"], generate_source_frame=["chirp_mass"])
    gmm = GMM_Generator(transformer=transformer, outdir = result + '/', label = result)
    gmm.generate_gmm()
    gmm.plot_corner()
