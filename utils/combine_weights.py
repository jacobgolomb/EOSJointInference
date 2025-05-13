import pickle
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--psr-weights', '-p', type=str, help="Conditioned weights for the radio pulsars")
parser.add_argument('--nicer-weights', '-n', type=str, help="Conditioned weights for the NICER observations")
parser.add_argument('--gw-weights', '-g', type=str, help="Conditioned weights for the GW observations")
parser.add_argument('--label', '-l', type=str, help="Label for this dataset")
parser.add_argument('--skip-nicer', action=argparse.BooleanOptionalAction, default=False)
args=parser.parse_args()
weights = dict()
weights["PSR"] = pickle.load(open(args.psr_weights, 'rb'))
if not args.skip_nicer:
    weights["NICER"] = pickle.load(open(args.nicer_weights, 'rb'))
weights["GW"] = pickle.load(open(args.gw_weights, 'rb'))

EM_log_likelihood = weights["PSR"]["log_likelihoods_sorted"]
if not args.skip_nicer:
     EM_log_likelihood += weights["NICER"]["log_likelihoods_sorted"]
GW_log_likelihood = weights["GW"]["log_likelihoods_sorted"]

EM_log_likelihood_eos = np.log(np.average(
    np.exp(EM_log_likelihood - weights["PSR"]["log_proposal"]), axis=-1) 
                          ) 
GW_log_likelihood_eos = np.log(np.average(
    np.exp(GW_log_likelihood - weights["GW"]["log_proposal"]), axis=-1) 
                          )
log_likelihood_eos = EM_log_likelihood_eos + GW_log_likelihood_eos

log_likelihood_EM = EM_log_likelihood + GW_log_likelihood_eos[:,None]
log_likelihood_GW = GW_log_likelihood + EM_log_likelihood_eos[:,None]

log_likelihood_GW_pop = np.log(np.average(np.exp(
    EM_log_likelihood_eos[:,None] + GW_log_likelihood), axis=0)
                             )
log_likelihood_EM_pop = np.log(np.average(np.exp(
    GW_log_likelihood_eos[:,None] + EM_log_likelihood), axis=0)
                             )

final_weights = dict()
final_weights["GW_log_proposal"] = weights["GW"]["log_proposal"]
final_weights["EM_log_proposal"] = weights["PSR"]["log_proposal"]
final_weights["EM_log_likelihood"] = EM_log_likelihood
final_weights["GW_log_likelihood"] = GW_log_likelihood
final_weights["EM_log_likelihood_eos"] = EM_log_likelihood_eos
final_weights["GW_log_likelihood_eos"] = GW_log_likelihood_eos

final_weights["log_likelihood_EM"] = log_likelihood_EM
final_weights["log_likelihood_GW"] = log_likelihood_GW
final_weights["log_likelihood_GW_pop"] = log_likelihood_GW_pop
final_weights["log_likelihood_EM_pop"] = log_likelihood_EM_pop
final_weights["log_likelihood_eos"] = log_likelihood_eos
final_weights["command_line"] = vars(args)

final_weights["EM_population_parameters"] = weights["PSR"]["mass_population_list"]
final_weights["GW_population_parameters"] = weights["GW"]["mass_population_list"]
if not args.skip_nicer:
    assert all([all(np.isclose(weights["PSR"]["mass_population_list"][param].flatten(), weights["NICER"]["mass_population_list"][param].flatten())) for param in
            weights["PSR"]["mass_population_list"].keys() if "Unnamed" not in param])
    for dset in ["NICER", "GW"]:
        assert all(np.sort(weights["PSR"]["eos_list"]) == np.sort(weights[dset]["eos_list"]))
    
final_weights["eoses"] = np.sort(weights["PSR"]["eos_list"])
with open(f"{args.label}.pkl", "wb") as ww:
    pickle.dump(final_weights, ww)