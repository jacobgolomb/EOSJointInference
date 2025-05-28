import corner
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
import bilby
import argparse
import matplotlib.pyplot as plt
import os
cwd = os.getcwd()
import sys
sys.path.append(cwd)

parser=argparse.ArgumentParser()
parser.add_argument("--result-file", "-r", help="bilby result file with hyperposterior", type=str)
parser.add_argument("--n-samples", "-n", help="Number of population hyperposterior samples to draw", type=int, default=5000)
parser.add_argument('--output-file', '-o', default="proposal_population_samples.csv", help="Where to save files", type=str)
parser.add_argument("--kde-bw", '-bw', default="0.1", type=str, help="KDE bandwidth to use. Either a number or one of the allowed methods in scipy")
args=parser.parse_args()

try:
    bw = float(args.kde_bw)
except:
    bw = args.kde_bw
    
output_file = args.output_file.split(".csv")[0]+".csv"
    
result = bilby.result.read_in_result(args.result_file)
samples = result.posterior
exclude = ["log_likelihood", "log_prior", "m", "b", "mmin"]
hyper_parameter_keys = [key for key in samples if key not in exclude]
print(hyper_parameter_keys)

proposal_array = np.array([np.array(samples[key]) for key in hyper_parameter_keys])
proposal_kde = gaussian_kde(proposal_array, bw_method=bw)
kde_samples = proposal_kde.resample(args.n_samples)
proposal_vals = proposal_kde(kde_samples)
kde_samples = pd.DataFrame({key: kde_samples[i, :] for i, key in enumerate(hyper_parameter_keys)})
kde_samples['proposal_log_prob'] = np.log(proposal_vals)
if 'mmin' in exclude:
    kde_samples['mmin'] = np.ones_like(kde_samples['proposal_log_prob']) * np.mean(result.posterior['mmin'])

kde_samples.to_csv(output_file, index=False)