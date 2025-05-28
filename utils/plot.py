
import sys
import pickle
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import corner
from tqdm import tqdm
from sampling_utils import UtilFuncs
import models
from models import get_pulsar_mass_weights
util_funcs = UtilFuncs('numpy')
models.set_backend("numpy")
import glob
from scipy import stats
from matplotlib.colors import LinearSegmentedColormap, colorConverter
from scipy.stats import gaussian_kde


weights_file = sys.argv[1]
default_weights = pickle.load(open(weights_file, 'rb'))

def get_defaults(matplotlib, fontsize=18):
    print("Warning! Modifying matplotlib defaults.")
    matplotlib.rcParams['figure.figsize'] = (9.7082039325, 6.0)
    matplotlib.rcParams['xtick.labelsize'] = fontsize
    matplotlib.rcParams['ytick.labelsize'] = fontsize
    matplotlib.rcParams['axes.labelsize'] = fontsize
    matplotlib.rcParams['legend.fontsize'] = fontsize
    plt.rcParams['font.family'] = 'DeJavu Serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    matplotlib.rcParams['font.sans-serif']= ['Bitstream Vera Sans']
    matplotlib.rcParams['text.usetex']= True
    matplotlib.rcParams['mathtext.fontset']= 'dejavuserif'
    matplotlib.rcParams['xtick.top'] = True
    matplotlib.rcParams['ytick.right'] = True
get_defaults(matplotlib)

macros = json.load(open("gp_mrgagn_eos_macros.json", "r"))
for eos in macros:
    macros[eos]['m'] = np.round(macros[eos]['m'], 2)


# In[6]:


idxs = dict()
traces=dict()
pop_samples = dict()
masses = np.linspace(1, 3, 3000)

for dset in ["GW", "EM"]:
    rng = np.random.RandomState(seed=100)

    weights_here = np.exp(default_weights[f'log_likelihood_{dset}_pop']- default_weights[f"{dset}_log_proposal"])
    idxs[dset] = rng.choice(len(default_weights[f'log_likelihood_{dset}_pop']),replace=True,
                                  p=weights_here/sum(weights_here)
                                  ,size=1000)
traces["EM"] = []
pop_samples["EM"] = []
em_params = ['mmin', 'mpop', 'mu1', 'mu2', 'sigma1', 'sigma2', 'frac1']
for idx in idxs["EM"]:
    sample_here = {param: default_weights["EM_population_parameters"][param][idx] for param in em_params}
    pop_samples["EM"].append(sample_here)
    trace = get_pulsar_mass_weights(masses, 1, **sample_here)
    traces["EM"].append(trace)

traces["EM"] = np.asarray(traces["EM"])

traces["GW"] = []
for idx in idxs["GW"]:
    trace = util_funcs.powerlaw(xx=masses,alpha=default_weights["GW_population_parameters"]['alpha'][idx],
                                high= default_weights["GW_population_parameters"]['mpop'][idx], low=1)
    traces["GW"].append(trace)
traces["GW"]=np.asarray(traces["GW"])


# In[7]:


def weighted_percentile(data, weights, percentile):
    """
    Calculate the weighted percentile of an array of data given the weights.
    
    :param data: numpy array of data
    :param weights: numpy array of weights
    :param percentile: desired percentile (between 0 and 100)
    :return: the weighted percentile value
    """
    data, weights, percentile = map(np.asarray, (data, weights, percentile))
    # Sort the data and weights
    sorted_indices = np.argsort(data)
    sorted_data = data[sorted_indices]
    sorted_weights = weights[sorted_indices]
    
    # Calculate the cumulative sum of the weights
    cumulative_weights = np.cumsum(sorted_weights)
    
    # Normalize the cumulative weights to be between 0 and 1
    normalized_weights = cumulative_weights / cumulative_weights[-1]
    
    # Find the percentile position
    percentile_position = percentile / 100.0
    
    return np.interp(percentile_position, normalized_weights, sorted_data)


# In[ ]:


trace_colors = {"GW": 'limegreen', "EM": 'navy'}
for run in ["GW", "EM"]:
    plt.fill_between(masses, *np.percentile(traces[run], q=[5,95], axis=0), alpha=0.2, 
                     color=trace_colors[run])
    plt.plot(masses, np.percentile(traces[run], q=50, axis=0), color=trace_colors[run], label = run + " Population")
    if run == "GW":
        for i in np.arange(0, 50, 1):
            plt.plot(masses, traces[run][i], alpha=0.2, color=trace_colors[run])
plt.xlim(1, 2.25)
plt.ylim(0, 5.5)
plt.legend()
plt.yticks([])
plt.xticks(fontsize=24)
plt.ylabel(r"$p(m)$", fontsize=24)
plt.xlabel(r"$m \ [m_\odot]$", fontsize=24)
plt.savefig(r"figs/mass_traces.pdf", bbox_inches='tight')


# In[25]:


def make_corner_contour(x_data_, y_data_, ax, weights, levels, color, sample=2000):
    levels = [0.0001] + levels
    rgba_color = colorConverter.to_rgba(color)
    contour_cmap = [list(rgba_color) for _ in levels]
    for i, l in enumerate(levels):
        contour_cmap[i][-1] = (i+1)/len(levels) * 0.99       
    indxs = np.random.choice(len(x_data_), p=weights/sum(weights), replace=True, size=sample)
    x_data = x_data_[indxs]
    y_data = y_data_[indxs]
        
    kde = gaussian_kde(np.array([x_data, y_data]), bw_method=0.24)
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    
    xx, yy = np.meshgrid(np.linspace(xmin,xmax,100), np.linspace(ymin,ymax,100))
    vals = kde(np.array([xx.flatten(), yy.flatten()])).reshape(xx.shape)
    vals[xx > yy] = 0

    Z_flat = vals.flatten()
    Z_flat_sorted = np.sort(Z_flat)[::-1]  # Sort in descending order

    # Compute cumulative sum of the probabilities
    cumsum = np.cumsum(Z_flat_sorted)
    cumsum /= cumsum[-1]  # Normalize to 1

    # Find the thresholds for 50% and 90% credible regions
    level_list = [Z_flat_sorted[np.searchsorted(cumsum, level)] for level in sorted(levels)[::-1]]

    # Plot the KDE contours for 50% and 90% credible regions
    ax.contourf(xx,yy, vals, levels=level_list, colors=contour_cmap)
    ax.contour(xx,yy, vals, levels=level_list, colors=color)

# In[ ]:


mpopGW, mtovs = np.meshgrid(default_weights["GW_population_parameters"]["mpop"], [macros[eos]['mtov'] for eos in default_weights['eoses']])
levels = [0.5, 0.9]
corner_kwargs = dict(smooth=None, plot_datapoints=False, range=[(1.5,2.5), (1.5, 3.5)],
                 plot_density=False, levels=levels,fill_contours=False, bins=25, plot_contours=False
                    )
fig=corner.corner({'mpop_GW': mpopGW.flatten(),
               'mtovs_GW': mtovs.flatten()},
              weights=np.exp(default_weights["log_likelihood_GW"]-default_weights["GW_log_proposal"]).flatten(),
                  hist_kwargs=dict(color = trace_colors['GW'], density=True, alpha=0.6), color=trace_colors['GW'], **corner_kwargs
)

fig.axes[2].plot([],[], color=trace_colors["GW"], label='GW')
make_corner_contour(mpopGW.flatten(), mtovs.flatten(), fig.axes[2], weights=np.exp(default_weights["log_likelihood_GW"]-default_weights["GW_log_proposal"]).flatten(),
                    levels=levels, color=trace_colors["GW"])
mpopEM, mtovs = np.meshgrid(default_weights["EM_population_parameters"]["mpop"], [macros[eos]['mtov'] for eos in default_weights['eoses']])

corner.corner({'mpop_EM': mpopEM.flatten(),
               'mtovs_EM': mtovs.flatten()},
              weights=np.exp(default_weights["log_likelihood_EM"]-default_weights["EM_log_proposal"]).flatten(), 
              fig=fig, hist_kwargs=dict(color=trace_colors["EM"], density=True, alpha=0.6),
              color=trace_colors['EM'], 
              labels=[r'$M_{\rm pop} \ [M_\odot]$', r'$M_{\rm TOV} \ [M_\odot]$'],
             **corner_kwargs)

fig.axes[2].plot([],[], color=trace_colors["EM"], label='EM')

fig.axes[2].axline([0, 0], [1, 1], linestyle=':', color='k')

make_corner_contour(mpopEM.flatten(), mtovs.flatten(), fig.axes[2], weights=np.exp(default_weights["log_likelihood_EM"]-default_weights["EM_log_proposal"]).flatten(),
                    levels=levels, color=trace_colors["EM"])

fig.legend(loc="upper right")
fig.savefig("figs/mpop_mtov_corner.pdf", bbox_inches='tight')



# mpopGW, mtovs = np.meshgrid(weights["lowspin"]["GW_population_parameters"]["mpop"], [macros[eos]['mtov'] for eos in weights["lowspin"]['eoses']])
# levels = [0.5, 0.9]
# corner_kwargs = dict(smooth=None, plot_datapoints=False, range=[(1.5,2.5), (1.5, 3.5)],
#                  plot_density=False, levels=levels,fill_contours=False, bins=25, plot_contours=False
#                     )
# fig=corner.corner({'mpop_GW': mpopGW.flatten(),
#                'mtovs_GW': mtovs.flatten()},
#               weights=np.exp(weights["lowspin"]["log_likelihood_GW"]-weights["lowspin"]["GW_log_proposal"]).flatten(),
#                   hist_kwargs=dict(color = trace_colors['GW'], density=True, alpha=0.6), color=trace_colors['GW'], **corner_kwargs
# )

# fig.axes[2].plot([],[], color=trace_colors["GW"], label='GW')
# make_corner_contour(mpopGW.flatten(), mtovs.flatten(), fig.axes[2], weights=np.exp(weights["lowspin"]["log_likelihood_GW"]-weights["lowspin"]["GW_log_proposal"]).flatten(),
#                     levels=levels, color=trace_colors["GW"])
# mpopEM, mtovs = np.meshgrid(weights["lowspin"]["EM_population_parameters"]["mpop"], [macros[eos]['mtov'] for eos in weights["lowspin"]['eoses']])

# corner.corner({'mpop_EM': mpopEM.flatten(),
#                'mtovs_EM': mtovs.flatten()},
#               weights=np.exp(weights["lowspin"]["log_likelihood_EM"]-weights["lowspin"]["EM_log_proposal"]).flatten(), 
#               fig=fig, hist_kwargs=dict(color=trace_colors["EM"], density=True, alpha=0.6),
#               color=trace_colors['EM'], 
#               labels=[r'$M_{\rm pop} \ [M_\odot]$', r'$M_{\rm TOV} \ [M_\odot]$'],
#              **corner_kwargs)

# fig.axes[2].plot([],[], color=trace_colors["EM"], label='EM')

# fig.axes[2].axline([0, 0], [1, 1], linestyle=':', color='k')

# make_corner_contour(mpopEM.flatten(), mtovs.flatten(), fig.axes[2], weights=np.exp(weights["lowspin"]["log_likelihood_EM"]-weights["lowspin"]["EM_log_proposal"]).flatten(),
#                     levels=levels, color=trace_colors["EM"])

# fig.legend(loc="upper right")
# fig.savefig("figs/lowspin_mpop_mtov_corner.pdf", bbox_inches='tight')