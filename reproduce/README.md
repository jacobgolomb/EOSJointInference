The shell scripts in this directory reproduce the results from Golomb et al. 2024 (10.1103/PhysRevD.111.023029), specifically under the low-spin GW source assumption. This requires several steps and familiarity with the instructions in the parent directory and the procedure outlined in the paper.

Running the shell scripts in the following order produce the output files in this directory:

1. run_EM_inference.sh
2. run_GW_inference.sh
3. draw_proposal_single_event_samples.sh
4. calculate_likelihoods_GWs.sh
5. calculate_likelihoods_NICER.sh
6. weight_populations.sh
7. condition_weights.sh
8. combine_weights.sh
9. make_plots.sh

The outputs that are currently in this directory were from actually running these files in this order.

Note that there may be environment or path issues that you need to resolve when running this on your own, but in terms of operations this works. There are also complications that may arise when going between using and not using jax, but I tried to alleviate some of this by including flags indicating the option to use jax in some of the steps.

There are a few auxiliary files in here that help with specifics of this analysis, and should be changed for other analyses. For example, the paths of the GW and NICER GMM fits are specific to the specific events with the specific PE we chose for the paper. The injection set (csv file) is an altered version of the LVK's injection set, which I rejection sampled to the PE prior for the parts of parameter space we did not model in the population. The file `gp_mrgagn_macros.json` is a convenience file that stores some summary properties of each EoS (like TOV mass, R and Lambda at various masses, etc) which can be useful to call when plotting. This information is redundant with some of the information accessed via the branched interpolator, but is more convenient to access pre-computed.

The first issues you are likely to encounter may be from not having your local checkout of the `utils` subdirectory in your $PYTHONPATH. 
I recommend first pointing your $PYTHONPATH to the `utils` directory of this repository.