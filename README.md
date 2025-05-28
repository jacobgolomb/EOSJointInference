# EOSJointInference

## Data Preparation

The first step is to create GMMs for all the events which need this kind of likelihood evaluation. For example, we make GMM firs fot GW170817 and GW190425, which can be seen in the samples directory. I have some of these fits pre-saved for these events. I created these by running the `fit_GMM.py` code, and pointed to the bilby result files for the PE of these two events. Note, I think there was a bug in an older version of bilby that prevented the uniform source frame distance prior from being read in proeprly, so I have a line commented out which should fix this if necessary. This will save the GMM objects and some summary statistics.

We include custom PE for GW170817, which is fixed to the location of NGC4993 (sky location fixed, distance prior narrow based on the estimated distance to the galaxy from a sruvey data, need to double check which survey I used). The PE for GW190425 has two options: high spin and low spin, from the LVK data release.

The fits for the NICER observations are a little more complicated. We want the likelihood of the data given a mass and compactness. For the NCICER observations with radio mass observations, we want to weight the mass posterior include this radio information. In other words, we want the NICER likelihoods to effectively include radio and X-ray data for the masses. We can do this by attaching an extra weight to the NICER posterior that is proportional to the likelihood of the pulsar mass given the radio data. You can see exactly how this was done in `fit_NICER.py`.

The data from J0437 are from https://arxiv.org/abs/2407.06789. The posterior released in that work uses a mass prior of N(1.418, 0.044^2), which is the mass measurement from Reardon et al. 2024. Since we want to keep that mass information (i.e., the radio mass measurements) in the posterior and not divide it out in the hierarchical inference, we apply a dummy flat prior to this observation. So during the hierarchical inference, instead of dividing out the radio mass information, we divide out the flat prior.

Similar case for J0740. The data are from Miller et al. 2021, which release the posterior for mass and inverse compactness. They apply a mass prior of N(2.08, 0.09^2), which is slightly wider than the mass measurement in Fonseca et al 2021, which reports M ~ N(2.08, 0.07^2). We therefore use an effective prior on this analysis, which gets divided out in the hierarchical inference, of the ratio:

$\pi_{\textrm{J0740}}(m) = \frac{\mathcal{N}(\mu = 2.18, \sigma = 0.09)}{\mathcal{N}(\mu = 2.18, \sigma=0.07)}$.

So in the hierarchical inference step where we divide out the above prior, it divides out the prior applied in Miller et al. 2021 (numerator), and attaches an extra weight for the radio measurement (denominator). The prior in Miller et al. 2019 is uniform in inverse compactness. We assign the equivalent prior on compactness, which is a power law.

J0030 is much simpler. We include no independent radio mass measurement information, and just the posterior from Miller et al. 2019, which applied a prior already in terms of mass and compactness.

I will include the GMM fit files used in https://arxiv.org/abs/2410.14597 in this repo.

At the end of this process, we will have GMM fits for the GW likelihoods (effectively), which can be evaluated for points of [`chirp_mass_source`, `mass_ratio`, `lambda_1`, `lambda_2`]. The NICER fits can be evaluated for [`mass`, `compactness`], and effectively include the radio data for the mass likelihood as well.

The radio pulsar data come from the data release of Will and Katerina's paper. the file `pulsars_noNICER.h5` includes all of the pulsars in the original Alsing+/Farr+ dataset except for the pulsars with NICER observations, to avoid double counting.

## Infering the Mass Distribution under a Fiducial EOS Model

We first infer the mass distribution under a universal relation EoS model in order to capture the basic correlations between the mass-radius measurements in the population inference, as described in the paper. This is done with the `run_inference.py` script, which should be run for the GW events. The parser includes the necessary options and descriptions of these options. This step gives our proposal mass distribution for each dataset (almost correct, but does not include information from the GP EoSes) Some important ones to consider:

`--gmm-file-list` should point to a text file listing the paths to the GMM fits used in this analysis. For example, the text files for the GW GMMs are incldued in this repo. This tells the script where to load the single-event fits from.

`--selection-samples` is used for the GWs analysis. This should point to an injection set to estimate sensitivity ("pdet" term). See the `selection` function to see how these samples are reweighted. It assumes the `pdraw` is in terms of `m1` and `m2` (in the source frame). Note that I used a custom file that I did some rejection or importance sampling on so the parameters that are not being reweighted in the population model are distributed according to the PE prior.

`--eos` is the fiducial EoS to use in the analysis. If this is "ur" then it uses the ILoveQ universal relation, and assumes we sample in "m" and "b", the slope and intercept of the moment of inertia relation, as described in our paper.

Note that this script is written for the power law population model, which assumes we sample in the slope `alpha` and maximum mass `mpop`.
This step results in a posterior for the population hyperparameters `alpha` and `mpop`.

A similar step is done for the EM data, with `run_inference-EM.py`, which runs mass distribution inference independent of the GP EoSes, to generate the propsal distribution for $\eta_{\textrm{EM}}$ given all of the EM data. This assumes a truncated double Gaussian model, which has a variable upper mass truncation `mpop` (different `mpop` than above). This requires specifying the paths to both the NICER GMMs and the radio pulsars data. The radio pulsars data are fed into a custom likelihood object that, when initialized, marginalizes over the companion mass for each pulsar, and returns an object with a function that can be evaluated to return the likelihood of each pulsar given a mass. This is computed via reweighting; so that this function does not need to be called each time, it is called once for a bunch of proposal masses, and then everytime the sampler wants to calculate the likelihood under a new populaiton model, it just reweights these proposals.

The end of this stage is a posterior for $\eta_{\textrm{GW}}$ and $\eta_{\textrm{EM}}$ which we will use as a proposal in later steps.

## Drawing Proposal Single-Event Samples

Since the EoS gives different mass-radius or mass-lambda combinations, we need to draw proposal single-event samples for which we will evaluate the per-event likelihoods. We don't want to evaluate the single-event likelihoods and the $R(m)$ and $\Lambda(m)$ relationships for each mass distribution and each EoS. So the strategy is to draw a bunch of proposal masses and pre-compute the $R(m)$ and $\Lamdba(m)$ values for each EoS for the proposal masses.

The script `draw_pop_samples.py` (which I now see is poorly named) is used to draw these fiducial samples for masses and radius/lambda, targeted around the support of the GW observations. I think I named it pop_samples because it is proposal single-event samples from a fiducial (targeted) population. Basically, because we are going to be reweighting masses to a population model, we want to make sure we have enough effective samples, so we target the samples around where the chirp mass/mass ratio support lies for the events. This step results in a dataframe with combinations of source frame component masses and lambdas for each EoS specified by `--eos`. In our paper, we uses 10,000 EoSes from the GP and 750 mass samples for each observation. Note that this also saves a pdraw for the mass and the appropriate jacobian to transform the draw probability between chirp mass/mass ratio (PE prior) to component masses (population prior). The number of samples per observation can be set by `--n-samples-per-event`.

`draw_pop_samples-NICER.py` does a similar thing for the NICER observations, except the mass-compactnesses targeted around the support of the NICER observations.

Again, the output of the `draw_pop_samples` step is some set of (mass, lambda or compactness) coordinates targeted around the support of the observations. To be specific, "targeted" means drawn from a Gaussian around the estimated masses for each observation. For GWs, we draw from a Gaussian around the source frame chirp mass chirp mass and a power law mamss ratio. For NICER, we draw from a Gaussian around the estimated mass. The widths of the Gaussians are estimated from the samples themselves, but are scaled by some factor `scale-chirp` to make the proposal support broader. The `pdraws` are saved so they can be divided out later.

## Calculating Single-Event Likelihoods

Now that we have samples of mass and lambda or compactness for each GW or NICER observation, we will calculate the likelihood of the data given those points. To do this, we use the GMM fits from the first step to estimate $\mathcal{L}(d_i|m_1, m_2,\Lambda_1, \Lambda_2)$ for the $i$th GW obsveration and $\mathcal{L}(d_i|m, C)$ for the $i$th NICER observation. You could also call this $\mathcal{L}(d_I|m,\epsilon)$ since the EoS $\epsilon$ specifies the other observable, given $m$. This uses the method described in Golomb an Talbot 2021 to evaluate the likelihoods, effectively dividing out the priors discussed in the "Data" step.

The scripts `calculate_likelihood.py` and `calculate_likelihoods-NICER.py` do this step, and save a bunch of pickle files containing the calculated likelihoods. Each pickle file contains some number of EOSes (e.g., we use 500 per file), and for each eos, we calculate $\mathcal{L}(d_i|m, \epsilon)$. These are saved as a dictionary where they key is the EoS name (as specified in the `draw_pop_samples` step). So assuming we told the script to save 500 EoSes per file, each file would contain a dictionary with 500 elements, where each element has an array of shape (N_observations, N_mass_samples).

## Calculating the Population Likelihoods

The next step is to calculate (the log of) $\mathcal{L}(d_{\textrm{NICER}}|\eta)$ and $\mathcal{L}(d_{\textrm{GW}}|\eta)$. For this step, we run `weight_population` (for the GWs) and `weight_population-NICER` (for the NICER observations). This requires specifiying a `--proposal-result`, which is a set os samples for the population hyperparameters (i.e., from the mass distribution inference). For each population, we calculate the weight for the proposal masses, meaning the ratio between the populaiton model evaluated at the proposal masses and the pdraw of the proposal masses. This is explained more technically in the paper. Recall the proposal masses are the mass coordinates where we evaluated the likelihood. We then marginalize over the masses, so for each EoS, we get an array of $\mathcal{L}(\{d\}_{\textrm{GW/NICER}}|\eta)$, the likelihood of the NICER or GW dataset, given the EoS and the array of mass hyperparameters. This saves a dictionary as a pickle file, which has keys corresponding to the EoS, and each element of the dictionary is an array of length of the number of $\eta$ samples (the number of population hyperposterior samples). This saves a samples_with_weights pickle file (for each data set it is run on).

A similar operation is done for the radio pulsars (but a little different because this dataset only knows about the mass distribution at this point, not any EoS information). This is done with the `reweight-PSR.py` script, which saves its own samples_with_weights file for the radio pulsars.

## Calculating the New Weights for the Population Hyperparameters

Now we have a set of proposal population hyperparameters (from the original mass distribution + UR inference), as well as likelihoods for each of those population models under the combined mass + GP EoSes. Since the porposals were obtained under the mass distribution + UR model, but we want the probability for just the mass distribution hyperparameters $\eta$, we estimate this probability by making a KDE of the $\eta$ samples. This is done with the `get_conditioned_weights.py` script, which has an option `--kde`. Once again, this is done separately for the datasets (GWs, radio pulsars, NICER). Note that the "conditioned" refers to this step enforcing MTOV > Mpop.

Finally, running `combine_weights.py` will combine the weights into one file, where the reweighted GW mass distribution, EM mass distribution, and EoSes inform each other. Specifically, this will save the log likelihood for each combination of mass distribution and EoS. See the `plot.py` script to see how to visualize the results after.