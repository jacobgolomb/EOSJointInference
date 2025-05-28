python ../utils/weight_population.py --log-likelihood-files "likelihoods_GW/*" --proposal-result ./outdir_GW/None_output_result.json --selection-samples newprior_lvk_bns_injection_o1o2o3.csv --gw-gmm-files GW_fits.txt --output GW_samples_with_weights.pkl --samples-from-population mass_lambdas.h5

python ../utils/weight_population-NICER.py --log-likelihood-files "likelihoods_NICER/*" --proposal-result ./outdir_EM/NICER_radio_UR_result.json --gmm-files NICER_fits.txt --output NICER_samples_with_weights.pkl --samples-from-population mass_compactness.h5

python ../utils/reweight-PSR.py --radio-data ../data/Alsing_pulsars/pulsars_noNICER.h5 --samples-from-population mass_compactness.h5 --proposal-result ./outdir_EM/NICER_radio_UR_result.json 