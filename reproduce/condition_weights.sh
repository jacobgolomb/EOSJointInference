python ../utils/get_conditioned_weights.py --weighted-samples GW_samples_with_weights.pkl --samples-from-population mass_lambdas.h5 --output-file GW_conditioned_weights.pkl --kde
python ../utils/get_conditioned_weights.py --weighted-samples NICER_samples_with_weights.pkl --samples-from-population mass_compactness.h5 --output-file NICER_conditioned_weights.pkl --kde
python ../utils/get_conditioned_weights.py --weighted-samples radio_pulsar_samples_with_weights.pkl --samples-from-population mass_compactness.h5 --output-file PSR_conditioned_weights.pkl --kde
