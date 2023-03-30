import pandas as pd
import processor as pr
import loader as load
import predictor as pred
import config
import time
import sys
import os
import warnings


# CHECKLIST: * Reset iterations * Increase batch timeout * Reset n_jobs * Upload integration files (refresh) * Change pred model
# * Update AE integrated feature number (restart kernel) * Check stad_exp=False * Check cancers in pipeline
# Ignore warnings https://stackoverflow.com/questions/53784971/how-to-disable-convergencewarning-using-sklearn#comment111709503_55595680
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    # os.environ["PYTHONWARNINGS"] = ('ignore::UserWarning, ignore::ConvergenceWarning, ignore::RuntimeWarning') # Also affect subprocesses
    os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses

# STAGE/STAD: 7p * 2sel * 4mod * iterations = 56 * iterations
# With only linreg and 2 iterations,It takes 2 minutes and 50 seconds with layer parallelization
# 7 minutes and 12 seconds when specifying -N1
# it's not paralyzing possibly because the first run is using all the memory
# with everything it takes about 2 hours 32 course
# 8/16/16 200 iter is about  30 mins / 1h / 1h

# # With only linreg and 2 iterationsAnd 4 cores
# ge: 3:19 915 mb
# gen: 1:05 410mb
# with ae with 30 features same settings it's 15 seconds

# Linreg, 2 iter, 4/8/8 cores, RF, RandomizedSearch
# gen/ge/genge = 3:56/7:56/7:30/
# 15 iter
# gen/ge/genge = 25:14/53:10/53:38

# 2 iter, changed config
# genge = 10:20

# 100 iter, 4/16/16
# gen/ge/genge = >6/9/9

# 200 iter, 8/16/16
# gen/ge/genge = 4h:45m/14h:40m


# TUMOR/ALL CANCERS
# SVM, chi2, hyperp tuning, 2 iterations and 32/_/64 cores and 3000/_/6000 gb memory
# gen/ge/genge(+parity) = 1:30/_/2:05

# STAGE/STAD
# elasticnet, lasso, hyperp tuning, 2 iterations and 8/_/16 cores and 2000/_/2000 gb memory
# gen/ge/genge(+parity) = 0:47/_/2:00
# elasticnet, enet, hyperp tuning, 2 iterations and 8/_/16 cores and 2000/_/2000 gb memory
# gen/ge/genge(+parity) = 0:35/_/1:41
def main():
    stad_stage_exp = True

    prediction_models = {
        "tumor": "SVC",
        # "stage": "RandomForestRegressor"
        "stage": "ElasticNet"
    }

    # aak_ge takes a while. chokes during feature selection COAD even with 5
    for target in config.prediction_targets[1:]:

        if len(sys.argv) > 1:
            chosen_layer = sys.argv[1]
            print("Running for layer", chosen_layer)
            if chosen_layer != "aak_ge":
                data, files = load.loadAll(includeStage=(target=="stage"), sameSamples=True, skipGenes=True)
            else:
                data, files = load.loadAll(includeStage=(target=="stage"), sameSamples=True, skipGenes=False)
            specific_data = load.getSpecificData(chosen_layer, data, files)
            data = [specific_data]
            files = [chosen_layer]
            # print("SPec data,", specific_data)
        else:
            print("Running for all layers")
            data, files = load.loadAll(includeStage=(target=="stage"), sameSamples=True, skipGenes=False)

        print("Using model", prediction_models[target], "for", target)
        for sampling in config.sampling[:]:
            for selection in ["rfreg", "elasticnet", "pearson"][:1]:#config.selection_types[-2:-1]: #["linreg", "chi2", "elasticnet", "lasso", "anova", "pearson"]
                # pred.runExperiments(data[1:2], files[1:2], target=target, sampling=sampling, selection=selection)
                
                for parity in config.modality_parities[:]:
                    enforce_modality_parity = (parity == "parity")

                    # Only run for overlap set if you run with parity
                    if enforce_modality_parity:
                        if len(sys.argv) > 1 and sys.argv[1] != "tcma_gen_aak_ge":
                            print(f"Skipping modality parity enforcement because layer {sys.argv[1]} is invalid")
                            continue
                        genus_overlapping_ge_name = "tcma_gen_aak_ge"
                        genus_overlapping_ge  = load.getSpecificData(genus_overlapping_ge_name, data, files)
                        pred.runExperiments([genus_overlapping_ge], [genus_overlapping_ge_name], target=target, sampling=sampling, selection=selection, modality_selection_parity=enforce_modality_parity, stad_exp=stad_stage_exp, selected_model=prediction_models[target])
                    else:
                        pred.runExperiments(data[:], files[:], target=target, sampling=sampling, selection=selection, modality_selection_parity=enforce_modality_parity, stad_exp=stad_stage_exp, selected_model=prediction_models[target])
                    
if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print("--- %.2f seconds ---" % elapsed_time)