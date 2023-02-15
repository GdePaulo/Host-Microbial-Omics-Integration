import pandas as pd
import processor as pr
import loader as load
import predictor as pred
import config
import time
import sys
import os
import warnings

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

# # With only linreg and 2 iterationsAnd 4 cores
# ge: 3:19 915 mb
# gen: 1:05 410mb
# with ae with 30 features same settings it's 15 seconds
def main():
    stad_stage_exp = True
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
        else:
            print("Running for all layers")
            data, files = load.loadAll(includeStage=(target=="stage"), sameSamples=True, skipGenes=False)

        for sampling in config.sampling[:]:
            for selection in config.selection_types[:1]:
                # pred.runExperiments(data[1:2], files[1:2], target=target, sampling=sampling, selection=selection)
                
                for parity in config.modality_parities[:1]:
                    enforce_modality_parity = (parity == "parity")

                    # Only run for overlap set if you run with parity
                    if enforce_modality_parity:
                        genus_overlapping_ge_name = "tcma_gen_aak_ge"
                        genus_overlapping_ge  = load.getSpecificData(genus_overlapping_ge_name, data, files)
                        pred.runExperiments([genus_overlapping_ge], [genus_overlapping_ge_name], target=target, sampling=sampling, selection=selection, modality_selection_parity=enforce_modality_parity, stad_exp=stad_stage_exp)
                    else:
                        pred.runExperiments(data[:], files[:], target=target, sampling=sampling, selection=selection, modality_selection_parity=enforce_modality_parity, stad_exp=stad_stage_exp)
                    
if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print("--- %.2f seconds ---" % elapsed_time)