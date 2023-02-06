import pandas as pd
import processor as pr
import loader as load
import predictor as pred
import config
import time

def main():
    stad_stage_exp = True
    # aak_ge takes a while. chokes during feature selection COAD even with 5
    for target in config.prediction_targets[1:]:
        data, files = load.loadAll(includeStage=(target=="stage"), sameSamples=True, skipGenes=False)
        for sampling in config.sampling[:]:
            for selection in config.selection_types[:2]:
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
    start_time = time.process_time()
    main()
    elapsed_time = time.process_time() - start_time
    print("--- %.2f seconds ---" % elapsed_time)