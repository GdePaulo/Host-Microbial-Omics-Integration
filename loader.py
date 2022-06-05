import pandas as pd
import re
import os

def loadGE():
    return pd.read_csv('Data/Aak_Study/tcga_scaled01_rnaseq.tsv', delimiter = "\t", index_col=0)

# Load aakash gene expression data with project
def loadGEWithClinical(includeStage=False):
    aak_ge_clinical = pd.read_csv('Data/Aak_Study/GE(Aakash)_Clinical_Data.tsv', delimiter = "\t", index_col=0)
    
    clinical_columns = ["acronym", "portion_id"]
    if includeStage:
        clinical_columns.append("stage")

    aak_ge_clinical_types = aak_ge_clinical[clinical_columns]
    aak_ge = pd.read_csv('Data/Aak_Study/tcga_scaled01_rnaseq.tsv', delimiter = "\t", index_col=0)

    # There are duplicates also in portion_id eg different measuring tech
    # aak_ge["patient_id"] = aak_ge.apply(lambda row: row.portion_id[:-4], axis=1)   
    aak_ge_clinical_types["portion_id"] = aak_ge_clinical_types.apply(lambda row: str(row.portion_id)[:-4], axis=1)   
    aak_ge_clinical_types.drop_duplicates(["portion_id"], inplace=True)
    # TCGA-AA-3495-01

    aak_ge_clinical_types.reset_index(drop=True, inplace=True)
    # IMPORTANT to prevent dupes
    aak_ge_clinical_types.set_index("portion_id", inplace=True)
    # print(aak_ge_clinical_types.columns)
    # aak_ge_clinical_types.drop(["sample_id"], axis=1, inplace=True)
    aak_ge_and_clinical = aak_ge.join(aak_ge_clinical_types, how="left")

    final = aak_ge_and_clinical.rename(columns={"acronym":"project"})
    return final

def loadTCMA(tcma_type):
    all_cancer_tcma = pd.read_csv(f'Data/TCMA/all_cancers_{tcma_type}.csv', index_col=0)
    return all_cancer_tcma

def loadGEOverlappingTCMA(tcma_type, includeStage=False):
    if includeStage:
        overlapping = pd.read_csv(f'Data/Integration/all_cancers_{tcma_type}_ge(aakash)_stage.csv', index_col=0)
    else:
        overlapping = pd.read_csv(f'Data/Integration/all_cancers_{tcma_type}_ge(aakash).csv', index_col=0)
    return overlapping

# Create data set overlapping TCMA data and GE (Aak)
def createGEOverlappingTCMA(tcma_type, includeStage=False):
    all_cancer_tcma = pd.read_csv(f'Data/TCMA/all_cancers_{tcma_type}.csv', index_col=0)
    all_cancer_tcma.rename(index= lambda s: s[:-1] if s[-1]=="A"else s, inplace=True)

    file = f"Data/Integration/all_cancers_{tcma_type}_ge(aakash)"

    if includeStage:
        ge = loadGEWithClinical(includeStage=True)
        ge = ge.drop(["project"], axis=1)
        file += "_stage"
    else:
        ge = loadGE()
    
    file += ".csv"

    tcma_ge = all_cancer_tcma.join(ge, how="inner")
    tcma_ge.index.name = "patient"

    tcma_ge.to_csv(file)

def attachTumorStatus(pd):
    pd["tumor"] = pd.apply(lambda row: 1 if re.search(r"[0][0-9][a-zA-Z]?$",row.name) else 0, axis=1)


def saveDescriptor(descriptor, file):
    directory = os.path.dirname(file)
    if not os.path.exists(directory):
            os.makedirs(directory)
    with open(file, 'w') as f:
        print(descriptor, file=f) 
    
if __name__ == "__main__":

    createGEOverlappingTCMA("phylum", includeStage=True)
