import pandas as pd
import re


def loadGE():
    return pd.read_csv('Data/Aak_Study/tcga_scaled01_rnaseq.tsv', delimiter = "\t", index_col=0)

# Load aakash gene expression data with project
def loadGEWithClinical():
    aak_ge_clinical = pd.read_csv('Data/Aak_Study/GE(Aakash)_Clinical_Data.tsv', delimiter = "\t", index_col=0)
    aak_ge_clinical_types = aak_ge_clinical[["acronym"]]

    aak_ge = pd.read_csv('Data/Aak_Study/tcga_scaled01_rnaseq.tsv', delimiter = "\t", index_col=0)

    aak_ge["patient_id"] = aak_ge.apply(lambda row: row.name[:-3], axis=1)   
    aak_ge_and_clinical = aak_ge.join(aak_ge_clinical_types, on=["patient_id"], how="inner")

    final = aak_ge_and_clinical.drop(["patient_id"], axis=1).rename(columns={"acronym":"project"})
    return final

def loadTCMA(tcma_type):
    all_cancer_tcma = pd.read_csv(f'Data/TCMA/all_cancers_{tcma_type}.csv', index_col=0)
    return all_cancer_tcma

def loadGEOverlappingTCMA(tcma_type):
    overlapping = pd.read_csv(f'Data/Integration/all_cancers_{tcma_type}_ge(aakash).csv', index_col=0)
    return overlapping

# Create data set overlapping TCMA data and GE (Aak)
def createGEOverlappingTCMA(tcma_type):
    all_cancer_tcma = pd.read_csv(f'Data/TCMA/all_cancers_{tcma_type}.csv', index_col=0)
    all_cancer_tcma.rename(index= lambda s: s[:-1] if s[-1]=="A"else s, inplace=True)

    ge = loadGE()

    tcma_ge = all_cancer_tcma.join(ge, how="inner")
    tcma_ge.index.name = "patient"
    tcma_ge.to_csv(f"Data/Integration/all_cancers_{tcma_type}_ge(aakash).csv")

def attachTumorStatus(pd):
    pd["tumor"] = pd.apply(lambda row: 1 if re.search(r"[0][0-9][a-zA-Z]?$",row.name) else 0, axis=1)
    
if __name__ == "__main__":

    createGEOverlappingTCMA("phylum")
