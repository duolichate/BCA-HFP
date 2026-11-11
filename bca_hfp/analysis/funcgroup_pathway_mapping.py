# bca_hfp/analysis/funcgroup_pathway_mapping.py
# Contains functional group mapping, aggregation functions, and pathway abbreviation dictionary

import numpy as np
from rdkit import Chem

# Functional group SMARTS list (47 entries)
FG_SMARTS = [
    '*N#N', '*P', '*N=N', '*N', '*N(O)=O', '*N=C', '*I', '*N=C=O', '*O', '*S',
    '*NC(C)=O', '*#N', '*SC', '*C(=O)OC', '*[Si]', '*P=O', '*C(C)=O', '*F',
    '*C(N)=O', '*S(=O)(=O)O', '*S(C)(=O)=O', '*N=O', '*=NC', '*C#C', '*OC',
    '*C(F)(F)F', '*S(=O)(=O)Cl', '*N=C=S', '*N=NC', '*=N', '[Na+]', '*C=C', '*Cl',
    '*C1CC1', '*[Se]', '*S(C)=O', '*C(O)=O', '*Br', '*=O', '*C(C)(C)C', '*NS(C)(=O)=O',
    '*OCC', '*S(=O)(=O)OC', '*=NO', '*=S', '*C#N', '*S(N)(=O)=O'
]

def get_fg_mapping(smiles):
    """Return mapping from functional group name (SMARTS) to atom index list in molecule"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}
    mapping = {}
    for smarts in FG_SMARTS:
        patt = Chem.MolFromSmarts(smarts)
        if patt is None:
            continue
        matches = mol.GetSubstructMatches(patt)
        if matches:
            atoms = set()
            for match in matches:
                atoms.update(match)
            mapping[smarts] = sorted(atoms)
    return mapping

def merge_fg_mapping(fg_mapping):
    """Merge functional groups with identical atom sets into a single representative"""
    atom_set_to_fgs = {}
    for fg_name, indices in fg_mapping.items():
        key = tuple(sorted(indices))
        if key not in atom_set_to_fgs:
            atom_set_to_fgs[key] = []
        atom_set_to_fgs[key].append(fg_name)
    merged_mapping = {}
    for key, fgs in atom_set_to_fgs.items():
        # Prefer '*O' as representative, then '*OC', otherwise take the first
        if '*O' in fgs:
            rep = '*O'
        elif '*OC' in fgs:
            rep = '*OC'
        else:
            rep = fgs[0]
        merged_mapping[rep] = list(key)
    return merged_mapping

def aggregate_to_fg(atom_weights, fg_mapping, num_atoms):
    """Aggregate atom weights to functional group level"""
    fg_weights = {}
    for fg_name, indices in fg_mapping.items():
        valid = [i for i in indices if i < num_atoms]
        if valid:
            fg_weights[fg_name] = np.mean(atom_weights[valid], axis=0)
        else:
            fg_weights[fg_name] = np.zeros(atom_weights.shape[1])
    return fg_weights

# Pathway name simplification dictionary
PATHWAY_SIMPLE_NAMES = {
    # Metabolism
    "KEGG_N_GLYCAN_BIOSYNTHESIS": "N-Glycan",
    "KEGG_OTHER_GLYCAN_DEGRADATION": "Other Glycan Deg",
    "KEGG_O_GLYCAN_BIOSYNTHESIS": "O-Glycan",
    "KEGG_GLYCOSAMINOGLYCAN_DEGRADATION": "GAG Deg",
    "KEGG_GLYCOSAMINOGLYCAN_BIOSYNTHESIS_KERATAN_SULFATE": "Keratan Sulfate GAG",
    "KEGG_GLYCEROLIPID_METABOLISM": "Glycerolipid Met",
    "KEGG_GLYCOSYLPHOSPHATIDYLINOSITOL_GPI_ANCHOR_BIOSYNTHESIS": "GPI Anchor",
    "KEGG_GLYCEROPHOSPHOLIPID_METABOLISM": "Glycerophospholipid Met",
    "KEGG_ETHER_LIPID_METABOLISM": "Ether Lipid Met",
    "KEGG_ARACHIDONIC_ACID_METABOLISM": "Arachidonic Acid Met",
    "KEGG_LINOLEIC_ACID_METABOLISM": "Linoleic Acid Met",
    "KEGG_ALPHA_LINOLENIC_ACID_METABOLISM": "a-Linolenic Acid Met",
    "KEGG_SPHINGOLIPID_METABOLISM": "Sphingolipid Met",
    "KEGG_GLYCOSPHINGOLIPID_BIOSYNTHESIS_LACTO_AND_NEOLACTO_SERIES": "Lacto/Neolacto",
    "KEGG_GLYCOSPHINGOLIPID_BIOSYNTHESIS_GLOBO_SERIES": "Globo Series",
    "KEGG_GLYCOSPHINGOLIPID_BIOSYNTHESIS_GANGLIO_SERIES": "Ganglio Series",
    "KEGG_RIBOFLAVIN_METABOLISM": "Riboflavin Met",
    "KEGG_NICOTINATE_AND_NICOTINAMIDE_METABOLISM": "Nicotinate Met",
    "KEGG_PANTOTHENATE_AND_COA_BIOSYNTHESIS": "Pantothenate/CoA",
    "KEGG_GLYCOLYSIS_GLUCONEOGENESIS": "Glycolysis/Gluconeo",
    "KEGG_CITRATE_CYCLE_TCA_CYCLE": "TCA Cycle",
    "KEGG_PENTOSE_PHOSPHATE_PATHWAY": "PPP",
    "KEGG_PENTOSE_AND_GLUCURONATE_INTERCONVERSIONS": "Pentose/Glucuronate",
    "KEGG_FRUCTOSE_AND_MANNOSE_METABOLISM": "Fructose/Mannose Met",
    "KEGG_GALACTOSE_METABOLISM": "Galactose Met",
    "KEGG_ASCORBATE_AND_ALDARATE_METABOLISM": "Ascorbate Met",
    "KEGG_FATTY_ACID_METABOLISM": "Fatty Acid Met",
    "KEGG_STEROID_BIOSYNTHESIS": "Steroid Bio",
    "KEGG_PRIMARY_BILE_ACID_BIOSYNTHESIS": "Primary Bile Acid",
    "KEGG_STEROID_HORMONE_BIOSYNTHESIS": "Steroid Hormone",
    "KEGG_OXIDATIVE_PHOSPHORYLATION": "OxPhos",
    "KEGG_PURINE_METABOLISM": "Purine Met",
    "KEGG_PYRIMIDINE_METABOLISM": "Pyrimidine Met",
    "KEGG_ALANINE_ASPARTATE_AND_GLUTAMATE_METABOLISM": "Ala/Asp/Glu Met",
    "KEGG_GLYCINE_SERINE_AND_THREONINE_METABOLISM": "Gly/Ser/Thr Met",
    "KEGG_CYSTEINE_AND_METHIONINE_METABOLISM": "Cys/Met Met",
    "KEGG_VALINE_LEUCINE_AND_ISOLEUCINE_DEGRADATION": "Val/Leu/Ile Deg",
    "KEGG_VALINE_LEUCINE_AND_ISOLEUCINE_BIOSYNTHESIS": "Val/Leu/Ile Bio",
    "KEGG_LYSINE_DEGRADATION": "Lysine Deg",
    "KEGG_ARGININE_AND_PROLINE_METABOLISM": "Arg/Pro Met",
    "KEGG_HISTIDINE_METABOLISM": "Histidine Met",
    "KEGG_TYROSINE_METABOLISM": "Tyrosine Met",
    "KEGG_PHENYLALANINE_METABOLISM": "Phenylalanine Met",
    "KEGG_TRYPTOPHAN_METABOLISM": "Tryptophan Met",
    "KEGG_BETA_ALANINE_METABOLISM": "b-Alanine Met",
    "KEGG_TAURINE_AND_HYPOTAURINE_METABOLISM": "Taurine Met",
    "KEGG_SELENOAMINO_ACID_METABOLISM": "Selenoamino Acid Met",
    "KEGG_GLUTATHIONE_METABOLISM": "Glutathione Met",
    "KEGG_STARCH_AND_SUCROSE_METABOLISM": "Starch/Sucrose Met",
    "KEGG_AMINO_SUGAR_AND_NUCLEOTIDE_SUGAR_METABOLISM": "Amino Sugar Met",
    "KEGG_GLYCOSAMINOGLYCAN_BIOSYNTHESIS_CHONDROITIN_SULFATE": "Chondroitin Sulfate",
    "KEGG_GLYCOSAMINOGLYCAN_BIOSYNTHESIS_HEPARAN_SULFATE": "Heparan Sulfate",
    "KEGG_INOSITOL_PHOSPHATE_METABOLISM": "Inositol Phosphate",
    "KEGG_PYRUVATE_METABOLISM": "Pyruvate Met",
    "KEGG_GLYOXYLATE_AND_DICARBOXYLATE_METABOLISM": "Glyoxylate Met",
    "KEGG_PROPANOATE_METABOLISM": "Propanoate Met",
    "KEGG_BUTANOATE_METABOLISM": "Butanoate Met",
    "KEGG_ONE_CARBON_POOL_BY_FOLATE": "One-Carbon Pool",
    "KEGG_FOLATE_BIOSYNTHESIS": "Folate Bio",
    "KEGG_RETINOL_METABOLISM": "Retinol Met",
    "KEGG_PORPHYRIN_AND_CHLOROPHYLL_METABOLISM": "Porphyrin Met",
    "KEGG_TERPENOID_BACKBONE_BIOSYNTHESIS": "Terpenoid Backbone",
    "KEGG_LIMONENE_AND_PINENE_DEGRADATION": "Limonene/Pinene Deg",
    "KEGG_NITROGEN_METABOLISM": "Nitrogen Met",
    "KEGG_SULFUR_METABOLISM": "Sulfur Met",
    "KEGG_METABOLISM_OF_XENOBIOTICS_BY_CYTOCHROME_P450": "Xenobiotics P450",
    "KEGG_DRUG_METABOLISM_CYTOCHROME_P450": "Drug Met P450",
    "KEGG_DRUG_METABOLISM_OTHER_ENZYMES": "Drug Met Other Enzymes",
    "KEGG_BIOSYNTHESIS_OF_UNSATURATED_FATTY_ACIDS": "Unsaturated FA Bio",

    # Genetic information processing
    "KEGG_AMINOACYL_TRNA_BIOSYNTHESIS": "Aminoacyl-tRNA",
    "KEGG_BASAL_TRANSCRIPTION_FACTORS": "Basal Transcription",
    "KEGG_RNA_DEGRADATION": "RNA Degradation",
    "KEGG_RNA_POLYMERASE": "RNA Polymerase",
    "KEGG_SPLICEOSOME": "Spliceosome",
    "KEGG_RIBOSOME": "Ribosome",
    "KEGG_PROTEASOME": "Proteasome",
    "KEGG_PROTEIN_EXPORT": "Protein Export",

    # DNA repair / replication
    "KEGG_BASE_EXCISION_REPAIR": "BER",
    "KEGG_NUCLEOTIDE_EXCISION_REPAIR": "NER",
    "KEGG_MISMATCH_REPAIR": "MMR",
    "KEGG_HOMOLOGOUS_RECOMBINATION": "HR",
    "KEGG_NON_HOMOLOGOUS_END_JOINING": "NHEJ",
    "KEGG_DNA_REPLICATION": "DNA Replication",

    # Signaling pathways
    "KEGG_MAPK_SIGNALING_PATHWAY": "MAPK Signaling",
    "KEGG_ERBB_SIGNALING_PATHWAY": "ErbB Signaling",
    "KEGG_WNT_SIGNALING_PATHWAY": "Wnt Signaling",
    "KEGG_NOTCH_SIGNALING_PATHWAY": "Notch Signaling",
    "KEGG_TGF_BETA_SIGNALING_PATHWAY": "TGF-b Signaling",
    "KEGG_MTOR_SIGNALING_PATHWAY": "mTOR Signaling",
    "KEGG_JAK_STAT_SIGNALING_PATHWAY": "JAK-STAT Signaling",
    "KEGG_PPAR_SIGNALING_PATHWAY": "PPAR Signaling",
    "KEGG_VEGF_SIGNALING_PATHWAY": "VEGF Signaling",
    "KEGG_CALCIUM_SIGNALING_PATHWAY": "Calcium Signaling",
    "KEGG_PHOSPHATIDYLINOSITOL_SIGNALING_SYSTEM": "PI3K Signaling",
    "KEGG_NEUROACTIVE_LIGAND_RECEPTOR_INTERACTION": "Neuroactive Ligand",
    "KEGG_CHEMOKINE_SIGNALING_PATHWAY": "Chemokine Signaling",
    "KEGG_CYTOKINE_CYTOKINE_RECEPTOR_INTERACTION": "Cytokine-CytokineR",
    "KEGG_T_CELL_RECEPTOR_SIGNALING_PATHWAY": "TCR Signaling",
    "KEGG_B_CELL_RECEPTOR_SIGNALING_PATHWAY": "BCR Signaling",
    "KEGG_TOLL_LIKE_RECEPTOR_SIGNALING_PATHWAY": "TLR Signaling",
    "KEGG_NOD_LIKE_RECEPTOR_SIGNALING_PATHWAY": "NLR Signaling",
    "KEGG_RIG_I_LIKE_RECEPTOR_SIGNALING_PATHWAY": "RLR Signaling",
    "KEGG_CYTOSOLIC_DNA_SENSING_PATHWAY": "Cytosolic DNA Sensing",
    "KEGG_HEDGEHOG_SIGNALING_PATHWAY": "Hedgehog Signaling",
    "KEGG_AXON_GUIDANCE": "Axon Guidance",
    "KEGG_FOCAL_ADHESION": "Focal Adhesion",
    "KEGG_ADHERENS_JUNCTION": "Adherens Junction",
    "KEGG_TIGHT_JUNCTION": "Tight Junction",
    "KEGG_GAP_JUNCTION": "Gap Junction",
    "KEGG_ECM_RECEPTOR_INTERACTION": "ECM-Receptor",
    "KEGG_REGULATION_OF_ACTIN_CYTOSKELETON": "Actin Cytoskeleton",
    "KEGG_INSULIN_SIGNALING_PATHWAY": "Insulin Signaling",
    "KEGG_GNRH_SIGNALING_PATHWAY": "GnRH Signaling",
    "KEGG_PROGESTERONE_MEDIATED_OOCYTE_MATURATION": "Progesterone Oocyte",
    "KEGG_MELANOGENESIS": "Melanogenesis",
    "KEGG_ADIPOCYTOKINE_SIGNALING_PATHWAY": "Adipocytokine",
    "KEGG_ALDOSTERONE_REGULATED_SODIUM_REABSORPTION": "Aldosterone Na+",
    "KEGG_VASOPRESSIN_REGULATED_WATER_REABSORPTION": "Vasopressin H2O",

    # Cellular processes
    "KEGG_CELL_CYCLE": "Cell Cycle",
    "KEGG_OOCYTE_MEIOSIS": "Oocyte Meiosis",
    "KEGG_P53_SIGNALING_PATHWAY": "p53 Signaling",
    "KEGG_APOPTOSIS": "Apoptosis",
    "KEGG_REGULATION_OF_AUTOPHAGY": "Autophagy",
    "KEGG_UBIQUITIN_MEDIATED_PROTEOLYSIS": "Ubiquitin Proteolysis",
    "KEGG_ENDOCYTOSIS": "Endocytosis",
    "KEGG_PEROXISOME": "Peroxisome",
    "KEGG_LYSOSOME": "Lysosome",
    "KEGG_SNARE_INTERACTIONS_IN_VESICULAR_TRANSPORT": "SNARE Interactions",
    "KEGG_ABC_TRANSPORTERS": "ABC Transporters",

    # Diseases
    "KEGG_PATHWAYS_IN_CANCER": "Pathways in Cancer",
    "KEGG_COLORECTAL_CANCER": "CRC",
    "KEGG_RENAL_CELL_CARCINOMA": "RCC",
    "KEGG_PANCREATIC_CANCER": "Pancreatic Cancer",
    "KEGG_ENDOMETRIAL_CANCER": "Endometrial Cancer",
    "KEGG_GLIOMA": "Glioma",
    "KEGG_PROSTATE_CANCER": "Prostate Cancer",
    "KEGG_THYROID_CANCER": "Thyroid Cancer",
    "KEGG_BASAL_CELL_CARCINOMA": "BCC",
    "KEGG_MELANOMA": "Melanoma",
    "KEGG_BLADDER_CANCER": "Bladder Cancer",
    "KEGG_CHRONIC_MYELOID_LEUKEMIA": "CML",
    "KEGG_ACUTE_MYELOID_LEUKEMIA": "AML",
    "KEGG_SMALL_CELL_LUNG_CANCER": "SCLC",
    "KEGG_NON_SMALL_CELL_LUNG_CANCER": "NSCLC",
    "KEGG_SYSTEMIC_LUPUS_ERYTHEMATOSUS": "SLE",
    "KEGG_PRIMARY_IMMUNODEFICIENCY": "Primary Immunodeficiency",
    "KEGG_ASTHMA": "Asthma",
    "KEGG_AUTOIMMUNE_THYROID_DISEASE": "Autoimmune Thyroid",
    "KEGG_ALLOGRAFT_REJECTION": "Allograft Rejection",
    "KEGG_GRAFT_VERSUS_HOST_DISEASE": "GVHD",
    "KEGG_VIRAL_MYOCARDITIS": "Viral Myocarditis",
    "KEGG_TYPE_I_DIABETES_MELLITUS": "T1DM",
    "KEGG_TYPE_II_DIABETES_MELLITUS": "T2DM",
    "KEGG_MATURITY_ONSET_DIABETES_OF_THE_YOUNG": "MODY",
    "KEGG_ALZHEIMERS_DISEASE": "AD",
    "KEGG_PARKINSONS_DISEASE": "PD",
    "KEGG_AMYOTROPHIC_LATERAL_SCLEROSIS_ALS": "ALS",
    "KEGG_HUNTINGTONS_DISEASE": "HD",
    "KEGG_PRION_DISEASES": "Prion Diseases",

    # Infection / Immunity
    "KEGG_PATHOGENIC_ESCHERICHIA_COLI_INFECTION": "E. coli Infection",
    "KEGG_VIBRIO_CHOLERAE_INFECTION": "Cholera Infection",
    "KEGG_EPITHELIAL_CELL_SIGNALING_IN_HELICOBACTER_PYLORI_INFECTION": "H. pylori Signaling",
    "KEGG_LEISHMANIA_INFECTION": "Leishmania Infection",
    "KEGG_NATURAL_KILLER_CELL_MEDIATED_CYTOTOXICITY": "NK Cell Cytotoxicity",
    "KEGG_FC_EPSILON_RI_SIGNALING_PATHWAY": "FceRI Signaling",
    "KEGG_FC_GAMMA_R_MEDIATED_PHAGOCYTOSIS": "FcgR Phagocytosis",
    "KEGG_LEUKOCYTE_TRANSENDOTHELIAL_MIGRATION": "Leukocyte Migration",
    "KEGG_INTESTINAL_IMMUNE_NETWORK_FOR_IGA_PRODUCTION": "IgA Immune Network",

    # Others
    "KEGG_CARDIAC_MUSCLE_CONTRACTION": "Cardiac Muscle",
    "KEGG_VASCULAR_SMOOTH_MUSCLE_CONTRACTION": "Vascular Smooth Muscle",
    "KEGG_RENIN_ANGIOTENSIN_SYSTEM": "Renin-Angiotensin",
    "KEGG_CIRCADIAN_RHYTHM_MAMMAL": "Circadian Rhythm",
    "KEGG_TASTE_TRANSDUCTION": "Taste Transduction",
    "KEGG_PROXIMAL_TUBULE_BICARBONATE_RECLAMATION": "Bicarbonate Reclamation",
    "KEGG_HYPERTROPHIC_CARDIOMYOPATHY_HCM": "HCM",
    "KEGG_ARRHYTHMOGENIC_RIGHT_VENTRICULAR_CARDIOMYOPATHY_ARVC": "ARVC",
    "KEGG_DILATED_CARDIOMYOPATHY": "Dilated Cardiomyopathy",
    "KEGG_LONG_TERM_POTENTIATION": "LTP",
    "KEGG_LONG_TERM_DEPRESSION": "LTD",
    "KEGG_NEUROTROPHIN_SIGNALING_PATHWAY": "Neurotrophin",
    "KEGG_OLFACTORY_TRANSDUCTION": "Olfactory Transduction",
    "KEGG_DORSO_VENTRAL_AXIS_FORMATION": "Dorso-Ventral Axis",
    "KEGG_COMPLEMENT_AND_COAGULATION_CASCADES": "Complement/Coagulation",
    "KEGG_ANTIGEN_PROCESSING_AND_PRESENTATION": "Antigen Presentation",
    "KEGG_CELL_ADHESION_MOLECULES_CAMS": "Cell Adhesion Molecules"
}
