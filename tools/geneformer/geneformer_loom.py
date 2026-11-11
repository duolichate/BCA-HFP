import pandas as pd
import numpy as np
import loompy

import pandas as pd
import numpy as np
import loompy

def csv_to_raw_loom(csv_path, loom_path, cell_id_col='GDSC_ID', mName = None, nName = None):
    """
    convert a CSV with Ensembl IDs to a loom file holding the raw expression values
    all metadata columns go into col_attrs
    """


    df = pd.read_csv(csv_path)

    # metadata columns: adjust to the actual column names in the CSV
    meta_cols = ['GDSC_ID', 'CCLE_ID', 'CELL_LINES', 'CELL_LINES_STD', 'ORGAN']
    gene_cols = [c for c in df.columns if c not in meta_cols]

    # expression matrix (genes, cells)
    exp_matrix = df[gene_cols].values.T.astype(np.float32)
    exp_matrix = np.nan_to_num(exp_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    # row attributes: 'ensembl_id' is required
    row_attrs = {'ensembl_id': np.array(gene_cols, dtype='str')}

    # column attributes: all metadata columns
    cell_ids = df[cell_id_col].astype(str).values
    n_counts = df[gene_cols].sum(axis=1).values.astype(np.float32)

    col_attrs = {
        f'{mName}_ID': cell_ids,                     # cell id, used for alignment
        'n_counts': n_counts,                   # required
        f'{nName}_ID': df[f'{nName}_ID'].values.astype(str),
        'CELL_LINES': df['CELL_LINES'].values.astype(str),
        'CELL_LINES_STD': df['CELL_LINES_STD'].values.astype(str),
        'ORGAN': df['ORGAN'].values.astype(str)
    }

    loompy.create(loom_path, exp_matrix, row_attrs, col_attrs)
    print(f"loom file written: {loom_path}, shape {exp_matrix.shape}")

if __name__ == "__main__":

    mName = 'GDSC'
    nName = 'CCLE'

    csv_to_raw_loom(
        csv_path=f'/home/xxin/my_model/Data/{mName}_EXP_ensembl.csv',
        loom_path=f'/home/xxin/my_model/Data/{mName}_EXP.loom',   # should use the .loom extension
        cell_id_col=f'{mName}_ID',
        mName=mName,
        nName=nName
    )