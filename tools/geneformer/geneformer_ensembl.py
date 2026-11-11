import pandas as pd
import mygene
import pickle
def convert_with_mygene(csv_path, output_path, gene_start_col=5, mapping_path=None):
    """
    - csv_path: input CSV path
    - output_path: output CSV path
    - gene_start_col: index of the first gene column (0-based), default 5 means the first 5 columns are metadata
    """
    if mapping_path is None:
        mapping_path = "/home/xxin/my_model/Gene_Feature/geneformer/gene_dictionaries_30m/gene_name_id_dict_gc30M.pkl"

    # load the mapping file: {symbol: ensembl_id_with_version}
    with open(mapping_path, 'rb') as f:
        symbol_to_ensembl = pickle.load(f)

    # build symbol -> ensembl_id (version suffix stripped)
    ensembl_map = {}
    for symbol, ens_id in symbol_to_ensembl.items():
        ens_no_version = ens_id.split('.')[0]
        ensembl_map[symbol.upper()] = ens_no_version

        # read the input CSV
    df = pd.read_csv(csv_path)
    gene_symbols = df.columns[gene_start_col:].tolist()
    print(f"{len(gene_symbols)} genes to convert")

    # rename columns: replace symbols with Ensembl IDs, keep the symbol when nothing matches
    new_columns = list(df.columns[:gene_start_col])
    success = 0
    for symbol in gene_symbols:
        ens = ensembl_map.get(symbol.upper())  # look up in upper case
        if ens is not None:
            new_columns.append(ens)
            success += 1
        else:
            print(f"gene {symbol} has no Ensembl ID, keeping the original name")
            new_columns.append(symbol)

    df.columns = new_columns
    df.to_csv(output_path, index=False)
    print(f"conversion rate: {success}/{len(gene_symbols)}")
    print(f"converted file saved: {output_path}")
    return ensembl_map  # mapping used for the column rename (optional)

# example usage
if __name__ == "__main__":
    Name = 'GDSC'
    convert_with_mygene(
        csv_path=f"/home/xxin/my_model/Data/{Name}_EXP.csv",
        output_path=f"/home/xxin/my_model/Data/{Name}_EXP_ensembl.csv",
        gene_start_col=5
    )