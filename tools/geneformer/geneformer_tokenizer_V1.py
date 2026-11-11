from geneformer import TranscriptomeTokenizer

if __name__ == '__main__':
    Name = 'GDSC'
    tk = TranscriptomeTokenizer(
        custom_attr_name_dict={
            'GDSC_ID': 'GDSC_ID',
            'CCLE_ID': 'CCLE_ID',
            'CELL_LINES': 'CELL_LINES',
            'CELL_LINES_STD': 'CELL_LINES_STD',
            'ORGAN': 'ORGAN'
        },
        nproc=4,
        model_version='V1'   # default dictionary used automatically
    )
    tk.tokenize_data(
        data_directory='/home/xxin/my_model/Data',
        output_directory='/home/xxin/my_model/Data',
        output_prefix=f'{Name}_V1',
        file_format='loom'   # adjust to the actual file format
    )