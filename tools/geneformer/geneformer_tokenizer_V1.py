import os
from geneformer import TranscriptomeTokenizer

_ROOT = os.environ.get('BCA_HFP_ROOT',
                 os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_GENE = os.path.join(_ROOT, 'Data', 'Gene')


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
        data_directory=_GENE,
        output_directory=_GENE,
        output_prefix=f'{Name}_V1',
        file_format='loom'   # adjust to the actual file format
    )