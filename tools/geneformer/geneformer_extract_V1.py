from geneformer import EmbExtractor
import torch
from pathlib import Path
import json
from datasets import load_from_disk
import shutil

def batch_extract_and_save_V1(dataset_path, output_dir, model_directory, batch_size=100, **extractor_kwargs):
    full_dataset = load_from_disk(dataset_path)
    total_cells = len(full_dataset)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    embex = EmbExtractor(**extractor_kwargs)

    for start_idx in range(0, total_cells, batch_size):
        end_idx = min(start_idx + batch_size, total_cells)
        print(f"processing cells {start_idx} to {end_idx-1}...")

        batch_dataset = full_dataset.select(range(start_idx, end_idx))
        temp_path = output_dir / f"temp_{start_idx}_{end_idx}.dataset"
        batch_dataset.save_to_disk(str(temp_path))

        cell_data_list = embex.extract_embs(
            model_directory=model_directory,
            input_data_file=str(temp_path),
            output_directory=None,
            output_prefix=None
        )

        for local_idx, cell in enumerate(cell_data_list):
            global_idx = start_idx + local_idx
            cell['embeddings'] = cell['embeddings'].cpu().to(torch.float16)
            torch.save(cell, output_dir / f"{global_idx}.pt")

        shutil.rmtree(temp_path)
        torch.cuda.empty_cache()

    mapping = {}
    for file in output_dir.glob("*.pt"):
        idx = int(file.stem)
        cell = torch.load(file)
        mapping[idx] = cell['metadata']
    with open(output_dir / "mapping.json", 'w') as f:
        json.dump(mapping, f, indent=2)

    print(f"all done, results saved in {output_dir}")

if __name__ == '__main__':

    NAME = 'GDSC'
    extractor_kwargs = {
        'model_type': 'Pretrained',
        'num_classes': 0,
        'emb_mode': 'gene',
        'gene_emb_style': 'all',
        'filter_data': None,
        'max_ncells': None,
        'emb_layer': -1,
        'emb_label': ['GDSC_ID', 'CCLE_ID', 'CELL_LINES', 'CELL_LINES_STD', 'ORGAN'],
        'forward_batch_size': 8,
        'model_version': 'V1',       # back to V1
        'nproc': 4,
        'return_per_cell': True,
        # no manual token_dictionary_file any more
    }
    batch_extract_and_save_V1(
        dataset_path=f'/home/xxin/my_model/Data/{NAME}_V1.dataset',
        output_dir=f'/home/xxin/my_model/Data/{NAME}_EXP_GF_V1',
        model_directory=f'/home/xxin/my_model/Gene_Feature/geneformer_model_V1',
        batch_size=100,
        **extractor_kwargs
    )