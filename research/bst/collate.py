"""Run the repo's own collate step (Step 3 of prepare_train_on_shuttleset_merged.py)
without editing the repo file.

usage: venv/bin/python collate.py <raw_npy_dir> <out_collated_dir> <seq_len> [sets...]
e.g.   venv/bin/python collate.py data/ShuttleSet_data_merged/dataset_npy_between_2_hits_with_max_limits \
           data/ShuttleSet_data_merged/dataset_npy_collated_between_2_hits_with_max_limits_seq_100 100
"""
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent / 'repo' / 'stroke_classification'
sys.path.insert(0, str(REPO / 'preparing_data'))  # the module imports `shuttleset_dataset` flat

from prepare_train_on_shuttleset_merged import collate_npy  # noqa: E402

if __name__ == '__main__':
    raw, out, seq_len = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
    sets = sys.argv[4:] or ['train', 'val', 'test']
    out.parent.mkdir(parents=True, exist_ok=True)
    for s in sets:
        t0 = time.time()
        collate_npy(root_dir=raw, set_name=s, seq_len=seq_len, save_dir=out)
        print(f'{s}: {time.time() - t0:.1f} s')
