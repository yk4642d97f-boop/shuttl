"""Pack a ShuttleSet_data_merged raw .npy tree (<set>/<class>/<clip>_{joints,pos,shuttle}.npy)
into one float32 .npz per split, and unpack it again.

The repo's collate step (pad_and_augment_one_npy_video) starts with
`.astype(np.float32)` on all three arrays, so converting float64 -> float32
before transfer gives bit-identical collated arrays. Used only to shrink the
transfer over the slow relay link.

usage: python pack_raw.py pack   <raw_dir> <out_dir>
       python pack_raw.py unpack <packed_dir> <raw_dir>
"""
import sys
from pathlib import Path

import numpy as np


def pack(raw: Path, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    for split in ['train', 'val', 'test']:
        names, J, P, S, lens = [], [], [], [], []
        for typ in sorted((raw / split).iterdir()):
            for f in sorted(typ.glob('*_pos.npy')):
                b = str(f).replace('_pos.npy', '')
                j = np.load(b + '_joints.npy').astype(np.float32)
                p = np.load(b + '_pos.npy').astype(np.float32)
                s = np.load(b + '_shuttle.npy').astype(np.float32)
                assert len(j) == len(p) == len(s), b
                names.append(f'{typ.name}/{Path(b).name}')
                J.append(j); P.append(p); S.append(s); lens.append(len(j))
        np.savez(out / f'{split}.npz', names=np.array(names), lens=np.array(lens, dtype=np.int64),
                 joints=np.concatenate(J), pos=np.concatenate(P), shuttle=np.concatenate(S))
        print(split, len(names), 'clips', sum(lens), 'frames')


def unpack(packed: Path, raw: Path):
    for split in ['train', 'val', 'test']:
        z = np.load(packed / f'{split}.npz')
        off = np.concatenate([[0], np.cumsum(z['lens'])])
        J, P, S = z['joints'], z['pos'], z['shuttle']
        for i, name in enumerate(z['names']):
            typ, clip = str(name).split('/')
            d = raw / split / typ
            d.mkdir(parents=True, exist_ok=True)
            a, b = off[i], off[i + 1]
            np.save(d / f'{clip}_joints.npy', J[a:b])
            np.save(d / f'{clip}_pos.npy', P[a:b])
            np.save(d / f'{clip}_shuttle.npy', S[a:b])
        print(split, len(z['names']), 'clips unpacked')


if __name__ == '__main__':
    {'pack': pack, 'unpack': unpack}[sys.argv[1]](Path(sys.argv[2]), Path(sys.argv[3]))
