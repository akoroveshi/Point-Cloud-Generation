"""One-time conversion of the HF `ILSVRC/imagenet-1k` clone into a fixed-size uint8 cache.

Handles both layouts the HF repo has used: parquet shards, or *.tar.gz image shards.
Output (in --cache_dir):  {split}_{S}_imgs.npy  {split}_{S}_labels.npy  {split}_{S}_meta.json

Example:
  python prepare_data.py --cache_dir CACHE --img_size 64 --split val --stream     # stream from the Hub
  python prepare_data.py --cache_dir CACHE --hf_root /path/to/imagenet-1k --split val
  python prepare_data.py --cache_dir CACHE --hf_root /path/to/imagenet-1k --split train --workers 16
  python prepare_data.py --cache_dir CACHE --hf_root /path/to/imagenet-1k --split train --max_per_class 50
"""
import argparse
import glob
import io
import json
import os
import re
import tarfile
from multiprocessing import Pool

import numpy as np
from PIL import Image

from images.data import cache_paths, center_crop_arr

CAPACITY = {"train": 1281167, "val": 50000}
_SIZE = None


def _init(size):
    global _SIZE
    _SIZE = size


def _proc(item):
    """bytes -> (S,S,3) uint8 array (or None if the image is broken)."""
    b, label = item
    try:
        im = Image.open(io.BytesIO(b))
        im.draft("RGB", (_SIZE * 2, _SIZE * 2))  # fast JPEG downscale-on-decode
        return np.asarray(center_crop_arr(im.convert("RGB"), _SIZE)), label
    except Exception:
        return None, label


def check_lfs(path):
    with open(path, "rb") as f:
        if f.read(40).startswith(b"version https://git-lfs"):
            raise SystemExit(f"{path} is a git-lfs pointer, not the real file.\n"
                             f"Run `git lfs install && git lfs pull` inside the dataset repo.")


def find_files(root, split):
    pats = {"train": ["train*.parquet", "train*.tar.gz"],
            "val": ["val*.parquet", "validation*.parquet", "val*.tar.gz"]}[split]
    for pat in pats:  # first pattern that matches wins (parquet preferred)
        files = sorted(glob.glob(os.path.join(root, "**", pat), recursive=True))
        if files:
            return files
    raise SystemExit(f"No {split} shards found under {root}. Check `ls {root}/data | head`.")


def iter_parquet(files):
    import pyarrow.parquet as pq
    for f in files:
        for batch in pq.ParquetFile(f).iter_batches(batch_size=256, columns=["image", "label"]):
            for im, l in zip(batch.column("image").to_pylist(), batch.column("label").to_pylist()):
                if l >= 0:
                    yield im["bytes"], l


def load_wnids(root):
    txt = open(os.path.join(root, "classes.py")).read()
    wnids = list(dict.fromkeys(re.findall(r"n\d{8}", txt)))  # ordered, unique
    assert len(wnids) == 1000, f"expected 1000 wnids in classes.py, got {len(wnids)}"
    return {w: i for i, w in enumerate(wnids)}


def iter_tar(files, wnid2idx):
    for f in files:
        with tarfile.open(f, "r|gz") as tf:
            for m in tf:
                if not m.isfile():
                    continue
                mt = re.search(r"n\d{8}", os.path.basename(m.name))
                if mt is None:
                    raise SystemExit(f"Cannot read the class from file name {m.name}")
                yield tf.extractfile(m).read(), wnid2idx[mt.group(0)]


def iter_stream(split):
    """Stream from the HF Hub (no local copy needed). Requires `huggingface-cli login` + accepted terms."""
    from datasets import Image as HFImage, load_dataset
    ds = load_dataset("ILSVRC/imagenet-1k", split={"train": "train", "val": "validation"}[split],
                      streaming=True)
    ds = ds.cast_column("image", HFImage(decode=False))  # raw bytes; decoding happens in our workers
    for ex in ds:
        if ex["label"] >= 0:
            yield ex["image"]["bytes"], ex["label"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf_root", default=None,
                    help="path to the local HF imagenet-1k folder (not needed with --stream)")
    ap.add_argument("--cache_dir", required=True, help="where the preprocessed cache is written")
    ap.add_argument("--split", default="val", choices=["train", "val"])
    ap.add_argument("--img_size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--stream", action="store_true",
                    help="stream from the HF Hub instead of reading --hf_root (no raw download)")
    ap.add_argument("--max_per_class", type=int, default=0, help="0 = all images")
    args = ap.parse_args()

    if not args.stream and args.hf_root is None:
        ap.error("give --hf_root /path/to/imagenet-1k (local copy) or use --stream")
    os.makedirs(args.cache_dir, exist_ok=True)
    if args.stream:
        src = iter_stream(args.split)
    else:
        files = find_files(args.hf_root, args.split)
        for f in files[:2]:
            check_lfs(f)
        print(f"{len(files)} {args.split} shards, e.g. {files[0]}")
        src = iter_parquet(files) if files[0].endswith(".parquet") else iter_tar(files, load_wnids(args.hf_root))

    img_path, lab_path, meta_path = cache_paths(args.cache_dir, args.split, args.img_size)
    S, cap = args.img_size, CAPACITY[args.split]
    imgs = np.lib.format.open_memmap(img_path, mode="w+", dtype=np.uint8, shape=(cap, S, S, 3))
    labels = np.zeros(cap, np.int16)
    per_class = np.zeros(1000, np.int64)

    def gen():
        for b, l in src:
            if args.max_per_class:
                if (per_class >= args.max_per_class).all():
                    return
                if per_class[l] >= args.max_per_class:
                    continue
            per_class[l] += 1
            yield b, l

    n, bad = 0, 0
    with Pool(args.workers, initializer=_init, initargs=(S,)) as pool:
        for arr, l in pool.imap(_proc, gen(), chunksize=64):
            if arr is None:
                bad += 1
                continue
            if n >= cap:
                break
            imgs[n], labels[n] = arr, l
            n += 1
            if n % 10000 == 0:
                print(f"  {n} images", flush=True)
    imgs.flush()
    np.save(lab_path, labels)
    json.dump({"n": n, "size": S, "bad": bad}, open(meta_path, "w"))
    print(f"done: {n} images ({bad} unreadable) -> {img_path}")


if __name__ == "__main__":
    main()