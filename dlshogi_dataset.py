import ctypes
import glob
import os
import sys

import numpy as np
import torch
from torch.utils.data import Dataset


MAX_FEATURES1_NUM = 62
MAX_FEATURES2_NUM = 57
BOARD_SQUARES = 81


local_dllpath = [
    n
    for n in glob.glob("./*training_data_loader.*")
    if n.endswith(".so") or n.endswith(".dll") or n.endswith(".dylib")
]
if not local_dllpath:
    print("Cannot find data_loader shared library.")
    sys.exit(1)
dllpath = os.path.abspath(local_dllpath[0])
dll = ctypes.cdll.LoadLibrary(dllpath)


class DlshogiBatch(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_int),
        ("features1", ctypes.POINTER(ctypes.c_float)),
        ("features2", ctypes.POINTER(ctypes.c_float)),
        ("outcome", ctypes.POINTER(ctypes.c_float)),
        ("score", ctypes.POINTER(ctypes.c_float)),
        ("ply", ctypes.POINTER(ctypes.c_float)),
    ]

    def get_tensors(self):
        # Clone into PyTorch-owned memory before the underlying C++ batch is freed.
        features1 = torch.from_numpy(
            np.ctypeslib.as_array(
                self.features1,
                shape=(self.size, MAX_FEATURES1_NUM, BOARD_SQUARES),
            )
        ).clone().pin_memory()
        features2 = torch.from_numpy(
            np.ctypeslib.as_array(
                self.features2,
                shape=(self.size, MAX_FEATURES2_NUM, BOARD_SQUARES),
            )
        ).clone().pin_memory()
        outcome = torch.from_numpy(
            np.ctypeslib.as_array(self.outcome, shape=(self.size, 1))
        ).clone().pin_memory()
        score = torch.from_numpy(
            np.ctypeslib.as_array(self.score, shape=(self.size, 1))
        ).clone().pin_memory()
        ply = torch.from_numpy(
            np.ctypeslib.as_array(self.ply, shape=(self.size, 1))
        ).clone().pin_memory()
        return features1, features2, outcome, score, ply


DlshogiBatchPtr = ctypes.POINTER(DlshogiBatch)


create_dlshogi_batch_stream = dll.create_dlshogi_batch_stream
create_dlshogi_batch_stream.restype = ctypes.c_void_p
create_dlshogi_batch_stream.argtypes = [
    ctypes.c_char_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_bool,
    ctypes.c_bool,
    ctypes.c_int,
]

destroy_dlshogi_batch_stream = dll.destroy_dlshogi_batch_stream
destroy_dlshogi_batch_stream.argtypes = [ctypes.c_void_p]

fetch_next_dlshogi_batch = dll.fetch_next_dlshogi_batch
fetch_next_dlshogi_batch.restype = DlshogiBatchPtr
fetch_next_dlshogi_batch.argtypes = [ctypes.c_void_p]

destroy_dlshogi_batch = dll.destroy_dlshogi_batch
destroy_dlshogi_batch.argtypes = [DlshogiBatchPtr]


class TrainingDataProvider:
    def __init__(
        self,
        create_stream,
        destroy_stream,
        fetch_next,
        destroy_part,
        filename,
        cyclic,
        num_workers,
        batch_size,
        filtered=False,
        random_fen_skipping=0,
    ):
        self.create_stream = create_stream
        self.destroy_stream = destroy_stream
        self.fetch_next = fetch_next
        self.destroy_part = destroy_part
        self.filename = filename.encode("utf-8")
        self.cyclic = cyclic
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.filtered = filtered
        self.random_fen_skipping = random_fen_skipping
        self.stream = self.create_stream(
            self.filename,
            self.num_workers,
            self.batch_size,
            cyclic,
            filtered,
            random_fen_skipping,
        )

    def __iter__(self):
        return self

    def __next__(self):
        v = self.fetch_next(self.stream)
        if not v:
            raise StopIteration
        tensors = v.contents.get_tensors()
        self.destroy_part(v)
        return tensors

    def __del__(self):
        if getattr(self, "stream", None):
            self.destroy_stream(self.stream)


class DlshogiBatchProvider(TrainingDataProvider):
    def __init__(
        self,
        filename,
        batch_size,
        cyclic=True,
        num_workers=1,
        filtered=False,
        random_fen_skipping=0,
    ):
        super().__init__(
            create_dlshogi_batch_stream,
            destroy_dlshogi_batch_stream,
            fetch_next_dlshogi_batch,
            destroy_dlshogi_batch,
            filename,
            cyclic,
            num_workers,
            batch_size,
            filtered,
            random_fen_skipping,
        )


class DlshogiBatchDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        filename,
        batch_size,
        cyclic=True,
        num_workers=1,
        filtered=False,
        random_fen_skipping=0,
    ):
        super().__init__()
        self.filename = filename
        self.batch_size = batch_size
        self.cyclic = cyclic
        self.num_workers = num_workers
        self.filtered = filtered
        self.random_fen_skipping = random_fen_skipping

    def __iter__(self):
        return DlshogiBatchProvider(
            self.filename,
            self.batch_size,
            cyclic=self.cyclic,
            num_workers=self.num_workers,
            filtered=self.filtered,
            random_fen_skipping=self.random_fen_skipping,
        )


class FixedNumBatchesDataset(Dataset):
    def __init__(self, dataset, num_batches):
        super().__init__()
        self.dataset = dataset
        self.iter = None
        self.num_batches = num_batches

    def __len__(self):
        return self.num_batches

    def __getitem__(self, idx):
        if idx == 0 or self.iter is None:
            self.iter = iter(self.dataset)
        return next(self.iter)
