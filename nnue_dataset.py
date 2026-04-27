import numpy as np
import ctypes
import torch
import os
import sys
import glob
from torch.utils.data import Dataset

_dll = None

def get_dll():
    global _dll
    if _dll is None:
        local_dllpath = [n for n in glob.glob('./*training_data_loader.*') if n.endswith('.so') or n.endswith('.dll') or n.endswith('.dylib')]
        if not local_dllpath:
            print('Cannot find data_loader shared library.')
            sys.exit(1)
        dllpath = os.path.abspath(local_dllpath[0])
        _dll = ctypes.cdll.LoadLibrary(dllpath)
        _setup_dll_signatures(_dll)
    return _dll

def _setup_dll_signatures(dll):
    dll.create_sparse_batch_stream.restype = ctypes.c_void_p
    dll.destroy_sparse_batch_stream.argtypes = [ctypes.c_void_p]
    dll.fetch_next_sparse_batch.restype = SparseBatchPtr
    dll.fetch_next_sparse_batch.argtypes = [ctypes.c_void_p]
    dll.destroy_sparse_batch.argtypes = [SparseBatchPtr]
    dll.get_sparse_batch_from_fens.restype = SparseBatchPtr

def create_sparse_batch_stream(*args, **kwargs):
    return get_dll().create_sparse_batch_stream(*args, **kwargs)

def destroy_sparse_batch_stream(*args, **kwargs):
    return get_dll().destroy_sparse_batch_stream(*args, **kwargs)

def fetch_next_sparse_batch(*args, **kwargs):
    return get_dll().fetch_next_sparse_batch(*args, **kwargs)

def destroy_sparse_batch(*args, **kwargs):
    return get_dll().destroy_sparse_batch(*args, **kwargs)

def get_sparse_batch_from_fens(*args, **kwargs):
    return get_dll().get_sparse_batch_from_fens(*args, **kwargs)

class SparseBatch(ctypes.Structure):
    _fields_ = [
        ('num_inputs', ctypes.c_int),
        ('size', ctypes.c_int),
        ('is_white', ctypes.POINTER(ctypes.c_float)),
        ('outcome', ctypes.POINTER(ctypes.c_float)),
        ('score', ctypes.POINTER(ctypes.c_float)),
        ('num_active_white_features', ctypes.c_int),
        ('num_active_black_features', ctypes.c_int),
        ('white', ctypes.POINTER(ctypes.c_int)),
        ('black', ctypes.POINTER(ctypes.c_int)),
        ('white_values', ctypes.POINTER(ctypes.c_float)),
        ('black_values', ctypes.POINTER(ctypes.c_float)),
        ('ply', ctypes.POINTER(ctypes.c_float)),
    ]

    def get_tensors(self, device):
        white_values = torch.from_numpy(np.ctypeslib.as_array(self.white_values, shape=(self.num_active_white_features,))).pin_memory().to(device=device, non_blocking=True)
        black_values = torch.from_numpy(np.ctypeslib.as_array(self.black_values, shape=(self.num_active_black_features,))).pin_memory().to(device=device, non_blocking=True)
        iw = torch.transpose(torch.from_numpy(np.ctypeslib.as_array(self.white, shape=(self.num_active_white_features, 2))).pin_memory().to(device=device, non_blocking=True), 0, 1).long()
        ib = torch.transpose(torch.from_numpy(np.ctypeslib.as_array(self.black, shape=(self.num_active_white_features, 2))).pin_memory().to(device=device, non_blocking=True), 0, 1).long()
        us = torch.from_numpy(np.ctypeslib.as_array(self.is_white, shape=(self.size, 1))).pin_memory().to(device=device, non_blocking=True)
        them = 1.0 - us
        outcome = torch.from_numpy(np.ctypeslib.as_array(self.outcome, shape=(self.size, 1))).pin_memory().to(device=device, non_blocking=True)
        score = torch.from_numpy(np.ctypeslib.as_array(self.score, shape=(self.size, 1))).pin_memory().to(device=device, non_blocking=True)
        white = torch.sparse_coo_tensor(iw, white_values, (self.size, self.num_inputs))
        black = torch.sparse_coo_tensor(ib, black_values, (self.size, self.num_inputs))
        white._coalesced_(True)
        black._coalesced_(True)
        ply = torch.from_numpy(np.ctypeslib.as_array(self.ply, shape=(self.size, 1))).pin_memory().to(device=device, non_blocking=True)
        return us, them, white, black, outcome, score, ply

class PythonSparseBatch:
    def __init__(self, us, them, white, black, outcome, score, ply):
        self.contents = self
        self.tensors = (us, them, white, black, outcome, score, ply)

    def get_tensors(self, device):
        return tuple(t.to(device) for t in self.tensors)

def _extract_halfkp_indices(sfen, feature_set_name):
    import cshogi
    board = cshogi.Board(sfen)
    
    # HalfKP parameters from halfkp.py
    NUM_SQ = 81
    NUM_PLANES = 1548
    
    def iter_board_pieces():
        pieces = getattr(board, 'pieces', None)
        if callable(pieces):
            yield from pieces()
            return
        if isinstance(pieces, list):
            for sq, piece in enumerate(pieces):
                if piece:
                    yield sq, piece
            return
        for sq in range(NUM_SQ):
            piece = board.piece(sq)
            if piece:
                yield sq, piece

    def get_piece_type_and_color(sq, piece):
        piece_type = board.piece_type(sq)
        color = cshogi.WHITE if piece >= cshogi.WPAWN else cshogi.BLACK
        return piece_type, color

    def get_indices(color):
        is_white_pov = (color == cshogi.WHITE)
        king_sq = board.king_square(color)
        
        def orient(is_white_pov, sq):
            return (63 * (not is_white_pov)) ^ sq

        indices = []
        piece_count = 0
        
        for sq, p in iter_board_pieces():
            pt, pc = get_piece_type_and_color(sq, p)
            if pt == cshogi.KING:
                continue
            piece_count += 1
            p_idx = (pt - 1) * 2 + (pc != is_white_pov)
            
            idx = 1 + orient(is_white_pov, sq) + p_idx * NUM_SQ + orient(is_white_pov, king_sq) * NUM_PLANES
            indices.append((idx, 1.0))
        
        if feature_set_name == "HalfKP^":
            # HalfK uses the piece count as the feature value.
            indices.append((NUM_PLANES * NUM_SQ + orient(is_white_pov, king_sq), float(piece_count)))
            for sq, p in iter_board_pieces():
                pt, pc = get_piece_type_and_color(sq, p)
                if pt == cshogi.KING:
                    continue
                p_idx = (pt - 1) * 2 + (pc != is_white_pov)
                indices.append((NUM_PLANES * NUM_SQ + NUM_SQ + (p_idx + 1) * NUM_SQ + orient(is_white_pov, sq), 1.0))

        return indices

    return get_indices(cshogi.WHITE), get_indices(cshogi.BLACK)

SparseBatchPtr = ctypes.POINTER(SparseBatch)

class TrainingDataProvider:
    def __init__(
        self,
        feature_set,
        create_stream,
        destroy_stream,
        fetch_next,
        destroy_part,
        filename,
        cyclic,
        num_workers,
        batch_size=None,
        filtered=False,
        random_fen_skipping=0,
        device='cpu'):

        self.feature_set = feature_set.encode('utf-8')
        self.create_stream = create_stream
        self.destroy_stream = destroy_stream
        self.fetch_next = fetch_next
        self.destroy_part = destroy_part
        self.filename = filename.encode('utf-8')
        self.cyclic = cyclic
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.filtered = filtered
        self.random_fen_skipping = random_fen_skipping
        self.device = device

        if batch_size:
            self.stream = self.create_stream(self.feature_set, self.num_workers, self.filename, batch_size, cyclic, filtered, random_fen_skipping)
        else:
            self.stream = self.create_stream(self.feature_set, self.num_workers, self.filename, cyclic, filtered, random_fen_skipping)

    def __iter__(self):
        return self

    def __next__(self):
        v = self.fetch_next(self.stream)

        if v:
            tensors = v.contents.get_tensors(self.device)
            self.destroy_part(v)
            return tensors
        else:
            raise StopIteration

    def __del__(self):
        self.destroy_stream(self.stream)


def make_sparse_batch_from_fens(feature_set, fens, scores, plies, results):
    if feature_set.name in ["HalfKP", "HalfKP^"]:
        size = len(fens)
        num_inputs = feature_set.num_features
        
        us_list = []
        outcome_list = []
        score_list = []
        ply_list = []
        
        white_indices = []
        white_values = []
        black_indices = []
        black_values = []
        
        for i, (sfen, s, p, r) in enumerate(zip(fens, scores, plies, results)):
            import cshogi
            board = cshogi.Board(sfen)
            us_list.append(1.0 if board.turn == cshogi.WHITE else 0.0)
            outcome_list.append(float(r))
            score_list.append(float(s))
            ply_list.append(float(p))
            
            w_idx, b_idx = _extract_halfkp_indices(sfen, feature_set.name)
            for idx, value in w_idx:
                white_indices.append([i, idx])
                white_values.append(value)
            for idx, value in b_idx:
                black_indices.append([i, idx])
                black_values.append(value)
                
        us = torch.tensor(us_list).reshape(size, 1)
        them = 1.0 - us
        outcome = torch.tensor(outcome_list).reshape(size, 1)
        score = torch.tensor(score_list).reshape(size, 1)
        ply = torch.tensor(ply_list).reshape(size, 1)
        
        wi = torch.tensor(white_indices).t().long()
        wv = torch.tensor(white_values)
        white = torch.sparse_coo_tensor(wi, wv, (size, num_inputs)).coalesce()
        
        bi = torch.tensor(black_indices).t().long()
        bv = torch.tensor(black_values)
        black = torch.sparse_coo_tensor(bi, bv, (size, num_inputs)).coalesce()
        
        return PythonSparseBatch(us, them, white, black, outcome, score, ply)

    # Fallback to DLL for other feature sets
    results_ = (ctypes.c_int*len(scores))()
    scores_ = (ctypes.c_int*len(plies))()
    plies_ = (ctypes.c_int*len(results))()
    fens_ = (ctypes.c_char_p * len(fens))()
    fens_[:] = [fen.encode('utf-8') for fen in fens]
    for i, v in enumerate(scores):
        scores_[i] = int(v)
    for i, v in enumerate(plies):
        plies_[i] = int(v)
    for i, v in enumerate(results):
        results_[i] = int(v)
    b = get_sparse_batch_from_fens(feature_set.name.encode('utf-8'), len(fens), fens_, scores_, plies_, results_)
    return b

def destroy_sparse_batch(batch):
    if isinstance(batch, PythonSparseBatch):
        return
    return get_dll().destroy_sparse_batch(batch)

class SparseBatchProvider(TrainingDataProvider):
    def __init__(self, feature_set, filename, batch_size, cyclic=True, num_workers=1, filtered=False, random_fen_skipping=0, device='cpu'):
        super(SparseBatchProvider, self).__init__(
            feature_set,
            create_sparse_batch_stream,
            destroy_sparse_batch_stream,
            fetch_next_sparse_batch,
            destroy_sparse_batch,
            filename,
            cyclic,
            num_workers,
            batch_size,
            filtered,
            random_fen_skipping,
            device)

class SparseBatchDataset(torch.utils.data.IterableDataset):
  def __init__(self, feature_set, filename, batch_size, cyclic=True, num_workers=1, filtered=False, random_fen_skipping=0, device='cpu'):
    super(SparseBatchDataset).__init__()
    self.feature_set = feature_set
    self.filename = filename
    self.batch_size = batch_size
    self.cyclic = cyclic
    self.num_workers = num_workers
    self.filtered = filtered
    self.random_fen_skipping = random_fen_skipping
    self.device = device

  def __iter__(self):
    return SparseBatchProvider(self.feature_set, self.filename, self.batch_size, cyclic=self.cyclic, num_workers=self.num_workers, filtered=self.filtered, random_fen_skipping=self.random_fen_skipping, device=self.device)

class FixedNumBatchesDataset(Dataset):
  def __init__(self, dataset, num_batches):
    super(FixedNumBatchesDataset, self).__init__()
    self.dataset = dataset;
    self.iter = iter(self.dataset)
    self.num_batches = num_batches

  def __len__(self):
    return self.num_batches

  def __getitem__(self, idx):
    return next(self.iter)
