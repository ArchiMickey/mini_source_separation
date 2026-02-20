import random
from torch.utils.data import Sampler
import numpy as np
import torch


class RandomSongSamplerMix(Sampler):
    """Randomly sample indexes of different stems of a dataset without 
    replacement. Supports distributed training (DDP) and generates indices
    on-the-fly to reduce memory overhead.
    
    This sampler yields indices infinitely, reshuffling after each epoch.
    The __len__ method returns a large number to ensure the dataloader
    continues until training_steps is reached.
    """

    def __init__(self, dataset, max_intra_source_mix, num_replicas=None, rank=None, seed=0):
        self.dataset = dataset
        self.stems = dataset.stems
        self.mix_num = max_intra_source_mix
        self.seed = seed
        self.epoch = 0
        
        # DDP support - get world size and rank
        if num_replicas is None:
            if torch.distributed.is_initialized():
                num_replicas = torch.distributed.get_world_size()
            else:
                num_replicas = 1
        if rank is None:
            if torch.distributed.is_initialized():
                rank = torch.distributed.get_rank()
            else:
                rank = 0
                
        self.num_replicas = num_replicas
        self.rank = rank
        
        # Calculate samples per replica
        self.dataset_size = len(self.dataset)
        self.num_samples = self.dataset_size
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        """Yield an index_dict infinitely, reshuffling after each epoch."""
        
        while True:
            # Use generator with epoch-based seed for reproducibility
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            
            # Generate permutation for this epoch
            indices = torch.randperm(self.dataset_size, generator=g).tolist()
            
            # Pad indices for DDP (ensure equal size across all ranks)
            # Repeat indices enough times to reach total_size
            if len(indices) < self.total_size:
                repeat_needed = (self.total_size + len(indices) - 1) // len(indices)
                indices = (indices * repeat_needed)[:self.total_size]
            assert len(indices) == self.total_size
            
            # Subsample for this rank
            indices = indices[self.rank:self.total_size:self.num_replicas]
            assert len(indices) == self.num_samples
            
            # Yield index dicts for this epoch
            for idx in indices:
                yield self._get_mix_indices(idx)
            
            # Increment epoch for next iteration
            self.epoch += 1

    def _get_mix_indices(self, base_idx):
        """Generate mix indices on-the-fly for a given base index.
        
        Uses a deterministic but varied selection based on base_idx and stem.
        """
        out = {}
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch + base_idx * 1000)
        
        for stem in self.stems:
            # Generate random indices for mixing
            indices = []
            for m in range(self.mix_num):
                # Create varied but deterministic indices
                offset = torch.randint(0, self.dataset_size, (1,), generator=g).item()
                indices.append((base_idx + offset) % self.dataset_size)
            out[stem] = indices
            
        return out

    def __len__(self):
        """Return a large number to ensure dataloader continues until training_steps.
        
        Since this sampler yields infinitely, we return a very large number
        to prevent the dataloader from stopping early.
        """
        return 2**31 - 1  # Max int32 value, effectively infinite
    
    def set_epoch(self, epoch):
        """Set the epoch for reproducibility in distributed training.
        
        This should be called at the beginning of each epoch before iterating
        over the dataloader.
        """
        self.epoch = epoch


class RandomSongSamplerMixLegacy:
    """Legacy version for backward compatibility.
    
    Randomly sample indexes of different stems of a dataset without 
    replacement. Execute this process infinitely.
    """

    def __init__(self, dataset, max_intra_source_mix):
        self.dataset = dataset
        self.stems = dataset.stems
        self.mix_num = max_intra_source_mix

        self.indices = {stem: self.random_permutation(len(self.dataset), self.mix_num) for stem in self.stems}
        # E.g., {"bg": [3, 7, 0, ...], "target":, [4, 1, 9, ...]}

        self.ptrs = {stem: 0 for stem in self.indices.keys()}  # pointers

    def __iter__(self):
        """Yield an index_dict."""

        while True:

            out = {}

            for stem in self.indices.keys():

                # Reshuffle indices. Reset pointer.
                if self.ptrs[stem] == len(self.indices[stem]):
                    self.indices[stem] = self.random_permutation(len(self.dataset), self.mix_num)
                    self.ptrs[stem] = 0

                out[stem] = self.indices[stem][self.ptrs[stem]]
                self.ptrs[stem] += 1
            
            yield out  # E.g., {"vocals": [94, 13], "drums": [13, 26], "other": [0, 22], "vocals": [6, 88]}

    def random_permutation(self, n: int, mix_num: int):
        indices = np.zeros((n, mix_num), dtype=np.int64)
        for m in range(mix_num):
            x = list(range(n))
            random.shuffle(x)
            indices[:, m] = x
        
        return indices
