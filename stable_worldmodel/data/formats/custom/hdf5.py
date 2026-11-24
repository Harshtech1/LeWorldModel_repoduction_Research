"""Custom HDF5 format for stable-worldmodel."""

import h5py
import numpy as np
from typing import List, Dict, Any
from stable_worldmodel.data.format import Format, Writer


class HDF5Dataset:
    """HDF5 dataset reader compatible with stable-worldmodel Dataset interface."""
    
    def __init__(self, path: str):
        self.path = path
        self._h5 = h5py.File(path, 'r')
        
        # Get column names from datasets (excluding metadata)
        self.columns = [k for k in self._h5.keys() if k not in ('episode_idx', 'step_idx')]
        
        # Build episode structure
        episode_idx = self._h5['episode_idx'][:]
        unique_episodes, counts = np.unique(episode_idx, return_counts=True)
        
        self.episode_lengths = counts.astype(np.int64).tolist()
        self.num_episodes = len(self.episode_lengths)
    
    def __len__(self) -> int:
        # Total number of steps across all episodes
        return sum(self.episode_lengths)
    
    @property
    def column_names(self) -> List[str]:
        return self.columns
    
    def get_dim(self, col: str) -> int:
        """Get the dimension of a column."""
        shape = self._h5[col].shape
        if len(shape) == 1:
            return 1
        else:
            return int(shape[-1])
    
    def load_episode(self, ep_idx: int) -> Dict[str, Any]:
        """Load an episode and return as dict of {column: torch.Tensor}."""
        import torch
        
        start = sum(self.episode_lengths[:ep_idx])
        end = start + self.episode_lengths[ep_idx]
        
        data = {}
        for col in self.columns:
            arr = self._h5[col][start:end]
            # Convert to torch tensor
            if arr.dtype == np.uint8:
                data[col] = torch.from_numpy(arr).float() / 255.0  # Normalize images
            else:
                data[col] = torch.from_numpy(arr).float()
        
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single step by global index."""
        import torch
        
        # Find which episode and local index
        cumsum = np.cumsum([0] + self.episode_lengths)
        ep_idx = np.searchsorted(cumsum[1:], idx + 1)
        local_idx = idx - cumsum[ep_idx]
        
        data = {}
        for col in self.columns:
            arr = self._h5[col][idx:idx+1]
            if arr.dtype == np.uint8:
                data[col] = torch.from_numpy(arr).float() / 255.0
            else:
                data[col] = torch.from_numpy(arr).float()
        
        return data
    
    def close(self):
        self._h5.close()


class HDF5Format(Format):
    """HDF5 format reader."""
    
    name = 'hdf5'
    
    @classmethod
    def detect(cls, path) -> bool:
        import os
        if isinstance(path, str):
            return path.endswith('.h5') or path.endswith('.hdf5')
        return False
    
    @classmethod
    def open_reader(cls, path: str, **kwargs):
        return HDF5Dataset(path)


# Register the format
import stable_worldmodel.data.format as fmt_module
if hasattr(fmt_module, 'register_format'):
    fmt_module.register_format(HDF5Format)
