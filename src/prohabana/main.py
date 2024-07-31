# from logutils.capture_memory import get_hpu_memory_stats
from typing import Dict
import torch
import habana_frameworks.torch as ht
import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.hpu.graphs as htgraphs

def log_details(func):
    def wrapper(*args, **kwargs) -> Dict[str, float]:
        """
        Returns memory stats of HPU as a dictionary:
        - current memory allocated (GB)
        - maximum memory allocated (GB)
        - total memory available (GB)

        Returns:
            Dict[str, float]: memory stats.
        """
        from habana_frameworks.torch.hpu import memory_stats

        output = func(*args, **kwargs)

        mem_stats = memory_stats(torch.device('hpu'))

        mem_dict = {
            "memory_allocated (GB)": round(mem_stats["InUse"] / 1e9, 2),
            "max_memory_allocated (GB)": round(mem_stats["MaxInUse"] / 1e9, 2),
            "total_memory_available (GB)": round(mem_stats["Limit"] / 1e9, 2),
        }
        
        print(mem_dict)
        return output

    return wrapper 

