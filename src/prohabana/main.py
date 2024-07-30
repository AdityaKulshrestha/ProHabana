# from logutils.capture_memory import get_hpu_memory_stats


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

        mem_stats = memory_stats(device)

        mem_dict = {
            "memory_allocated (GB)": to_gb_rounded(mem_stats["InUse"]),
            "max_memory_allocated (GB)": to_gb_rounded(mem_stats["MaxInUse"]),
            "total_memory_available (GB)": to_gb_rounded(mem_stats["Limit"]),
        }

        return mem_dict

    return wrapper 

