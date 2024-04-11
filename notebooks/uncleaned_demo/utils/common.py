"""
common.py

A collection of commonly used utility functions.
"""

def detach_cpu_list(list_of_tensor):
    result = [item.detach().cpu() for item in list_of_tensor]
    return result


def compute_batch_sections(n_data: int, batch_size: int):
    """
    Computes the sections that divide the given data into batches.

    Args:
        n_data: int, number of data
        batch_size: int, size of a single batch 
    """
    assert isinstance(n_data, int) and n_data > 0
    assert isinstance(batch_size, int) and batch_size > 0

    batch_indices = list(range(0, n_data, batch_size))
    if batch_indices[-1] != n_data:
        batch_indices.append(n_data)
    
    batch_start = batch_indices[:-1]
    batch_end = batch_indices[1:]
    assert len(batch_start) == len(batch_end), f"Lengths are different {len(batch_start)} {len(batch_end)}"
    
    return batch_start, batch_end