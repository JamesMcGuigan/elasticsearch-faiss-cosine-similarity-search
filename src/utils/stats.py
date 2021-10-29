from typing import Any, Dict, Iterable, Union

import numpy as np


def stats_describe(data: np.ndarray):
    return {
        "min":    np.min(data),
        "1%":     np.quantile(data, 0.01),
        "10%":    np.quantile(data, 0.10),
        # "25%":    np.quantile(data, 0.25),
        "mean":   np.mean(data),
        "median": np.median(data),
        # "75%":    np.quantile(data, 0.75),
        "90%":    np.quantile(data, 0.90),
        "99%":    np.quantile(data, 0.99),
        "max":    np.max(data),
        "std":    np.std(data),
    }


def print_stats(data: Union[Dict,np.ndarray]):
    stats = stats_describe(data) if isinstance(data, np.ndarray) else data
    for k,v in stats.items():
        print(f'  {k:6s} = {v:6.3f}')
    print()


def count_duplicates( items: Iterable[Any] ) -> int:
    items = list(items) if not isinstance(items, (list,tuple,set)) else items
    return len(items) - len(set(items))
