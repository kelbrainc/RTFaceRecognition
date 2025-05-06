#system_metrics.py
# utils/system_metrics.py

import psutil

def get_system_metrics():
    """
    Returns a tuple of (cpu_percent, memory_percent).
    """
    return psutil.cpu_percent(interval=None), psutil.virtual_memory().percent
