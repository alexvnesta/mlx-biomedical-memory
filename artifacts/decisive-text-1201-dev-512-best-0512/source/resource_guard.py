"""Aggregate host telemetry; never stops or modifies another session."""

import re
import resource
import subprocess
import time

import mlx.core as mx


def telemetry():
    swap=subprocess.check_output(['sysctl','-n','vm.swapusage'],text=True)
    match=re.search(r'used\s*=\s*([\d.]+)M',swap)
    level=int(subprocess.check_output(['sysctl','-n','kern.memorystatus_vm_pressure_level'],text=True).strip())
    return dict(pressure_level=level,swap_used_gb=float(match[1])/1024 if match else None,
        process_peak_rss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e9,
        active_metal_gb=mx.get_active_memory()/1e9,peak_metal_gb=mx.get_peak_memory()/1e9)


class ResourceGuard:
    def __init__(self,limit_gb=27):
        mx.set_memory_limit(int(limit_gb*1e9));mx.set_cache_limit(64*1024**2)
        self.initial=telemetry();self.last=time.monotonic()

    def check(self):
        state=telemetry();delta=state['swap_used_gb']-self.initial['swap_used_gb']
        if state['pressure_level']>=4 or delta>2:
            raise RuntimeError('Host resource guard requested checkpoint/exit: '+str(state))
        return state
