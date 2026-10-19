"""Confirm that training changed the intended adapter and codec tensors only."""
import hashlib
import json
from pathlib import Path
import mlx.core as mx


def main():
    results={}
    for mode in ['text','memory']:
        for seed in [1201,1202]:
            run=Path(f'runs/decisive-{mode}-{seed}');initial=mx.load(str(run/'initial.safetensors'));final=mx.load(str(run/'final-1024.safetensors'))
            assert set(initial)==set(final)
            changes={name:float(mx.max(mx.abs(final[name].astype(mx.float32)-value.astype(mx.float32))).item()) for name,value in initial.items()}
            assert all(name.startswith('memory.') or '.lora_' in name for name in changes)
            assert any(value>0 for name,value in changes.items() if '.lora_' in name)
            result=dict(max_absolute_changes=changes,initial_sha256=hashlib.sha256((run/'initial.safetensors').read_bytes()).hexdigest(),
                final_sha256=hashlib.sha256((run/'final-1024.safetensors').read_bytes()).hexdigest())
            if mode=='memory':
                for field in range(7):assert changes[f'memory.tables.{field}.weight']>0
                assert changes['memory.projection.weight']>0
                # F10/F11 are deliberately absent from the ten-record training banks.
                table='memory.tables.0.weight'
                assert mx.array_equal(initial[table][10:],final[table][10:]).item()
                result.update(initial_gate=float(initial['memory.gate'].item()),final_gate=float(final['memory.gate'].item()),
                    record_id_rows_10_11_unchanged=True)
            results[f'{mode}-{seed}']=result;del initial,final;mx.clear_cache()
    Path('runs/decisive-parameter-audit.json').write_text(json.dumps(dict(completed=True,results=results,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')
    Path('runs/decisive-parameter-audit-source.py').write_bytes(Path(__file__).read_bytes())
    print('Adapter and codec parameter-change audit passed')


if __name__=='__main__':main()
