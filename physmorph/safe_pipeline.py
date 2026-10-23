"""External guarded native execution and disposable render workers."""
from contextlib import nullcontext
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
ARM = 'render_full_dt_iso_nn'
GATES = ('G2_guards', 'G3_rest', 'G4_holes_abs', 'G4_ejection')
INTERVAL = .2


def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    return value


def dump(path, value):
    Path(path).write_text(json.dumps(clean(value), ensure_ascii=False, indent=2)+'\n', encoding='utf8')


def sample_steps(end, stride=48):
    return sorted(set(range(0, end+1, stride)) | {end})


def delivered_end(history, deliver_n):
    return max([1, *(h['frame_end'] for h in history if h.get('frame_end') is not None
        and not h.get('null_commit') and not h.get('no_simulated_time') and h['frame_end'] <= deliver_n)])-1


def simulate(request, job, substeps=1):
    # Torch first also resolves the CUDA DLL search order on Windows.
    import torch  # noqa: F401
    import numpy as np
    from scripts import pipeline_run as P
    from scripts.experiments.material_state_safety import state_context
    from scripts.experiments.material_time_resolution import refine_time
    from physmorph.text_request import pipeline_argv
    captured, steps, events, safety, resolution = {}, [], [], {}, {}
    original_run = P.run_pipeline
    real_stack, real_save = np.stack, np.savez_compressed

    def run(*args, **kwargs):
        result = original_run(*args, **kwargs)
        captured['result'] = result
        return result

    def prepare(*args, **kwargs):
        prep = original_prepare(*args, **kwargs)
        captured['prepared'] = dict(mpm=dataclasses.asdict(prep.prm), arrays={})
        for name, value in [('src', prep.src), ('tgt', prep.tgt)]:
            value = np.ascontiguousarray(value)
            captured['prepared']['arrays'][name] = dict(shape=list(value.shape),
                sha256=hashlib.sha256(value.tobytes()).hexdigest())
        return prep

    class ArchiveOnly:
        def __getattr__(self, name):
            return getattr(np, name)

        def stack(self, arrays, *args, **kwargs):
            result = captured.get('result')
            if result is None:
                return real_stack(arrays, *args, **kwargs)
            if arrays is result['frames'].x:
                dt = captured['prepared']['mpm']['dt']
                steps[:] = sample_steps(delivered_end(result['history'], result['deliver_n']),
                                        max(1, round(INTERVAL/dt)))
                return real_stack([arrays[i] for i in steps])
            return np.empty((0,), np.float32)  # native evaluation has finished; omit archival F/J

        def savez(self, path, **values):
            real_save(job/'visual.npz', frames=values['frames'], tgt=values['tgt'],
                      deliver_n=np.int64(len(values['frames'])))

    def note(event):
        if len(events) < 3 and event.get('kind') != 'safety_summary':
            events.append({key: value for key, value in event.items()
                           if key not in ('nearby_states', 'timesteps')})

    local = {**request, 'simulation': {**request['simulation'], 'out': str(job/'native')}}
    argv = ['pipeline_run.py', *pipeline_argv(local)]
    result = dict(status='failed')
    try:
        with (refine_time(substeps) if substeps > 1 else nullcontext(dict(substeps=1))) as resolution:
            # Preserve the optional refinement wrapper when decorating prepare.
            original_prepare = P.prepare
            with state_context(note) as safety, patch.object(P, 'np', ArchiveOnly()), \
                 patch.object(P, 'run_pipeline', run), patch.object(P, 'prepare', prepare), \
                 patch.object(sys, 'argv', argv):
                P.main()
        native = json.loads((job/'native.json').read_text(encoding='utf8'))
        arm = native['arms'][ARM]
        dt = captured['prepared']['mpm']['dt']
        result.update(status='completed', iou=arm['metrics']['sil_iou'], chamfer=arm['metrics']['chamfer'],
            gates_passed=all(arm['gates'][gate] for gate in GATES), gates=arm['gates'],
            metrics=arm['metrics'], config=arm['config'], history=arm['history'],
            prepared=captured['prepared'], source_provenance=native['provenance'],
            raw_steps=steps, times=[step*dt for step in steps], seconds=arm['seconds'])
    except BaseException as error:
        result['error'] = type(error).__name__+': '+str(error)
        raise
    finally:
        result.update(state_safety=dict(safety), first_invalid=events, time_resolution=resolution)
        dump(job/'result.json', result)


def render(job, res=320):
    import torch  # noqa: F401
    import runpy
    from PIL import ImageDraw
    original = ImageDraw.ImageDraw.text

    def text(self, xy, value, *args, **kwargs):
        if 'splat frame' not in str(value):
            return original(self, xy, value, *args, **kwargs)

    argv = ['render_splat_gpu.py', str(job/'visual.npz'), str(job/'visual.mp4'),
            '--res', str(res), '--stride', '1', '--hold', '0', '--views', '35']
    with patch.object(sys, 'argv', argv), patch.object(ImageDraw.ImageDraw, 'text', text):
        runpy.run_path(str(ROOT/'scripts/render_splat_gpu.py'), run_name='__main__')


def child(script, arguments, job, *, python=sys.executable, seconds=3600, name='worker', render_stage=False):
    env = dict(os.environ, CUDA_MODULE_LOADING='EAGER', PYTHONFAULTHANDLER='1',
               PHYSMORPH_SAMPLE_CACHE='0')
    # Keep every renderer tempfile inside the disposable job.
    if render_stage:
        folder = job/'render'
        folder.mkdir()
        env.update(TEMP=str(folder), TMP=str(folder), TMPDIR=str(folder))
    log = job/(name+'.log')
    with log.open('w', encoding='utf8') as stream:
        proc = subprocess.Popen([python, '-B', str(script), *arguments], cwd=ROOT, env=env,
            stdout=stream, stderr=subprocess.STDOUT, start_new_session=os.name != 'nt')
        try:
            code = proc.wait(timeout=seconds)
        except BaseException:
            if os.name == 'nt':
                subprocess.run(['taskkill', '/PID', str(proc.pid), '/T', '/F'],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            raise
    if code:
        raise RuntimeError(name+' exit '+str(code)+': '+log.read_text(encoding='utf8', errors='replace')[-1800:])
