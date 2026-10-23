"""English text to local inference, guarded native simulation and GIF."""
import json
import math
from pathlib import Path
import sys
import subprocess
import tempfile
import time

from physmorph.text_request import ROOT, build_config, parse_response
from physmorph.material_prior import build_prior_config
from physmorph.material_semantics import build_material_config
from physmorph.safe_pipeline import child, dump


def validate_request(record, out):
    if 'error' in record:
        raise ValueError(record['error'])
    options = dict(n=record['simulation']['n'], seed=record['simulation']['seed'], out=str(out))
    if record['schema_version'] == 3:
        expected = build_material_config(record['prompt'], record['inference']['semantic'],
                                         record['inference']['properties'], **options)
    elif record['schema_version'] == 2:
        expected = build_prior_config(record['prompt'], record['inference'], **options)
    elif record['schema_version'] == 1:
        expected = build_config(record['prompt'], parse_response(record['inference']['response'], record['prompt']),
                                record['inference'], **options)
    else:
        raise ValueError('Unsupported request schema version.')
    for key in ('interpretation', 'material', 'simulation'):
        if record[key] != expected[key]:
            raise ValueError(key+' differs from the recorded model response.')


def validate_simulation(simulation, request):
    actual = simulation['source_provenance']
    for key in ('src', 'tgt', 'n', 'seed'):
        if actual[key] != request['simulation'][key]:
            raise ValueError('Actual solver '+key+' differs from the request.')
    for key, value in request['material']['effective'].items():
        if simulation['config'][key] != value or actual[key] != value:
            raise ValueError('Actual material differs from the request.')
    if not all(math.isfinite(simulation[key]) for key in ('iou', 'chamfer')):
        raise ValueError('Final metrics are nonfinite.')


def make_gif(folder, path, simulation, material):
    from PIL import Image, ImageDraw
    files = sorted((folder/'render').glob('splatgpu_*/f*.png'))
    times = simulation['times']
    if len(files) != len(times) or not files:
        raise ValueError('Rendered frame/timestamp count mismatch.')
    images = []
    for file, timestamp in zip(files, times):
        with Image.open(file) as source:
            image = source.convert('RGB')
        draw = ImageDraw.Draw(image)
        e = material['selected']['young_pa']
        nu = material['selected']['poisson']
        draw.text((8, 8), f'{material["name"]}  E={e:.4g} Pa  nu={nu:.3g}', fill=(20, 20, 20))
        draw.text((8, 25), f't={timestamp:.2f}  Final IoU {simulation["iou"]:.4f}', fill=(20, 20, 20))
        images.append(image)
    # 2x playback of the solver clock; display holds add no simulated time.
    durations = [max(10, round((b-a)*500)) for a, b in zip(times, times[1:])] + [2500]
    durations[0] += 1000
    images[0].save(path, save_all=True, append_images=images[1:], duration=durations, loop=0)
    return dict(frames=len(images), solver_times=times, playback_speed=2,
                initial_pause_ms=1000, final_pause_ms=2500, views=[35])


def save_summary(path, result):
    simulation = result.get('simulation', {})
    material = result.get('material', {})
    lines = ['# Text-to-material run', '', 'Status: '+result['status'],
             'Prompt: '+result['prompt'], '',
             '| Material | E Pa | nu | IoU | Chamfer | Native gates |',
             '|---|---:|---:|---:|---:|---|',
             '| '+' | '.join(str(v) for v in (material.get('name', ''),
                 material.get('selected', {}).get('young_pa', ''),
                 material.get('selected', {}).get('poisson', ''), simulation.get('iou', ''),
                 simulation.get('chamfer', ''), simulation.get('gates_passed', ''))) + ' |', '',
             'Model estimates are uncalibrated. Common E unit conversion; nu unchanged. '
             'Original native model/optimizer, with external state checks and rollback. '
             'Safety gates do not certify target fit. Time is the solver clock, not calibrated SI time.', '',
             '```json', json.dumps(result, ensure_ascii=False, indent=2), '```', '']
    path.write_text('\n'.join(lines), encoding='utf8')


def run_text_pipeline(prompt, out, *, n=40000, seed=97, material_mode='infer',
                      res=320, model_dir=None, substeps=1, python=sys.executable):
    if not prompt or not prompt.strip():
        raise ValueError('Provide a nonempty English description.')
    if n <= 0 or res <= 0 or res % 2 or substeps not in (1, 2):
        raise ValueError('Use positive N, an even render resolution and substeps 1 or 2.')
    if material_mode not in ('infer', 'preset'):
        raise ValueError('material_mode must be infer or preset.')
    out = Path(out).resolve()
    files = {key: Path(str(out)+suffix) for key, suffix in
             [('request', '_request.json'), ('summary', '.md'), ('gif', '.gif')]}
    for file in files.values():
        if file.exists():
            raise FileExistsError('Use a new output prefix; already exists: '+str(file))
    out.parent.mkdir(parents=True, exist_ok=True)
    result = dict(status='running', prompt=prompt, files={k: str(v) for k, v in files.items()}, stages=[])
    stage = 'language'
    worker = ROOT/'scripts/text_morph.py'
    started = time.perf_counter()
    try:
        with tempfile.TemporaryDirectory(prefix=out.name+'_job_', dir=out.parent) as folder:
            job = Path(folder)
            try:
                args = [prompt, '--out', str(out), '--n', str(n), '--seed', str(seed),
                        '--material-mode', material_mode]
                if model_dir is not None:
                    args += ['--model-dir', str(Path(model_dir).resolve())]
                print('[language] local Qwen inference', flush=True)
                child(ROOT/'scripts/prompt_config.py', args, job, python=python, name=stage, seconds=900)
                request = json.loads(files['request'].read_text(encoding='utf8'))
                validate_request(request, out)
                result.update(interpretation=request['interpretation'], material=request['material'])
                result['stages'].append(stage)
                stage = 'simulation'
                print('[simulation] native pipeline with safety checks', flush=True)
                child(worker, ['--worker', stage, '--job', str(job), '--request', str(files['request']),
                               '--substeps', str(substeps)], job, python=python, name=stage)
                simulation = json.loads((job/'result.json').read_text(encoding='utf8'))
                result['simulation'] = simulation
                validate_simulation(simulation, request)
                result['stages'].append(stage)
                stage = 'render'
                print('[render] native shader to GIF', flush=True)
                child(worker, ['--worker', stage, '--job', str(job), '--res', str(res)], job,
                      python=python, name=stage, render_stage=True, seconds=1800)
                result['render'] = make_gif(job, files['gif'], simulation, request['material'])
                result['stages'].append(stage)
                result['status'] = 'complete' if simulation['gates_passed'] else 'completed_with_failed_gates'
            finally:
                if (job/'result.json').exists() and 'simulation' not in result:
                    result['simulation'] = json.loads((job/'result.json').read_text(encoding='utf8'))
    except KeyboardInterrupt:
        result.update(status='interrupted', failed_stage=stage)
        raise
    except (ValueError, TypeError, KeyError, OSError, RuntimeError, TimeoutError, subprocess.SubprocessError) as error:
        result.update(status='failed', failed_stage=stage, error=str(error))
    finally:
        result['seconds'] = time.perf_counter()-started
        save_summary(files['summary'], result)
    return result
