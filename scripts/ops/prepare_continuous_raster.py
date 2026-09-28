"""Copy and patch the audited diff_gauss source into an isolated build directory.

Never installs into or modifies the original environment. Build the returned
directory with setup.py build_ext --inplace, then prepend that directory to
PYTHONPATH. The original license remains in the copied tree. This experimental
backend removes nonzero alpha cutoffs and uses log-transmittance to retain the
reverse recurrence even through opaque stacks; it does not remove depth sorting.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def tree_digest(root):
    files = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(root.rglob('*')) if p.is_file()
             and not any(part in ('.git', 'build', '__pycache__') or part.endswith('.egg-info')
                         for part in p.relative_to(root).parts) and p.suffix != '.so'}
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(), len(files)


def replace(root, name, old, new, count=1):
    path = root/name
    text = path.read_text()
    if text.count(old) != count:
        raise ValueError(f'{name}: expected {count} occurrences of {old!r}')
    path.write_text(text.replace(old, new), newline='\n')


def prepare(source, destination):
    manifest = json.loads(Path(__file__).with_name('continuous_raster_source.json').read_text())
    tree = manifest.pop('_input_tree')
    if tree_digest(source) != (tree['sha256'], tree['files']):
        raise ValueError('Unaudited full dependency tree, including compile headers')
    for name, digest in manifest.items():
        if hashlib.sha256((source/name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Unaudited dependency revision: {name}')
    if destination.exists() or source == destination or source in destination.parents:
        raise ValueError('Use a new isolated destination outside the source tree')
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns(
        '*.so', '__pycache__', '.git', 'build', '*.egg-info'))
    if tree_digest(destination) != (tree['sha256'], tree['files']):
        raise ValueError('Copied dependency inputs changed')
    (destination/'diff_gauss').rename(destination/'physmorph_diff_gauss')
    replace(destination, 'setup.py', 'diff_gauss', 'physmorph_diff_gauss', 3)
    fw = 'cuda_rasterizer/forward.cu'
    bw = 'cuda_rasterizer/backward.cu'
    impl = 'cuda_rasterizer/rasterizer_impl.cu'
    replace(destination, fw, '  float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));',
        '  if (!(opacities[idx] > 1.f / 255.f)) return;\n'
        '  float my_radius = ceil(sqrt(2.f * log(opacities[idx] * 255.f)) * sqrt(max(lambda1, lambda2))) + 1.f;')
    replace(destination, fw,
        '      float alpha = min(0.99f, con_o.w * exp(power));\n      if (alpha < 1.0f / 255.0f)',
        '      float alpha = min(0.99f, max(0.f, con_o.w * exp(power) - 1.f / 255.f));\n      if (alpha <= 0.f)')
    replace(destination, fw,
        '      float test_T = T * (1 - alpha);\n      if (test_T < 0.0001f)\n      {\n        done = true;\n        continue;\n      }',
        '      log_T += log1p(-double(alpha));\n      float test_T = float(exp(log_T));')
    replace(destination, fw, '  float T = 1.0f;', '  float T = 1.0f;\n  double log_T = 0.0;')
    replace(destination, fw, '    out_alpha[pix_id] = 1 - T;',
        '    out_alpha[pix_id] = 1 - T;\n    out_log_T[pix_id] = log_T;')
    replace(destination, fw, '  float* __restrict__ out_alpha,',
        '  float* __restrict__ out_alpha,\n  double* __restrict__ out_log_T,')
    replace(destination, fw, '  float* out_alpha,', '  float* out_alpha,\n  double* out_log_T,')
    replace(destination, fw, '    out_alpha,\n', '    out_alpha,\n    out_log_T,\n')
    replace(destination, 'cuda_rasterizer/forward.h', '    float* out_alpha,',
        '    float* out_alpha,\n    double* out_log_T,')
    replace(destination, 'cuda_rasterizer/rasterizer_impl.h', '    uint32_t* n_contrib;',
        '    uint32_t* n_contrib;\n    double* log_T;')
    replace(destination, impl, '  obtain(chunk, img.n_contrib, N, 128);',
        '  obtain(chunk, img.n_contrib, N, 128);\n  obtain(chunk, img.log_T, N, 128);')
    replace(destination, impl, '    out_alpha,\n    imgState.n_contrib,',
        '    out_alpha,\n    imgState.log_T,\n    imgState.n_contrib,')
    replace(destination, impl, '    accum_alphas,\n    imgState.n_contrib,',
        '    imgState.log_T,\n    imgState.n_contrib,')
    for file in (bw, 'cuda_rasterizer/backward.h'):
        path = destination/file
        text = path.read_text().replace('const float* __restrict__ accum_alphas', 'const double* __restrict__ final_log_T')
        text = text.replace('const float* accum_alphas', 'const double* final_log_T')
        text = text.replace('    accum_alphas,', '    final_log_T,')
        path.write_text(text, newline='\n')
    replace(destination, bw,
        '  const float T_final = inside ? (1 - accum_alphas[pix_id]) : 0;\n  float T = T_final;',
        '  double log_T = inside ? final_log_T[pix_id] : 0.;\n'
        '  const float T_final = inside ? float(exp(log_T)) : 0.f;\n  float T = T_final;')
    replace(destination, bw,
        '      const float alpha = min(0.99f, con_o.w * G);\n      if (alpha < 1.0f / 255.0f)',
        '      const float raw_alpha = con_o.w * G - 1.f / 255.f;\n'
        '      const float alpha = min(0.99f, max(0.f, raw_alpha));\n      if (alpha <= 0.f)')
    replace(destination, bw, '      T = T / (1.f - alpha);',
        '      log_T -= log1p(-double(alpha));\n      T = float(exp(log_T));')
    replace(destination, bw, '      const float dL_dG = con_o.w * dL_dalpha;',
        '      if (raw_alpha >= 0.99f) dL_dalpha = 0.f;\n      const float dL_dG = con_o.w * dL_dalpha;')
    replace(destination, bw, '(2 * h_x * t.x)', '((1.f + x_grad_mul) * h_x * t.x)')
    replace(destination, bw, '(2 * h_y * t.y)', '((1.f + y_grad_mul) * h_y * t.y)')
    aux = 'cuda_rasterizer/auxiliary.h'
    for axis, block in (('x', 'X'), ('y', 'Y')):
        replace(destination, aux, f'(int)((p.{axis} - max_radius) / BLOCK_{block})',
                f'(int)floorf((p.{axis} - max_radius) / BLOCK_{block})')
        replace(destination, aux, f'(int)((p.{axis} + max_radius + BLOCK_{block} - 1) / BLOCK_{block})',
                f'((int)floorf((p.{axis} + max_radius) / BLOCK_{block}) + 1)')
    # The inherited C++ kernels use stream 0. Fail explicitly instead of racing
    # a caller's side stream or allocating scratch on an unrelated device.
    package = 'physmorph_diff_gauss/__init__.py'
    replace(destination, package, 'from . import _C', 'from . import _C\n\n'
        'def _check_stream(x):\n'
        '    if x.device.type != "cuda" or x.device.index != torch.cuda.current_device():\n'
        '        raise RuntimeError("continuous raster requires the current CUDA device")\n'
        '    if torch.cuda.current_stream(x.device) != torch.cuda.default_stream(x.device):\n'
        '        raise RuntimeError("continuous raster requires the default CUDA stream")\n')
    replace(destination, package, '        # Invoke C++/CUDA rasterizer',
        '        _check_stream(means3D)\n        # Invoke C++/CUDA rasterizer')
    replace(destination, package, '        # Compute gradients for relevant tensors by invoking backward method',
        '        _check_stream(means3D)\n        # Compute gradients for relevant tensors by invoking backward method')
    replace(destination, package, '            visible = _C.mark_visible(',
        '            _check_stream(positions)\n            visible = _C.mark_visible(')
    evidence = dict(source_files=manifest, input_tree=tree,
                    patch_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    mode='shifted_alpha_tau_1_over_255_full_log_transmittance',
                    files={str(p.relative_to(destination)): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in sorted(destination.rglob('*')) if p.is_file()})
    (destination/'physmorph_build.json').write_text(json.dumps(evidence, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--destination', required=True, type=Path)
    args = parser.parse_args()
    prepare(args.source.resolve(), args.destination.resolve())
