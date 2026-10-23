"""English prompt -> local LLM -> guarded native simulation -> GIF."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if sys.platform == 'win32':
    import torch  # load its CUDA DLLs before the native package imports CuPy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prompt', nargs='?')
    parser.add_argument('--out', default='output/text_run', help='new output prefix')
    parser.add_argument('--n', type=int, default=40000)
    parser.add_argument('--seed', type=int, default=97)
    parser.add_argument('--material-mode', choices=('infer', 'preset'), default='infer')
    parser.add_argument('--model-dir', type=Path, help='local model snapshot; see scripts/download_language_model.py')
    parser.add_argument('--res', type=int, default=320)
    parser.add_argument('--substeps', type=int, choices=(1, 2), default=1,
                        help='explicit temporal refinement; steel comparison uses 2')
    parser.add_argument('--worker', choices=('simulation', 'render'), help=argparse.SUPPRESS)
    parser.add_argument('--job', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--request', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        from physmorph.safe_pipeline import render, simulate
        if args.worker == 'simulation':
            import json
            simulate(json.loads(args.request.read_text(encoding='utf8')), args.job, args.substeps)
        else:
            render(args.job, args.res)
        return 0
    from physmorph.text_pipeline import run_text_pipeline
    try:
        result = run_text_pipeline(args.prompt, args.out, n=args.n, seed=args.seed,
            material_mode=args.material_mode, model_dir=args.model_dir, res=args.res, substeps=args.substeps)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(result['status']+': '+result['files']['summary'])
    if result['status'] == 'failed':
        print(result['error'], file=sys.stderr)
        return 1
    simulation = result['simulation']
    print(f'IoU={simulation["iou"]:.4f}, Chamfer={simulation["chamfer"]:.4f}; '
          f'native gates={simulation["gates_passed"]}')
    print('GIF: '+result['files']['gif'])
    return 0 if simulation['gates_passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
