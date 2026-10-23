"""Parse an English morphing description on the local GPU and save a runnable config."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
if sys.platform == 'win32':
    import torch  # load its CUDA DLLs before the native package imports CuPy

from physmorph.language_model import LocalQwen  # noqa: E402
from physmorph.material_semantics import infer_material, semantic_instructions  # noqa: E402
from physmorph.text_request import build_config, parse_response  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("prompt")
    ap.add_argument("--out", default="output/text_run", help="simulation output prefix")
    ap.add_argument("--n", type=int, default=40000)
    ap.add_argument("--seed", type=int, default=97, help="simulation seed")
    ap.add_argument('--material-mode', choices=('infer', 'preset'), default='infer',
                    help='infer E/nu ranges (default), or select a documented preset')
    ap.add_argument('--model-dir', type=Path, help='local pinned model snapshot')
    args = ap.parse_args()
    if not args.prompt.strip() or args.n <= 0:
        ap.error("Provide a nonempty English description and a positive particle count.")
    model_options = ({'instructions': semantic_instructions(), 'max_new_tokens': 768}
                     if args.material_mode == 'infer' else {})
    model = LocalQwen(model_directory=args.model_dir, **model_options)
    inference = {}
    try:
        if args.material_mode == 'infer':
            record = infer_material(args.prompt, model, n=args.n, seed=args.seed, out=args.out)
        else:
            response, inference = model(args.prompt)
            request = parse_response(response, args.prompt)
            record = build_config(args.prompt, request, inference, n=args.n, seed=args.seed, out=args.out)
    except ValueError as error:
        record = {"schema_version": 3 if args.material_mode == 'infer' else 1,
                  "prompt": args.prompt, "inference": getattr(error, 'inference', inference), "error": str(error)}
    path = Path(f"{args.out}_request.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"Saved {path}")
    if "error" in record:
        print(record["error"], file=sys.stderr)
        return 2
    print(json.dumps(record["interpretation"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
