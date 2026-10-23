"""Download the pinned local language model once; inference stays offline."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from physmorph.language_model import MODEL_DIRECTORY, MODEL_ID, MODEL_REVISION


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, default=MODEL_DIRECTORY)
    args = parser.parse_args()
    from huggingface_hub import snapshot_download
    path = snapshot_download(MODEL_ID, revision=MODEL_REVISION, local_dir=args.model_dir,
                             allow_patterns=['*.json', '*.safetensors', '*.jinja', 'merges.txt', 'vocab.*'])
    print(path)


if __name__ == '__main__':
    main()
