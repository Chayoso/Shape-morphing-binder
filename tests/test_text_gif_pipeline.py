"""Small CPU checks for orchestration, request integrity and disposable outputs."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from physmorph import text_pipeline as T
from physmorph.text_request import MorphRequest, build_config


class PipelineTests(unittest.TestCase):
    def test_request_cannot_silently_change_material(self):
        record = build_config('A wood bunny becomes a dragon.', MorphRequest('bunny', 'dragon', 'wood-like'),
            {'response': json.dumps(dict(language='English', source=['bunny'], target=['dragon'], material=['wood']))},
            out='output/test')
        T.validate_request(record, 'output/test')
        record['material']['effective']['young'] *= 2
        with self.assertRaises(ValueError):
            T.validate_request(record, 'output/test')

    def test_complete_stages_use_safe_worker_then_keep_only_gif_and_records(self):
        from PIL import Image
        calls = []
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)/'run'
            record = build_config('A wood bunny becomes a dragon.', MorphRequest('bunny', 'dragon', 'wood-like'),
                {'response': json.dumps(dict(language='English', source=['bunny'], target=['dragon'], material=['wood']))},
                out=str(out))

            def worker(script, args, job, **options):
                calls.append((str(script), args, options))
                (job/'scratch.log').write_text('disposable')
                if options['name'] == 'language':
                    T.dump(str(out)+'_request.json', record)
                elif options['name'] == 'simulation':
                    T.dump(job/'result.json', dict(status='completed', iou=.7, chamfer=.2,
                        gates_passed=True, times=[0., .2], config=record['material']['effective'],
                        source_provenance=record['simulation'] | record['material']['effective']))
                else:
                    frames=job/'render'/'splatgpu_test';frames.mkdir(parents=True)
                    for i, color in enumerate(['white', 'gray']):
                        Image.new('RGB',(320,320),color).save(frames/f'f{i:04}.png')

            with patch.object(T, 'child', worker):
                result = T.run_text_pipeline(record['prompt'], out)
            self.assertEqual(result['status'], 'complete')
            self.assertEqual(result['stages'], ['language', 'simulation', 'render'])
            self.assertIn('--worker', calls[1][1])
            self.assertEqual(set(p.name for p in Path(folder).iterdir()), {'run_request.json', 'run.md', 'run.gif'})
            with Image.open(str(out)+'.gif') as image:
                self.assertEqual(image.n_frames, 2)
            with self.assertRaises(FileExistsError):
                T.run_text_pipeline(record['prompt'], out)

    def test_failed_stage_cleans_job_and_keeps_scalar_error(self):
        with tempfile.TemporaryDirectory() as folder:
            def fail(script, args, job, **options):
                (job/'dense.npz').write_bytes(b'temporary')
                raise RuntimeError('worker failed')
            out = Path(folder)/'run'
            with patch.object(T, 'child', fail):
                result = T.run_text_pipeline('A wood bunny becomes a dragon.', out)
            self.assertEqual(result['failed_stage'], 'language')
            self.assertEqual(result['status'], 'failed')
            self.assertEqual([p.name for p in Path(folder).iterdir()], ['run.md'])
            self.assertIn('worker failed', (Path(folder)/'run.md').read_text())


if __name__ == '__main__':
    unittest.main()
