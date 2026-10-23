"""CPU checks for the external safety and material-comparison handoff."""
from pathlib import Path
from types import SimpleNamespace as NS
import math
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.experiments import run_material_comparison as C


class ComparisonTests(unittest.TestCase):
    def test_fixed_parameters_need_no_language_model(self):
        self.assertTrue(hasattr(C, 'load_materials'), 'handoff must load bundled parameters, not a local LLM')
        data=C.load_materials()
        self.assertEqual((data['seed'],data['particles']), (97,40000))
        self.assertEqual(len(data['materials']),7)
        soft=data['materials'][0]
        self.assertAlmostEqual(soft['young_pa']/data['stress_unit_pa'],3.5417509793885875)
        self.assertIn('raw',soft,'raw model ranges and provenance must accompany selected values')
        self.assertEqual(soft['raw']['young']['range'],[1e5,1e6])
        self.assertEqual(data['selection']['young'],'geometric_midpoint')
        for material in data['materials']:
            low,high=material['raw']['young']['range']
            self.assertTrue(math.isclose(material['young_pa'],math.sqrt(low*high),rel_tol=1e-12))
            self.assertAlmostEqual(material['poisson'],sum(material['raw']['poisson']['range'])/2)
            self.assertTrue(material['source_prompt'] and material['estimator']['revision'])
        self.assertEqual([m['substeps'] for m in data['materials']], [1,1,1,1,1,1,2])

    def test_clock_uses_each_materials_actual_dt_and_holds_endpoints(self):
        self.assertTrue(hasattr(C, 'sample_times'))
        coarse=C.sample_times(120,1/240)
        fine=C.sample_times(240,1/480)
        self.assertEqual(coarse,fine)
        self.assertAlmostEqual(coarse[-1],.5)
        self.assertEqual(C.aligned_index(coarse,2),len(coarse)-1)
        self.assertEqual(C.aligned_index(coarse,.2),1)

    def test_rejected_windows_and_terminal_holds_add_no_time(self):
        history=[{'frame_end':41},{'frame_end':81,'null_commit':1},{'frame_end':81},
                 {'null_commit':1,'no_simulated_time':1}]
        self.assertEqual(C.delivered_end(history,82),80)
        self.assertEqual(C.delivered_end(history,41),40)


class SafetyTests(unittest.TestCase):
    def test_guarded_archive_wrapper_preserves_temporal_refinement(self):
        import numpy as np
        from physmorph import safe_pipeline as W
        from physmorph.prepare import Prepared
        from physmorph.mpm.state import MPMParams
        from scripts import pipeline_run as P
        import json
        import tempfile
        prep=Prepared(np.zeros((1,3),np.float32),np.zeros((1,3),np.float32),1.,1.,
                      MPMParams(dt=1/240),16,16,1.,1.)
        seen=[]
        def native():
            refined=P.prepare()
            cfg=P.run_config(None,None,refined)
            seen.append((refined.prm.dt,cfg.T))
            raise RuntimeError('preparation checked without simulation')
        request={'simulation':dict(src='assets/bunny.obj',tgt='assets/dragon.obj',n=40000,seed=97,out='unused'),
                 'material':{'effective':dict(young=100.,poisson=.3)}}
        with tempfile.TemporaryDirectory() as folder,patch.object(P,'prepare',return_value=prep), \
             patch.object(P,'run_config',return_value=P.PipelineConfig(T=20)),patch.object(P,'main',native):
            with self.assertRaisesRegex(RuntimeError,'preparation checked'):
                W.simulate(request,Path(folder),2)
            record=json.loads((Path(folder)/'result.json').read_text())
        self.assertEqual(seen,[(1/480,40)])
        self.assertEqual(record['time_resolution']['total_window_time'],40/240)

    def setUp(self):
        import numpy as np
        import torch
        import warp as wp
        from physmorph.mpm.state import MPMParams
        wp.init()
        x=np.zeros((1,3),np.float32);bad=x.copy();bad[0,0]=np.nan
        tr=NS(x=[wp.array(a,dtype=wp.vec3,device='cpu') for a in (x,bad)],
              v=[wp.zeros(1,dtype=wp.vec3,device='cpu') for _ in range(2)],
              F=[wp.array(np.eye(3,dtype=np.float32)[None],dtype=wp.mat33,device='cpu') for _ in range(2)],
              run=lambda:None)
        dc=torch.zeros(1,1,3,3)
        self.win=NS(N=1,T=1,prm=MPMParams(),tr=tr,dc_buf=dc,load=lambda *a:dc,
                    _adj=NS(traj=tr,dc=dc))
        self.events=[]

    def initialize(self,opt,*a,**kw):
        import torch
        opt.win=self.win
        opt.dFc=torch.ones(1,1,3,3)
        opt.leaves=[opt.dFc,torch.ones(1)]
        opt.mom=[torch.ones_like(t) for t in opt.leaves]
        opt.vel=[torch.ones_like(t) for t in opt.leaves]
        opt.adam_t=3;opt.tele={};opt.balancer=NS(value=2)
        opt.tgt=NS(settled_step=7,grid_ot=NS(repeat=True,warm=True,_solved='old'))

    def test_nan_trial_is_rejected_before_geometry_or_loss(self):
        from physmorph.pipeline.window import solve as S,rollout as R
        from physmorph.pipeline.window.objective import Objective
        from scripts.experiments.material_state_safety import state_context
        original=S.eval_terms
        def unsafe(*a,**kw):raise AssertionError('invalid geometry reached loss construction')
        with patch.object(S.WindowOptimizer,'__init__',self.initialize),patch.object(R,'_evaluate',unsafe):
            with state_context(self.events.append) as stats:
                opt=S.WindowOptimizer()
                e=S.eval_terms(opt.win,NS(),opt.dFc,opt.leaves[1])
                self.assertEqual(S.state_reason(e),'nonfinite')
                self.assertTrue(math.isinf(Objective.scalar(NS(),e,1.)))
                self.assertEqual(stats['invalid'],1)
        self.assertIs(S.eval_terms,original)

    def test_invalid_ad_forward_aborts_without_running_loss_and_clears_cache(self):
        import torch
        from physmorph.pipeline.window import solve as S,rollout as R
        from scripts.experiments.material_state_safety import state_context
        def unsafe(*a,**kw):raise AssertionError('invalid graph reached loss or AD')
        def native_run(opt,*a,**kw):
            opt.balancer.value=99;opt.tgt.settled_step=0
            return R._evaluate(NS(),None,None,None,self.win.dc_buf,None)
        with patch.object(S.WindowOptimizer,'__init__',self.initialize),patch.object(S.WindowOptimizer,'run',native_run),patch.object(R,'_evaluate',unsafe):
            with state_context(self.events.append) as stats:
                opt=S.WindowOptimizer();result=opt.run()
                self.assertIsNone(result.commit)
                self.assertTrue(result.stats['invalid_selection'])
                self.assertTrue(all(torch.count_nonzero(t)==0 for t in opt.leaves+opt.mom+opt.vel))
                self.assertEqual((opt.adam_t,opt.balancer.value,opt.tgt.settled_step),(0,2,7))
                self.assertIsNone(opt.win._adj)
                self.assertIsNone(opt.tgt.grid_ot._solved)
                self.assertFalse(opt.tgt.grid_ot.repeat or opt.tgt.grid_ot.warm)
                self.assertEqual((stats['invalid_graphs'],stats['aborted_windows']),(1,1))


if __name__=='__main__':unittest.main()
