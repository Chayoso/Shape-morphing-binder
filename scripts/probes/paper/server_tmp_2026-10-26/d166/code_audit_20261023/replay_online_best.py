"""CPU-only history audit; no displacement reconstruction or new selection decisions."""
from pathlib import Path
import ast,json,types,hashlib,math
import numpy as np
B=Path('/data/relcfd/chayo/physmorph_v2');D=B/'tmp/d166/code_audit_20261023'
source=B/'repo_d166/physmorph/pipeline/run/selection.py'
tree=ast.parse(source.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Selection')
ns={'np':np,'PipelineConfig':object};exec(compile(ast.Module(body=[cls],type_ignores=[]),str(source),'exec'),ns)
class NoDisplacement:
 def clone(self):return None
cases=['d167/runs/coin_TP_97','d167/runs/bunny_TP_97','d167/runs/band_TP_97','d145t_b1/dragon300k_L97','d161/runs40/coin40k_full_97']
results=[]
for case in cases:
 p=B/'output/gpu'/f'{case}.json'
 if not p.exists():
  results.append(dict(case=case,status='MISSING'));continue
 data=json.load(open(p));a=next(iter(data['arms'].values()));cfg=types.SimpleNamespace(**a['config'])
 required=('tol','patience','anneal_stale','outer_merit_tol','outer_reversal_cos','outer_reversal_gain','reject_stop')
 if any(not hasattr(cfg,k) for k in required):
  results.append(dict(case=case,status='INSUFFICIENT_CONFIG'));continue
 sel=ns['Selection'](cfg);epoch=None;accepted=[];events=[];agreement=[];gaps=[];all_improved_diff=[];counts=dict(judged=0,accepted=0,outer_rejected=0,null=0,settle=0,unusable=0)
 for index,h in enumerate(a['history']):
  if h.get('settle'):counts['settle']+=1;continue
  if 'selection_epoch' not in h or not math.isfinite(h.get('selection_merit',math.nan)):
   counts['null' if h.get('null_commit') else 'unusable']+=1
   # Best/last/render are untouched by null(), whose other counters are not audited here.
   if h.get('null_commit'):sel.null()
   continue
  ep=h['selection_epoch']
  if ep!=epoch:
   sel.new_epoch(count=False);sel.epoch=ep;epoch=ep;accepted=[]
  q=float(h['selection_merit']);lam=float(h.get('lambda') or 0.)
  render=float(h.get('d_render') or 0.)+cfg.w_pbr*float(h.get('d_pbr') or 0.)
  rec=dict(h);sel.check_lambda(rec,lam,render);counts['judged']+=1
  before_best=float(sel.best)
  recomputed_improved=bool(np.isfinite(q) and (sel.last is None or q<sel.best-cfg.tol*abs(sel.best)))
  if 'improved' in h:agreement.append(bool(h['improved'])==recomputed_improved)
  if accepted:
   scores=[r['q']+(lam-r['lambda_'])*r['render'] for r in accepted]
   true_best=float(min(scores));gap=before_best-true_best
   tolerance=cfg.tol*abs(before_best)
   gaps.append((max(0.,gap),max(0.,gap)/max(abs(true_best),1e-30)))
   true_improved=q<true_best-cfg.tol*abs(true_best)
   if recomputed_improved!=true_improved:
    all_improved_diff.append(dict(animation=h.get("animation"),gap=gap,online_improved=recomputed_improved,envelope_improved=true_improved))
   if gap>max(tolerance,1e-10):
    true_improved=q<true_best-cfg.tol*abs(true_best)
    events.append(dict(history_index=index,animation=h.get('animation'),epoch=epoch,lambda_=lam,
     tracked_best=before_best,all_accepted_min=true_best,gap=gap,relative_gap=gap/max(abs(true_best),1e-30),
     candidate=q,online_improved=recomputed_improved,envelope_improved=true_improved,
     changed_improved=recomputed_improved!=true_improved,outer_rejected=bool(h.get('outer_rejected')),
     earliest_better_animation=accepted[int(np.argmin(scores))]['animation']))
  if h.get('outer_rejected') or h.get('null_commit'):
   counts['outer_rejected']+=1
   # judge/rejected require per-particle displacement/replay tolerance absent from records.
   # They do not change best/last/best_render; preserve actual accept/reject lineage.
   continue
  if not h.get('frame_end'):
   counts['unusable']+=1;continue
  counts['accepted']+=1
  sel._score=float(h.get('outer_merit',q));sel._score_phys=float(h.get('outer_merit_phys',0))
  sel.accepted(rec,int(h['animation']),NoDisplacement(),recomputed_improved)
  accepted.append(dict(q=q,lambda_=lam,render=render,animation=h['animation']))
 result=dict(case=case,status='BEST_LEDGER_REPLAY_ONLY',history_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
  recorded_code_hash=data.get('provenance',{}).get('code_hash'),tol=cfg.tol,lambda_ema=a['config'].get('lambda_ema'),
  render_weight_scale=a['config'].get('render_weight_scale'),counts=counts,
  improved_records_checked=len(agreement),improved_record_disagreements=sum(not x for x in agreement),
  above_tol_mismatch_count=len(events),changed_improved_count=sum(e['changed_improved'] for e in events),
  all_envelope_improved_difference_count=len(all_improved_diff),all_envelope_improved_differences=all_improved_diff,
  max_all_positive_gap=max((g[0] for g in gaps),default=0),max_all_relative_gap=max((g[1] for g in gaps),default=0),
  max_absolute_gap=max((e['gap'] for e in events),default=0),max_relative_gap=max((e['relative_gap'] for e in events),default=0),
  examples=sorted(events,key=lambda e:e['gap'],reverse=True)[:3])
 results.append(result)
out=dict(selection_source=str(source),selection_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
 scope='Uses actual check_lambda/accepted with recorded tol and epoch resets. Recorded outer-rejected/null lineage retained; never selects a counterfactual path. Displacement-dependent judge/rejection, downstream stopping changes and render outcomes not replayed.',
 results=results)
(D/'online_best_history_replay.json').write_text(json.dumps(out,indent=2)+'\n')
for r in results:
 print(r['case'],r['status'],'counts',r.get('counts'),'above_tol',r.get('above_tol_mismatch_count'),
  'changed_improved',r.get('changed_improved_count'),'maxgap',r.get('max_absolute_gap'),
  'maxrel',r.get('max_relative_gap'),'recorddisagreements',r.get('improved_record_disagreements'))
