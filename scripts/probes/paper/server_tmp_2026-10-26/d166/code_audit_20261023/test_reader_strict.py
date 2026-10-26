import importlib.util,json,os,pathlib,shutil,hashlib
import numpy as np
B=pathlib.Path('/data/relcfd/chayo/physmorph_v2');D=B/'tmp/d166/code_audit_20261023'
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
old=load('reader_old',D/'d166_read.before_strict.py');new=load('reader_new',B/'tmp/d166/d166_read.py')
os.environ['D166_CACHE_ONLY']='1';os.environ['CUDA_VISIBLE_DEVICES']=''
results=[]
def fixture(label):
 root=D/'fixtures'/f'{os.getpid()}_{label}';root.mkdir(parents=True,exist_ok=False)
 (root/'repo_d162d').mkdir();run=root/'output/gpu/d166/runs';run.mkdir(parents=True)
 out=root/'output/gpu/d166/auto/61_fixture';out.mkdir(parents=True)
 hist=[dict(frame_end=41,v_mean=.001,**{'lambda':1.}),dict(frame_end=81,v_mean=.001,settle=True)]
 record=dict(arms={'full':dict(deliver_n=81,history=hist,metrics=dict(sil_iou=.99,chamfer=.01,thin_uncovered=.01),seconds=1)})
 colour=[]
 for sc in ('coin','bunny','band'):
  for arm in ('H','Twin'):
   for seed in (97,99):
    pref=run/f'{sc}_{arm}_{seed}';pref.with_suffix('.json').write_text(json.dumps(record));pref.with_suffix('.rc').write_text('0\n')
    if sc=='band':
     end=.1 if arm=='H' else 1.
     colour.append(dict(name=pref.name,deliver_n=81,t_end=80/240,floor=0,end=dict(mis=end),timeline=[[0,1],[40/240,0],[80/240,end]]))
 for sc in ('coin','bunny'):
  data=dict(refs={'relief_s197':dict(relief=dict(A=1))},runs={f'H{s}':dict(series=[dict(raw=r,relief=dict(A=v)) for r,v in ((0,0),(40,1),(80,.9))]) for s in (97,99)})
  (out/f'read_{sc}.json').write_text(json.dumps(data))
 (out/'read_colour.jsonl').write_text(''.join(json.dumps(c)+'\n' for c in colour))
 return root,run,out
def run(m,root,run,out):
 m.B=root;m.G=root/'output/gpu';m.scenes(str(out),str(run),'fixture','H:Twin')
 return json.load(open(out/'verdict.json'))['arms']['H']
# End-to-end regression against the backed-up current reader (including its colour repair).
root,r,o=fixture('missing_seed')
for sc in ('coin','bunny','band'):
 (r/f'{sc}_H_99.json').unlink()
vold=run(old,root,r,o);vnew=run(new,root,r,o)
assert vold['pass_'] and not vnew['pass_'] and vnew['status']=='INCOMPLETE'
results.append(dict(test='missing_seed99',old_pass=vold['pass_'],new_status=vnew['status'],counts=vnew['counts']))
root,r,o=fixture('complete_exact')
v=run(new,root,r,o);assert v['pass_'] and v['complete'] and v['kept_min']==.9
assert all(d['stop_raw']==40 and d['stop_exact'] and d['stop_share']==1 for d in v['diagnostics'])
results.append(dict(test='complete_exact_frame_end41_raw40',status=v['status'],stop=v['diagnostics'][0]))
for label,mut in (
 ('nan', lambda r,o: (r/'coin_H_99.json').write_text((r/'coin_H_99.json').read_text().replace('"v_mean": 0.001','"v_mean": NaN'))),
 ('failed_rc', lambda r,o: (r/'coin_H_99.rc').write_text('1\n')),
 ('missing_rc', lambda r,o: (r/'coin_H_99.rc').unlink()),
 ('empty_relief', lambda r,o: (o/'read_coin.json').write_text('')),
 ('empty_colour', lambda r,o: (o/'read_colour.jsonl').write_text('')),
 ('missing_delivered_commit', lambda r,o: (r/'coin_H_99.json').write_text((r/'coin_H_99.json').read_text().replace('"deliver_n": 81','"deliver_n": 80'))),
 ('wrong_colour_endpoint', lambda r,o: (o/'read_colour.jsonl').write_text((o/'read_colour.jsonl').read_text().replace('"deliver_n": 81','"deliver_n": 80'))),
 ('nonfinite_relief', lambda r,o: (o/'read_coin.json').write_text((o/'read_coin.json').read_text().replace('"A": 0.9','"A": NaN'))),
):
 root,r,o=fixture(label);mut(r,o);v=run(new,root,r,o)
 assert not v['pass_'] and v['status']=='INCOMPLETE',(label,v)
 results.append(dict(test=label,status=v['status']))
root,r,o=fixture('unsampled_stop')
for sc in ('coin','bunny'):
 p=o/f'read_{sc}.json';p.write_text(p.read_text().replace('"raw": 40','"raw": 39'))
v=run(new,root,r,o)
assert not v['pass_'] and v['status']=='INCOMPLETE'
assert v['diagnostics'][0]['stop_share'] is not None and not v['diagnostics'][0]['stop_exact']
results.append(dict(test='unsampled_stop_preserves_interpolation',status=v['status'],diagnostic=v['diagnostics'][0]))
# A later live state cannot replace the selected committed state.
p=r/'coin_H_97.json';a=json.loads(p.read_text());a['arms']['full']['deliver_n']=41;p.write_text(json.dumps(a))
rec=new.run_rec(str(p)[:-5]);assert rec['stop_fe']==41 and not rec['settled']
results.append(dict(test='selected_commit_ignores_later_live_state',record=rec))
# Legacy rc_NAME status is supported and failed legacy codes rejected.
p=r/'bunny_Twin_97';p.with_suffix('.rc').unlink();legacy=p.parent/f'rc_{p.name}';legacy.write_text('0\n')
assert new.run_rec(str(p));legacy.write_text('3\n');assert new.run_rec(str(p)) is None
results.append(dict(test='legacy_rc_name_zero_and_nonzero',passed=True))
# No big/300k evidence may adopt a D167 definition.
root,r,o=fixture('missing_big');v=run(new,root,r,o)
ga=root/'output/gpu/d167/auto/82_gallery';ga.mkdir(parents=True)
(ga/'summary.md').write_text('fixture gallery')
(ga/'verdict.json').write_text(json.dumps(dict(arms={'H':dict(kind='gallery',pass_=True,held=6,n=6,rest=True)})))
target=root/'master.md';new.aggregate(str(target))
assert '- D167 (weight definition): adopt ' not in target.read_text()
results.append(dict(test='missing_300k_no_adopt',passed=True))
assert new.sample_at([0,40,80],[0,.8,.4],40)==(.8,True)
assert new.sample_at([],[],40)==(None,False)
assert new.sample_at([0,40],[1,2],41)==(None,False)
assert 'mesh=bunny' in (B/'tmp/d166/d166_read.py').read_text()
(D/'fixture_results.json').write_text(json.dumps(dict(passed=len(results),reader_sha256=hashlib.sha256((B/'tmp/d166/d166_read.py').read_bytes()).hexdigest(),results=results),indent=2)+'\n')
print(json.dumps(dict(passed=len(results),results=results),indent=2))
