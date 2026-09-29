"""Independent archive reductions only; no MPM, loss, renderer, or FD."""
from pathlib import Path
import hashlib, json, math, os, sys, time
from datetime import datetime, timezone
import numpy as host

B=Path('/data/relcfd/chayo/physmorph_v2'); W=B/'work/p303'
CODE=W/'code_production_withdrawal1'; RUN=W/'p329_live1'; OUT=W/'p329_independent_audit3.json'
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='1' and Path.cwd().resolve()==CODE
assert (CODE/'VERSION').read_text().strip()=='19e725ec97e8229cafc8a1680bdebc23d45995fe'
assert not OUT.exists()
sys.path.insert(0,str(CODE))
from physmorph.compute import cuda_execution, cuda_module
import physmorph
assert Path(physmorph.__file__).resolve().parent.parent==CODE
started=time.perf_counter(); bindings={}; checks=[]

def sha_record(path):
    path=Path(path); a=path.stat(); h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''): h.update(block)
    z=path.stat(); assert (a.st_size,a.st_mtime_ns,a.st_ino)==(z.st_size,z.st_mtime_ns,z.st_ino)
    return dict(bytes=z.st_size,mtime_ns=z.st_mtime_ns,inode=z.st_ino,sha256=h.hexdigest())

def bind(path, expected=None):
    path=Path(path); v=sha_record(path)
    if expected is not None: assert v==expected,(str(path),'identity/hash mismatch')
    bindings[str(path)]=v; return v

def read_json(path):
    v=bind(path); raw=Path(path).read_bytes(); assert hashlib.sha256(raw).hexdigest()==v['sha256']
    result=json.loads(raw); json.dumps(result,allow_nan=False); return result

def check(name, condition, **detail):
    checks.append(dict(name=name,passed=bool(condition),**detail))
    assert condition,name

def near(name, actual, expected, rel=3e-11, abs_tol=1e-13):
    a=float(actual); e=float(expected)
    check(name,math.isfinite(a) and math.isfinite(e) and math.isclose(a,e,rel_tol=rel,abs_tol=abs_tol),actual=a,expected=e)

protocol=read_json(RUN/'protocol.json'); result=read_json(RUN/'result.json'); c=result['checkpoint']
checkpoint=read_json(RUN/'checkpoint.json'); quality=read_json(W/'p329_raw_coast1.json')
memory=read_json(W/'p329_live1.memory.json')
for p,v in protocol['bindings'].items(): bind(p,v)
assert {str(p) for p in (CODE/'physmorph').rglob('*.py')}=={p for p in protocol['bindings'] if Path(p).is_relative_to(CODE/'physmorph')}
for name,v in c['sidecars'].items(): bind(RUN/name,v)
for name in ('run.render_influence.json','run.render_influence.md'): bind(RUN/name)
for name in ('p329_live1.log','p329_live1.memory.stop','p329_live1.memory.log','sample_device_memory.py','run_p329_live.sh','p329_archive_quality.py','run_p329_quality.sh','refute_p329_v3.py','launch_p329_audit3.sh'):
    bind(W/name)
for name in ('hyde06_env.sh','gpu_env.sh','cuda_python.py'): bind(CODE/'scripts/ops'/name)
for p,v in quality['bindings'].items(): bind(p,v)
check('producer_protocol_sha',result['protocol_sha256']==bindings[str(RUN/'protocol.json')]['sha256'])
check('schema',protocol['schema']=='prepared_withdrawal_p329_v1')
from dataclasses import asdict
from physmorph.pipeline.config import PipelineConfig
expanded=json.loads(json.dumps(asdict(PipelineConfig(**protocol['source_config']))))
check('sole_recipe_override_after_frozen_defaults',dict(expanded,stop_after_windows=20)==protocol['effective_config'])
check('original_pass_and_isolation',result['passed'] and result['failure'] is None and result['bindings_unchanged'] and result['outer_accepted'] and c['measurement_passed'] and c['optimizer_state_after_callback_exact'])
check('reported_return_identity',set(c['original_return_preserved'])=={'x','v','F','C'} and all(c['original_return_preserved'].values()) and set(c['committed_original_endpoint'])=={'x','F','v'} and all(c['committed_original_endpoint'].values()))
check('callback_record_preserved',all(c[k]==v for k,v in checkpoint.items()))
check('binding_digest_stable',c['merit_binding_before']==c['merit_binding_after'] and c['merit_binding_unchanged'])
hist=result['history']; actual=[r for r in hist if 'animation' in r and not r.get('held') and not r.get('c2f')]
check('20_complete_commits',len(actual)==20 and [r['animation'] for r in actual]==list(range(20)) and [r.get('frame_end') for r in actual]==list(range(21,402,20)) and all(r.get('outer_accepted')==1 and not r.get('outer_rejected') and not r.get('null_commit') for r in actual))
check('outer_record_link',all(actual[-1][k]==v for k,v in result['outer_record'].items()))
check('manual_cap_not_rest',result['termination']['reason']=='manual_window_cap' and result['termination']['individual_rest']=='not_evaluated' and result['deliver_n']==401 and result['truncation'] is None and not any(result['guards'].values()))
check('selected_inner',c['window']==20 and c['iteration']==8 and c['history']['iter']==7 and c['inner_stats']['accepted']==8 and c['inner_stats']['rejected']==0)
for group in ('original_scalar_closure','scalar_closure'):
    for key,r in c[group].items():
        tol=32*2**-23*max(abs(r['expected']),1e-12)
        near(group+'.'+key+'.error',abs(r['actual']-r['expected']),r['absolute_difference'])
        near(group+'.'+key+'.tolerance',tol,r['tolerance'])
        check(group+'.'+key+'.gate',r['passed'] and abs(r['actual']-r['expected'])<=tol)
for group in ('original_merit','joint_merit'):
    r=c[group]; check(group+'.recombine',r['merit']==r['physical']+r['lambda_render']*r['render'] and r['lambda_render']==c['lambda_render'])
check('monitor_script_sha',memory['script_sha256']==bindings[str(W/'sample_device_memory.py')]['sha256'])
samples=memory['samples']; check('monitor_valid',memory['pid']==532136 and memory['gpu']==0 and not memory['errors'] and bool(samples) and any(s['process_seen'] for s in samples))
check('monitor_maxima',memory['sampled_process_peak_MiB']==max(s['process_used_MiB'] for s in samples) and memory['sampled_device_peak_MiB']==max(s['device_used_MiB'] for s in samples))
check('producer_exit_zero',(W/'p329_live1.memory.stop').read_text().strip()=='0')
check('output_budget',sum(p.stat().st_size for p in RUN.iterdir() if p.is_file())<=3_000_000_000)
check('render_report_bound',read_json(RUN/'run.render_influence.json')==result['render_influence'])
audit_protocol=dict(start_utc=datetime.now(timezone.utc).isoformat(),producer_commit=(CODE/'VERSION').read_text().strip(),gpu=1,
    global_launch_epoch=int((B/'maintenance/last_gpu_launch_epoch').read_text()),bindings_before=bindings,
    environment={k:os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES','WARP_CACHE_PATH','CUPY_CACHE_DIR','CUDA_CACHE_PATH','XDG_CACHE_HOME','TORCH_HOME','TMPDIR','PYTHONPATH')},scope='Independent archive-only audit; no MPM, renderer, loss or FD')
with OUT.with_suffix('.protocol.json').open('x') as f:json.dump(audit_protocol,f,indent=2,allow_nan=False)

def read_arrays(path):
    with host.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}

accepted_h=read_arrays(RUN/'accepted_head.npz'); private_h=read_arrays(RUN/'private_head.npz')
joint_h=read_arrays(RUN/'joint_state.npz'); gradients_h=read_arrays(RUN/'coast_gradients.npz')
owner_h=read_arrays(RUN/'prepared_owner.npz')
def decode(value):
    if not isinstance(value,dict):return value
    if value['type']=='tensor':
        a=owner_h[value['key']]; assert list(a.shape)==value['shape'] and str(a.dtype)==value['dtype']; return a
    if value['type']=='dict':return {k:decode(v) for k,v in value['items'].items()}
    return [decode(v) for v in value['items']]
owner=decode(json.loads(owner_h['manifest'].tobytes()))
with cuda_execution('cuda:0'):
    cp=cuda_module(); cp.cuda.Device(0).use()
    accepted={k:cp.asarray(v) for k,v in accepted_h.items()}; private={k:cp.asarray(v) for k,v in private_h.items()}
    joint={k:cp.asarray(v) for k,v in joint_h.items()}; gradients={k:cp.asarray(v) for k,v in gradients_h.items()}
    for label,arrays in (('accepted',accepted),('private',private),('joint',joint),('gradients',gradients)):
        for k,v in arrays.items(): check(label+'.'+k+'.finite',bool(cp.isfinite(v).all()))
    for k,a in owner_h.items():
        if k!='manifest':check('owner.'+k+'.finite',not a.dtype.hasobject and bool(cp.isfinite(cp.asarray(a)).all()))
    T=20;N=300000;dt=1/240;dx=.3062907543956724;sp=c['spacing']
    check('native_discretization',c['N']==N and c['T']==T and c['dt']==dt and c['dx']==dx)
    for group,a,b in (('original_closure',private,accepted),('joint_closure',joint,accepted),('private_joint_closure',joint,private)):
        for key,unit in (('positions',dx),('V',dx/(T*dt)),('F',1.),('C',1/(T*dt)),*([('Fg',1.)] if group=='private_joint_closure' else [])):
            x=a[key]; y=b[key].reshape(x.shape); check(group+'.'+key+'.layout',x.dtype==y.dtype==cp.float32 and x.shape==y.shape)
            err=cp.abs(x-y);tol=cp.float32(32*2**-23)*(cp.float32(unit)+cp.abs(y));ratio=float((err/tol).max());mx=float(err.max())
            check(group+'.'+key+'.elementwise_gate',bool((err<=tol).all()),max_abs=mx,max_ratio=ratio)
            near(group+'.'+key+'.reported_max',mx,c[group][key]['max_abs'],rel=0,abs_tol=0)
            near(group+'.'+key+'.reported_ratio',ratio,c[group][key]['max_tolerance_ratio'],rel=3e-7,abs_tol=1e-9)
    pins=accepted['pins'];arrived=accepted['start_arrived'];X=joint['coast_X'];V=joint['coast_V'];C=joint['coast_C']
    check('full_coast_layout',X.shape==V.shape==(T+1,N,3) and C.shape==(T+1,N,3,3) and joint['coast_F'].shape==joint['coast_Fg'].shape==(T+1,N,9))
    check('boundary_identity',bool(cp.array_equal(X[0],joint['x']) and cp.array_equal(V[0],joint['v']) and cp.array_equal(C[0],joint['C']) and cp.array_equal(joint['coast_F'][0],joint['F']) and cp.array_equal(joint['coast_Fg'][0],joint['Fg'])))
    check('head_endpoint_identity',bool(cp.array_equal(joint['x'],joint['positions'][-1]) and cp.array_equal(joint['v'],joint['V'][-1])))
    check('pinned_paths',bool((X[:,pins]==accepted['x0'][pins][None]).all() and (V[1:,pins]==0).all() and (C[1:,pins]==0).all() and (joint['positions'][:,pins]==accepted['x0'][pins][None]).all() and (joint['V'][:,pins]==0).all()))
    det=cp.linalg.det(joint['coast_F'].reshape(T+1,N,3,3).astype(cp.float64));check('coast_F_positive',bool((det>0).all()),minimum=float(det.min()))
    near('coast_min_det_report',det.min(),c['joint_health']['coast_min_det'],rel=3e-6)
    model=owner['model'];spec=owner['spec']
    check('prepared_x0_identity',bool(cp.array_equal(cp.asarray(spec['x0']),accepted['x0'])))
    check('fixed_coefficients',bool(cp.array_equal(accepted['coefficients'],cp.asarray(model['coefficients'])) and cp.array_equal(joint['displacement'],accepted['coefficients'][:,:3]) and cp.array_equal(joint['terminal'],accepted['coefficients'][:,3:])))
    check('fixed_start_pins',bool(cp.array_equal(cp.asarray(spec['pin'])>.5,pins)))
    check('same_body_energy',bool(cp.array_equal(private['body_energy'],joint['body_energy'])))
    for name,mask in (('start_free',~pins),('start_arrived_free',~pins & arrived)):
        ids=cp.flatnonzero(mask); check(name+'.ids',bool(cp.array_equal(ids,accepted[name+'_ids'])))
        r=c['cohorts'][name];check(name+'.count',len(ids)==r['particles']); xx=X[:,mask].astype(cp.float64); vv=V[1:,mask].astype(cp.float64);d=cp.diff(xx,axis=0);lengths=cp.linalg.norm(d,axis=-1)
        motion=dict(geometric_step_mean_square=float(((d/dt)**2).sum(-1).mean()),stored_speed_mean_square=float((vv**2).sum(-1).mean()))
        for k,v in motion.items():near(name+'.'+k,v,r['losses'][k])
        near(name+'.net',cp.sqrt(((xx[-1]-xx[0])**2).sum(-1).mean())/sp,r['net_rms_sp'])
        near(name+'.path',lengths.sum(0).mean()/sp,r['path_mean_sp'])
        qr=quality['motion'][name];near(name+'.quality_geometric',motion['geometric_step_mean_square'],qr['geometric_mean_speed_square']);near(name+'.quality_net',r['net_rms_sp'],qr['net_rms_sp']);near(name+'.quality_path',r['path_mean_sp'],qr['path_mean_sp'])
        eligible=(lengths[1:]>1e-4*sp)&(lengths[:-1]>1e-4*sp);rev=eligible&((d[1:]*d[:-1]).sum(-1)<0)
        check(name+'.reversal_counts',int(eligible.sum())==qr['adjacent_active_pairs'] and int(rev.sum())==qr['reversed_pairs'])
        for loss,channels in r['gradients'].items():
            for ch,g in channels.items():
                z=gradients[name+'__'+loss+'__'+ch];zd=z.astype(cp.float64)
                near(name+'.'+loss+'.'+ch+'.norm',cp.sqrt((zd*zd).sum()),g['norm']);near(name+'.'+loss+'.'+ch+'.rms',cp.sqrt((zd*zd).mean()),g['rms']);near(name+'.'+loss+'.'+ch+'.absmax',cp.abs(z).max(),g['absmax'],rel=0,abs_tol=0)
                check(name+'.'+loss+'.'+ch+'.flags',bool(cp.isfinite(z).all())==g['finite'] and bool((z!=0).any())==g['nonzero'])
    # Independent count/projection construction, same audited CuPy fill/KDTree libraries.
    from physmorph.compute import KDTree
    from cupyx.scipy.ndimage import binary_fill_holes
    source_path=B/'repro/current_pair/source_render_full_dt_iso_nn.npz'
    with host.load(source_path,allow_pickle=False) as z:source=cp.asarray(z['src']);target=cp.asarray(z['tgt'])
    st=KDTree(source);tt=KDTree(target);s=float(cp.median(st.query(source,k=2)[0][:,1]));tn=tt.query(target,k=9)[0];ts=float(cp.median(tn[:,1]));radius=float(cp.median(tn[:,8]));extent=1.15*float(cp.linalg.norm(target.astype(cp.float32),axis=1).max())
    near('source_spacing',s,sp,rel=0,abs_tol=0);near('target_spacing',ts,quality['target_spacing'],rel=0,abs_tol=0);near('supply_radius',radius,quality['supply_radius'],rel=0,abs_tol=0)
    count=st.query_ball_point(source,2*s,return_length=True);ids=cp.flatnonzero((source[:,1]>=(source[:,1].min()+source[:,1].max())/2)&(count<.6*cp.median(count)))
    check('source_supply_cohort',len(ids)==quality['fixed_source_ids_count'] and int(pins[ids].sum())==quality['fixed_source_ids_pinned_count'] and hashlib.sha256(cp.asnumpy(ids.astype(cp.int64)).tobytes()).hexdigest()==quality['fixed_source_ids_sha256'])
    top=target[:,1]>2.3;tip=target[target[:,1].argmax()]
    views=[(2*math.pi*i/8,e) for e in (0.,.5,-.5) for i in range(8)]
    def project(x,res,az,el):
        right=cp.asarray([math.cos(az),0.,-math.sin(az)],cp.float32);up=cp.asarray([-math.sin(el)*math.sin(az),math.cos(el),-math.sin(el)*math.cos(az)],cp.float32)
        ij=cp.floor((cp.stack((x@right,x@up),1)+extent)/(2*extent)*res).astype(cp.int64);ok=(ij>=0).all(1)&(ij<res).all(1);ij=ij[ok];buf=cp.zeros(res*res,cp.int64)
        for ox in (-1,0,1):
            for oy in (-1,0,1):cp.add.at(buf,cp.clip(ij[:,0]+ox,0,res-1)*res+cp.clip(ij[:,1]+oy,0,res-1),1)
        return buf.reshape(res,res)>0,int((~ok).sum())
    for phase in (0,10,20):
        x=X[phase];qr=quality['rows'][phase];tree=KDTree(x);dist=tree.query(target)[0];counts=tree.query_ball_point(x[ids],radius,return_length=True)-1
        for key,val in dict(target_support=(dist<=2*ts).mean(),upper_target_support=(dist[top]<=2*ts).mean(),target_gap_p95_sp=cp.percentile(dist/ts,95),fixed_source_supply=counts.mean()/8,source_supply_under_half=(counts<4).mean(),tip_count=(cp.linalg.norm(x-tip,axis=1)<.25).sum()).items():near('phase'+str(phase)+'.'+key,val,qr[key])
        # Independent blocked dense distances for64 target and64 supplier queries.
        pick=cp.asarray(host.linspace(0,N-1,64,dtype=host.int64)); sid=ids[cp.asarray(host.linspace(0,len(ids)-1,64,dtype=host.int64))]
        for begin in range(0,64,8):
            td=cp.linalg.norm(target[pick[begin:begin+8],None].astype(cp.float64)-x[None].astype(cp.float64),axis=-1).min(1)
            check('phase'+str(phase)+'.brute_target'+str(begin),bool(cp.allclose(td,dist[pick[begin:begin+8]],rtol=2e-12,atol=1e-14)))
            sd=cp.linalg.norm(x[sid[begin:begin+8],None].astype(cp.float64)-x[None].astype(cp.float64),axis=-1)
            observed=(sd<=radius).sum(1)-1
            expected=counts[cp.asarray(host.linspace(0,len(ids)-1,64,dtype=host.int64))[begin:begin+8]]
            check('phase'+str(phase)+'.brute_supplier'+str(begin),bool(cp.array_equal(observed,expected)))
        for res in (128,256):
            ious=[];holes=[];extra=[];clipped=[]
            for az,el in views:
                body,cl=project(x,res,az,el);targetmask,_=project(target,res,az,el);h=int((binary_fill_holes(body)&~body).sum());th=int((binary_fill_holes(targetmask)&~targetmask).sum());u=(body|targetmask).sum();ious.append((body&targetmask).sum()/cp.maximum(u,1));holes.append(h);extra.append(max(0,h-th));clipped.append(cl)
            q=qr['projections'][str(res)];near('phase'+str(phase)+'.iou'+str(res),cp.stack(ious).mean(),q['mean_iou']);check('phase'+str(phase)+'.masks'+str(res),holes==q['per_view_body_hole_pixels'] and extra==q['per_view_extra_hole_pixels'] and max(extra)==q['max_extra_hole_pixels'] and clipped==q['per_view_clipped_particle_centers'])
    cp.cuda.get_current_stream().synchronize()

for p,v in bindings.items():assert sha_record(p)==v,('changed after audit',p)
report=dict(passed=True,completed_utc=datetime.now(timezone.utc).isoformat(),elapsed_s=time.perf_counter()-started,check_count=len(checks),checks=checks,bindings=bindings,
    producer_commit='19e725ec97e8229cafc8a1680bdebc23d45995fe',gpu=1,global_launch_epoch=int((B/'maintenance/last_gpu_launch_epoch').read_text()),
    scope='Archive reductions only. All full X/V+terminal F/C closure arrays, private Fg, full coast finite/pins/F determinants, masks/control identity, cohort motion/loss and saved gradient reductions. Raw geometry independently reconstructed only phases0/10/20, all24views128/256 and full target/support cohorts. No MPM/loss/renderer/FD or full head F/C sequence audit. Original return and optimizer isolation are corroborated metadata, not independently replayed. Full KD queries reuse frozen bounded physmorph.compute.KDTree and CuPy fill; independent blocked FP64 brute checks64 target plus64 supplier queries per selected phase. Memory is a sampled process peak, not exact high-water mark.',
    source_n=300000,T=20,dt=dt,dx=dx,raw_geometry_phases=[0,10,20],memory_peak_MiB=memory['sampled_process_peak_MiB'])
with OUT.open('x') as f:json.dump(report,f,indent=2,allow_nan=False)
print(json.dumps(dict(passed=True,checks=len(checks),output=str(OUT),elapsed_s=report['elapsed_s'])),flush=True)
