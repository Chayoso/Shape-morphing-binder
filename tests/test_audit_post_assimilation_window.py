"""P335 auditor metadata/precision guards; no pipeline or GPU execution."""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.probes import audit_post_assimilation_window as audit


@pytest.fixture(autouse=True)
def isolated_audit():
    audit.checks.clear();audit.bindings.clear();audit.observations.clear()
    yield
    audit.checks.clear();audit.bindings.clear();audit.observations.clear()


def protocol_fixture(tmp_path):
    code=tmp_path/'code';root=tmp_path/'project'
    (code/'physmorph').mkdir(parents=True)
    (code/'physmorph'/'test.py').write_text('a=1\n')
    cfg=dict(T=20,iters=8,loss_res=36,render_res=64,stop_after_windows=21,
             target_reference=str(root/'target.npz'),compute_backend='cuda',device='cuda:0',
             body_ctrl=True,body_terminal_ctrl=True,layer_ctrl=True,layer_relax=True,
             lambda_auto=.5,loss_units='density')
    cfg.update({k:False for k in ('commit_pic','commit_pic_objective','shift_sub','opt_material',
        'geometric_rest','geometric_variance','grad_dump','render_F_geom','use_gauss_loss',
        'surface_gs_loss','continuity','settle_pin_kkt')})
    paths=[code/'physmorph'/'test.py',*(code/k for k in audit.EXTRAS),root/'work/p303/raw24a.json',
           root/'repro/current_pair/source_render_full_dt_iso_nn.npz',root/'target.npz']
    protocol=dict(schema='post_assimilation_window_p335_v1',N=300000,window=20,successor=21,
                  effective_config=cfg,mpm=dict(dt=audit.DT,dx=audit.DX),
                  boundary_and_coast_rule=audit.RULE,max_output_bytes=3_000_000_000,
                  bindings={str(p):{} for p in paths})
    return protocol,dict(files={k:{} for k in audit.ARCHIVES}),code,root


def test_exact_protocol_accepts_and_failure_can_remain_partial(tmp_path):
    protocol,result,code,root=protocol_fixture(tmp_path)
    assert audit.validate_protocol(protocol,result,code,root)[0]['stop_after_windows']==21
    result.update(passed=False,failure={'message':'original callback failed'},files={})
    audit.validate_protocol(protocol,result,code,root)
    audit.check('producer_failed',False,fatal=False)
    audit.check('remaining_array_metadata',True)
    assert audit.checks[-2]['passed'] is False and audit.checks[-1]['passed'] is True


@pytest.mark.parametrize('device',['cuda','cuda:0'])
def test_declared_logical_cuda_aliases_accept_same_actual_device(tmp_path,device):
    protocol,result,code,root=protocol_fixture(tmp_path)
    protocol['effective_config']['device']=device
    audit.validate_protocol(protocol,result,code,root)
    # The same predicate validates serialized owner.spec.device. Actual captured
    # metadata remains independently required to equal canonical cuda:0.
    assert audit.cuda_zero_alias(device)


@pytest.mark.parametrize('device',['cuda:1','cpu',None,0])
def test_alias_handling_does_not_relax_device_identity(device):
    assert not audit.cuda_zero_alias(device)


@pytest.mark.parametrize('field,value',[('T',1),('iters',9),('loss_res',72),('render_res',128),
    ('stop_after_windows',20),('commit_pic',True),('body_terminal_ctrl',False),
    ('lambda_auto',0),('device','cpu'),('loss_units','mass')])
def test_reject_recipe_or_scope_changes(tmp_path,field,value):
    protocol,result,code,root=protocol_fixture(tmp_path)
    protocol['effective_config'][field]=value
    with pytest.raises(AssertionError):audit.validate_protocol(protocol,result,code,root)


@pytest.mark.parametrize('operation',['drop_core','add_other','wrong_input','wrong_archive','wrong_rule'])
def test_reject_binding_membership_or_evidence_scope(tmp_path,operation):
    protocol,result,code,root=protocol_fixture(tmp_path)
    if operation=='drop_core':del protocol['bindings'][str(code/'physmorph/test.py')]
    elif operation=='add_other':protocol['bindings'][str(code/'other.py')]={}
    elif operation=='wrong_input':
        del protocol['bindings'][str(root/'repro/current_pair/source_render_full_dt_iso_nn.npz')]
        protocol['bindings'][str(root/'repro/current_pair/other.npz')]={}
    elif operation=='wrong_archive':result['files']['joint_head_not_actually_saved.npz']={}
    else:protocol['boundary_and_coast_rule']='relaxed'
    with pytest.raises(AssertionError):audit.validate_protocol(protocol,result,code,root)


def test_parsed_json_is_bound_and_later_tamper_rejected(tmp_path,monkeypatch):
    monkeypatch.setattr(audit,'BASE',tmp_path)
    path=tmp_path/'proof.json';path.write_text('{"passed":true}')
    assert audit.read_json(path)=={'passed':True}
    before=audit.bindings[str(path)]
    path.write_text('{"passed":false}')
    assert audit.identity(path)!=before
    with pytest.raises(AssertionError):audit.bind(path,before)


def encoded_owner(extra=False,bad_shape=False,reuse=False):
    ref=dict(type='tensor',key='array_0',shape=[3] if not bad_shape else [4],dtype='float32')
    tree=dict(type='dict',items=dict(version=1,spec=ref,
        observations=dict(type='dict',items={}),model=deepcopy(ref) if reuse else dict(type='dict',items={})))
    data=dict(array_0=np.arange(3,dtype=np.float32),
              manifest=np.frombuffer(json.dumps(tree).encode(),dtype=np.uint8))
    if extra:data['unused']=np.zeros(1,np.float32)
    blob=io.BytesIO();np.savez(blob,**data);blob.seek(0)
    return np.load(blob,allow_pickle=False)


def test_owner_tree_exact_members_and_upload_are_explicit():
    uploaded=[]
    with encoded_owner() as archive:
        value=audit.decode_owner(archive,lambda a:uploaded.append(a.copy()) or 'device-array')
    assert len(uploaded)==1 and value['spec']=='device-array'


def test_real_owner_archive_preserves_tuple_metadata_and_scalar_material(tmp_path):
    # Constructing/saving this fixture performs no rollout. It guards the actual
    # archive serializer, including tuple-vs-JSON-list and scalar material cases.
    from test_frozen_withdrawal_window import make_owner
    owner=make_owner()
    path=tmp_path/'owner.npz'
    owner.save(path,{'binding':'test'})
    with np.load(path,allow_pickle=False) as archive:
        decoded=audit.decode_owner(archive,lambda a:a.copy())
    assert np.array_equal(decoded['model']['coefficients'],owner.coefficients.numpy())
    assert decoded['spec']['prm']['grid_min']==owner.spec.prm.grid_min
    assert json.loads(json.dumps(decoded['spec']['prm']))['grid_min']==list(owner.spec.prm.grid_min)
    assert decoded['spec']['body_modes']==2 and decoded['observations']=={'binding':'test'}
    owner.close()


@pytest.mark.parametrize('option',['extra','bad_shape','reuse'])
def test_owner_manifest_rejects_tamper(option):
    with encoded_owner(**{option:True}) as archive:
        with pytest.raises(AssertionError):audit.decode_owner(archive,lambda a:a)


def test_array_rule_rejects_actual_outside_even_if_receipt_claims_pass():
    expected=np.array([0.,1.],np.float32)
    actual=expected.copy();actual[0]=np.float32(64*2**-23)
    row=audit.array_closure(np,'example',actual,expected,1.)
    assert row['passed'] is False and row['max_tolerance_ratio']==2.
    assert any(not item['passed'] for item in audit.checks)


def test_array_rule_uses_each_reference_scale_not_maximum():
    expected=np.array([0.,1e5],np.float32)
    actual=expected.copy();actual[0]=1e-4
    assert not audit.array_closure(np,'example',actual,expected,1.)['passed']


def test_nonfinite_failure_evidence_is_serializable_without_false_pass():
    row=audit.array_closure(np,'example',np.array([np.nan],np.float32),np.zeros(1,np.float32),1.)
    assert not row['passed'] and not row['finite']
    saved=json.loads(json.dumps(audit.json_safe(row),allow_nan=False))
    assert saved['max_abs'] is None and saved['passed'] is False


def test_scalar_closure_does_not_trust_saved_flag():
    row=dict(actual=2.,expected=1.,absolute_difference=1.,tolerance=32*2**-23,passed=True)
    audit.scalar_receipt('merit',row)
    assert audit.checks[0]['passed'] is False


def test_scalar_closure_uses_original_precision_and_absolute_reference():
    value=-2.;tol=32*2**-23*abs(value)
    row=dict(actual=value+tol,expected=value,absolute_difference=tol,tolerance=tol,passed=True)
    audit.scalar_receipt('merit',row)
    assert all(c['passed'] for c in audit.checks)
    row['tolerance']*=2
    audit.scalar_receipt('tampered',row)
    assert not audit.checks[-2]['passed']


def test_cofactor_reference_distinguishes_reflection():
    f=np.stack((np.diag([2.,3.,4.]),np.diag([-2.,3.,4.]))).astype(np.float32)
    assert np.array_equal(audit.determinant(np,f),[24.,-24.])


def test_partial_coast_failure_retains_phase_scope_without_claiming_array_replay():
    rows={name:[dict(passed=True,finite=True,native_scale=unit,max_abs=0.,max_tolerance_ratio=0.,
                    rule=audit.RULE) for _ in range(21)]
          for name,unit in (('x',audit.DX),('v',audit.DX/(20*audit.DT)),('C',12.),('F',1.))}
    rows['C'][2].update(passed=False,max_abs=.0002453,max_tolerance_ratio=5.296)
    rows['v'][10].update(passed=False,max_abs=.000022367,max_tolerance_ratio=1.542)
    result=audit.reported_coast_summary({'joint':{'independent_coast':rows}})
    assert result['passed'] is False
    assert result['fields']['C']['failed_phases']==[2]
    assert result['fields']['v']['failed_phases']==[10]
    assert result['fields']['x']['failed_phases']==[]
    assert 'not an independent numerical reproduction' in result['scope']
    assert audit.checks[-1]['passed'] is False


def test_reported_coast_rejects_missing_phase_and_changed_unit():
    rows={name:[dict(passed=True,finite=True,native_scale=unit,max_abs=0.,max_tolerance_ratio=0.,
                    rule=audit.RULE) for _ in range(21)]
          for name,unit in (('x',audit.DX),('v',audit.DX/(20*audit.DT)),('C',12.),('F',1.))}
    rows['x'].pop()
    with pytest.raises(AssertionError):audit.reported_coast_summary({'joint':{'independent_coast':rows}})
    rows['x'].append(deepcopy(rows['x'][0]));rows['C'][0]['native_scale']=120.
    with pytest.raises(AssertionError):audit.reported_coast_summary({'joint':{'independent_coast':rows}})


def test_reported_nonfinite_coast_cannot_claim_pass_and_null_remains_failure():
    rows={name:[dict(passed=True,finite=True,native_scale=unit,max_abs=0.,max_tolerance_ratio=0.,
                    rule=audit.RULE) for _ in range(21)]
          for name,unit in (('x',audit.DX),('v',audit.DX/(20*audit.DT)),('C',12.),('F',1.))}
    rows['C'][2].update(finite=False,max_abs=None,max_tolerance_ratio=None)
    with pytest.raises(AssertionError):audit.reported_coast_summary({'joint':{'independent_coast':rows}})
    rows['C'][2]['passed']=False
    result=audit.reported_coast_summary({'joint':{'independent_coast':rows}})
    assert result['passed'] is False and result['fields']['C']['failed_phases']==[2]
