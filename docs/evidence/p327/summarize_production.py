"""Reduce completed JSON evidence only; no geometry or simulation is evaluated."""
import hashlib
import json
from pathlib import Path

ROOT = Path('C:/dev/physmorph_runtime/p303/p327_evidence')
inputs = {}


def read(name):
    path = ROOT / name
    raw = path.read_bytes()
    inputs[name] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


q = read('p327_quality1.json')
p = read('p327_phase1.json')
result = dict(discretization=dict(N=q['n'], T=q['T'], mpm=q['mpm'],
              loss_res=q['loss_res'], source_spacing_wu=q['native_spacing']),
              scope='Descriptive pair; pre-exposure trajectory nonidentity prevents sole-causal attribution',
              accepted_interval=p['accepted_interval'], arms={}, cohorts={})
keys = ('commit', 'frame', 'sil_iou', 'target_near_frac', 'top_target_near_frac',
        'top_density', 'top_under_half', 'tip_n', 'pinned_frac')
for label, arm in (('baseline', 'legacy'), ('candidate', 'retained')):
    a = read(f'p327_{arm}1.fragment_activity.json')
    t = read(f'p327_{arm}1.rest_trace.json')
    r = read(f'p327_{arm}1.render_influence.json')
    m = read(f'{arm}_motion.json')
    s = read(f'{arm}_shape.json')
    changed = [f for f in a['forwards'] if sum(f['cohorts']['all']['different_from_final'])]
    row = q[label]
    shape = {}
    for res in s['resolutions']:
        candidates = [(count, sample['kind'], sample['archive_frame'], view)
                      for sample in s['samples']
                      for view, count in enumerate(sample['views'][str(res)]['hole_pixels_outside_target_holes'])]
        maximum = max(candidates, key=lambda item: item[0])
        shape[str(res)] = dict(max_extra_projected_hole_pixels=maximum[0], kind=maximum[1],
                              archive_frame=maximum[2], view=maximum[3],
                              nonzero_view_samples=sum(item[0] > 0 for item in candidates),
                              total_view_samples=len(candidates))
    final_free = m['windows'][-1]['cohorts']['final_free']
    result['arms'][arm] = dict(attempts=row['attempts'], commits=row['commits'],
        seconds=row['seconds'], min_accepted_detF=row['min_accepted_detF'],
        guards=row['guards'], termination=t['termination'],
        actual_frames=row['delivery_scope']['actual_archive_frames'],
        held_rows=row['delivery_scope']['held_suffix_rows'],
        own_endpoint={k: row['actual_endpoint'][k] for k in keys},
        common_endpoint={k: row['curve'][29][k] for k in keys},
        activity=dict(models=len(a['models']), forwards=len(a['forwards']),
            changed_forwards=len(changed), attempts_zero_based=sorted({f['attempt'] for f in changed}),
            max_particle_time_differences=max((sum(f['cohorts']['all']['different_from_final']) for f in changed), default=0)),
        own_last_window_final_free=dict(particles=final_free['particles'],
            step_rms_wu=final_free['step_rms_wu']['rms'],
            terminal_geometric_rms_wu_s=final_free['raw_terminal_geometric_speed_wu_s']['rms'],
            terminal_stored_rms_wu_s=final_free['optimizer_terminal_stored_speed_wu_s']['rms'],
            endpoint_correction_rms_wu=final_free['endpoint_correction_wu']['rms']),
        shape=dict(samples=len(s['samples']), resolutions=shape,
                   scope='All sampled archive and raw endpoint projections; not 3D watertightness or 4K rendering'),
        rendering=r,
        phase=dict(final_phase=p[label]['final_phase'], reversal_groups=p[label]['reversal_groups']))
for name in ('source_upper_surface', 'common_endpoint_free_both'):
    result['cohorts'][name] = q['cohorts'][name]
result['pre_exposure_position_difference_sp'] = [
    dict(commit=row['commit'], **row['position_difference_sp'])
    for row in q['equal_accepted_commits'] if row['commit'] <= 10]
result['input_sha256'] = inputs
result['summary_script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
for name, expected in inputs.items():
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == expected, name
out = ROOT / 'production_summary.json'
with out.open('x', encoding='utf-8') as stream:
    json.dump(result, stream, indent=2, allow_nan=False)
print(out)
