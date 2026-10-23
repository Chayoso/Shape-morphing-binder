"""Seven fixed material estimates, external safety, and one shared-clock GIF."""
import argparse
import bisect
import json
import math
from pathlib import Path
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
if sys.platform=='win32':
    import torch  # load its CUDA DLLs before the native package imports CuPy
INPUT=Path(__file__).with_name('material_comparison.json')
from physmorph.safe_pipeline import INTERVAL, clean, delivered_end, dump, render, sample_steps


def load_materials():
    return json.loads(INPUT.read_text(encoding='utf8'))


def sample_times(end,dt):
    return [step*dt for step in sample_steps(end,max(1,round(INTERVAL/dt)))]


def aligned_index(times,t):
    return max(0,bisect.bisect_right(times,t+1e-10)-1)


def save_summary(out,state):
    lines=['# Material comparison','',
        'Fixed LLM estimates; not measured material constants. No runtime LLM required.',
        'E_solver = E_Pa / '+str(state['parameters']['stress_unit_pa'])+'; nu unchanged.',
        'Seed97/N40000/bunny-to-dragon; native CLI defaults, with external state safety.',
        'Steel uses two temporal substeps; native20 control bins, same window duration.',
        'One solver clock, not calibrated SI time; short runs hold their delivered endpoint.',
        'Final safety gates do not certify target fit. No ABC or added/reference force.',
        '', 'Stage: '+state['stage'],'',
        '| Material | E Pa | nu | Substeps | IoU | Chamfer | Gates |',
        '|---|---:|---:|---:|---:|---:|---|']
    for m in state['parameters']['materials']:
        r=state['results'].get(m['key'],{})
        lines.append('| '+' | '.join(str(v) for v in (m['label'],m['young_pa'],m['poisson'],m['substeps'],
            r.get('iou',''),r.get('chamfer',''),r.get('gates_passed','')))+' |')
    lines+=['','```json',json.dumps(clean(state),ensure_ascii=False,separators=(',',':')),'```','']
    (out/'results.md').write_text('\n'.join(lines),encoding='utf8')


def simulate(key,job):
    from physmorph.safe_pipeline import simulate as guarded_simulate
    data=load_materials();m=next(m for m in data['materials'] if m['key']==key)
    request={'simulation':dict(src=data['source'],tgt=data['target'],n=data['particles'],
                              seed=data['seed'],out=str(job/'native')),
             'material':{'effective':dict(young=m['young_pa']/data['stress_unit_pa'],poisson=m['poisson'])}}
    try:
        guarded_simulate(request,job,m['substeps'])
    finally:
        path=job/'result.json'
        if path.exists():
            row=json.loads(path.read_text(encoding='utf8'));row['material']=key;dump(path,row)


def child(action,key,job,python,seconds):
    from physmorph.safe_pipeline import child as worker
    worker(__file__,[action,'--case',key,'--job',str(job)],job,python=python,seconds=seconds,
           name=action,render_stage=action=='render')


def font(size):
    from PIL import ImageFont
    for name in ('DejaVuSans.ttf','Arial.ttf'):
        try:return ImageFont.truetype(name,size)
        except OSError:pass
    return ImageFont.load_default()


def comparison_frame(state,t,pictures):
    from PIL import Image,ImageDraw
    image=Image.new('RGB',(1280,798),(247,248,250));draw=ImageDraw.Draw(image)
    title,small=font(17),font(14)
    draw.text((18,9),f't = {t:.2f}',font=title,fill=(35,45,60))
    for k,m in enumerate(state['parameters']['materials']):
        x=(k%4)*320+(160 if k>=4 else 0);y=42+(k//4)*378
        draw.rectangle((x+2,y+2,x+317,y+375),fill='white',outline=(210,215,220))
        draw.text((x+12,y+8),m['label'],font=title,fill=(28,91,147))
        e=m['young_pa'];unit='GPa' if e>=1e9 else 'MPa'
        draw.text((x+12,y+31),f'E={e/(1e9 if e>=1e9 else 1e6):.3g} {unit}, nu={m["poisson"]:.3g}',font=small,fill=(70,75,85))
        row=state['results'][m['key']]
        image.paste(pictures[m['key']][aligned_index(row['times'],t)],(x,y+52))
        draw.text((x+12,y+356),f'Final IoU {row["iou"]:.4f}',font=small,fill=(40,90,65))
    return image


def compose(out,state,pictures):
    end=max(row['times'][-1] for row in state['results'].values())
    timeline=[i*INTERVAL for i in range(math.floor(end/INTERVAL)+1)]
    if end-timeline[-1]>1e-9:timeline.append(end)
    frames=[comparison_frame(state,t,pictures) for t in timeline]
    durations=[100]*len(frames);durations[0]=1000;durations[-1]=2500
    frames[0].save(out/'materials.gif',save_all=True,append_images=frames[1:],duration=durations,loop=0,optimize=False)
    state['visualization']=dict(frames=len(frames),end_solver_time=end,sample_interval=INTERVAL,
        playback_speed=2,initial_pause_ms=1000,final_pause_ms=2500)


def run(out,python):
    from PIL import Image
    out=out.resolve();out.mkdir(parents=True,exist_ok=False)
    state=dict(stage='registered',parameters=load_materials(),results={})
    pictures={};save_summary(out,state)
    try:
        for m in state['parameters']['materials']:
            key=m['key'];print('START '+key,flush=True)
            with tempfile.TemporaryDirectory(prefix='job_',dir=out) as folder:
                job=Path(folder)
                try:
                    child('simulate',key,job,python,2700)
                    row=json.loads((job/'result.json').read_text(encoding='utf8'))
                    state['results'][key]=row;save_summary(out,state)
                    child('render',key,job,python,1800)
                    images=sorted((job/'render').glob('splatgpu_*/f*.png'))
                    if len(images)!=len(row['times']):raise RuntimeError('Rendered frame/timestamp count mismatch')
                    pictures[key]=[]
                    for file in images:
                        with Image.open(file) as image:pictures[key].append(image.convert('RGB').crop((0,0,320,300)))
                except BaseException as error:
                    path=job/'result.json'
                    row=json.loads(path.read_text(encoding='utf8')) if path.exists() else dict(status='failed')
                    row['error']=type(error).__name__+': '+str(error);state['results'][key]=row;raise
            state['stage']='running';save_summary(out,state)
            print('DONE '+key,flush=True)
        compose(out,state,pictures);state['stage']='complete'
    except BaseException:
        state['stage']='attention_required';raise
    finally:save_summary(out,state)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('run','simulate','render'))
    parser.add_argument('--out',type=Path,default=ROOT/'output/material_comparison')
    parser.add_argument('--python',default=sys.executable,help='GPU worker interpreter; defaults to this interpreter')
    parser.add_argument('--case');parser.add_argument('--job',type=Path)
    args=parser.parse_args()
    if args.action=='run':run(args.out,args.python)
    elif args.action=='simulate':simulate(args.case,args.job.resolve())
    else:render(args.job.resolve())
