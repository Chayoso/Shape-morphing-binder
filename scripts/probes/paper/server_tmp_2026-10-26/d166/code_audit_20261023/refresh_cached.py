import importlib.util,json,os,pathlib,shutil,time
from pathlib import Path
B=pathlib.Path('/data/relcfd/chayo/physmorph_v2');D=B/'tmp/d166/code_audit_20261023'
s=importlib.util.spec_from_file_location('reader',B/'tmp/d166/d166_read.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
os.environ['D166_CACHE_ONLY']='1';os.environ['CUDA_VISIBLE_DEVICES']=''
specs={'61_ulast':['Ul:T'],'62_relaxoff':['Rx:T'],'63_assim10':['A10:TA10'],'64_assim08':['A08:TA08'],
 '65_cross_a10':['UlA10:TA10','RxA10:TA10'],'66_assim025':['A025:TA025'],'67_stiff3':['UlE3:TE3','RxE3:TE3']}
def active_reader(out):
 for proc in Path("/proc").iterdir():
  if not proc.name.isdigit():
   continue
  try:
   if proc.stat().st_uid != os.getuid():
    continue
   args=proc.joinpath("cmdline").read_bytes().split(b"\0")
   if str(out).encode() in args and any(b"/d166/after.sh" in x or b"/d166/d166_read.py" in x for x in args):
    return int(proc.name)
  except OSError:
   continue
 return None
selected=os.environ.get('D166_REFRESH_JOB')
changes=[]
for job,sp in specs.items():
 if selected and job!=selected:continue
 out=B/'output/gpu/d166/auto'/job
 if not (out/'verdict.json').exists():continue
 active=active_reader(out)
 if active:
  print('DEFER active reader',job,active,flush=True);continue
 pidf=out/'after.pid' 
 if pidf.exists():
  try:
   pid=int(pidf.read_text());cmd=pathlib.Path(f'/proc/{pid}/cmdline').read_bytes()
   if b'after.sh' in cmd:
    print('DEFER active reader',job,pid,flush=True);continue
  except (OSError,ValueError):pass
 backup=out/'codex_strict_backup_20261023'
 backup.mkdir(exist_ok=True)
 for fn in ('summary.md','verdict.json'):
  if (out/fn).exists() and not (backup/fn).exists():shutil.copy2(out/fn,backup/fn)
 before=json.load(open(backup/'verdict.json'))
 m.scenes(str(out),str(B/'output/gpu/d166/runs'),before['title'],*sp)
 after=json.load(open(out/'verdict.json'))
 for arm,v in after['arms'].items():
  prev=before['arms'][arm]
  changes.append(dict(job=job,arm=arm,before=prev,after=v,
   cause='Reader completeness/exact-stop requirement; authority/end geometry/speed inputs reused unchanged; stop interpolation now raw=frame_end-1.',
   numerical_change={k:[prev.get(k),v.get(k)] for k in ('share_min','colour_min','vx_max','kept_min','sil_mean') if prev.get(k)!=v.get(k)}))
master=B/'output/gpu/RENDER_REST_WEIGHT_SUMMARY.md'
if master.exists() and not (D/'master.before_strict.md').exists():shutil.copy2(master,D/'master.before_strict.md')
m.aggregate(str(master))
out=D/(('refresh_'+selected) if selected else 'refresh_completed')
out.with_suffix('.json').write_text(json.dumps(dict(time=time.strftime('%Y-%m-%d %H:%M:%S %Z'),cache_only=True,changes=changes),indent=2)+'\n')
for c in changes:
 a=c['after'];print(c['job'],c['arm'],('PASS' if c['before']['pass_'] else 'FAIL'),'->',a['status'],'counts',a['counts'],'change',c['numerical_change'],flush=True)
