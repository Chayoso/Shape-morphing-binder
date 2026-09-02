import math, time, sys
import numpy as np, torch
sys.path.insert(0, ".")
from diff_gauss import GaussianRasterizationSettings, GaussianRasterizer

dev = "cuda"
torch.manual_seed(0)
N = 20000
x = (torch.rand(N, 3, device=dev) - 0.5) * 2.0
x.requires_grad_(True)
cov6 = torch.tensor([0.01,0,0,0.01,0,0.01], device=dev).repeat(N,1)
cov6.requires_grad_(True)
norms = torch.nn.functional.normalize(torch.randn(N,3,device=dev), dim=1)
colors = torch.ones(N,3,device=dev)*0.8
opac = torch.full((N,1), 0.9, device=dev)

def cam(az, el, radius=4.0, res=128, fov=0.7):
    campos = radius*np.array([math.sin(az)*math.cos(el), math.sin(el), math.cos(az)*math.cos(el)], np.float32)
    fwd = -campos/np.linalg.norm(campos)
    right = np.cross(fwd,[0,1,0]); right/=np.linalg.norm(right)
    up = np.cross(right,fwd)
    A = np.stack([right,-up,fwd],0).astype(np.float32)
    W = np.eye(4,dtype=np.float32); W[:3,:3]=A; W[:3,3]=-A@campos
    tx = math.tan(fov/2)
    P = np.zeros((4,4),np.float32); P[0,0]=1/tx; P[1,1]=1/tx
    P[2,2]=100/(100-0.01); P[2,3]=-100*0.01/(100-0.01); P[3,2]=1
    return GaussianRasterizationSettings(res,res,tx,tx,
        torch.zeros(3,device=dev),1.0,
        torch.tensor(W.T,device=dev), torch.tensor((P@W).T,device=dev),
        0, torch.tensor(campos,device=dev), False, False)

st = cam(0.5, 0.3)
rast = GaussianRasterizer(raster_settings=st)
t0=time.time()
scales = torch.full((N,3), 0.05, device=dev, requires_grad=True)
rots = torch.zeros(N,4,device=dev); rots[:,0]=1.0
out = rast(means3D=x, means2D=torch.zeros_like(x), opacities=opac,
           colors_precomp=colors, scales=scales, rotations=rots)
img = out[0] if isinstance(out,tuple) else out
torch.cuda.synchronize(); t1=time.time()
loss = (img - 0.3).pow(2).mean()
loss.backward()
torch.cuda.synchronize(); t2=time.time()
print("outputs:", len(out) if isinstance(out,tuple) else 1, "img", tuple(img.shape))
gx = x.grad; gc = scales.grad
print("grad means3D: finite %s  norm %.3e  nonzero %.1f%%" % (bool(torch.isfinite(gx).all()), float(gx.norm()), 100*float((gx.abs().sum(1)>0).float().mean())))
print("grad scales:  finite %s  norm %.3e" % (bool(torch.isfinite(gc).all()), float(gc.norm())))
print("fwd %.1f ms  bwd %.1f ms  (N=20k, 128px)" % (1000*(t1-t0), 1000*(t2-t1)))
# 18-view loss timing at 96px
sts=[cam(a*0.35, e) for a in range(6) for e in (-0.4,0,0.4)]
t0=time.time()
for st_ in sts:
    r=GaussianRasterizer(raster_settings=st_)
    o=r(means3D=x, means2D=torch.zeros_like(x), opacities=opac, colors_precomp=colors, scales=scales, rotations=rots)
    (o[0]-0.3).pow(2).mean().backward()
torch.cuda.synchronize()
print("18 views fwd+bwd: %.0f ms" % (1000*(time.time()-t0)))
