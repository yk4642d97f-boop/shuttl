"""Parity in the SHIPPED contract: raw 0..255 RGB into RTMPoseM.mlpackage."""
import glob, os, json, warnings
import cv2, numpy as np
warnings.filterwarnings("ignore")
HERE=os.path.dirname(os.path.abspath(__file__)); H,W=256,192; SPLIT=2.0
MEAN=np.array([123.675,116.28,103.53],np.float32).reshape(3,1,1)
STD=np.array([58.395,57.12,57.375],np.float32).reshape(3,1,1)
import onnxruntime as ort, coremltools as ct
sess=ort.InferenceSession(os.path.join(HERE,"RTMPoseM.patched.onnx"),providers=["CPUExecutionProvider"])
INAME=sess.get_inputs()[0].name
def raw(bgr):
    r=cv2.resize(bgr,(W,H),interpolation=cv2.INTER_LINEAR).astype(np.float32)
    return np.ascontiguousarray(r[:,:,::-1].transpose(2,0,1)[None]).astype(np.float32)
def norm(x): return np.ascontiguousarray((x-MEAN)/STD).astype(np.float32)
files=sorted(glob.glob(os.path.join(HERE,"real_crops","*.png")))
real=[(os.path.basename(f),raw(cv2.imread(f))) for f in files if cv2.imread(f) is not None]
rng=np.random.default_rng(12345)
rand=[(f"rand{i}",(rng.random((1,3,H,W),dtype=np.float32)*255).astype(np.float32)) for i in range(24)]
out={}
for pn,pf,needs_norm in (("shipped","RTMPoseM.mlpackage",False),
                         ("rawinput_ref","RTMPoseM_rawinput.mlpackage",True),
                         ("fp16","RTMPoseM_fp16.mlpackage",False)):
    p=os.path.join(HERE,pf)
    if not os.path.exists(p): continue
    for un,cu in (("CPU_ONLY",ct.ComputeUnit.CPU_ONLY),("CPU_AND_GPU",ct.ComputeUnit.CPU_AND_GPU),
                  ("CPU_AND_NE",ct.ComputeUnit.CPU_AND_NE),("ALL",ct.ComputeUnit.ALL)):
        ml=ct.models.MLModel(p,compute_units=cu)
        for cn,cs in (("real",real),("rand",rand)):
            worst=0.0; nf=0; nkp=0; nan=0; px=0.0; fl=[]
            for nm,xr in cs:
                rx,ry=sess.run(None,{INAME:norm(xr)})
                o=ml.predict({"image":norm(xr) if needs_norm else xr})
                gx,gy=np.asarray(o["simcc_x"]),np.asarray(o["simcc_y"])
                nan+=int(np.isnan(gx).sum()+np.isnan(gy).sum())
                for lbl,r,g in (("x",rx,gx),("y",ry,gy)):
                    worst=max(worst,float(np.abs(r.astype(np.float64)-g.astype(np.float64)).max()))
                    a,b=np.argmax(r,-1),np.argmax(g,-1); d=a!=b
                    nkp+=a.size; nf+=int(d.sum())
                    for i in np.argwhere(d.ravel()).ravel():
                        s=np.sort(r,axis=-1); m=(s[...,-1]-s[...,-2]).ravel()
                        fl.append((nm,lbl,int(i),abs(int(b.ravel()[i])-int(a.ravel()[i])),float(m[i])))
                px=max(px,float(np.hypot(np.argmax(gx,-1)/SPLIT-np.argmax(rx,-1)/SPLIT,
                                         np.argmax(gy,-1)/SPLIT-np.argmax(ry,-1)/SPLIT).max()))
            k=f"{pn}/{un}/{cn}"
            ok=worst<1e-2 and nf==0 and nan==0
            print(f"{k:30s} maxdiff={worst:10.6f} agree={100*(1-nf/max(nkp,1)):8.3f}% "
                  f"flips={nf:4d}/{nkp:4d} nan={nan:5d} px_max={px:6.3f} -> {'PASS' if ok else 'FAIL'}")
            for f in fl[:3]: print(f"      {f[0]} simcc_{f[1]} kp={f[2]} binΔ={f[3]} margin={f[4]:.2e}")
            out[k]=dict(worst=worst,nflips=nf,nkp=nkp,nan=nan,px_max=px,flips=fl)
json.dump(out,open(os.path.join(HERE,"parity_final.json"),"w"),indent=1,default=str)
