"""Prepare fixed-window RTMPose BST train/validation inputs; never read test examples."""
from collections import Counter
from pathlib import Path
import hashlib,json,sys
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'backend'))
import stroke_chain as R
from app.services.tracking.bst import model_inputs,CLASSES
OUT=HERE/'data/bst_fixed30'


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
 if OUT.exists(): raise SystemExit('Dataset exists; keep immutable inputs or choose a new dataset version')
 OUT.mkdir(parents=True)
 official={}
 for split in ('train','val'):
  for typ in (Path(R.CV.OFFICIAL_RAW)/split).iterdir():
   for p in typ.glob('*_joints.npy'): official[p.name[:-11]]=typ.name
 manifest={'version':1,'classes':CLASSES,'window':'hit +/- round(fps)//2 frames; no shift; BST seq30 transform',
           'test_access':False,'note':'Released checkpoint pretraining provenance is separate; no new blind-test claim.',
           'excluded_matches':{'007':'Repeated global frame IDs in extracted rally rows; requires per-rally reconciliation before use'},
           'splits':{}}
 for split,mids in [('train',[f'{i:03}' for i in range(1,7)]),('val',['041'])]:
  clips=[];labels=[];names=[];reasons=Counter();sources=[]
  for mid in mids:
   frames=R.MatchFrames(mid);match=R.E.C.load_match(mid)
   for rally in match['rallies']:
    for label in frames.labels(rally,official):
     if not label['class']:
      reasons['unknown_or_ambiguous_label']+=1;continue
     clip,reason=frames.raw_clip(label['frame'])
     if clip is None: reasons[reason]+=1;continue
     clips.append(clip);labels.append(label['class'])
     names.append(f"{mid}/{rally['set']}/{rally['rally']}/{label['frame']}")
   for base in ('research/extract/out','research/rtmpose/pose'):
    p=ROOT/base/f'{mid}.npz';sources.append({'path':str(p.relative_to(ROOT)),'sha256':sha(p)})
   print(split,mid,'clips',len(clips),flush=True)
  assert len(names)==len(set(names))
  j,s,p,n=model_inputs(clips)
  target=OUT/f'{split}.npz'
  np.savez_compressed(target,jnb=j,shuttle=s,pos=p,length=n.astype(np.int64),label=np.array(labels,np.int64),name=np.array(names))
  manifest['splits'][split]={'file':target.name,'sha256':sha(target),'matches':mids,'clips':len(clips),
                            'excluded':dict(reasons),'sources':sources,'class_counts':dict(Counter(labels))}
 (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
 print('Saved',OUT,flush=True)
if __name__=='__main__':main()
