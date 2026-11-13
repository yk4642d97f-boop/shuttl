"""RTMPose-m body7 refinement around hits, using the audited BGR crop contract."""
from dataclasses import replace
from functools import lru_cache
import hashlib
from pathlib import Path
import numpy as np
from app.services.tracking import video_io


class Pose:
    def __init__(self, weights, device='cpu'):
        import onnxruntime as ort
        from app.services.tracking.rtmpose_core import make_session
        path=Path(weights).resolve()
        self.sha256=hashlib.sha256(path.read_bytes()).hexdigest()
        if device.startswith('cuda'):
            if 'CUDAExecutionProvider' not in ort.get_available_providers():
                raise RuntimeError('ONNX Runtime CUDA provider unavailable')
            device_id=int(device.partition(':')[2] or 0)
            providers=['CUDAExecutionProvider','CPUExecutionProvider']
            options=[{'device_id':str(device_id)},{}]
        elif device == 'cpu':
            providers=['CPUExecutionProvider'];options=None
        else:
            raise ValueError('RTMPose supports explicit cpu or cuda[:index]; MPS is not an ONNX provider')
        self.session=make_session(str(path),providers,provider_options=options,threads=4)
        if providers[0] not in self.session.get_providers():
            raise RuntimeError('Requested ONNX provider failed to initialize')
        self.session.disable_fallback()
        self.input=self.session.get_inputs()[0]
        if self.input.shape[1:] != [3,256,192]:
            raise ValueError('Expected RTMPose-m body7 192x256 input')
        self.outputs=[o.name for o in self.session.get_outputs()]
        self.providers=self.session.get_providers()

    def estimate(self,image,boxes):
        from app.services.tracking.rtmpose_core import run_batch, pad_box
        if not len(boxes): return np.empty((0,17,3),np.float32)
        h,w=image.shape[:2]
        boxes=np.asarray(boxes,np.float32)
        if boxes.shape != (len(boxes),4) or not np.isfinite(boxes).all() or np.any(boxes[:,2:]<=boxes[:,:2]):
            raise ValueError('Invalid pose boxes')
        padded=np.stack([pad_box(b,w,h) for b in boxes])
        if np.any(padded[:,2:]<=padded[:,:2]): raise ValueError('Pose box outside frame')
        kp,sc=run_batch(self.session,self.input.name,self.outputs,image,padded)
        result=np.concatenate((kp,sc[:,:,None]),-1)
        if result.shape != (len(boxes),17,3) or not np.isfinite(result).all():
            raise ValueError('Invalid pose model output')
        return result

    def refine(self,path,meta,players,hits):
        half=int(round(meta.fps))//2
        wanted=set()
        for hit in hits: wanted.update(range(max(0,hit.frame-half),min(meta.n_frames,hit.frame+half+1)))
        by_frame={}
        for obs in players:
            if obs.frame in wanted and obs.side in ('far','near') and np.isfinite(obs.bbox).all() and np.all(obs.bbox[2:]>obs.bbox[:2]):
                by_frame.setdefault(obs.frame,[]).append(obs)
        out=[]
        if by_frame:
            for frame,img in video_io.iter_frames(path,start=min(by_frame),stop=max(by_frame)+1):
                rows=by_frame.pop(frame,[])
                if not rows: continue
                kp=self.estimate(img,[r.bbox for r in rows])
                out.extend(replace(row,keypoints=keypoints) for row,keypoints in zip(rows,kp))
        if by_frame: raise ValueError('Video ended before pose windows were decoded')
        return out


@lru_cache(maxsize=1)
def _cached(path,size,mtime,device): return Pose(path,device)


def load(weights,device='cpu'):
    path=Path(weights).resolve();stat=path.stat()
    return _cached(str(path),stat.st_size,stat.st_mtime_ns,device)
