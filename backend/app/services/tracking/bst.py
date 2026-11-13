"""Portable, singles-only BST seq30 logit ensemble and production feature adapter."""
from __future__ import annotations
from collections import Counter
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from app.services.tracking.bst_vendor.features import create_bones, get_bone_pairs, get_merged_stroke_types, make_seq_len_same
from app.services.tracking.hitnet import player_index
from app.services.tracking.video_io import pts_gaps

CLASSES = get_merged_stroke_types()
TYPES = ('net','block','smash','lift','clear','drive','drop','push','kill','cross_net','short_serve','long_serve')


def frame_features(track, meta, players):
    """Far first. Box normalization uses original tracking boxes, not padded crops."""
    if meta.width <= 0 or meta.height <= 0 or track.n != meta.n_frames:
        raise ValueError('Invalid video/track dimensions')
    index,multiple = player_index(players)
    if multiple:
        raise ValueError('BST checkpoint supports two players only')
    n = track.n
    box = np.full((n,2,4),np.nan,np.float32)
    kp = np.full((n,2,17,3),np.nan,np.float32)
    foot = np.full((n,2,2),np.nan,np.float32)
    for (f,side),obs in index.items():
        if not 0 <= f < n:
            continue
        p = 0 if side == 'far' else 1
        if np.shape(obs.bbox) == (4,): box[f,p] = obs.bbox
        if np.shape(obs.keypoints) == (17,3): kp[f,p] = obs.keypoints
        if np.shape(obs.foot_world) == (2,): foot[f,p] = obs.foot_world
    covered = np.isfinite(kp[:,:,0,0]).any(1)
    good = np.isfinite(box).all(-1) & np.isfinite(kp[:,:,:17,:2]).all((-1,-2))
    good &= (kp[:,:,15,2] > 0) & (kp[:,:,16,2] > 0)
    good[:,1] &= good[:,0]  # same early-exit semantics as released extraction
    resolved = good.all(1)
    centre = (box[:,:,:2]+box[:,:,2:])/np.float32(2)
    diag = np.maximum(np.hypot(box[:,:,2]-box[:,:,0],box[:,:,3]-box[:,:,1]),1e-6)
    joints = np.where(good[:,:,None,None],(kp[:,:,:17,:2]-centre[:,:,None])/diag[:,:,None,None],0).astype(np.float32)
    fx,fy = foot[:,:,0],foot[:,:,1]
    pos = np.stack((np.float32(.5)-fy/np.float32(6.1),(np.float32(6.7)-fx)/np.float32(13.4)),-1)
    pos[~np.isfinite(foot).all(-1)] = 0
    shuttle = np.zeros((n,2),np.float32)
    visible = resolved & track.visible & np.isfinite(track.xy).all(1)
    shuttle[visible] = np.asarray(track.xy,dtype=np.float32)[visible]/np.array([meta.width,meta.height],np.float32)
    return joints,pos,shuttle,covered,resolved


def clip_at(frame, fps, arrays, *, lower=0, upper=None):
    joints,pos,shuttle,covered,resolved = arrays
    half = int(round(fps))//2
    a,b = max(lower,frame-half),min(len(joints) if upper is None else upper,frame+half+1)
    if b-a < 2:
        return None,'missing_window'
    if not covered[a:b].all():
        return None,'missing_pose_coverage'
    if not resolved[a:min(b,a+30)].any():
        return None,'no_resolved_players'
    return (joints[a:b],pos[a:b],shuttle[a:b]),None


def model_inputs(clips):
    arrays = []
    for joints,pos,shuttle in clips:
        if not 1 <= len(pos) or any(not np.isfinite(a).all() for a in (joints,pos,shuttle)):
            raise ValueError('BST requires finite nonempty clips')
        j,p,s,n = make_seq_len_same(30,joints.astype(np.float32),pos.astype(np.float32),shuttle.astype(np.float32))
        jnb = np.concatenate((j,create_bones(j,get_bone_pairs('coco'))),axis=-2).reshape(30,2,72)
        arrays.append((jnb,s,p,n))
    return [np.stack(a) for a in zip(*arrays)] if arrays else []


class BST:
    def __init__(self, manifest, device='cpu'):
        from app.services.tracking.bst_vendor.bst import BST_CG_AP
        path = Path(manifest).resolve()
        doc = json.loads(path.read_text())
        if doc.get('architecture') != 'BST_CG_AP_seq30' or doc.get('classes') != CLASSES or doc.get('ensemble') != 'mean_logits':
            raise ValueError('BST architecture/class/ensemble contract mismatch')
        if len(doc.get('weights',[])) != 10:
            raise ValueError('Frozen baseline requires all 10 checkpoints')
        self.device = torch.device(device)
        self.nets = []
        self.sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        for item in doc['weights']:
            weight = (path.parent/item['file']).resolve()
            if not weight.is_relative_to(path.parent) or weight.suffix != '.pt':
                raise ValueError('Weights must be inside the manifest directory')
            if hashlib.sha256(weight.read_bytes()).hexdigest() != item['sha256']:
                raise ValueError(f'BST checksum mismatch: {weight.name}')
            net = BST_CG_AP(in_dim=72,n_class=25,seq_len=30,depth_tem=2,depth_inter=1)
            net.load_state_dict(torch.load(weight,map_location='cpu',weights_only=True),strict=True)
            self.nets.append(net.to(self.device).eval())

    @torch.inference_mode()
    def logits(self, clips, batch_size=64):
        if not clips: return np.empty((0,25),np.float32)
        arrays = model_inputs(clips)
        output = np.zeros((len(clips),25),np.float32)
        for net in self.nets:
            for start in range(0,len(clips),batch_size):
                args = [torch.from_numpy(a[start:start+batch_size]).to(self.device) for a in arrays]
                output[start:start+batch_size] += net(*args).float().cpu().numpy()/len(self.nets)
        if not np.isfinite(output).all(): raise ValueError('Non-finite BST model output')
        return output

    def refine(self, rallies, track, meta, players):
        arrays = frame_features(track,meta,players)
        gaps = pts_gaps(meta.timestamps)
        clips,targets = [],[]
        reasons = Counter()
        for rally in rallies:
            for shot in rally.shots:
                f = shot.hit.frame
                if shot.hit.source != 'hitnet':
                    reasons['unvalidated_hit_source'] += 1
                    continue
                # Never join frames across a cut in the presentation clock or rally boundary.
                lo = max(rally.start_frame,int(gaps[gaps<=f][-1]) if (gaps<=f).any() else 0)
                hi = min(rally.end_frame+1,int(gaps[gaps>f][0]) if (gaps>f).any() else track.n)
                clip,reason = clip_at(f,meta.fps,arrays,lower=lo,upper=hi)
                if clip is None: reasons[reason] += 1
                else:
                    clips.append(clip)
                    targets.append(shot)
        logits = self.logits(clips)
        predictions = []
        for shot,row in zip(targets,logits):
            cls = int(row.argmax())
            if not cls:
                reasons['unknown_class'] += 1
                continue
            prob = np.exp(row-row.max());prob /= prob.sum()
            typ = TYPES[(cls-1)%12]
            side = 'far' if cls<=12 else 'near'
            predictions.append({'frame':shot.hit.frame,'class':cls,'type':typ,'side':side,'probability':float(prob[cls])})
            # Hitter/geometry are already linked. Keep that contract and expose any
            # BST disagreement in diagnostics instead of rewriting a fitted flight.
            shot.shot_type = typ
            shot.shot_type_confidence = float(prob[cls])
        return {'used':'BST_CG_AP_seq30' if predictions else 'rules',
                'evaluated_model':'BST_CG_AP_seq30',
                **({'fallback_reason':'no_accepted_predictions'} if not predictions else {}),
                'manifest_sha256':self.sha256,'refined':len(predictions),
                'kept_rules':dict(reasons),'predictions':predictions,
                'validation_scope':'annotated singles rallies; continuous video pending'}


@lru_cache(maxsize=1)
def _cached(path,size,mtime,device):
    return BST(path,device)


def load(manifest,device='cpu'):
    path = Path(manifest).resolve();stat = path.stat()
    return _cached(str(path),stat.st_size,stat.st_mtime_ns,device)
