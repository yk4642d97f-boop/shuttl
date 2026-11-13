"""Portable copy of the project RTMPose reference; preprocessing delegated to rtmlib 0.0.16."""
import numpy as np
from rtmlib.tools.pose_estimation.rtmpose import bbox_xyxy2cs, top_down_affine, get_simcc_maximum
INPUT_SIZE=(192,256)
MEAN=np.array([123.675,116.28,103.53],dtype=np.float32)
STD=np.array([58.395,57.12,57.375],dtype=np.float32)

def make_session(onnx_path, providers, provider_options=None, threads=None,
                 batch_hint=None):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if threads is not None:
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
    if batch_hint:                       # fixed batch so CoreML can fuse
        fn = getattr(so, "add_free_dimension_override_by_name", None)
        if fn is None:
            raise RuntimeError("this onnxruntime has no free-dimension override API")
        fn("batch", batch_hint)
    kw = dict(sess_options=so, providers=providers)
    if provider_options is not None or 'CUDAExecutionProvider' in providers:
        # onnxruntime requires provider_options to be the same length as
        # providers, otherwise it silently falls back to CPU only.
        po = [dict(option) for option in (provider_options or [])]
        while len(po) < len(providers):
            po.append({})
        for index, provider in enumerate(providers):
            if provider == 'CUDAExecutionProvider':
                # SimCC argmax can jump tens of pixels under ORT's default
                # TF32 math. Keep the audited float32 inference contract.
                po[index].setdefault('use_tf32', '0')
        kw["provider_options"] = po
    return ort.InferenceSession(onnx_path, **kw)

def run_batch(sess, inp_name, out_names, img, boxes_pad):
    """boxes_pad: (P,4) already padded xyxy.  Returns kp (P,17,2) px, sc (P,17)."""
    P = len(boxes_pad)
    x = np.empty((P, 3, INPUT_SIZE[1], INPUT_SIZE[0]), dtype=np.float32)
    centers = np.empty((P, 2), dtype=np.float32)
    scales = np.empty((P, 2), dtype=np.float32)
    for j in range(P):
        center, scale = bbox_xyxy2cs(boxes_pad[j], padding=1.25)
        crop, scale = top_down_affine(INPUT_SIZE, scale, center, img)
        crop = (crop.astype(np.float32) - MEAN) / STD
        x[j] = crop.transpose(2, 0, 1)
        centers[j] = center
        scales[j] = scale
    outs = sess.run(out_names, {inp_name: x})
    locs, scores = get_simcc_maximum(outs[0], outs[1])   # (P,17,2) (P,17)
    kp = locs / 2.0
    kp = kp / np.array(INPUT_SIZE, dtype=np.float32) * scales[:, None, :]
    kp = kp + centers[:, None, :] - scales[:, None, :] / 2
    return kp.astype(np.float32), scores.astype(np.float32)

def pad_box(b, w, h):
    x1, y1, x2, y2 = b
    mw, mh = 0.10 * (x2 - x1), 0.10 * (y2 - y1)
    return np.array([max(x1 - mw, 0), max(y1 - mh, 0),
                     min(x2 + mw, w - 1), min(y2 + mh, h - 1)], dtype=np.float32)
