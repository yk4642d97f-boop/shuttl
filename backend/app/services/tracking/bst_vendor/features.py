"""Subset of BST (Jing-Yuan Chang, MIT); see LICENSE and provenance.json."""
import numpy as np

def get_merged_stroke_types(pad_to_same_len=False):
    class_ls = [
        '放小球', '擋小球', '殺球', '挑球',
        '長球', '平球', '切球', '推球',
        '撲球', '勾球', '發短球', '發長球'
    ]
    if pad_to_same_len:
        max_len = max([len(e) for e in class_ls])
        class_ls = [e.ljust(max_len, '　') for e in class_ls]
        class_ls = ['未知球種'.ljust(max_len, '　')+' '*7] + ['Top_'+s+' '*3 for s in class_ls] + ['Bottom_'+s for s in class_ls]
        return class_ls

    class_ls = ['未知球種'] + ['Top_'+s for s in class_ls] + ['Bottom_'+s for s in class_ls]
    return class_ls

def get_bone_pairs(skeleton_format='coco'):
    match skeleton_format:
        case 'coco':
            pairs = [
                (0,1),(0,2),(1,2),(1,3),(2,4),   # head
                (3,5),(4,6),                     # ears to shoulders
                (5,7),(7,9),(6,8),(8,10),        # arms
                (5,6),(5,11),(6,12),(11,12),     # torso
                (11,13),(13,15),(12,14),(14,16)  # legs
            ]
        case _:
            raise NotImplementedError
    return pairs

def make_seq_len_same(
    target_len: int,
    joints: np.ndarray,
    pos: np.ndarray,
    shuttle: np.ndarray
):
    video_len = len(pos)

    if video_len > target_len:
        need_padding = (video_len % target_len) > (target_len // 2)
        stride = video_len // target_len + int(need_padding)

        joints = joints[::stride][:target_len]
        pos = pos[::stride][:target_len]
        shuttle = shuttle[::stride][:target_len]

        new_video_len = len(pos)

        if need_padding:
            pad_len = target_len - new_video_len
            joints = np.pad(joints, ((0, pad_len), *([(0, 0)]*3)))
            pos = np.pad(pos, ((0, pad_len), *([(0, 0)]*2)))
            shuttle = np.pad(shuttle, ((0, pad_len), (0, 0)))

    else:
        # Since they have been normalized, we don't interpolate them.
        new_video_len = video_len

        pad_len = target_len - new_video_len
        joints = np.pad(joints, ((0, pad_len), *([(0, 0)]*3)))
        pos = np.pad(pos, ((0, pad_len), *([(0, 0)]*2)))
        shuttle = np.pad(shuttle, ((0, pad_len), (0, 0)))

    return joints, pos, shuttle, new_video_len

def create_bones(joints: np.ndarray, pairs) -> np.ndarray:
    '''Same as create_bones_robust in TemPose.'''
    # joints: (t, m, J, 2)
    bones = []
    for start, end in pairs:
        start_j = joints[:, :, start, :]
        end_j = joints[:, :, end, :]
        bone = np.where((start_j != 0.0) & (end_j != 0.0), end_j - start_j, 0.0)
        # bone: (t, m, 2)
        bones.append(bone)
    return np.stack(bones, axis=-2)
