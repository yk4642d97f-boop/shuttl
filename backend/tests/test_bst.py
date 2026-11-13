"""BST semantics, abstention and context boundaries without requiring checkpoints."""
import numpy as np
import pytest
from app.services.tracking import bst
from app.services.tracking.data_types import HitEvent,RallyV2,ShotEventV2
from app.services.tracking.serialize import legacy_shot_type
from tests.test_hitnet import sample,observation


def posed(frame,side,tid):
    row=observation(frame,side,tid,foot=[4 if side=='far' else -4,1])
    row.keypoints=np.tile([20.,40.,1.],(17,1))
    return row


def test_no_pose_abstains_and_four_players_rejected():
    track,meta=sample()
    arrays=bst.frame_features(track,meta,[])
    clip,reason=bst.clip_at(6,meta.fps,arrays)
    assert clip is None and reason=='missing_pose_coverage'
    with pytest.raises(ValueError,match='two players'):
        bst.frame_features(track,meta,[posed(0,'far',1),posed(0,'far',2)])


def test_position_and_joint_contract():
    track,meta=sample(2)
    far,near=posed(0,'far',1),posed(0,'near',2)
    joints,pos,shuttle,covered,resolved=bst.frame_features(track,meta,[near,far])
    np.testing.assert_array_equal(joints[0],np.zeros((2,17,2),np.float32))
    np.testing.assert_allclose(pos[0,:,1],[(6.7-4)/13.4,(6.7+4)/13.4])
    assert resolved.tolist()==[True,False] and covered.tolist()==[True,False]
    np.testing.assert_allclose(shuttle[0],[80/320,60/180])
    np.testing.assert_array_equal(shuttle[1],[0,0])


def test_short_padding_long_resampling_and_empty_rejection():
    short=(np.ones((2,2,17,2)),np.ones((2,2,2)),np.ones((2,2)))
    args=bst.model_inputs([short])
    assert args[3].tolist()==[2] and args[0].shape==(1,30,2,72)
    assert not args[0][0,2:].any()
    long=(np.ones((60,2,17,2)),np.ones((60,2,2)),np.arange(120).reshape(60,2))
    args=bst.model_inputs([long])
    assert args[3].tolist()==[30]
    np.testing.assert_array_equal(args[1][0],long[2][::2])
    with pytest.raises(ValueError): bst.model_inputs([tuple(a[:0] for a in short)])


def test_refinement_respects_clock_gap_and_preserves_hitter_geometry():
    track,meta=sample(12)
    track.t[6:]+=2;meta.timestamps=track.t.copy()
    players=[posed(i,s,p) for i in range(12) for p,s in enumerate(('far','near'))]
    hit=HitEvent(5,float(track.t[5]),(80,60),'far',source='hitnet')
    shot=ShotEventV2(hit,None,shot_type='unknown')
    rally=RallyV2(0,11,[shot])
    model=object.__new__(bst.BST);model.sha256='fixture'
    lengths=[]
    def predict(clips):
        lengths.extend(len(c[0]) for c in clips)
        out=np.zeros((len(clips),25),np.float32);out[:,15]=10
        return out
    model.logits=predict
    result=model.refine([rally],track,meta,players)
    assert lengths==[6] and shot.shot_type=='smash'
    assert shot.hit.hitter=='far' and result['predictions'][0]['side']=='near'


@pytest.mark.parametrize('name,expected',[('cross_net','net'),('short_serve','serve'),('long_serve','serve')])
def test_fine_classes_keep_legacy_contract(name,expected):
    assert legacy_shot_type(name)==expected


def test_unknown_class_records_rules_as_actual_output_source():
    track,meta=sample(12)
    players=[posed(i,s,p) for i in range(12) for p,s in enumerate(('far','near'))]
    shot=ShotEventV2(HitEvent(5,float(track.t[5]),(80,60),'far',source='hitnet'),None,shot_type='drop')
    model=object.__new__(bst.BST);model.sha256='fixture'
    model.logits=lambda clips:np.zeros((len(clips),25),np.float32)
    diag=model.refine([RallyV2(0,11,[shot])],track,meta,players)
    assert diag['used']=='rules' and diag['evaluated_model']=='BST_CG_AP_seq30'
    assert diag['refined']==0 and diag['kept_rules']=={'unknown_class':1}
    assert shot.shot_type=='drop'
