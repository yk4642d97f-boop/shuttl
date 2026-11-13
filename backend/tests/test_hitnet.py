"""Input semantics and unsafe-input handling, independent of trained weights."""
import numpy as np
import pytest
import torch
from app.services.tracking import hitnet
from app.services.tracking.data_types import PlayerObservation, Track2D, VideoMeta


def sample(n=12):
    t = np.arange(n, dtype=float) / 30
    return (Track2D(t, np.full((n, 2), [80., 60.]), np.ones(n),
                    np.ones(n, bool), np.zeros(n, bool)),
            VideoMeta(30., 320, 180, n, t.copy()))


def observation(frame=0, side='far', tid=1, box=None, foot=None):
    return PlayerObservation(frame, tid, np.array(box if box is not None else [10,20,30,60.]),
                             .9, side, np.array(foot) if foot is not None else None)


def test_semantic_order_and_independent_foot():
    track, meta = sample()
    players = [observation(side='near', box=[100,100,180,160], foot=[-4,1]),
               observation(side='far', box=[float('nan')]*4, foot=[4,-1])]
    raw = hitnet.raw_matrix(track, meta, players)
    assert np.isnan(raw[0,4:8]).all()
    np.testing.assert_array_equal(raw[0,12:16], [4,-1,-4,1])
    np.testing.assert_allclose(raw[0,8:12], [140/320,130/180,80/320,60/180])
    assert np.isnan(raw[1,4:]).all()


def test_hidden_coordinates_are_not_promoted_to_detections():
    track, meta = sample()
    track.visible[0] = False
    track.interpolated[0] = True
    raw = hitnet.raw_matrix(track, meta, players=[])
    assert np.isnan(raw[0,:3]).all() and raw[0,3] == 1


def fake_model():
    model = object.__new__(hitnet.HitNet)
    model.decoder = {'thr': .4, 'gap_s': .3, 'lam': 0.}
    calls = []
    def logits(raw, fps):
        calls.append(len(raw))
        out = np.full((len(raw),2), -10.,np.float32)
        out[len(raw)//2] = [10.,10.]
        return out
    model.logits = logits
    return model, calls


def test_gap_splits_context_and_keeps_real_clock():
    track, meta = sample()
    track.t[6:] += 2
    meta.timestamps = track.t.copy()
    model, calls = fake_model()
    events = model.detect(track, meta, [observation(frame=3,tid=7)])
    assert calls == [6,6]
    assert [h.frame for h in events] == [3,9]
    assert [h.t for h in events] == [track.t[3],track.t[9]]
    assert events[0].player_track_id == 7 and events[0].hitter == 'far'


def test_doubles_and_simultaneous_players_rejected():
    track, meta = sample()
    model,_ = fake_model()
    for players, double in [([],True),([observation(tid=3),observation(tid=1)],False)]:
        with pytest.raises(ValueError, match='two players'):
            model.detect(track,meta,players,doubles=double)
    index, multiple = hitnet.player_index([observation(tid=3),observation(tid=1)])
    assert multiple and index[(0,'far')].track_id == 1
    assert not hitnet.player_index([observation(tid=1),observation(frame=1,tid=3)])[1]


def test_empty_missing_and_misaligned():
    model,calls = fake_model()
    track,meta = sample(0)
    assert model.detect(track,meta,[]) == []
    track,meta = sample()
    track.visible[:] = False
    assert model.detect(track,meta,[]) == [] and not calls
    track.t += .01
    with pytest.raises(ValueError,match='timestamps differ'):
        model.detect(track,meta,[])


def test_short_nonfinite_output_rejected():
    model = object.__new__(hitnet.HitNet)
    model.device = torch.device('cpu')
    model.net = lambda x: torch.full((1,len(x[0]),2),float('nan'))
    with pytest.raises(ValueError,match='Non-finite'):
        model.logits(np.zeros((12,16),np.float32),30.)
