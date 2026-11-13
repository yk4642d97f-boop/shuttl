"""Contract consumed by ShuttlCore; the emitted fixture is also decoded by iOS tests."""
import json
import numpy as np
from tests.test_serialize import _analysis
from app.services.tracking.serialize import mobile_archive
from app.services.tracking.data_types import PlayerObservation


def test_archive_preserves_timestamps_masks_camera_and_rich_reports():
    res = _analysis()
    res.meta.timestamps[2] = 0.034123456789  # VFR timing must not be rounded to nominal FPS.
    res.players = [PlayerObservation(10, 7, np.array([1, 2, 3, 4]), .9, 'near',
                                      np.array([-4, .5]), np.ones((17, 3)))]
    doc = json.loads(json.dumps(mobile_archive(res), allow_nan=False))
    assert doc['meta']['timestamps'][2] == 0.034123456789
    assert doc['track']['xy'][0] == [None, None]
    assert doc['track']['visible'] == res.track.visible.tolist()
    assert doc['track']['interpolated'] == res.track.interpolated.tolist()
    np.testing.assert_array_equal(doc['camera']['K'], res.camera.K)
    assert doc['players'][0]['foot_world'] == [-4, .5]
    assert 'keypoints' not in doc['players'][0]
    shot = doc['rallies'][0]['shots'][0]
    assert shot['shot_type'] == 'kill'
    assert shot['segment']['frames'] == list(range(10, 41))
    np.testing.assert_array_equal(shot['segment']['positions'], res.rallies[0].shots[0].segment.positions)
    assert shot['segment']['net_crossing_height'] == 1.9


def test_archive_preserves_optional_missing_analysis():
    res = _analysis()
    res.track = res.camera = None
    res.rallies = []
    doc = mobile_archive(res)
    assert doc['track'] is None and doc['camera'] is None and doc['rallies'] == []
    json.dumps(doc, allow_nan=False)
