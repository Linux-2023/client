"""Recording correctness and failure/interrupt paths without controlling hardware."""
import json
from pathlib import Path
import sys
import threading

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'packages/openpi-client/src'))
import tshape_video_saver as video


def observation():
    images = {}
    for i, name in enumerate(video.CAMERAS):
        images[name] = np.zeros((3, 16, 16), dtype=np.uint8)
        images[name][i] = 255
    return {'images': images, 'timestamps': dict(zip(video.CAMERAS, [1.0, 1.01, 1.02])),
            'sync_error': .02, 'state': np.zeros(14)}


def test_layout_and_rgb_are_preserved():
    frame = video.compose_tshape(observation()['images'])
    assert frame.shape == (48, 32, 3)
    assert frame[0, 0].tolist() == [255, 0, 0]
    assert frame[40, 0].tolist() == [0, 255, 0]
    assert frame[40, 31].tolist() == [0, 0, 255]


def test_close_without_episode_end_writes_playable_video_and_timestamps(tmp_path):
    saver = video.TShapeVideoSaver(tmp_path, fps=17)
    saver.on_episode_start()
    for _ in range(3):
        saver.on_step(observation(), {'actions': np.ones(14)})
    saver.close()  # Same path used on SIGINT/SIGTERM/runtime exceptions.
    saver.close()
    reader = video.imageio.get_reader(saver.video_path)
    assert reader.get_meta_data()['fps'] == 17
    assert reader.count_frames() == 3
    reader.close()
    rows = [json.loads(line) for line in saver.metadata_path.read_text().splitlines()]
    assert rows[1]['sensor_timestamps'] == observation()['timestamps']
    assert rows[-1]['frames'] == 3
    assert rows[-1]['dropped_frames'] == 0
    first = saver.video_path
    saver.on_episode_start()
    saver.close()
    assert saver.video_path != first
    assert not saver.video_path.exists()  # Empty episode: metadata only.


def test_slow_encoder_drops_recording_without_waiting_for_encoder(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    class Writer:
        def append_data(self, frame):
            entered.set()
            assert release.wait(5)
        def close(self): pass
    monkeypatch.setattr(video.imageio, 'get_writer', lambda *a, **kw: Writer())
    saver = video.TShapeVideoSaver(tmp_path, queue_size=1)
    saver.on_episode_start()
    try:
        saver.on_step(observation(), {})
        assert entered.wait(5)
        saver.on_step(observation(), {})
        saver.on_step(observation(), {})
        assert saver.dropped == 1
    finally:
        release.set()
        saver.close()
    rows = [json.loads(line) for line in saver.metadata_path.read_text().splitlines()]
    assert rows[-1]['frames'] == 2
    assert rows[-1]['dropped_frames'] == 1


def test_encoder_errors_are_not_silenced(tmp_path, monkeypatch):
    def fail(*a, **kw): raise OSError('disk full')
    monkeypatch.setattr(video.imageio, 'get_writer', fail)
    saver = video.TShapeVideoSaver(tmp_path)
    saver.on_episode_start()
    saver.on_step(observation(), {})
    with pytest.raises(RuntimeError, match='writer failed'):
        saver.close()


@pytest.mark.parametrize('failure', [KeyboardInterrupt, RuntimeError])
def test_entrypoint_finalizes_video_after_stopping_environment(tmp_path, monkeypatch, failure):
    import main_dual_eef_xyz3d_ros as main
    from types import SimpleNamespace
    order = []
    def run(): raise failure('stopped')
    runtime = SimpleNamespace(run=run, video_recorder=SimpleNamespace(close=lambda: order.append('video')))
    monkeypatch.setattr(main, '_build_environment', lambda args: SimpleNamespace(close=lambda: order.append('environment')))
    monkeypatch.setattr(main, '_build_policy', lambda args: object())
    monkeypatch.setattr(main, '_build_runtime', lambda *a, **kw: runtime)
    monkeypatch.setattr(main.signal, 'signal', lambda *a: None)
    assert main.main(main.Args(out_dir=tmp_path)) == (130 if failure is KeyboardInterrupt else 1)
    assert order == ['environment', 'video']
