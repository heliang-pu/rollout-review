from __future__ import annotations

from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import shutil
import subprocess
import threading
import urllib.error
import urllib.request

import pytest

from rollout_review import review

pytestmark = pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="no ffmpeg")
CAMERAS = ["observation.images.top", "observation.images.wrist"]
TASK = 'Pick: the "cup", 100%'


def make(root: Path, lengths: list[int], cameras: list[str] = CAMERAS, fps: int = 10) -> Path:
    features = {c: {"dtype": "video", "shape": [48, 64, 3]} for c in cameras}
    features["observation.state"] = {"dtype": "float32", "shape": [2]}
    info = {"codebase_version": "v2.1", "fps": fps, "chunks_size": 1000, "features": features,
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"}  # fmt: skip
    (root / "meta").mkdir(parents=True)
    (root / "meta/info.json").write_text(json.dumps(info))
    rows = []
    for number, length in enumerate(lengths):
        rows.append({"episode_index": number, "tasks": [TASK], "length": length})
        data = root / info["data_path"].format(episode_chunk=0, episode_index=number)
        data.parent.mkdir(parents=True, exist_ok=True)
        data.write_bytes(f"rows of episode {number} with {length} frames".encode())  # only its bytes matter here
        for camera in cameras:
            video = root / info["video_path"].format(episode_chunk=0, video_key=camera, episode_index=number)
            video.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", f"testsrc=size=64x48:rate={fps}",
                            "-frames:v", str(length), "-pix_fmt", "yuv420p", str(video)], check=True)  # fmt: skip
    (root / "meta/episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return root


def probe(path: Path) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
                          "stream=codec_name,width,height,nb_read_frames,pix_fmt", "-of", "json", str(path)],
                         capture_output=True, text=True, check=True).stdout  # fmt: skip
    return json.loads(out)["streams"][0]


@pytest.fixture
def served(tmp_path):
    root = make(tmp_path / "demo", [8, 6, 5])
    server = ThreadingHTTPServer(("127.0.0.1", 0), review.make_handler(review.App([root])))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"

    def call(path, body=None, headers=None):
        request = urllib.request.Request(base + path, data=json.dumps(body).encode() if body is not None else None,
                                         headers=headers or {})  # fmt: skip
        try:
            with urllib.request.urlopen(request) as response:
                return response.status, response.read(), dict(response.headers)
        except urllib.error.HTTPError as error:
            return error.code, error.read(), dict(error.headers)

    yield root, call
    server.shutdown()


def test_review_videos_sit_next_to_the_dataset_and_play_in_a_browser(tmp_path):
    root = make(tmp_path / "demo", [8])
    dataset = review.Dataset(root)
    video = dataset.stitched(0)
    assert video == tmp_path / "demo_review/episode_000000.mp4"
    assert not list(root.rglob("*review*"))  # nothing was written inside the dataset
    stream = probe(video)
    assert (stream["codec_name"], stream["pix_fmt"], stream["nb_read_frames"]) == ("h264", "yuv420p", "8")
    assert (stream["width"], stream["height"]) == (2 * 480, 360)  # two cameras side by side
    made = video.stat().st_mtime_ns
    assert dataset.stitched(0).stat().st_mtime_ns == made  # kept, not made again

    four = review.Dataset(make(tmp_path / "four", [4], cameras=[f"observation.images.c{i}" for i in range(4)]))
    stream = probe(four.stitched(0))
    assert (stream["width"], stream["height"]) == (2 * 480, 2 * 360)  # a 2 x 2 grid


def test_judging_over_http(served):
    root, call = served
    status, body, _ = call("/api/episodes")
    data = json.loads(body)
    assert status == 200 and data["count"] == {"ok": 0, "problem": 0, "unsure": 0, "none": 3, "all": 3}
    assert data["episodes"][1] == {"dataset": "demo", "episode": 1, "length": 6, "seconds": 0.6, "task": TASK,
                                   "verdict": None, "tags": [], "note": "", "stale": None}  # fmt: skip
    assert (
        call(
            "/api/judge",
            {"dataset": "demo", "episode": 1, "verdict": "problem", "tags": ["hand", "hand"], "note": " x "},
        )[0]
        == 200
    )
    assert call("/api/judge", {"dataset": "demo", "episode": 2, "verdict": "ok"})[0] == 200
    row = json.loads(call("/api/episodes")[1])["episodes"][1]
    assert (row["verdict"], row["tags"], row["note"]) == ("problem", ["hand"], "x")
    saved = json.loads((root.parent / "demo_review/judgements.json").read_text())
    assert set(saved) == {"1", "2"} and saved["1"]["fingerprint"]
    assert len((root.parent / "demo_review/judgements.log.jsonl").read_text().splitlines()) == 2

    assert call("/api/judge", {"dataset": "demo", "episode": 9, "verdict": "ok"})[0] == 400
    assert call("/api/judge", {"dataset": "demo", "episode": 0, "verdict": "great"})[0] == 400
    assert call("/api/judge", {"dataset": "nope", "episode": 0, "verdict": "ok"})[0] == 400
    assert call("/api/judge", {"dataset": "demo", "episode": 2, "verdict": None})[0] == 200  # cleared
    assert set(json.loads((root.parent / "demo_review/judgements.json").read_text())) == {"1"}

    status, body, _ = call("/api/export.csv")
    assert status == 200 and "demo,1,6" in body.decode("utf-8-sig") and "problem,hand,x" in body.decode("utf-8-sig")
    assert b"<title>" in call("/")[1]
    assert call("/nothing")[0] == 404


def test_video_is_served_with_ranges(served):
    _, call = served
    status, body, headers = call("/video?dataset=demo&episode=0")
    assert status == 200 and headers["Content-Type"] == "video/mp4" and headers["Accept-Ranges"] == "bytes"
    status, part, headers = call("/video?dataset=demo&episode=0", headers={"Range": "bytes=10-19"})
    assert status == 206 and part == body[10:20] and headers["Content-Range"] == f"bytes 10-19/{len(body)}"
    status, tail, _ = call("/video?dataset=demo&episode=0", headers={"Range": "bytes=-5"})
    assert status == 206 and tail == body[-5:]
    assert call("/video?dataset=demo&episode=0", headers={"Range": f"bytes={len(body)}-"})[0] == 416
    assert call("/video?dataset=demo&episode=7")[0] == 400


def test_a_verdict_goes_stale_when_the_episode_changes(tmp_path, capsys):
    root = make(tmp_path / "demo", [8, 6, 5, 5])
    dataset = review.Dataset(root)
    for episode, verdict in [(0, "problem"), (2, "problem"), (3, "problem"), (1, "unsure")]:
        dataset.judge(episode, verdict, [], "")
    assert review.main(["list", str(root)]) == 0
    assert capsys.readouterr().out.strip() == "0,2-3"
    dataset.data_path(2).write_bytes(b"another episode now has this number")
    fresh = review.Dataset(root)
    row = fresh.rows()[2]
    assert row["verdict"] is None and row["stale"] == "problem"
    assert review.main(["list", str(root)]) == 0
    assert capsys.readouterr().out.strip() == "0,3"
    assert review.main(["list", str(root), "--verdict", "stale"]) == 0
    assert capsys.readouterr().out.strip() == "2"
    assert review.main(["export", str(root)]) == 0
    assert "demo,2,5" in capsys.readouterr().out


def test_stitch_all_and_bad_input(tmp_path, capsys):
    root = make(tmp_path / "demo", [4, 4])
    assert review.main(["stitch", str(root), "--workers", "2"]) == 0
    assert "2 review videos ready" in capsys.readouterr().out
    assert len(list((tmp_path / "demo_review").glob("episode_*.mp4"))) == 2
    assert review.main(["list", str(tmp_path / "nowhere")]) == 2
    assert "not a LeRobot dataset" in capsys.readouterr().err
    assert review.ranges([5, 1, 2, 3, 9, 9]) == "1-3,5,9" and review.ranges([]) == ""
