"""rollout-review: look through the episodes of LeRobot v2.1 datasets in a browser and judge them by hand.

    rollout-review serve  <dataset>... [--port 8790] [--host 127.0.0.1]   the page
    rollout-review stitch <dataset>... [--workers 4]                      make all review videos ahead of time
    rollout-review list   <dataset> [--verdict problem]                   episode numbers, e.g. "3,10-12"
    rollout-review export <dataset>... > judgements.csv

A review video shows all cameras of an episode in one picture with the episode number, the task and a frame counter.
It is made with ffmpeg when first asked for and kept in ``<dataset>_review/`` NEXT TO the dataset, never inside it
(lerobot counts every mp4 under a dataset). Verdicts (ok / problem / unsure, tags, a note) go to
``<dataset>_review/judgements.json``; every change is also appended to ``judgements.log.jsonl``.
The datasets themselves are only read.

A verdict is tied to the episode's number and to a fingerprint of its data file. When episodes are renumbered or
re-recorded, verdicts whose fingerprint no longer matches are shown as stale and are not counted.

Python standard library only (3.10+); needs ffmpeg and ffprobe for the videos.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import datetime
import hashlib
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
import io
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import urllib.parse

VERSION = "0.1.0"
VERDICTS = ("ok", "problem", "unsure")
CELL_HEIGHT = 360
lock = threading.Lock()
stitch_locks: dict[Path, threading.Lock] = {}


class ReviewError(Exception):
    pass


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


class Dataset:
    def __init__(self, root: Path | str):
        self.root = Path(root).resolve()
        info = self.root / "meta/info.json"
        if not info.exists():
            raise ReviewError(f"{root} is not a LeRobot dataset (no meta/info.json)")
        self.info = json.loads(info.read_text())
        self.name = self.root.name
        self.review = self.root.parent / f"{self.root.name}_review"
        self.cameras = [k for k, v in self.info["features"].items() if v.get("dtype") == "video"]
        self.chunks = int(self.info.get("chunks_size", 1000))
        self._fingerprints: dict[int, tuple[float, str]] = {}

    def episodes(self) -> list[dict]:
        return sorted(read_jsonl(self.root / "meta/episodes.jsonl"), key=lambda e: e["episode_index"])

    def data_path(self, episode: int) -> Path:
        return self.root / self.info["data_path"].format(episode_chunk=episode // self.chunks, episode_index=episode)

    def video_path(self, episode: int, camera: str) -> Path:
        return self.root / self.info["video_path"].format(
            episode_chunk=episode // self.chunks, video_key=camera, episode_index=episode
        )

    def fingerprint(self, episode: int) -> str:
        """Changes when the episode's rows change (another episode got this number, or it was recorded again)."""
        path = self.data_path(episode)
        try:
            stat = path.stat()
        except OSError:
            return ""
        hit = self._fingerprints.get(episode)
        if hit and hit[0] == stat.st_mtime:
            return hit[1]
        digest = hashlib.sha1(path.read_bytes()).hexdigest()[:16]  # noqa: S324  (an identity, not security)
        self._fingerprints[episode] = (stat.st_mtime, digest)
        return digest

    def judgements(self) -> dict:
        path = self.review / "judgements.json"
        try:
            return json.loads(path.read_text()) if path.exists() else {}
        except ValueError as error:
            raise ReviewError(f"{path} is not valid JSON; it is left alone ({error})") from error

    def rows(self) -> list[dict]:
        """One row per episode with its verdict; a verdict given to other data under this number is stale."""
        mine = self.judgements()
        out = []
        for episode in self.episodes():
            number = episode["episode_index"]
            entry = mine.get(str(number))
            stale = bool(entry) and entry.get("fingerprint") != self.fingerprint(number)
            out.append({
                "dataset": self.name, "episode": number, "length": episode.get("length"),
                "seconds": round(episode.get("length", 0) / self.info["fps"], 1),
                "task": (episode.get("tasks") or [""])[0],
                "verdict": None if not entry or stale else entry["verdict"],
                "tags": [] if not entry or stale else entry.get("tags", []),
                "note": "" if not entry or stale else entry.get("note", ""),
                "stale": entry["verdict"] if stale else None,
            })  # fmt: skip
        return out

    def judge(self, episode: int, verdict: str | None, tags: list[str], note: str) -> dict:
        if verdict not in (*VERDICTS, None):
            raise ReviewError(f"verdict must be one of {VERDICTS} or null")
        if episode not in {e["episode_index"] for e in self.episodes()}:
            raise ReviewError(f"{self.name} has no episode {episode}")
        now = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
        with lock:
            mine = self.judgements()
            if verdict is None:
                mine.pop(str(episode), None)
                entry = {"verdict": None}
            else:
                entry = {"verdict": verdict, "tags": sorted({t.strip() for t in tags if t.strip()}), "note": note.strip(),
                         "fingerprint": self.fingerprint(episode), "time": now}  # fmt: skip
                mine[str(episode)] = entry
            self.review.mkdir(exist_ok=True)
            tmp = self.review / ".judgements.json.tmp"
            tmp.write_text(json.dumps(mine, indent=1, ensure_ascii=False) + "\n")
            tmp.replace(self.review / "judgements.json")
            with open(self.review / "judgements.log.jsonl", "a", encoding="utf-8") as log:
                log.write(json.dumps({"episode": episode, **entry, "time": now}, ensure_ascii=False) + "\n")
        return entry

    def stitched(self, episode: int) -> Path:
        """The episode's review video; made now if it is missing or older than its sources."""
        if not self.cameras:
            raise ReviewError(f"{self.name} has no video features")
        sources = [self.video_path(episode, camera) for camera in self.cameras]
        missing = [p for p in sources if not p.exists()]
        if missing:
            raise ReviewError(f"missing video {missing[0]}")
        out = self.review / f"episode_{episode:06d}.mp4"
        with lock:
            mine = stitch_locks.setdefault(out, threading.Lock())
        with mine:  # two requests for one episode: the second waits and finds it done
            if out.exists() and out.stat().st_mtime >= max(p.stat().st_mtime for p in sources):
                return out
            self.review.mkdir(exist_ok=True)
            task = next((e.get("tasks") or [""] for e in self.episodes() if e["episode_index"] == episode), [""])[0]
            stitch(sources, out, f"{self.name}  episode {episode}  {task}")
        return out


def font_option() -> str:
    for path in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                 "/System/Library/Fonts/Helvetica.ttc"):  # fmt: skip
        if Path(path).exists():
            return f"fontfile={path}:"
    return ""


def stitch(sources: list[Path], out: Path, title: str) -> None:
    """All sources side by side (a grid from four on) in one H.264 video that browsers play, with the title on top
    and a frame counter. The title goes through a text file, so any characters are safe."""
    if shutil.which("ffmpeg") is None:
        raise ReviewError("ffmpeg is not installed")
    n = len(sources)
    columns = n if n <= 3 else math.ceil(math.sqrt(n))
    width = CELL_HEIGHT * 4 // 3
    filters = [f"[{i}:v]scale={width}:{CELL_HEIGHT}:force_original_aspect_ratio=decrease,"
               f"pad={width}:{CELL_HEIGHT}:(ow-iw)/2:(oh-ih)/2,setsar=1[v{i}]" for i in range(n)]  # fmt: skip
    if n == 1:
        grid = "[v0]"
    else:
        layout = "|".join(f"{(i % columns) * width}_{(i // columns) * CELL_HEIGHT}" for i in range(n))
        filters.append("".join(f"[v{i}]" for i in range(n)) + f"xstack=inputs={n}:layout={layout}:fill=black[grid]")
        grid = "[grid]"
    font = font_option()
    with tempfile.TemporaryDirectory() as tmp:
        text = Path(tmp) / "title.txt"
        text.write_text(title.replace("%", "%%")[:160])
        box = "fontcolor=white:fontsize=16:box=1:boxcolor=black@0.55:boxborderw=4"
        filters.append(f"{grid}drawtext={font}textfile={text}:x=6:y=6:{box},"
                       f"drawtext={font}text='%{{n}}':x=w-tw-6:y=6:{box},format=yuv420p[out]")  # fmt: skip
        tmp_out = out.with_name(f".{out.name}.part.mp4")
        command = ["ffmpeg", "-loglevel", "error", "-y"]
        for source in sources:
            command += ["-i", str(source)]
        command += ["-filter_complex", ";".join(filters), "-map", "[out]", "-an", "-c:v", "libx264", "-preset", "veryfast",
                    "-crf", "26", "-movflags", "+faststart", str(tmp_out)]  # fmt: skip
        done = subprocess.run(command, capture_output=True, text=True)
        if done.returncode != 0:
            tmp_out.unlink(missing_ok=True)
            raise ReviewError(f"ffmpeg failed on {sources[0].name}: {done.stderr.strip()[-300:]}")
        tmp_out.replace(out)


def ranges(numbers: list[int]) -> str:
    """[0, 2, 3, 4, 9] -> "0,2-4,9" (the form lerobot-surgery drop --drop takes)."""
    parts, numbers = [], sorted(set(numbers))
    i = 0
    while i < len(numbers):
        j = i
        while j + 1 < len(numbers) and numbers[j + 1] == numbers[j] + 1:
            j += 1
        parts.append(str(numbers[i]) if i == j else f"{numbers[i]}-{numbers[j]}")
        i = j + 1
    return ",".join(parts)


class App:
    def __init__(self, roots: list[Path]):
        self.datasets: dict[str, Dataset] = {}
        for root in roots:
            dataset = Dataset(root)
            if dataset.name in self.datasets:
                raise ReviewError(f"two datasets are called {dataset.name}; give them different directory names")
            self.datasets[dataset.name] = dataset

    def dataset(self, name: str) -> Dataset:
        if name not in self.datasets:
            raise ReviewError(f"no dataset {name!r}")
        return self.datasets[name]

    def episodes(self) -> dict:
        rows = [row for dataset in self.datasets.values() for row in dataset.rows()]
        count = {v: sum(1 for r in rows if r["verdict"] == v) for v in VERDICTS}
        return {"episodes": rows, "count": {**count, "none": sum(1 for r in rows if r["verdict"] is None), "all": len(rows)},
                "datasets": [{"name": d.name, "path": str(d.root), "cameras": d.cameras, "fps": d.info["fps"]}
                             for d in self.datasets.values()]}  # fmt: skip

    def export(self) -> str:
        out = io.StringIO()
        writer = csv.writer(out)
        writer.writerow(["dataset", "episode", "length", "task", "verdict", "tags", "note", "stale_verdict"])
        for row in self.episodes()["episodes"]:
            writer.writerow([row["dataset"], row["episode"], row["length"], row["task"], row["verdict"] or "",
                             " ".join(row["tags"]), row["note"], row["stale"] or ""])  # fmt: skip
        return out.getvalue()


def make_handler(app: App):
    page = (Path(__file__).parent / "index.html").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):  # quiet
            pass

        def send(self, body: bytes, kind: str, status: int = 200, extra: dict | None = None) -> None:
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for key, value in (extra or {}).items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(body)

        def send_json(self, value, status: int = 200) -> None:
            self.send(json.dumps(value, ensure_ascii=False).encode(), "application/json; charset=utf-8", status)

        def send_file(self, path: Path) -> None:
            """With Range support: browsers need it to seek in a video."""
            size = path.stat().st_size
            start, end = 0, size - 1
            header = self.headers.get("Range", "")
            partial = header.startswith("bytes=")
            if partial:
                first, _, last = header[6:].split(",")[0].partition("-")
                if first:
                    start, end = int(first), min(int(last), size - 1) if last else size - 1
                else:
                    start = max(0, size - int(last))
                if start > end or start >= size:
                    self.send(b"", "video/mp4", 416, {"Content-Range": f"bytes */{size}"})
                    return
            self.send_response(206 if partial else 200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            if partial:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            with open(path, "rb") as f:
                f.seek(start)
                left = end - start + 1
                while left > 0 and (chunk := f.read(min(1 << 20, left))):
                    self.wfile.write(chunk)
                    left -= len(chunk)

        def do_GET(self):  # noqa: N802
            url = urllib.parse.urlparse(self.path)
            query = {k: v[0] for k, v in urllib.parse.parse_qs(url.query).items()}
            try:
                if url.path == "/":
                    self.send(page, "text/html; charset=utf-8")
                elif url.path == "/api/episodes":
                    self.send_json(app.episodes())
                elif url.path == "/api/export.csv":
                    self.send(app.export().encode("utf-8-sig"), "text/csv; charset=utf-8",
                              extra={"Content-Disposition": "attachment; filename=judgements.csv"})  # fmt: skip
                elif url.path == "/video":
                    self.send_file(app.dataset(query.get("dataset", "")).stitched(int(query.get("episode", "-1"))))
                else:
                    self.send_json({"error": "not found"}, 404)
            except (ReviewError, ValueError) as error:
                self.send_json({"error": str(error)}, 400)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_POST(self):  # noqa: N802
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))) or b"{}")
                if self.path == "/api/judge":
                    entry = app.dataset(body.get("dataset", "")).judge(
                        int(body["episode"]),
                        body.get("verdict"),
                        list(body.get("tags") or []),
                        str(body.get("note") or ""),
                    )
                    self.send_json({"saved": entry})
                else:
                    self.send_json({"error": "not found"}, 404)
            except (ReviewError, ValueError, KeyError, TypeError) as error:
                self.send_json({"error": str(error)}, 400)

    return Handler


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="rollout-review", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--version", action="version", version=VERSION)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("serve", help="the review page")
    p.add_argument("datasets", nargs="+", type=Path)
    p.add_argument("--port", type=int, default=8790)
    p.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to open it to the network (there is no login)")
    p = sub.add_parser("stitch", help="make all review videos now")
    p.add_argument("datasets", nargs="+", type=Path)
    p.add_argument("--workers", type=int, default=4)
    p = sub.add_parser("list", help='episode numbers with a verdict, as "3,10-12"')
    p.add_argument("dataset", type=Path)
    p.add_argument("--verdict", default="problem", choices=[*VERDICTS, "none", "stale"])
    p = sub.add_parser("export", help="all verdicts as CSV on standard output")
    p.add_argument("datasets", nargs="+", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "list":
            rows = Dataset(args.dataset).rows()
            pick = {"none": lambda r: r["verdict"] is None and not r["stale"], "stale": lambda r: bool(r["stale"])}.get(
                args.verdict, lambda r: r["verdict"] == args.verdict
            )
            print(ranges([r["episode"] for r in rows if pick(r)]))
            return 0
        app = App(args.datasets)
        if args.command == "export":
            sys.stdout.write(app.export())
            return 0
        if args.command == "stitch":
            jobs = [(d, e["episode_index"]) for d in app.datasets.values() for e in d.episodes()]
            failed = 0
            with ThreadPoolExecutor(args.workers) as pool:
                for (dataset, episode), result in zip(
                    jobs, pool.map(lambda j: _try(j[0].stitched, j[1]), jobs), strict=True
                ):
                    if isinstance(result, Exception):
                        failed += 1
                        print(f"{dataset.name} episode {episode}: {result}", file=sys.stderr)
            print(f"{len(jobs) - failed} review videos ready" + (f", {failed} failed" if failed else ""))
            return 1 if failed else 0
        server = ThreadingHTTPServer((args.host, args.port), make_handler(app))
        total = sum(len(d.episodes()) for d in app.datasets.values())
        print(
            f"rollout-review: {len(app.datasets)} datasets, {total} episodes at http://{args.host}:{args.port}/",
            flush=True,
        )
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        return 0
    except ReviewError as error:
        print(f"rollout-review: {error}", file=sys.stderr)
        return 2


def _try(function, *args):
    try:
        return function(*args)
    except ReviewError as error:
        return error


if __name__ == "__main__":
    sys.exit(main())
