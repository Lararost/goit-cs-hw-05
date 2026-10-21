"""Відтворювана демонстрація обох завдань із локальним HTTP URL."""

import asyncio
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import Thread

from task_01.sort_files import sort_files


def main():
    root = Path(__file__).resolve().parent
    with TemporaryDirectory() as directory:
        stats = asyncio.run(sort_files(root / "samples/source",
                                       Path(directory) / "sorted"))
        if stats.errors or stats.copied != 4:
            raise RuntimeError("Демонстрація сортування не пройшла.")
        print(f"Асинхронне сортування: {stats.copied} файли, помилок 0.",
              flush=True)

    class QuietHandler(SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass

    handler = partial(QuietHandler, directory=str(root / "samples"))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}/text.txt"
    try:
        subprocess.run(
            [sys.executable, str(root / "task_02/mapreduce_words.py"), url],
            cwd=root, check=True,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    payload = json.loads((root / "word_counts.json").read_text())
    assert payload["frequencies"]["python"] == 4
    assert payload["frequencies"]["mapreduce"] == 3
    (root / "demo_summary.json").write_text(
        json.dumps({"sorting_copied": stats.copied,
                    "sorting_errors": stats.errors,
                    "http_download": "passed",
                    "mapreduce_known_counts": "passed",
                    "total_words": payload["total_words"]}, indent=2) + "\n",
        encoding="utf-8",
    )
    print("Демонстрація обох завдань — OK.")


if __name__ == "__main__":
    main()
