"""Перевірки реального async I/O та HTTP + MapReduce."""

import asyncio
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Thread
import unittest

from task_01.sort_files import sort_files
from task_02.mapreduce_words import (
    download_text, map_reduce, visualize_top_words,
)


class SortingTests(unittest.TestCase):
    def test_bytes_extensions_and_collisions(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "source", root / "output"
            (source / "nested").mkdir(parents=True)
            files = {"report.txt": b"first", "nested/report.txt": b"second",
                     "nested/a.CSV": b"a,b\n1,2", "LICENSE": b"license",
                     "empty.bin": b"", "large.bin": bytes(range(256)) * 9000}
            for name, content in files.items():
                (source / name).write_bytes(content)
            stats = asyncio.run(sort_files(source, output, workers=3))
            self.assertEqual(stats.copied, len(files))
            self.assertEqual(stats.errors, 0)
            for name, content in files.items():
                extension = Path(name).suffix.lower().lstrip(".")
                extension = extension or "no_extension"
                destination = output / extension / name
                self.assertEqual(destination.read_bytes(), content)
            with self.assertLogs("task_01.sort_files", level=logging.ERROR):
                second = asyncio.run(sort_files(source, output, workers=2))
            self.assertEqual(second.errors, len(files))
            saved = (output / "txt/report.txt").read_bytes()
            self.assertEqual(saved, b"first")

    def test_invalid_paths_and_symlink(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            for output in (source, source / "inside", root):
                with self.assertRaises(ValueError):
                    asyncio.run(sort_files(source, output))
            with self.assertRaises(ValueError):
                asyncio.run(sort_files(root / "missing", root / "out"))
            with self.assertRaises(ValueError):
                asyncio.run(sort_files(source, root / "out", workers=0))
            (source / "loop").symlink_to(source, target_is_directory=True)
            with self.assertLogs("task_01.sort_files", level=logging.WARNING):
                stats = asyncio.run(sort_files(source, root / "out"))
            self.assertEqual(stats.skipped, 1)
            self.assertEqual(stats.copied, 0)


class WordTests(unittest.TestCase):
    def test_mapreduce_unicode_and_chart(self):
        text = "Python PYTHON python3; кіт КІТ; п'ять п’ять пʼять. 123 _"
        expected = {"python": 3, "кіт": 2, "п'ять": 3}
        for workers in (1, 4):
            self.assertEqual(map_reduce(text, workers), expected)
        self.assertEqual(map_reduce("123 !!!"), {})
        with self.assertRaises(ValueError):
            map_reduce("test", workers=0)
        with TemporaryDirectory() as directory:
            output = Path(directory) / "chart.png"
            top = visualize_top_words(expected, 2, output)
            self.assertEqual(top, [("python", 3), ("п'ять", 3)])
            self.assertEqual(output.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            visualize_top_words({}, 10, output)
            self.assertGreater(output.stat().st_size, 1000)

    def test_real_http_and_network_errors(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "text.txt"
            path.write_text("Python Python кіт", encoding="utf-8")

            class QuietHandler(SimpleHTTPRequestHandler):
                def log_message(self, *args):
                    pass

            handler = partial(QuietHandler, directory=directory)
            server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
            thread = Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                url = f"http://127.0.0.1:{server.server_port}/text.txt"
                self.assertEqual(map_reduce(download_text(url)),
                                 {"python": 2, "кіт": 1})
                with self.assertRaises(ValueError):
                    download_text(url, max_bytes=2)
                from urllib.error import HTTPError
                with self.assertRaises(HTTPError):
                    download_text(url + ".missing")
            finally:
                server.shutdown()
                server.server_close()
                thread.join()
        with self.assertRaises(ValueError):
            download_text("file:///tmp/text.txt")


if __name__ == "__main__":
    unittest.main()
