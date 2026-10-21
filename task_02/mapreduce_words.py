"""Завантажити текст за URL, порахувати слова MapReduce і зберегти графік."""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
from itertools import chain
import logging
from pathlib import Path
import re
from time import perf_counter
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


LOGGER = logging.getLogger(__name__)
WORD_PATTERN = re.compile(r"[^\W\d_]+(?:['’ʼ][^\W\d_]+)*", re.UNICODE)


def download_text(url, timeout=30, max_bytes=20 * 1024 * 1024):
    """Завантажити HTTP(S) текст із тайм-аутом і перевіркою кодування."""
    if urlsplit(url).scheme not in {"http", "https"}:
        raise ValueError("URL має починатися з http:// або https://")
    request = Request(url, headers={"User-Agent": "GoIT-CS-HW05/1.0"})
    with urlopen(request, timeout=timeout) as response:
        raw = response.read(max_bytes + 1)
        if len(raw) > max_bytes:
            raise ValueError("Текст перевищує ліміт 20 MiB.")
        encoding = response.headers.get_content_charset() or "utf-8-sig"
        return raw.decode(encoding)


def tokenize(text):
    """Unicode-слова: регістр не враховуємо; апострофи нормалізуємо."""
    normalized = text.casefold().replace("’", "'").replace("ʼ", "'")
    return WORD_PATTERN.findall(normalized)


def map_function(word):
    """Map: перетворити слово на пару (слово, 1)."""
    return word, 1


def shuffle_function(mapped):
    """Shuffle: згрупувати одиниці за ключем-словом."""
    grouped = defaultdict(list)
    for word, count in mapped:
        grouped[word].append(count)
    return grouped.items()


def reduce_function(item):
    """Reduce: підсумувати значення для одного слова."""
    word, counts = item
    return word, sum(counts)


def map_chunk(words):
    """Обробити пакет слів, зменшуючи витрати на задачі пулу."""
    return [map_function(word) for word in words]


def map_reduce(text, workers=4):
    """Виконати Map -> Shuffle -> Reduce з пулом потоків."""
    if workers < 1:
        raise ValueError("Кількість workers має бути більшою за нуль.")
    words = tokenize(text)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        chunks = (words[i:i + 1000] for i in range(0, len(words), 1000))
        mapped = chain.from_iterable(executor.map(map_chunk, chunks))
        grouped = shuffle_function(mapped)
        reduced = executor.map(reduce_function, grouped)
        return dict(sorted(reduced))


def visualize_top_words(frequencies, top_n=10, output="top_words.png"):
    """Зберегти горизонтальну діаграму; повернути топ (слово, частота)."""
    if top_n < 1:
        raise ValueError("top_n має бути більшим за нуль.")
    ranking = sorted(frequencies.items(), key=lambda item: (-item[1], item[0]))
    top = ranking[:top_n]
    figure, axis = plt.subplots(figsize=(10, max(4, len(top) * 0.42)))
    try:
        if top:
            words, counts = zip(*top)
            bars = axis.barh(words, counts, color="#297d9c")
            axis.invert_yaxis()
            axis.bar_label(bars, padding=4)
            axis.set_xlim(0, max(counts) * 1.18)
        else:
            axis.text(0.5, 0.5, "No words found", ha="center", va="center")
        axis.set_title(f"Top {len(top)} words by frequency")
        axis.set_xlabel("Occurrences")
        axis.set_ylabel("Word")
        axis.spines[["top", "right"]].set_visible(False)
        figure.tight_layout()
        figure.savefig(output, dpi=180)
    finally:
        plt.close(figure)
    return top


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url", help="URL звичайного тексту (не HTML)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--chart", type=Path, default=Path("top_words.png"))
    parser.add_argument("--json", type=Path, default=Path("word_counts.json"))
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--log-file", default="analysis_errors.log")
    args = parser.parse_args()
    try:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(message)s",
            handlers=[logging.FileHandler(args.log_file, encoding="utf-8"),
                      logging.StreamHandler()],
        )
        if args.top < 1 or args.workers < 1 or args.timeout <= 0:
            raise ValueError("top, workers і timeout мають бути додатними.")
        started = perf_counter()
        text = download_text(args.url, args.timeout)
        frequencies = map_reduce(text, args.workers)
        top = visualize_top_words(frequencies, args.top, args.chart)
        payload = {"url": args.url, "total_words": sum(frequencies.values()),
                   "unique_words": len(frequencies), "top_words": top,
                   "frequencies": frequencies}
        serialized = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        args.json.write_text(serialized, encoding="utf-8")
        for word, count in top:
            print(f"{word}: {count}")
        total = payload["total_words"]
        print(f"Слів: {total}; унікальних: {len(frequencies)}")
        print(f"Графік: {args.chart}; JSON: {args.json}; "
              f"час: {perf_counter() - started:.4f} с")
        return 0
    except (OSError, HTTPError, URLError, ValueError,
            UnicodeError, LookupError):
        LOGGER.exception("Не вдалося завантажити або проаналізувати текст")
        return 1
    except KeyboardInterrupt:
        LOGGER.warning("Аналіз перервано користувачем")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
