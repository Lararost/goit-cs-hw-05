"""Асинхронне копіювання файлів у підпапки за розширеннями."""

import argparse
import asyncio
from dataclasses import dataclass
import logging
from time import perf_counter

from anyio import Path, open_file


LOGGER = logging.getLogger(__name__)
CHUNK_SIZE = 1024 * 1024


@dataclass
class Statistics:
    """Лічильники змінюються лише в одному event loop між await."""

    copied: int = 0
    errors: int = 0
    skipped: int = 0


async def copy_file(source, output, root, stats):
    """Копіювати блоками, зберігаючи відносний шлях і всі байти.

    Наприклад a/report.TXT -> output/txt/a/report.TXT.
    Режим xb забороняє перезапис наявного файлу.
    """
    destination = None
    created = False
    try:
        extension = source.suffix.lower().lstrip(".") or "no_extension"
        destination = output / extension / source.relative_to(root)
        await destination.parent.mkdir(parents=True, exist_ok=True)
        async with await open_file(source, "rb") as reader:
            async with await open_file(destination, "xb") as writer:
                created = True
                while block := await reader.read(CHUNK_SIZE):
                    await writer.write(block)
        stats.copied += 1
    except OSError:
        stats.errors += 1
        LOGGER.exception("Не вдалося скопіювати %s у %s", source, destination)
        if created:
            try:
                await destination.unlink(missing_ok=True)
            except OSError:
                LOGGER.exception(
                    "Не вдалося прибрати неповну копію %s", destination
                )


async def read_folder(source, queue, stats):
    """Рекурсивно читати асинхронні Path та передавати файли workers."""
    try:
        async for entry in source.iterdir():
            try:
                if await entry.is_symlink():
                    stats.skipped += 1
                    LOGGER.warning("Пропущено символічне посилання: %s", entry)
                elif await entry.is_dir():
                    await read_folder(entry, queue, stats)
                elif await entry.is_file():
                    await queue.put(entry)
                else:
                    stats.skipped += 1
                    LOGGER.warning("Пропущено спеціальний файл: %s", entry)
            except OSError:
                stats.errors += 1
                LOGGER.exception("Не вдалося перевірити %s", entry)
    except OSError:
        stats.errors += 1
        LOGGER.exception("Не вдалося прочитати папку %s", source)


async def sort_files(source, output, workers=8):
    """Запустити обхід та обмежену кількість одночасних копіювань."""
    if workers < 1:
        raise ValueError("Кількість workers має бути більшою за нуль.")
    source = await Path(source).resolve()
    output = await Path(output).resolve()
    if not await source.is_dir():
        raise ValueError(f"Вихідної папки не існує: {source}")
    if (source == output or source in output.parents
            or output in source.parents):
        raise ValueError("Source та output мають бути окремими папками.")
    await output.mkdir(parents=True, exist_ok=True)
    stats = Statistics()
    queue = asyncio.Queue(maxsize=workers * 2)

    async def worker():
        while True:
            entry = await queue.get()
            try:
                if entry is None:
                    return
                await copy_file(entry, output, source, stats)
            finally:
                queue.task_done()

    # TaskGroup скасовує решту задач, якщо виникне неочікувана помилка.
    async with asyncio.TaskGroup() as group:
        for _ in range(workers):
            group.create_task(worker())
        await read_folder(source, queue, stats)
        for _ in range(workers):
            await queue.put(None)
        await queue.join()
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="Вихідна папка")
    parser.add_argument("output", help="Папка призначення")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--log-file", default="sort_errors.log")
    args = parser.parse_args()
    try:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(message)s",
            handlers=[logging.FileHandler(args.log_file, encoding="utf-8"),
                      logging.StreamHandler()],
        )
        started = perf_counter()
        stats = asyncio.run(sort_files(args.source, args.output, args.workers))
        elapsed = perf_counter() - started
        print(f"Скопійовано: {stats.copied}; помилок: {stats.errors}; "
              f"пропущено: {stats.skipped}; час: {elapsed:.4f} с")
        return 1 if stats.errors else 0
    except (OSError, ValueError, ExceptionGroup):
        LOGGER.exception("Сортування не завершено")
        return 1
    except KeyboardInterrupt:
        LOGGER.warning("Сортування перервано користувачем")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
