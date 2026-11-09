#!/usr/bin/env python3
"""Live MinerU progress dashboard for the Russian chemistry collection.

Serves a page that reloads the ledger, Markdown outputs, and cloud log.
Token counts use the same 16k BPE as training.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mineru_ledger as ledger
from run_mineru_cloud import (
    CHEM_TEXT,
    CYRILLIC,
    EXTENSIONS,
    LATIN,
    SRC,
    tesseract_pending,
)

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "logs" / "mineru_cloud.log"
CACHE_PATH = ledger.OUT / "_progress_cache.json"
TOKENIZER_PATH = ROOT / "tokenizers" / "fineweb2_rus_v001" / "chosen" / "tokenizer.json"
CULTURA_TRAIN_TOKENS = 625_624_621
MILESTONES = [
    (10_000_000, "Короткое дообучение 10.9M-модели"),
    (50_000_000, "Плотный химический корпус для этой сети"),
    (100_000_000, "Сильный доменный набор"),
    (CULTURA_TRAIN_TOKENS, "Масштаб исходного pretrain Cultura-Ru-Edu"),
]

_CACHE_LOCK = threading.Lock()
_SNAP_LOCK = threading.Lock()
_CACHE: dict = {"outputs": {}, "pages": {}}
_TOKENIZER = None
_SNAPSHOT: dict | None = None
_SNAP_AT = 0.0
_TOKEN_THREAD_STARTED = False
PAGE_MARK = re.compile(r"(?:----- page\s+(\d+) -----|<!--\s*page\s+(\d+))", re.I)
START_RE = re.compile(r"^START\s+(\d+)/(\d+)\s+\[([^\]]+)\]\s+(.+)$")
DONE_RE = re.compile(r"^(OK|ERROR|SKIP latin)\s+(\d+)/(\d+)\s+(?:\[([^\]]+)\]\s+)?(.+?)(?:\s+sec=([\d.]+))?$")


def load_cache() -> None:
    global _CACHE
    if not CACHE_PATH.is_file():
        return
    try:
        data = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if isinstance(data, dict):
        _CACHE = {
            "outputs": data.get("outputs") or {},
            "pages": data.get("pages") or {},
        }


def save_cache() -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = CACHE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(_CACHE), encoding="utf-8")
    tmp.replace(CACHE_PATH)


def tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        from tokenizers import Tokenizer

        _TOKENIZER = Tokenizer.from_file(str(TOKENIZER_PATH))
    return _TOKENIZER


def parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def chem_text_pages() -> dict[str, int]:
    pages: dict[str, int] = {}
    for record in ledger.iter_records(CHEM_TEXT / "_manifest.jsonl"):
        source = record.get("source")
        count = record.get("pages")
        if source and isinstance(count, int) and count > 0:
            pages[str(source)] = count
    return pages


def latest_full_ok() -> dict[str, dict]:
    found: dict[str, dict] = {}
    for record in ledger.iter_records():
        if record.get("status") != "ok":
            continue
        if record.get("page_ranges"):
            continue
        source = record.get("source")
        output = record.get("output")
        if not source or not output:
            continue
        found[str(source)] = record
    usable: dict[str, dict] = {}
    for source, record in found.items():
        path = ledger.resolve_output(record.get("output"))
        if ledger._usable(path):
            usable[source] = record
    return usable


def cyr_ratio(text: str) -> float:
    cyr = len(CYRILLIC.findall(text))
    lat = len(LATIN.findall(text))
    total = cyr + lat
    return (cyr / total) if total else 0.0


def name_is_russian(name: str) -> bool:
    return bool(CYRILLIC.search(name) or "(ru)" in name.lower())


def chem_text_russian(source: str) -> bool | None:
    txt = CHEM_TEXT / Path(source).with_suffix(".txt")
    if not txt.is_file():
        return None
    sample = txt.read_text(encoding="utf-8", errors="replace")[:120_000]
    ratio = cyr_ratio(sample)
    if ratio == 0 and not sample.strip():
        return None
    return ratio >= 0.5


def is_russian_book(source: str, md_meta: dict | None) -> bool:
    if name_is_russian(Path(source).name):
        return True
    if md_meta and md_meta.get("cyr_ratio", 0) >= 0.5:
        return True
    flag = chem_text_russian(source)
    if flag is not None:
        return flag
    return False


def count_md(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    marks = PAGE_MARK.findall(text)
    pages = 0
    if marks:
        nums = [int(a or b) for a, b in marks]
        pages = max(nums) if nums else 0
    tokens = len(tokenizer().encode(text).ids)
    return {
        "chars": len(text),
        "tokens": tokens,
        "pages_marked": pages,
        "cyr_ratio": round(cyr_ratio(text[:80_000]), 4),
        "mtime_ns": path.stat().st_mtime_ns,
        "size": path.stat().st_size,
    }


def cached_output(rel: str, path: Path) -> dict | None:
    with _CACHE_LOCK:
        hit = _CACHE["outputs"].get(rel)
        if not hit:
            return None
        try:
            st = path.stat()
        except OSError:
            return None
        if hit.get("mtime_ns") == st.st_mtime_ns and hit.get("size") == st.st_size:
            return hit
    return None


def store_output(rel: str, meta: dict) -> None:
    with _CACHE_LOCK:
        _CACHE["outputs"][rel] = meta
        save_cache()


def token_worker() -> None:
    while True:
        try:
            done = latest_full_ok()
            pending = []
            for record in done.values():
                rel = str(record["output"])
                path = ledger.resolve_output(rel)
                if path is None or not path.is_file():
                    continue
                if cached_output(rel, path) is None:
                    pending.append((rel, path))
            for rel, path in pending[:8]:
                try:
                    store_output(rel, count_md(path))
                except Exception:
                    continue
            time.sleep(0.4 if pending else 2.0)
        except Exception:
            time.sleep(3.0)


def runner_status() -> dict:
    import subprocess

    try:
        result = subprocess.run(
            ["pgrep", "-af", "run_mineru_cloud.py"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return {"alive": False, "detail": "pgrep unavailable"}
    lines = [
        line
        for line in result.stdout.splitlines()
        if "run_mineru_cloud.py" in line and "mineru_dashboard" not in line
    ]
    return {"alive": bool(lines), "detail": lines[0][:180] if lines else ""}


def parse_cloud_log(limit: int = 4000) -> dict:
    if not LOG.is_file():
        return {"active": [], "last_ok": [], "queue_total": None}
    data = LOG.read_bytes()
    if len(data) > 2_000_000:
        data = data[-2_000_000:]
    text = data.decode("utf-8", errors="replace")
    lines = text.splitlines()[-limit:]
    started: dict[str, dict] = {}
    last_ok: list[dict] = []
    queue_total = None
    for line in lines:
        stripped = line.strip()
        start = START_RE.match(stripped)
        if start:
            index, total, key, name = start.groups()
            queue_total = int(total)
            started[name] = {"name": name, "index": int(index), "key": key, "status": "running"}
            continue
        done = DONE_RE.match(stripped)
        if done:
            status, index, total, key, rest, sec = done.groups()
            queue_total = int(total)
            name = rest.strip()
            started.pop(name, None)
            if status == "OK":
                last_ok.append(
                    {
                        "name": name,
                        "index": int(index),
                        "key": key or "",
                        "seconds": float(sec) if sec else None,
                    }
                )
    return {
        "active": list(started.values())[-16:],
        "last_ok": last_ok[-12:],
        "queue_total": queue_total,
    }


def in_progress_parts() -> list[dict]:
    items = []
    if not ledger.OUT.is_dir():
        return items
    for parts in ledger.OUT.rglob("*.parts"):
        if not parts.is_dir():
            continue
        files = list(parts.glob("*.md"))
        if not files:
            continue
        newest = max(f.stat().st_mtime for f in files)
        items.append(
            {
                "name": parts.with_suffix("").name,
                "parts": len(files),
                "age_min": round((time.time() - newest) / 60, 1),
            }
        )
    items.sort(key=lambda row: row["age_min"])
    return items[:20]


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def build_snapshot() -> dict:
    files = [
        path
        for path in SRC.rglob("*")
        if path.is_file() and path.suffix.lower() in EXTENSIONS
    ]
    chem_pages = chem_text_pages()
    done_map = latest_full_ok()
    tess = tesseract_pending()
    log_info = parse_cloud_log()
    runner = runner_status()

    russian_done = []
    other_done = []
    tokens_ready = 0
    chars_ready = 0
    pages_done_ru = 0
    tokens_uncached = 0
    now = datetime.now(timezone.utc)
    recent_pages = defaultdict(int)
    recent_books = defaultdict(int)

    for source, record in done_map.items():
        rel = str(record["output"])
        path = ledger.resolve_output(rel)
        meta = cached_output(rel, path) if path else None
        russian = is_russian_book(source, meta)
        pages = int(record.get("pages") or chem_pages.get(source) or (meta or {}).get("pages_marked") or 0)
        chars = int((meta or {}).get("chars") or record.get("chars") or 0)
        tokens = int((meta or {}).get("tokens") or 0)
        if meta is None:
            tokens_uncached += 1
        row = {
            "source": source,
            "name": Path(source).name,
            "pages": pages,
            "chars": chars,
            "tokens": tokens,
            "finished": record.get("finished"),
            "backend": record.get("backend") or "local",
        }
        if russian:
            russian_done.append(row)
            pages_done_ru += pages
            chars_ready += chars
            tokens_ready += tokens
            finished = parse_ts(record.get("finished"))
            if finished:
                hours = (now - finished).total_seconds() / 3600
                if 0 <= hours <= 24:
                    bucket = int(hours)
                    recent_pages[bucket] += pages
                    recent_books[bucket] += 1
        else:
            other_done.append(row)

    pending_ru = []
    pending_pages = 0
    known_page_counts = []
    latin_skip = 0
    for path in files:
        key = ledger.source_key(path)
        if key in done_map:
            continue
        if is_russian_book(key, None):
            pages = int(chem_pages.get(key) or 0)
            pending_ru.append(key)
            if pages:
                pending_pages += pages
                known_page_counts.append(pages)
        else:
            latin_skip += 1

    avg_pending_pages = mean(known_page_counts) if known_page_counts else mean(
        [row["pages"] for row in russian_done if row["pages"]]
    )
    unknown_page_books = len(pending_ru) - len(known_page_counts)
    est_pending_pages = pending_pages + int(round(unknown_page_books * avg_pending_pages))

    counted = [row for row in russian_done if row["tokens"] > 0 and row["pages"] > 0]
    tok_per_page = mean([row["tokens"] / row["pages"] for row in counted]) if counted else 0.0
    chars_per_tok = mean(
        [row["chars"] / row["tokens"] for row in russian_done if row["tokens"] > 0 and row["chars"] > 0]
    )
    if tokens_ready == 0 and chars_ready and chars_per_tok == 0:
        chars_per_tok = 2.8
    est_done_tokens = tokens_ready
    if tokens_uncached and chars_per_tok:
        missing_chars = sum(row["chars"] for row in russian_done if row["tokens"] == 0)
        est_done_tokens = tokens_ready + int(missing_chars / chars_per_tok)
    est_remaining_tokens = int(est_pending_pages * tok_per_page) if tok_per_page else 0
    est_full_tokens = est_done_tokens + est_remaining_tokens

    pages_last_6h = sum(recent_pages[h] for h in range(6))
    books_last_6h = sum(recent_books[h] for h in range(6))
    pages_per_hour = pages_last_6h / 6 if pages_last_6h else 0.0
    eta_hours = (est_pending_pages / pages_per_hour) if pages_per_hour else None

    russian_done.sort(key=lambda row: row.get("finished") or "", reverse=True)
    next_milestone = None
    for value, label in MILESTONES:
        if est_done_tokens < value:
            next_milestone = {
                "tokens": value,
                "label": label,
                "remaining": value - est_done_tokens,
                "frac": est_done_tokens / value,
            }
            break

    return {
        "updated": now.isoformat(),
        "runner": runner,
        "collection": {
            "source_files": len(files),
            "latin_skipped": latin_skip,
            "tesseract_first_left": len(tess),
        },
        "russian": {
            "done_books": len(russian_done),
            "pending_books": len(pending_ru),
            "target_books": len(russian_done) + len(pending_ru),
            "done_pages": pages_done_ru,
            "pending_pages_est": est_pending_pages,
            "pending_pages_known": pending_pages,
            "chars": chars_ready,
            "tokens_counted": tokens_ready,
            "tokens_est": est_done_tokens,
            "tokens_uncached_books": tokens_uncached,
            "tokens_remaining_est": est_remaining_tokens,
            "tokens_full_est": est_full_tokens,
            "tokens_per_page": round(tok_per_page, 1),
            "chars_per_token": round(chars_per_tok, 2),
        },
        "other_done_books": len(other_done),
        "rate": {
            "pages_last_6h": pages_last_6h,
            "books_last_6h": books_last_6h,
            "pages_per_hour": round(pages_per_hour, 1),
            "eta_hours": round(eta_hours, 1) if eta_hours is not None else None,
        },
        "milestones": [
            {
                "tokens": value,
                "label": label,
                "frac": min(1.0, est_done_tokens / value) if value else 0,
                "reached": est_done_tokens >= value,
            }
            for value, label in MILESTONES
        ],
        "next_milestone": next_milestone,
        "log": log_info,
        "in_progress_parts": in_progress_parts(),
        "recent_books": [
            {
                "name": row["name"],
                "pages": row["pages"],
                "tokens": row["tokens"],
                "finished": row["finished"],
            }
            for row in russian_done[:15]
        ],
        "tokenizer": "tokenizers/fineweb2_rus_v001/chosen (BPE 16k)",
    }


def snapshot(max_age: float = 3.0) -> dict:
    global _SNAPSHOT, _SNAP_AT
    now = time.time()
    with _SNAP_LOCK:
        if _SNAPSHOT is not None and now - _SNAP_AT < max_age:
            return _SNAPSHOT
        _SNAPSHOT = build_snapshot()
        _SNAP_AT = now
        return _SNAPSHOT


HTML = r"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MinerU — русская коллекция</title>
<style>
  :root {
    --bg: #111214;
    --fg: #e8e6e3;
    --muted: #9a958c;
    --line: #2a2b2e;
    --card: #191a1c;
    --accent: #c4a574;
    --ok: #7d9b76;
    --warn: #c4a050;
    --bad: #c07070;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    font: 14px/1.45 ui-sans-serif, system-ui, sans-serif;
    background: var(--bg);
    color: var(--fg);
  }
  header {
    padding: 28px 32px 12px;
    border-bottom: 1px solid var(--line);
  }
  header h1 {
    margin: 0 0 6px;
    font-size: 22px;
    font-weight: 600;
    letter-spacing: -0.02em;
  }
  header p { margin: 0; color: var(--muted); }
  main { padding: 24px 32px 48px; max-width: 1180px; }
  .hero {
    display: grid;
    grid-template-columns: 1.4fr 1fr;
    gap: 28px;
    margin-bottom: 28px;
    align-items: end;
  }
  .big {
    font-size: 52px;
    font-weight: 600;
    letter-spacing: -0.04em;
    line-height: 1;
  }
  .big span { color: var(--muted); font-size: 22px; font-weight: 500; }
  .caption { margin-top: 8px; color: var(--muted); }
  .grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin: 22px 0 28px;
  }
  .stat {
    background: var(--card);
    border: 1px solid var(--line);
    padding: 14px 16px;
  }
  .stat .k { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.04em; }
  .stat .v { font-size: 26px; font-weight: 600; margin-top: 4px; letter-spacing: -0.03em; }
  .bar {
    height: 10px;
    background: #2a2b2e;
    margin-top: 10px;
    overflow: hidden;
  }
  .bar > i {
    display: block;
    height: 100%;
    background: var(--accent);
    width: 0%;
  }
  .milestones { display: grid; gap: 10px; }
  .ms {
    display: grid;
    grid-template-columns: 140px 1fr 72px;
    gap: 12px;
    align-items: center;
    font-size: 13px;
  }
  .ms .track { height: 6px; background: #2a2b2e; }
  .ms .track > i { display: block; height: 100%; background: var(--ok); }
  .ms.todo .track > i { background: var(--accent); }
  table { width: 100%; border-collapse: collapse; }
  th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--line); }
  th { color: var(--muted); font-weight: 500; font-size: 12px; }
  .pill {
    display: inline-block;
    padding: 2px 8px;
    border: 1px solid var(--line);
    font-size: 12px;
  }
  .pill.ok { color: var(--ok); border-color: #35503a; }
  .pill.off { color: var(--bad); border-color: #5a3030; }
  .cols { display: grid; grid-template-columns: 1fr 1fr; gap: 28px; }
  h2 { font-size: 15px; font-weight: 600; margin: 0 0 12px; }
  .note { color: var(--muted); font-size: 13px; margin-top: 18px; max-width: 70ch; }
  @media (max-width: 900px) {
    .hero, .grid, .cols { grid-template-columns: 1fr; }
    .big { font-size: 40px; }
    main, header { padding-left: 16px; padding-right: 16px; }
  }
</style>
</head>
<body>
<header>
  <h1>Русская химическая коллекция</h1>
  <p id="sub">MinerU Markdown · токены тем же BPE 16k, что в обучении</p>
</header>
<main>
  <div class="hero">
    <div>
      <div class="big" id="tokens">—</div>
      <div class="caption" id="tokensCap">считаю токены по готовому Markdown</div>
      <div class="bar" title="доля русских книг"><i id="bookBar"></i></div>
      <div class="caption" id="bookCap"></div>
    </div>
    <div>
      <div id="runner"></div>
      <div class="caption" id="eta"></div>
    </div>
  </div>
  <div class="grid">
    <div class="stat"><div class="k">Книги готовы</div><div class="v" id="sBooks">—</div></div>
    <div class="stat"><div class="k">Страницы готовы</div><div class="v" id="sPages">—</div></div>
    <div class="stat"><div class="k">Ещё страниц</div><div class="v" id="sLeft">—</div></div>
    <div class="stat"><div class="k">Темп, стр/час</div><div class="v" id="sRate">—</div></div>
  </div>
  <div class="cols">
    <section>
      <h2>Когда хватит для обучения</h2>
      <div class="milestones" id="ms"></div>
      <p class="note" id="enough"></p>
    </section>
    <section>
      <h2>Сейчас в работе</h2>
      <div id="active"></div>
    </section>
  </div>
  <section style="margin-top:32px">
    <h2>Последние готовые русские книги</h2>
    <table>
      <thead><tr><th>Книга</th><th>Страницы</th><th>Токены</th><th>Готово</th></tr></thead>
      <tbody id="recent"></tbody>
    </table>
  </section>
  <p class="note" id="foot"></p>
</main>
<script>
const fmt = n => n == null ? "—" : Number(n).toLocaleString("ru-RU");
const hours = h => {
  if (h == null) return "темп за 6 ч пока слабый — ETA не считаю";
  if (h < 36) return "осталось около " + h.toFixed(1) + " ч при текущем темпе";
  return "осталось около " + (h/24).toFixed(1) + " суток при текущем темпе";
};
function render(d) {
  const r = d.russian;
  const tok = r.tokens_est;
  document.getElementById("tokens").innerHTML =
    fmt(tok) + ' <span>токенов</span>';
  const cap = r.tokens_uncached_books
    ? ("посчитано точно " + fmt(r.tokens_counted) + ", ещё " + r.tokens_uncached_books + " книг в очереди подсчёта")
    : ("BPE 16k · " + r.chars_per_token + " символа/токен · " + r.tokens_per_page + " токенов/страница");
  document.getElementById("tokensCap").textContent = cap;
  const frac = r.target_books ? r.done_books / r.target_books : 0;
  document.getElementById("bookBar").style.width = (frac * 100).toFixed(1) + "%";
  document.getElementById("bookCap").textContent =
    fmt(r.done_books) + " из " + fmt(r.target_books) + " русских книг · ещё " +
    fmt(r.pending_books) + " · оценка полного корпуса " + fmt(r.tokens_full_est) + " токенов";
  document.getElementById("sBooks").textContent = fmt(r.done_books);
  document.getElementById("sPages").textContent = fmt(r.done_pages);
  document.getElementById("sLeft").textContent = fmt(r.pending_pages_est);
  document.getElementById("sRate").textContent = fmt(d.rate.pages_per_hour);
  const alive = d.runner.alive;
  document.getElementById("runner").innerHTML =
    '<span class="pill ' + (alive ? "ok" : "off") + '">' +
    (alive ? "cloud runner работает" : "cloud runner не запущен") + "</span>";
  document.getElementById("eta").textContent = hours(d.rate.eta_hours);
  document.getElementById("ms").innerHTML = d.milestones.map(m =>
    '<div class="ms ' + (m.reached ? "done" : "todo") + '">' +
    "<div>" + fmt(m.tokens) + "</div>" +
    '<div class="track"><i style="width:' + (m.frac*100).toFixed(1) + '%"></i></div>' +
    "<div>" + Math.round(m.frac*100) + "%</div></div>"
  ).join("");
  const nxt = d.next_milestone;
  let enough = "Исходный pretrain 10.9M-модели шёл на 626 млн токенов Cultura. Химия — доменный корпус, его не нужно дотягивать до того же объёма.";
  if (nxt) enough += " Ближайшая отметка: " + nxt.label.toLowerCase() + " (" + fmt(nxt.tokens) + ").";
  else enough += " Все отмеченные пороги уже пройдены.";
  document.getElementById("enough").textContent = enough;
  const active = (d.log.active || []).map(a =>
    "<div>" + a.name + " <span class='caption'>#" + a.index + " · " + a.key + "</span></div>"
  ).join("") || "<div class='caption'>нет активных START в хвосте лога</div>";
  const parts = (d.in_progress_parts || []).slice(0,6).map(p =>
    "<div class='caption'>" + p.name + " · " + p.parts + " частей · " + p.age_min + " мин назад</div>"
  ).join("");
  document.getElementById("active").innerHTML = active + parts;
  document.getElementById("recent").innerHTML = (d.recent_books || []).map(b =>
    "<tr><td>" + b.name + "</td><td>" + fmt(b.pages) + "</td><td>" +
    (b.tokens ? fmt(b.tokens) : "…") + "</td><td>" + (b.finished || "") + "</td></tr>"
  ).join("");
  document.getElementById("foot").textContent =
    "Обновлено " + d.updated + " · " + d.tokenizer +
    " · английские книги в знаменатель русской очереди не входят (" +
    d.other_done_books + " уже сняты MinerU раньше)";
  document.getElementById("sub").textContent =
    "обновление каждые 4 с · " + (d.collection.tesseract_first_left || 0) +
    " Tesseract-книг ещё впереди";
}
async function tick() {
  try {
    const r = await fetch("/api/progress", {cache: "no-store"});
    render(await r.json());
  } catch (e) {}
}
tick();
setInterval(tick, 4000);
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._send(200, HTML.encode("utf-8"), "text/html; charset=utf-8")
            return
        if path == "/api/progress":
            payload = json.dumps(snapshot(), ensure_ascii=False).encode("utf-8")
            self._send(200, payload, "application/json; charset=utf-8")
            return
        self._send(404, b"not found", "text/plain")


def main() -> int:
    global _TOKEN_THREAD_STARTED
    parser = argparse.ArgumentParser(description="Live dashboard for MinerU Russian corpus progress.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    load_cache()
    if not _TOKEN_THREAD_STARTED:
        thread = threading.Thread(target=token_worker, daemon=True)
        thread.start()
        _TOKEN_THREAD_STARTED = True
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"MinerU dashboard http://{args.host}:{args.port}/", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
