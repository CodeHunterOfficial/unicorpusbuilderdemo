# tools/strategy_hitrate.py
"""
Measure strategy-level precision and cost, plus body-extraction quality.

Per field (author, title, date, category, content) the script reports:
  * first-matching strategy distribution;
  * per-strategy precision p — fraction of correct extractions among
    firings, evaluated against meta-tag references;
  * per-strategy cost c — mean wall-clock time per document attempt;
  * score = p / c, used to justify the chain order in the article.

Body-extraction quality for UniCorpusBuilder is measured against an
independently built gold text.

ROUGE tokenizer
---------------
``rouge_score.RougeScorer`` accepts a ``tokenizer`` object and calls
``tokenizer.tokenize(text)``. The default ``tokenize.tokenize`` keeps
only ``[a-z0-9]``, so ROUGE returns 0 on identical Cyrillic text. This
module passes ``_UnicodeTokenizer``, a subclass of
``rouge_score.tokenizers.Tokenizer`` whose ``tokenize`` splits on
``\\w+`` under ``re.UNICODE``.

Author comparison
-----------------
``_author_correct`` compares token sets, not substrings. The old rule
``ref[:15].lower() in extracted.lower()`` accepted different names as
matches: "Aleksandr Alekseev" matched "Aleksandr Aleksandrov".

``_looks_like_author_name`` rejects non-name strings that some sites
expose through ``author_selectors`` or ``meta``: reading times, dates,
photo credits, CMS logins, relative days, URLs and e-mail addresses.
Keep this filter in sync with
``pipeline_extraction._looks_like_author_name`` so that the measurement
matches the extractor.

Date correctness
----------------
``_date_correct`` uses ``dateparser.parse(..., ignoretz=True)`` on
both values, matching ``pipeline_core.parse_datetime_value``.

URL sources
-----------
Config mode — sites selected from ``config/universal.yaml``, either by
language (``--langs`` + ``--sites-per-lang``) or by explicit site keys
(``--sites``). File mode — URLs read from local JSONL files. URL-file
mode — URLs read from a CSV with ``url``, ``site``, ``lang`` columns;
discovery is skipped entirely. Use it to rerun the analysis on the
same documents after a code change.

Usage
-----
    python tools/strategy_hitrate.py
    python tools/strategy_hitrate.py --langs tg tt ba os --sites-per-lang 2 --articles-per-site 20
    python tools/strategy_hitrate.py --sites khovar tatar-inform vatantat akbuzat yanshishma iryston
    python tools/strategy_hitrate.py --exclude sssr ozodi
    python tools/strategy_hitrate.py --input-files "D:\\...\\file1.jsonl" "D:\\...\\file2.jsonl"
    python tools/strategy_hitrate.py --url-file trafilatura_comparison_baseline.csv

Installation
------------
    pip install trafilatura rouge-score python-dateutil tqdm
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, _TOOLS_DIR)

from bs4 import BeautifulSoup
from dateutil import parser as dateparser
from tqdm import tqdm

from config.loader import load_modular_config
from pipeline.pipeline_extraction import ExtractionEngine
from logger_setup import get_file_logger

from _common import WORD_RE, discover_urls_for_site, stats_summary


logger = get_file_logger("strategy_hitrate", "logs/strategy_hitrate.log")


# ---------------------------------------------------------------------------
# ROUGE with a Unicode-aware tokenizer
# ---------------------------------------------------------------------------

try:
    from rouge_score.tokenizers import Tokenizer as _RougeTokenizerBase
except ImportError:  # pragma: no cover - rouge-score not installed
    class _RougeTokenizerBase:  # type: ignore[no-redef]
        def tokenize(self, text: str) -> list[str]:
            raise NotImplementedError


_UNICODE_WORD_RE = re.compile(r"\w+", re.UNICODE)


class _UnicodeTokenizer(_RougeTokenizerBase):
    r"""
    Unicode-aware tokenizer compatible with rouge_score.

    Splits on ``\w+`` under ``re.UNICODE`` and lowercases the input.
    Ignores extra keyword arguments that some rouge-score versions
    forward to ``tokenize``.
    """

    def tokenize(self, text: str, *args: Any, **kwargs: Any) -> list[str]:
        return _UNICODE_WORD_RE.findall((text or "").lower())

    def __call__(self, text: str, *args: Any, **kwargs: Any) -> list[str]:
        return self.tokenize(text, *args, **kwargs)


_ROUGE_ERROR_LOGGED = False


def _build_scorer() -> Optional[Any]:
    """Return a configured RougeScorer, or None if rouge-score is unavailable."""
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        print("[WARN] rouge-score is not installed; ROUGE will be skipped.")
        print("       Install with: pip install rouge-score")
        return None

    try:
        return rouge_scorer.RougeScorer(
            ["rouge1", "rougeL"],
            use_stemmer=False,
            tokenizer=_UnicodeTokenizer(),
        )
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[WARN] ROUGE scorer init failed: {type(exc).__name__}: {exc}")
        return None


_SCORER: Optional[Any] = _build_scorer()


# ---------------------------------------------------------------------------
# Text metrics
# ---------------------------------------------------------------------------

def _clean_text(value: Optional[str]) -> str:
    """Collapse whitespace and strip."""
    return re.sub(r"\s+", " ", value or "").strip()


def _tokenize(value: str) -> list[str]:
    """Split text into lowercase word units via the shared WORD_RE."""
    return WORD_RE.findall((value or "").lower())


def extract_gold_text(soup: BeautifulSoup) -> str:
    """
    Return the gold article body used to score UniCorpusBuilder.

    Resolution order:
      1. A fixed list of narrow selectors for common CMS layouts.
      2. ``<article>`` / ``<main>`` / ``[role=main]``.
      3. ``<body>`` with boilerplate tags removed.
    """
    narrow_selectors = [
        "div.page-main__text",
        "div.shortcode-content",
        "section.cols > div",
        ".news-single .content",
        ".single__content",
        "div.tdb-block-inner.td-fix-index",
        "div.td-post-content",
        "[itemprop=articleBody]",
        ".article-body",
        ".post-content",
        ".entry-content",
        ".article-content",
        ".news-text",
        ".article__body",
        ".story-body",
        ".content-body",
        ".article-body__content",
        ".content",
    ]
    for selector in narrow_selectors:
        try:
            node = soup.select_one(selector)
        except Exception:
            node = None
        if node:
            text = _clean_text(node.get_text(" ", strip=True))
            if len(text) > 200:
                return text

    for selector in ("article", "main", "[role=main]"):
        try:
            node = soup.select_one(selector)
        except Exception:
            node = None
        if node:
            text = _clean_text(node.get_text(" ", strip=True))
            if len(text) > 300:
                return text

    body = soup.find("body")
    if body:
        soup_copy = BeautifulSoup(str(body), "html.parser")
        for tag in soup_copy(
            ["script", "style", "nav", "header", "footer",
             "aside", "form", "button", "noscript"]
        ):
            tag.decompose()
        return _clean_text(soup_copy.get_text(" ", strip=True))
    return ""


def word_prf1(prediction: str, gold: str) -> tuple[float, float, float]:
    """Word-level precision, recall and F1 over unique words."""
    pred_set = set(_tokenize(prediction))
    gold_set = set(_tokenize(gold))
    if not pred_set or not gold_set:
        return 0.0, 0.0, 0.0
    intersection = pred_set & gold_set
    precision = len(intersection) / len(pred_set)
    recall = len(intersection) / len(gold_set)
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return precision, recall, f1


def compute_quality(gold: str, prediction: str) -> dict[str, Any]:
    """
    Compute ROUGE-1, ROUGE-L, word-level P/R/F1 and length ratio.

    Fields are None when they cannot be computed (gold text shorter
    than 300 characters). ROUGE scoring failures are logged once and
    do not abort the run.
    """
    global _ROUGE_ERROR_LOGGED

    out: dict[str, Any] = {
        "gold_len": len(gold or ""),
        "pred_len": len(prediction or ""),
        "len_ratio": None,
        "word_precision": None,
        "word_recall": None,
        "word_f1": None,
        "rouge_1_f1": None,
        "rouge_l_f1": None,
    }
    if len(gold or "") < 300 or not prediction:
        return out

    out["len_ratio"] = round(len(prediction) / len(gold), 3)

    precision, recall, f1 = word_prf1(prediction, gold)
    out["word_precision"] = round(precision, 4)
    out["word_recall"] = round(recall, 4)
    out["word_f1"] = round(f1, 4)

    if _SCORER is not None:
        try:
            scores = _SCORER.score(gold, prediction)
            out["rouge_1_f1"] = round(scores["rouge1"].fmeasure, 4)
            out["rouge_l_f1"] = round(scores["rougeL"].fmeasure, 4)
        except Exception as exc:
            if not _ROUGE_ERROR_LOGGED:
                logger.warning(
                    "ROUGE scoring failed once: %s: %s",
                    type(exc).__name__,
                    exc,
                )
                _ROUGE_ERROR_LOGGED = True

    return out


# ---------------------------------------------------------------------------
# Reference metadata (meta tags) and correctness checks
# ---------------------------------------------------------------------------

def ref_meta_author(soup: BeautifulSoup) -> Optional[str]:
    """Return the reference author from meta tags only."""
    for name in (
        "author",
        "article:author",
        "DC.creator",
        "sailthru.author",
        "twitter:creator",
    ):
        tag = soup.find("meta", attrs={"name": name})
        if tag and tag.get("content"):
            return tag["content"]
    for prop in ("article:author", "og:article:author"):
        tag = soup.find("meta", attrs={"property": prop})
        if tag and tag.get("content"):
            return tag["content"]
    return None


def ref_meta_date(soup: BeautifulSoup) -> Optional[str]:
    """Return the reference date from meta tags only."""
    for prop in (
        "article:published_time",
        "og:published_time",
        "article:modified_time",
    ):
        tag = soup.find("meta", attrs={"property": prop})
        if tag and tag.get("content"):
            return tag["content"]
    for name in (
        "datePublished",
        "pubdate",
        "publishdate",
        "date",
        "DC.date",
        "sailthru.date",
    ):
        tag = soup.find("meta", attrs={"name": name})
        if tag and tag.get("content"):
            return tag["content"]
    return None


def ref_meta_title(soup: BeautifulSoup) -> Optional[str]:
    """Return the reference title from OpenGraph / Twitter / generic tags."""
    for prop in ("og:title", "twitter:title"):
        tag = soup.find("meta", attrs={"property": prop})
        if tag and tag.get("content"):
            return tag["content"]
    tag = soup.find("meta", attrs={"name": "title"})
    if tag and tag.get("content"):
        return tag["content"]
    return None


def _author_tokens(value: str) -> list[str]:
    """Lowercase word tokens with punctuation removed."""
    cleaned = re.sub(r"[^\w\s]", " ", (value or "").lower(), flags=re.UNICODE)
    return [token for token in cleaned.split() if token]


def _author_correct(extracted: Optional[str], reference: Optional[str]) -> bool:
    """
    Strict author comparison.

    Token-level comparison:
      * two or more shared full tokens → match;
      * single token on both sides and equal → match;
      * otherwise → no match.

    Empty values never match.
    """
    if not reference or not extracted:
        return False
    extracted_tokens = _author_tokens(extracted)
    reference_tokens = _author_tokens(reference)
    if not extracted_tokens or not reference_tokens:
        return False
    shared = set(extracted_tokens) & set(reference_tokens)
    if len(shared) >= 2:
        return True
    if (
        len(extracted_tokens) == 1
        and len(reference_tokens) == 1
        and extracted_tokens[0] == reference_tokens[0]
    ):
        return True
    return False


def _date_correct(extracted: Optional[str], reference: Optional[str]) -> bool:
    """
    Compare two date strings by calendar day.

    ``ignoretz=True`` suppresses ``UnknownTimezoneWarning`` on tokens
    that look like timezone names but are not. Only the calendar day
    is compared, so the timezone is lossless to ignore.
    """
    if not reference or not extracted:
        return False
    try:
        left = dateparser.parse(str(extracted), fuzzy=True, ignoretz=True)
        right = dateparser.parse(str(reference), fuzzy=True, ignoretz=True)
        if not left or not right:
            return False
        return left.date() == right.date()
    except Exception:
        return False


def _title_correct(extracted: Optional[str], reference: Optional[str]) -> bool:
    """Return True if the first 30 characters of the reference appear in the extract."""
    if not reference or not extracted:
        return False
    return reference[:30].lower() in extracted.lower()


_MONTH_NAMES = (
    "январ", "феврал", "март", "апрел", "мая", "май", "июн", "июл",
    "август", "сентябр", "октябр", "ноябр", "декабр",
    "январь", "февраль", "апрель", "июнь", "июль",
    "август", "сентябрь", "октябрь", "ноябрь", "декабрь",
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
)

_READING_TIME_RE = re.compile(
    r"\s*\d+\s*(мин|мин\.|минут[аыу]?|час(а|ов)?|ч|сек(унд[аыу]?)?"
    r"|min(ute)?s?|hour(s)?|hrs?|sec(ond)?s?)\b"
    r"(\s+(чтения|просмотра|на\s+чтение|чтение|(?:to\s+)?read(?:ing)?))?\s*\.?\s*",
    re.I,
)

_DATE_ONLY_RE = re.compile(
    r"\s*(\d{1,4}[-./]\d{1,2}([-./]\d{1,4})?)"
    r"(?:[T\s]\d{1,2}:\d{2}(?::\d{2})?(?:[+-]\d{2}:?\d{2}|Z)?)?\s*",
    re.I,
)

_TECH_LOGIN_RE = re.compile(
    r"^\s*(?:admin|administrator|webmaster|root|guest|anonymous|noreply|no-reply)\s*$",
    re.I,
)

_CONTENT_LABEL_RE = re.compile(
    r"^\s*(?:"
    r"партн[её]рск(?:ий|ая|ое|ие)\s+материал"
    r"|маводи\s+шарикона"
    r"|спонсорск(?:ий|ая|ое|ие)\s+материал"
    r"|рекламн(?:ый|ая|ое|ые)\s+материал"
    r"|реклама|промо"
    r"|advertorial|sponsored(?:\s+content|\s+post)?"
    r"|partner\s+material|paid\s+post"
    r"|reading\s+time|time\s+to\s+read|время\s+чтения"
    r")\s*$",
    re.I,
)

_PHOTO_CREDIT_RE = re.compile(
    r"^\s*(?:АКС|Акс|ФОТО|Фото|ФОТОГРАФИЯ|Фотография|Photo|Credit)\s*[:：\-—]",
    re.IGNORECASE,
)


def _looks_like_author_name(value: Optional[str]) -> bool:
    """
    Return True if ``value`` plausibly names a person.

    Rejects non-name strings that some sites expose through
    ``author_selectors`` or ``<meta>``:

      * reading times: "1 минута", "2 минуты", "5 минут чтения";
      * dates: "15.03.2024", "2024-03-15", "15 марта 2024";
      * photo credits: "АКС: Вазорати...", "Фото: «Татар-информ»";
      * CMS logins: "admin", "administrator", "webmaster", "noreply";
      * relative days: "сегодня", "вчера", "позавчера";
      * URLs, e-mail addresses;
      * strings with no letters at all.

    Editorial bylines ("Редакция", "Пресс-служба") pass this filter;
    they are distinguished from personal names downstream by
    ``tools/author_verification.py`` (verdict ``editorial``).
    """
    if not value:
        return False
    text = value.strip()
    if len(text) < 2 or len(text) > 120:
        return False
    if _CONTENT_LABEL_RE.match(text):
        return False
    if _PHOTO_CREDIT_RE.match(text):
        return False
    if _TECH_LOGIN_RE.match(text):
        return False
    if _READING_TIME_RE.fullmatch(text):
        return False
    if _DATE_ONLY_RE.fullmatch(text):
        return False
    text_low = text.lower()
    if re.search(r"\d", text_low) and any(month in text_low for month in _MONTH_NAMES):
        return False
    if text_low in ("сегодня", "вчера", "позавчера"):
        return False
    if text.startswith(("http://", "https://", "www.")) or "@" in text:
        return False
    if len(re.findall(r"[A-Za-zА-Яа-яЁёӨөҮүҚқҒғҲҳҶҷӢӣ]{2,}", text)) < 1:
        return False
    return True


# ---------------------------------------------------------------------------
# Strategy tracker
# ---------------------------------------------------------------------------

@dataclass
class ProbeResult:
    strategy: Optional[str] = None
    value: Optional[str] = None
    time_ms: float = 0.0
    is_correct: Optional[bool] = None


class StrategyTracker:
    """
    Collect per-strategy precision and cost.

    For each (field, strategy) pair, accumulate:
      attempts        — documents where the strategy was invoked;
      fired           — invocations that returned a value;
      ref_available   — firings where a meta-tag reference existed;
      correct         — firings whose value matched the reference;
      time_all_total  — wall-clock milliseconds across all attempts;
      time_fire_total — wall-clock milliseconds on firings only.

    Derived:
      p        = correct / ref_available       (precision among firings);
      c        = time_all_total / attempts     (mean ms per document);
      score    = p / c.
    """

    FIELDS = ("author", "title", "date", "category", "content")

    def __init__(self) -> None:
        self.data: dict[tuple[str, str], dict[str, Any]] = defaultdict(
            lambda: {
                "attempts": 0,
                "fired": 0,
                "correct": 0,
                "ref_available": 0,
                "time_all_total": 0.0,
                "time_fire_total": 0.0,
            }
        )

    def record(
        self,
        field: str,
        strategy: str,
        time_ms: float,
        is_correct: Optional[bool],
        fired: bool,
    ) -> None:
        entry = self.data[(field, strategy)]
        entry["attempts"] += 1
        entry["time_all_total"] += time_ms
        if fired:
            entry["fired"] += 1
            entry["time_fire_total"] += time_ms
            if is_correct is not None:
                entry["ref_available"] += 1
                if is_correct:
                    entry["correct"] += 1

    def rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for (field, strategy), entry in sorted(self.data.items()):
            p = (
                entry["correct"] / entry["ref_available"]
                if entry["ref_available"]
                else None
            )
            c_all = (
                entry["time_all_total"] / entry["attempts"]
                if entry["attempts"]
                else None
            )
            c_fire = (
                entry["time_fire_total"] / entry["fired"]
                if entry["fired"]
                else None
            )
            score = (
                p / c_all
                if (p is not None and c_all and c_all > 0)
                else None
            )
            rows.append(
                {
                    "field": field,
                    "strategy": strategy,
                    "attempts": entry["attempts"],
                    "fired": entry["fired"],
                    "ref_available": entry["ref_available"],
                    "correct": entry["correct"],
                    "p": round(p, 4) if p is not None else None,
                    "c_all_ms": round(c_all, 4) if c_all is not None else None,
                    "c_fire_ms": round(c_fire, 4) if c_fire is not None else None,
                    "p_over_c": round(score, 4) if score is not None else None,
                }
            )
        return rows


# ---------------------------------------------------------------------------
# URL collection
# ---------------------------------------------------------------------------

def gather_from_config(
    engine: ExtractionEngine,
    langs: list[str],
    sites_per_lang: int,
    articles_per_site: int,
    min_strategies: int,
    exclude: Optional[list[str]] = None,
    sites_filter: Optional[list[str]] = None,
) -> list[dict[str, str]]:
    """Select sites by language and collect article URLs from each.

    If ``sites_filter`` is provided, only those site keys are processed;
    ``sites_per_lang`` is ignored.
    """
    exclude_set = set(exclude or [])
    sites_set = set(sites_filter) if sites_filter else None
    config = load_modular_config(engine.yaml_path)
    sites = config.get("sites", {})

    by_lang: dict[str, list[tuple[str, dict, int]]] = defaultdict(list)
    for site_key, site_cfg in sites.items():
        if not isinstance(site_cfg, dict):
            continue
        if site_key in exclude_set:
            continue
        if sites_set is not None and site_key not in sites_set:
            continue
        lang = site_cfg.get("default_language")
        if lang not in langs:
            continue
        n_strategies = len(site_cfg.get("author_strategy", []) or [])
        if n_strategies < min_strategies:
            continue
        if sites_set is None and len(by_lang[lang]) >= sites_per_lang:
            continue
        by_lang[lang].append((site_key, site_cfg, n_strategies))

    print("\n[PLAN] Sites to process:")
    if exclude_set:
        print(f"    (excluded: {', '.join(sorted(exclude_set))})")
    if sites_set is not None:
        print(f"    (explicit --sites: {', '.join(sorted(sites_set))})")
    for lang in sorted(by_lang):
        names = [key for key, _, _ in by_lang[lang]]
        print(f"    {lang}: {', '.join(names)}")

    urls: list[dict[str, str]] = []
    for lang in sorted(by_lang):
        print(f"\n[{lang.upper()}]")
        for site_key, site_cfg, n_strategies in by_lang[lang]:
            match = site_cfg.get("match", [])
            domain = match[0] if match else "?"
            print(
                f"  [DISCOVER] {site_key} ({domain}) — "
                f"{n_strategies} author strategies"
            )
            site_urls = discover_urls_for_site(
                engine, site_key, site_cfg, articles_per_site
            )
            for url in site_urls:
                urls.append({"url": url, "site": site_key, "lang": lang})
            print(f"    -> {len(site_urls)} URLs")
    return urls


def iter_articles_from_files(
    files: list[Path],
    per_site: int,
):
    """Yield one ``{url, site, lang}`` dict per unique article, capped per site."""
    seen: set[str] = set()
    counter: dict[str, int] = defaultdict(int)
    for file_path in files:
        with open(file_path, "r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                url = obj.get("url")
                if not url or url in seen:
                    continue
                site = obj.get("site") or file_path.stem.split("-")[0]
                if counter[site] >= per_site:
                    continue
                seen.add(url)
                counter[site] += 1
                yield {
                    "url": url,
                    "site": site,
                    "lang": obj.get("language", "?"),
                }


def iter_articles_from_url_file(path: Path) -> list[dict[str, str]]:
    """Load ``{url, site, lang}`` dicts from a CSV file."""
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: list[dict[str, str]] = []
        for row in reader:
            url = (row.get("url") or "").strip()
            if not url:
                continue
            if (row.get("error") or "").strip():
                continue
            rows.append(
                {
                    "url": url,
                    "site": (row.get("site") or "?").strip() or "?",
                    "lang": (row.get("lang") or "?").strip() or "?",
                }
            )
    return rows


# ---------------------------------------------------------------------------
# Per-field strategy probes
# ---------------------------------------------------------------------------

_AUTHOR_LABEL_RE = re.compile(
    r"^(Author|By|From the author)\s*[:\-]?\s*", re.I
)


def _clean_probe_value(raw: Optional[str], max_len: int = 300) -> Optional[str]:
    """Reject JSON blobs and over-long strings."""
    if not raw:
        return None
    text = raw.strip()
    if not text or len(text) > max_len:
        return None
    if text.startswith(("{", "[")):
        return None
    return text


def probe_author(
    engine: ExtractionEngine,
    soup: BeautifulSoup,
    site_cfg: dict[str, Any],
    url: str,
    tracker: Optional[StrategyTracker] = None,
    reference: Optional[str] = None,
) -> ProbeResult:
    """
    Try each author strategy in the configured order and return the
    first value that passes ``_looks_like_author_name``.
    """
    strategies = (
        site_cfg.get("author_strategy")
        or engine.global_cfg().get("author_strategies_order")
        or []
    )
    strategies = list(dict.fromkeys(strategies))
    real_strategies = [s for s in strategies if s != "default_fallback"]

    for strategy in real_strategies:
        single_cfg = dict(site_cfg)
        single_cfg["author_strategy"] = [strategy]
        single_cfg["default_author"] = None
        single_cfg["author_priority"] = None

        start = time.perf_counter()
        try:
            raw = engine.extract_author_from_soup(soup, single_cfg, url)
        except Exception:
            raw = None
        elapsed_ms = (time.perf_counter() - start) * 1000

        value = _clean_probe_value(raw, max_len=300)
        if value:
            value = _AUTHOR_LABEL_RE.sub("", value).strip() or None
        if value and not _looks_like_author_name(value):
            value = None

        if tracker is not None:
            if value:
                correct = (
                    _author_correct(value, reference) if reference else None
                )
                tracker.record(
                    "author", strategy, elapsed_ms, correct, fired=True
                )
            else:
                tracker.record(
                    "author", strategy, elapsed_ms, None, fired=False
                )

        if value:
            return ProbeResult(strategy, value, elapsed_ms, None)

    if "default_fallback" in strategies and site_cfg.get("default_author"):
        value = _clean_probe_value(site_cfg["default_author"], max_len=300)
        if value and not _looks_like_author_name(value):
            value = None
        if value and tracker is not None:
            correct = (
                _author_correct(value, reference) if reference else None
            )
            tracker.record(
                "author", "default_fallback", 0.0, correct, fired=True
            )
        if value:
            return ProbeResult("default_fallback", value, 0.0, None)

    return ProbeResult()


def probe_title(
    engine: ExtractionEngine,
    soup: BeautifulSoup,
    site_cfg: dict[str, Any],
    tracker: Optional[StrategyTracker] = None,
    reference: Optional[str] = None,
) -> ProbeResult:
    """Try each title selector in order and return the first non-empty value."""
    for selector in site_cfg.get("title_selectors", []):
        start = time.perf_counter()
        try:
            element = engine._safe_select_one(soup, selector)
        except Exception:
            element = None
        elapsed_ms = (time.perf_counter() - start) * 1000

        value = None
        if element:
            if element.name == "meta" and element.get("content"):
                value = _clean_probe_value(element.get("content"))
            else:
                value = _clean_probe_value(element.get_text(" ", strip=True))

        if tracker is not None:
            if value:
                correct = (
                    _title_correct(value, reference) if reference else None
                )
                tracker.record(
                    "title", selector, elapsed_ms, correct, fired=True
                )
            else:
                tracker.record(
                    "title", selector, elapsed_ms, None, fired=False
                )

        if value:
            return ProbeResult(selector, value, elapsed_ms, None)

    if soup.title and soup.title.string and soup.title.string.strip():
        value = _clean_probe_value(soup.title.string)
        if tracker is not None and value:
            correct = (
                _title_correct(value, reference) if reference else None
            )
            tracker.record("title", "<title>", 0.0, correct, fired=True)
        if value:
            return ProbeResult("<title>", value, 0.0, None)

    return ProbeResult()


def probe_date(
    engine: ExtractionEngine,
    soup: BeautifulSoup,
    site_cfg: dict[str, Any],
    url: str,
    tracker: Optional[StrategyTracker] = None,
    reference: Optional[str] = None,
) -> ProbeResult:
    """Try JSON-LD first, then each date selector in order."""
    from pipeline.pipeline_core import parse_datetime_value

    locale_map = engine.get_date_locale_map(url)

    start = time.perf_counter()
    try:
        jsonld_date = engine.extract_jsonld_date(soup)
    except Exception:
        jsonld_date = None
    elapsed_ms = (time.perf_counter() - start) * 1000

    if jsonld_date:
        parsed = parse_datetime_value(jsonld_date, locale_map)
        if parsed:
            if tracker is not None:
                correct = (
                    _date_correct(parsed, reference) if reference else None
                )
                tracker.record(
                    "date",
                    "json_ld:datePublished",
                    elapsed_ms,
                    correct,
                    fired=True,
                )
            return ProbeResult(
                "json_ld:datePublished", parsed, elapsed_ms, None
            )
    if tracker is not None:
        tracker.record(
            "date", "json_ld:datePublished", elapsed_ms, None, fired=False
        )

    for selector in site_cfg.get("date_selectors", []):
        if "ld+json" in selector:
            continue
        start = time.perf_counter()
        try:
            element = engine._safe_select_one(soup, selector)
        except Exception:
            element = None
        elapsed_ms = (time.perf_counter() - start) * 1000

        parsed = None
        if element and element.name != "script":
            raw = (
                element.get("content")
                or element.get("datetime")
                or element.get_text(" ", strip=True)
            )
            if raw:
                parsed = parse_datetime_value(raw, locale_map)

        if tracker is not None:
            if parsed:
                correct = (
                    _date_correct(parsed, reference) if reference else None
                )
                tracker.record(
                    "date", selector, elapsed_ms, correct, fired=True
                )
            else:
                tracker.record(
                    "date", selector, elapsed_ms, None, fired=False
                )

        if parsed:
            return ProbeResult(selector, parsed, elapsed_ms, None)

    return ProbeResult()


def probe_category(
    engine: ExtractionEngine,
    soup: BeautifulSoup,
    url: str,
    site_cfg: dict[str, Any],
    tracker: Optional[StrategyTracker] = None,
) -> ProbeResult:
    """Try each category strategy in order and return the first value."""
    strategies = site_cfg.get("category_strategy") or []

    for strategy in strategies:
        if strategy == "meta_tag":
            for selector in site_cfg.get("category_selectors", []):
                start = time.perf_counter()
                try:
                    element = engine._safe_select_one(soup, selector)
                except Exception:
                    element = None
                elapsed_ms = (time.perf_counter() - start) * 1000
                value = None
                if element:
                    raw = (
                        element.get("content")
                        if element.name == "meta"
                        else element.get_text(" ", strip=True)
                    )
                    value = _clean_probe_value(raw, max_len=200)
                label = f"{strategy}:{selector[:40]}"
                if tracker is not None:
                    tracker.record(
                        "category", label, elapsed_ms, None, fired=bool(value)
                    )
                if value:
                    return ProbeResult(label, value, elapsed_ms, None)

        elif strategy == "url_path_parsing":
            start = time.perf_counter()
            try:
                category = engine.extract_category(soup, url, site_cfg)
            except Exception:
                category = None
            elapsed_ms = (time.perf_counter() - start) * 1000
            value = _clean_probe_value(category, max_len=200)
            if tracker is not None:
                tracker.record(
                    "category", strategy, elapsed_ms, None, fired=bool(value)
                )
            if value:
                return ProbeResult(strategy, value, elapsed_ms, None)

        elif strategy == "breadcrumb":
            start = time.perf_counter()
            value = None
            for element in soup.select(
                ".breadcrumb, .breadcrumbs, [class*=breadcrumb]"
            ):
                text = element.get_text(" ", strip=True)
                value = _clean_probe_value(text, max_len=200)
                if value:
                    break
            elapsed_ms = (time.perf_counter() - start) * 1000
            if tracker is not None:
                tracker.record(
                    "category", strategy, elapsed_ms, None, fired=bool(value)
                )
            if value:
                return ProbeResult(strategy, value, elapsed_ms, None)

        elif strategy == "context_passed":
            continue

    start = time.perf_counter()
    try:
        category = engine.extract_category(soup, url, site_cfg)
    except Exception:
        category = None
    elapsed_ms = (time.perf_counter() - start) * 1000
    value = _clean_probe_value(category, max_len=200)
    if tracker is not None:
        tracker.record(
            "category", "fallback", elapsed_ms, None, fired=bool(value)
        )
    if value:
        return ProbeResult("fallback", value, elapsed_ms, None)

    return ProbeResult()


def probe_content(
    engine: ExtractionEngine,
    soup: BeautifulSoup,
    site_cfg: dict[str, Any],
    tracker: Optional[StrategyTracker] = None,
) -> ProbeResult:
    """Try each content selector in order, then the heuristic fallback."""
    for selector in site_cfg.get("content_selectors", []):
        start = time.perf_counter()
        try:
            node = engine._safe_select_one(soup, selector)
        except Exception:
            node = None
        elapsed_ms = (time.perf_counter() - start) * 1000
        value = None
        if node:
            text = node.get_text(" ", strip=True)
            if text and len(text) > 200:
                value = selector
        if tracker is not None:
            tracker.record(
                "content", selector, elapsed_ms, None, fired=bool(value)
            )
        if value:
            return ProbeResult(selector, value, elapsed_ms, None)

    start = time.perf_counter()
    container = None
    try:
        container = engine.find_best_content_container(soup, site_cfg)
    except Exception:
        container = None
    elapsed_ms = (time.perf_counter() - start) * 1000
    value = (
        "auto:heuristic"
        if (container is not None and container != soup)
        else None
    )
    if tracker is not None:
        tracker.record(
            "content", "auto:heuristic", elapsed_ms, None, fired=bool(value)
        )
    if value:
        return ProbeResult("auto:heuristic", value, elapsed_ms, None)

    return ProbeResult()


# ---------------------------------------------------------------------------
# Main analysis loop
# ---------------------------------------------------------------------------

def run_analysis(
    engine: ExtractionEngine,
    urls_to_process: list[dict[str, str]],
    sleep: float,
):
    """
    Walk the URL list, probe every field on every document, and return
    per-strategy counters plus per-site/per-language quality samples.
    """
    stats = {
        name: Counter()
        for name in ("author", "title", "date", "category", "content")
    }
    coverage_total: Counter = Counter()
    coverage_ok: dict[str, Counter] = defaultdict(Counter)
    tracker = StrategyTracker()

    quality_by_site: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {
            "rouge_l": [],
            "rouge_1": [],
            "word_f1": [],
            "word_p": [],
            "word_r": [],
            "len_ratio": [],
        }
    )
    quality_by_lang: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {
            "rouge_l": [],
            "rouge_1": [],
            "word_f1": [],
            "word_p": [],
            "word_r": [],
            "len_ratio": [],
        }
    )

    total = 0
    failed_fetch = 0

    print(f"\n[FETCH] Processing {len(urls_to_process)} URLs...\n")

    for article in tqdm(urls_to_process, desc="[PROCESS]", unit="doc"):
        url = article["url"]
        site = article["site"]
        lang = article.get("lang", "?")

        try:
            html = engine.fetch_html(url)
        except Exception:
            failed_fetch += 1
            continue

        try:
            soup = BeautifulSoup(html, "html.parser")
            site_cfg = engine.site_cfg(url)
        except Exception:
            failed_fetch += 1
            continue

        reference_author = ref_meta_author(soup)
        reference_date = ref_meta_date(soup)
        reference_title = ref_meta_title(soup)

        coverage_total[site] += 1

        author = probe_author(
            engine, soup, site_cfg, url, tracker, reference_author
        )
        stats["author"][author.strategy or "[missed]"] += 1
        if author.strategy:
            coverage_ok[site]["author"] += 1

        title = probe_title(engine, soup, site_cfg, tracker, reference_title)
        stats["title"][title.strategy or "[missed]"] += 1
        if title.strategy:
            coverage_ok[site]["title"] += 1

        date = probe_date(
            engine, soup, site_cfg, url, tracker, reference_date
        )
        stats["date"][date.strategy or "[missed]"] += 1
        if date.strategy:
            coverage_ok[site]["date"] += 1

        category = probe_category(engine, soup, url, site_cfg, tracker)
        stats["category"][category.strategy or "[missed]"] += 1
        if category.strategy:
            coverage_ok[site]["category"] += 1

        content = probe_content(engine, soup, site_cfg, tracker)
        stats["content"][content.strategy or "[missed]"] += 1
        if content.strategy:
            coverage_ok[site]["content"] += 1

        try:
            data = engine.extract_article_fields(html, url)
            ucb_content = data.get("content") or ""
        except Exception:
            ucb_content = ""

        gold = extract_gold_text(soup)
        quality = compute_quality(gold, ucb_content)

        if quality["rouge_l_f1"] is not None:
            quality_by_site[site]["rouge_l"].append(quality["rouge_l_f1"])
            quality_by_lang[lang]["rouge_l"].append(quality["rouge_l_f1"])
        if quality["rouge_1_f1"] is not None:
            quality_by_site[site]["rouge_1"].append(quality["rouge_1_f1"])
            quality_by_lang[lang]["rouge_1"].append(quality["rouge_1_f1"])
        if quality["word_f1"] is not None:
            quality_by_site[site]["word_f1"].append(quality["word_f1"])
            quality_by_lang[lang]["word_f1"].append(quality["word_f1"])
            quality_by_site[site]["word_p"].append(quality["word_precision"])
            quality_by_site[site]["word_r"].append(quality["word_recall"])
            quality_by_lang[lang]["word_p"].append(quality["word_precision"])
            quality_by_lang[lang]["word_r"].append(quality["word_recall"])
        if quality["len_ratio"] is not None:
            quality_by_site[site]["len_ratio"].append(quality["len_ratio"])
            quality_by_lang[lang]["len_ratio"].append(quality["len_ratio"])

        total += 1
        time.sleep(sleep)

    return (
        stats,
        coverage_total,
        coverage_ok,
        quality_by_site,
        quality_by_lang,
        total,
        failed_fetch,
        tracker,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_stats(
    name: str,
    counter: Counter,
    total: int,
) -> list[dict[str, Any]]:
    """Print the strategy distribution for one field and return CSV rows."""
    print(f"\n--- {name.upper()} ---")
    if not counter:
        print("  (no data)")
        return []
    print(f"  {'Value':<45} {'Hits':>10} {'Share':>8}")
    print("  " + "-" * 65)

    rows: list[dict[str, Any]] = []
    for value, hits in counter.most_common():
        if value == "[missed]":
            continue
        share = hits / total * 100 if total else 0
        print(f"  {value[:45]:<45} {hits:>10} {share:>7.1f}%")
        rows.append(
            {
                "metric": name,
                "value": value,
                "hits": hits,
                "share_pct": round(share, 2),
            }
        )

    missed = counter.get("[missed]", 0)
    if missed:
        share = missed / total * 100 if total else 0
        print(f"  {'[no strategy matched]':<45} {missed:>10} {share:>7.1f}%")
        rows.append(
            {
                "metric": name,
                "value": "[missed]",
                "hits": missed,
                "share_pct": round(share, 2),
            }
        )
    return rows


def print_strategy_scores(rows: list[dict[str, Any]]) -> None:
    """Print per-strategy precision p, cost c, and the p/c score."""
    print(f"\n{'=' * 70}")
    print("[STRATEGY PRECISION AND COST]")
    print(f"{'=' * 70}")
    print("  p = correct / ref_available   (precision among firings)")
    print("  c = time_all_total / attempts (mean ms per document attempt)")
    print("  score = p / c\n")
    print(
        f"  {'Field':<10} {'Strategy':<28} {'Fired':>6} {'Ref':>5} "
        f"{'Corr':>5} {'p':>7} {'c, ms':>8} {'p/c':>7}"
    )
    print("  " + "-" * 82)

    for row in rows:
        p_str = f"{row['p']:.3f}" if row["p"] is not None else "—"
        c_str = (
            f"{row['c_all_ms']:.3f}" if row["c_all_ms"] is not None else "—"
        )
        s_str = (
            f"{row['p_over_c']:.3f}" if row["p_over_c"] is not None else "—"
        )
        print(
            f"  {row['field']:<10} {row['strategy'][:28]:<28} "
            f"{row['fired']:>6} {row['ref_available']:>5} "
            f"{row['correct']:>5} {p_str:>7} {c_str:>8} {s_str:>7}"
        )


def _ensure_parent_dir(path: str) -> None:
    """Create the parent directory of ``path`` if it does not exist."""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    """Run the analysis and return a process exit code."""
    parser = argparse.ArgumentParser(
        description=(
            "Measure strategy-level precision and cost, plus "
            "body-extraction quality."
        )
    )
    parser.add_argument("--config", default="config/universal.yaml")
    parser.add_argument("--out", default="strategy_hitrate.csv")
    parser.add_argument("--out-coverage", default="strategy_coverage.csv")
    parser.add_argument("--out-quality", default="strategy_quality.csv")
    parser.add_argument("--out-scores", default="strategy_scores.csv")
    parser.add_argument("--sleep", type=float, default=0.3)

    parser.add_argument("--langs", nargs="+", default=["tg", "tt", "ba", "os"])
    parser.add_argument("--sites-per-lang", type=int, default=2)
    parser.add_argument(
        "--sites",
        nargs="+",
        default=None,
        help="Explicit site keys (overrides --sites-per-lang)",
    )
    parser.add_argument("--articles-per-site", type=int, default=20)
    parser.add_argument("--min-strategies", type=int, default=1)
    parser.add_argument("--exclude", nargs="*", default=["sssr", "ozodi"])

    parser.add_argument("--input-files", nargs="*", default=None)
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--per-site", type=int, default=30)
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument(
        "--url-file",
        default=None,
        help="CSV with url,site,lang columns; skips discovery",
    )

    args = parser.parse_args(argv)

    engine = ExtractionEngine(yaml_path=args.config)

    if args.url_file:
        url_path = Path(args.url_file)
        if not url_path.exists():
            print(f"[ERROR] URL file not found: {url_path}")
            return 1
        urls_to_process = iter_articles_from_url_file(url_path)
        if not urls_to_process:
            print(f"[ERROR] No usable URLs in {url_path}")
            return 1
        print(f"[URLS] Loaded {len(urls_to_process)} from {url_path}")
    elif args.input_files or args.input_dir:
        if args.input_files:
            files = [Path(p) for p in args.input_files]
        else:
            files = sorted(Path(args.input_dir).glob("*.jsonl"))
        files = [f for f in files if f.exists()]
        if not files:
            print("[ERROR] No JSONL files found.")
            return 1
        print(f"[FILES] Found: {len(files)} files")
        for file_path in files:
            print(f"        {file_path.name}")

        articles = list(iter_articles_from_files(files, args.per_site))
        if len(articles) > args.limit:
            articles = articles[: args.limit]
        urls_to_process = articles
    else:
        mode_desc = (
            f"explicit sites={args.sites}"
            if args.sites
            else f"languages={args.langs}, {args.sites_per_lang} sites/language"
        )
        print(
            f"[MODE] Config mode: {mode_desc}, "
            f"{args.articles_per_site} articles/site"
        )
        urls_to_process = gather_from_config(
            engine,
            langs=args.langs,
            sites_per_lang=args.sites_per_lang,
            articles_per_site=args.articles_per_site,
            min_strategies=args.min_strategies,
            exclude=args.exclude,
            sites_filter=args.sites,
        )

    if not urls_to_process:
        print("[ERROR] No URLs collected.")
        return 1

    print(f"\n[TOTAL] URLs to process: {len(urls_to_process)}")

    (
        stats,
        coverage_total,
        coverage_ok,
        quality_by_site,
        quality_by_lang,
        total,
        failed_fetch,
        tracker,
    ) = run_analysis(engine, urls_to_process, args.sleep)

    print(f"\n{'=' * 70}")
    print(f"Processed:      {total}")
    print(f"Fetch failures: {failed_fetch}")
    print(f"{'=' * 70}")

    all_rows: list[dict[str, Any]] = []
    for field_name in ("author", "title", "date", "category", "content"):
        rows = print_stats(field_name, stats[field_name], total)
        if rows:
            all_rows.extend(rows)

    print(f"\n{'=' * 70}")
    print("[BODY-EXTRACTION QUALITY] — gold: <article> / <main> / <body>")
    print(f"{'=' * 70}")

    all_rouge_l: list[float] = []
    all_rouge_1: list[float] = []
    all_word_f1: list[float] = []
    all_word_p: list[float] = []
    all_word_r: list[float] = []
    all_len_ratio: list[float] = []
    for _site, samples in quality_by_site.items():
        all_rouge_l.extend(samples["rouge_l"])
        all_rouge_1.extend(samples["rouge_1"])
        all_word_f1.extend(samples["word_f1"])
        all_word_p.extend(samples["word_p"])
        all_word_r.extend(samples["word_r"])
        all_len_ratio.extend(samples["len_ratio"])

    rouge_l_summary = stats_summary(all_rouge_l)
    rouge_1_summary = stats_summary(all_rouge_1)
    word_f1_summary = stats_summary(all_word_f1)
    word_p_summary = stats_summary(all_word_p)
    word_r_summary = stats_summary(all_word_r)
    len_ratio_summary = stats_summary(all_len_ratio)

    if rouge_l_summary:
        print("\n  ROUGE-L F1:")
        print(
            f"    N = {rouge_l_summary['n']}   "
            f"mean = {rouge_l_summary['mean']:.4f}   "
            f"median = {rouge_l_summary['median']:.4f}   "
            f"min/max = {rouge_l_summary['min']:.4f}/"
            f"{rouge_l_summary['max']:.4f}"
        )
    else:
        print(
            "\n  ROUGE-L F1: no data "
            "(install rouge-score, or no gold text >= 300 chars)"
        )

    if rouge_1_summary:
        print("\n  ROUGE-1 F1:")
        print(
            f"    mean = {rouge_1_summary['mean']:.4f}   "
            f"median = {rouge_1_summary['median']:.4f}"
        )

    if word_f1_summary:
        print("\n  Word-level (unique words):")
        print(f"    Precision:  mean = {word_p_summary['mean']:.4f}")
        print(f"    Recall:     mean = {word_r_summary['mean']:.4f}")
        print(
            f"    F1:         mean = {word_f1_summary['mean']:.4f}   "
            f"median = {word_f1_summary['median']:.4f}"
        )

    if len_ratio_summary:
        print("\n  Length ratio (UCB / gold):")
        print(
            f"    mean = {len_ratio_summary['mean']:.2f}   "
            f"median = {len_ratio_summary['median']:.2f}   "
            f"min/max = {len_ratio_summary['min']:.2f}/"
            f"{len_ratio_summary['max']:.2f}"
        )

    print(f"\n{'=' * 70}")
    print("[COVERAGE BY SITE]")
    print(f"{'=' * 70}")
    print(
        f"  {'Site':<20} {'N':>4} {'Author':>8} {'Title':>8} "
        f"{'Date':>8} {'Categ':>8} {'Cont':>8} "
        f"{'ROUGE-L':>9} {'WordF1':>8}"
    )
    print("  " + "-" * 92)

    coverage_rows: list[dict[str, Any]] = []
    for site in sorted(coverage_total):
        site_total = coverage_total[site]
        if site_total == 0:
            continue
        ok = coverage_ok[site]
        author_pct = ok["author"] / site_total * 100
        title_pct = ok["title"] / site_total * 100
        date_pct = ok["date"] / site_total * 100
        category_pct = ok["category"] / site_total * 100
        content_pct = ok["content"] / site_total * 100

        site_rouge_l = stats_summary(quality_by_site[site]["rouge_l"])
        site_word_f1 = stats_summary(quality_by_site[site]["word_f1"])
        rouge_display = (
            f"{site_rouge_l['mean']:.3f}" if site_rouge_l else "—"
        )
        word_display = (
            f"{site_word_f1['mean']:.3f}" if site_word_f1 else "—"
        )

        print(
            f"  {site:<20} {site_total:>4} "
            f"{author_pct:>7.1f}% {title_pct:>7.1f}% "
            f"{date_pct:>7.1f}% {category_pct:>7.1f}% "
            f"{content_pct:>7.1f}% {rouge_display:>9} {word_display:>8}"
        )

        coverage_rows.append(
            {
                "site": site,
                "articles": site_total,
                "author_pct": round(author_pct, 1),
                "title_pct": round(title_pct, 1),
                "date_pct": round(date_pct, 1),
                "category_pct": round(category_pct, 1),
                "content_pct": round(content_pct, 1),
                "rouge_l_mean": (
                    round(site_rouge_l["mean"], 4) if site_rouge_l else None
                ),
                "rouge_l_n": site_rouge_l["n"] if site_rouge_l else 0,
                "word_f1_mean": (
                    round(site_word_f1["mean"], 4) if site_word_f1 else None
                ),
            }
        )

    print(f"\n{'=' * 70}")
    print("[BY LANGUAGE]")
    print(f"{'=' * 70}")
    print(
        f"  {'Lang':<6} {'N':>5} {'Author':>10} {'Date':>10} "
        f"{'Content':>10} {'ROUGE-L':>10} {'WordF1':>10}"
    )
    print("  " + "-" * 70)

    lang_rows: list[dict[str, Any]] = []
    lang_sites: dict[str, set[str]] = defaultdict(set)
    for article in urls_to_process:
        lang_sites[article.get("lang", "?")].add(article["site"])

    for lang in sorted(lang_sites):
        lang_total = 0
        author_hits = title_hits = date_hits = 0
        category_hits = content_hits = 0
        for site in lang_sites[lang]:
            lang_total += coverage_total.get(site, 0)
            author_hits += coverage_ok[site]["author"]
            title_hits += coverage_ok[site]["title"]
            date_hits += coverage_ok[site]["date"]
            category_hits += coverage_ok[site]["category"]
            content_hits += coverage_ok[site]["content"]
        if lang_total == 0:
            continue

        lang_rouge_l = stats_summary(quality_by_lang[lang]["rouge_l"])
        lang_word_f1 = stats_summary(quality_by_lang[lang]["word_f1"])
        rouge_display = (
            f"{lang_rouge_l['mean']:.3f}" if lang_rouge_l else "—"
        )
        word_display = (
            f"{lang_word_f1['mean']:.3f}" if lang_word_f1 else "—"
        )

        print(
            f"  {lang:<6} {lang_total:>5} "
            f"{author_hits / lang_total * 100:>9.1f}% "
            f"{date_hits / lang_total * 100:>9.1f}% "
            f"{content_hits / lang_total * 100:>9.1f}% "
            f"{rouge_display:>10} {word_display:>10}"
        )

        lang_rows.append(
            {
                "lang": lang,
                "articles": lang_total,
                "author_pct": round(author_hits / lang_total * 100, 1),
                "date_pct": round(date_hits / lang_total * 100, 1),
                "content_pct": round(content_hits / lang_total * 100, 1),
                "rouge_l_mean": (
                    round(lang_rouge_l["mean"], 4) if lang_rouge_l else None
                ),
                "word_f1_mean": (
                    round(lang_word_f1["mean"], 4) if lang_word_f1 else None
                ),
            }
        )

    strategy_rows = tracker.rows()
    print_strategy_scores(strategy_rows)

    print(f"\n{'=' * 70}")
    print("[FINAL TABLE FOR THE ARTICLE]")
    print(f"{'=' * 70}")
    print(f"  {'Metric':<40} {'Value':>18}")
    print("  " + "-" * 60)

    if total:
        print(f"  {'Documents processed':<40} {total:>18}")
        print(
            f"  {'Author extracted':<40} "
            f"{sum(coverage_ok[s]['author'] for s in coverage_total) / total * 100:>17.1f}%"
        )
        print(
            f"  {'Title extracted':<40} "
            f"{sum(coverage_ok[s]['title'] for s in coverage_total) / total * 100:>17.1f}%"
        )
        print(
            f"  {'Date extracted':<40} "
            f"{sum(coverage_ok[s]['date'] for s in coverage_total) / total * 100:>17.1f}%"
        )
        print(
            f"  {'Category extracted':<40} "
            f"{sum(coverage_ok[s]['category'] for s in coverage_total) / total * 100:>17.1f}%"
        )
        print(
            f"  {'Content extracted':<40} "
            f"{sum(coverage_ok[s]['content'] for s in coverage_total) / total * 100:>17.1f}%"
        )
    if rouge_l_summary:
        print(
            f"  {'ROUGE-L F1 (N=' + str(rouge_l_summary['n']) + ')':<40} "
            f"{rouge_l_summary['mean']:>18.4f}"
        )
    if word_f1_summary:
        print(
            f"  {'Word-level F1':<40} {word_f1_summary['mean']:>18.4f}"
        )
    if len_ratio_summary:
        print(
            f"  {'Length ratio (mean)':<40} {len_ratio_summary['mean']:>18.2f}"
        )

    _ensure_parent_dir(args.out)
    with open(args.out, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["metric", "value", "hits", "share_pct"]
        )
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\n[OK] Strategy distribution: {args.out}")

    _ensure_parent_dir(args.out_coverage)
    with open(args.out_coverage, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "site", "articles", "author_pct", "title_pct", "date_pct",
                "category_pct", "content_pct",
                "rouge_l_mean", "rouge_l_n", "word_f1_mean",
            ],
        )
        writer.writeheader()
        writer.writerows(coverage_rows)
    print(f"[OK] Coverage by site:     {args.out_coverage}")

    _ensure_parent_dir(args.out_quality)
    with open(args.out_quality, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "lang", "articles", "author_pct", "date_pct", "content_pct",
                "rouge_l_mean", "word_f1_mean",
            ],
        )
        writer.writeheader()
        writer.writerows(lang_rows)
    print(f"[OK] Quality by language:  {args.out_quality}")

    _ensure_parent_dir(args.out_scores)
    with open(args.out_scores, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "field", "strategy", "attempts", "fired", "ref_available",
                "correct", "p", "c_all_ms", "c_fire_ms", "p_over_c",
            ],
        )
        writer.writeheader()
        writer.writerows(strategy_rows)
    print(f"[OK] Strategy scores:      {args.out_scores}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
    