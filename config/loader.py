# config/loader.py — Config loader for monolithic and modular YAML files
from __future__ import annotations

import copy
import io
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from logger_setup import get_file_logger

logger = get_file_logger("loader", "logs/loader.log")

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_SOCIAL_CONFIG = os.path.join(_PROJECT_ROOT, "social", "config", "social.yaml")

# Ordered selector chains. The first non-empty value wins during extraction,
# so site-specific selectors must come before defaults. deep_merge() appends
# override lists after base lists, which is correct for union-style fields
# (noise words, URL excludes) but wrong for these ordered chains.
_ORDERED_SELECTOR_FIELDS = (
    "content_selectors",
    "author_selectors",
    "title_selectors",
    "date_selectors",
    "category_selectors",
    "image_selectors",
    "remove_selectors",
    "card_selectors",
    "article_link_selectors",
    "pagination_selectors",
    "rubric_selectors",
)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dicts; lists are concatenated, not replaced."""
    result = copy.deepcopy(base) if base else {}
    if not override:
        return result
    for key, value in override.items():
        if key in result:
            if isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = deep_merge(result[key], value)
            elif isinstance(result[key], list) and isinstance(value, list):
                result[key] = result[key] + copy.deepcopy(value)
            else:
                result[key] = copy.deepcopy(value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def clean_list(value: Any) -> list:
    """Convert any value to a list."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def normalize_site_match(match_value: Any) -> List[str]:
    """Normalize a site's match patterns to a lowercase string list."""
    items = clean_list(match_value)
    return [str(x).strip().lower() for x in items if x is not None and str(x).strip()]


def safe_domain(url: str) -> str:
    """Return the netloc of a URL, or an empty string."""
    try:
        return urlparse(url).netloc.lower().strip()
    except Exception:
        return ""


def extract_root_config(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Unwrap a config nested under 'v6_5' if present."""
    if not isinstance(raw, dict):
        return {}
    if "v6_5" in raw and isinstance(raw["v6_5"], dict):
        return raw["v6_5"]
    return raw


def _prioritize_site_selectors(
    merged: Dict[str, Any],
    site_cfg: Dict[str, Any],
    default_profile: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Put site-specific selectors before defaults for ordered chains.

    Only invoked when the site defines its own list. For fields the site
    does not override, the merged value is left as produced by deep_merge.
    Duplicates are removed while preserving the site-then-default order.
    """
    for field in _ORDERED_SELECTOR_FIELDS:
        site_list = list(site_cfg.get(field) or [])
        if not site_list:
            continue
        default_list = list(default_profile.get(field) or [])
        merged[field] = list(dict.fromkeys(site_list + default_list))
    return merged


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def normalize_text_rules(text_rules: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(text_rules, dict):
        return {}

    normalized = copy.deepcopy(text_rules)

    if "noise_words" in normalized and isinstance(normalized["noise_words"], dict):
        for lang, words in normalized["noise_words"].items():
            normalized["noise_words"][lang] = clean_list(words)

    if "stopwords" in normalized and isinstance(normalized["stopwords"], dict):
        sw = normalized["stopwords"]
        if "enabled" not in sw:
            sw["enabled"] = False
        if "per_language" not in sw:
            sw["per_language"] = True
        for key in list(sw.keys()):
            if key not in ("enabled", "per_language"):
                sw[key] = clean_list(sw.get(key))

    if "filters" in normalized and isinstance(normalized["filters"], dict):
        f = normalized["filters"]
        f.setdefault("min_word_length", 2)
        f.setdefault("remove_numeric_only", True)
        f.setdefault("drop_if_only_noise", True)
        f.setdefault("min_text_length", 20)
        f.setdefault("deduplicate_lines", True)

    if "patterns" in normalized and isinstance(normalized["patterns"], dict):
        for k in ("ui_patterns", "ad_patterns"):
            if k in normalized["patterns"]:
                normalized["patterns"][k] = clean_list(normalized["patterns"][k])

    if "language_detection" in normalized and isinstance(normalized["language_detection"], dict):
        ld = normalized["language_detection"]
        ld.setdefault("enabled", False)
        ld.setdefault("fallback", "unknown")

    if "normalize" in normalized and isinstance(normalized["normalize"], dict):
        n = normalized["normalize"]
        n.setdefault("lowercase", True)
        n.setdefault("trim", True)
        n.setdefault("remove_punctuation", False)

    return normalized


def normalize_profile(profile: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    profile = profile or {}
    profile = copy.deepcopy(profile)

    profile.setdefault("match", [])
    profile.setdefault("default_author", None)
    profile.setdefault("default_language", None)
    profile.setdefault("cookie_lang", None)
    profile.setdefault("amp_mode", "none")
    profile.setdefault("verify_ssl", None)
    profile.setdefault("date_locale", None)
    profile.setdefault("start_url", None)
    profile.setdefault("api_source", {})
    profile.setdefault("noise_language", ["common", "ru"])

    for key in [
        "rubric_selectors", "card_selectors", "article_link_selectors",
        "pagination_selectors", "title_selectors", "date_selectors",
        "author_selectors", "category_selectors", "image_selectors",
        "content_selectors", "remove_selectors",
        "article_url_filters", "article_url_excludes",
    ]:
        profile[key] = clean_list(profile.get(key))

    for key in ["rubric_strategy", "author_strategy", "category_strategy", "card_exclude_classes"]:
        if key in profile:
            profile[key] = clean_list(profile[key])
        else:
            profile.pop(key, None)

    # article_url_regex is a single optional regex, not a list.
    regex = profile.get("article_url_regex")
    profile["article_url_regex"] = (
        regex.strip() if isinstance(regex, str) and regex.strip() else None
    )

    profile["match"] = normalize_site_match(profile.get("match"))

    if "text_rules" in profile:
        profile["text_rules"] = normalize_text_rules(profile["text_rules"])

    return profile


def normalize_config(config: Dict[str, Any]) -> Dict[str, Any]:
    config = copy.deepcopy(config or {})

    config.setdefault("version", "6.5")
    config.setdefault("name", "optimal_pipeline")
    config.setdefault("global", {})
    config.setdefault("default_profile", {})
    config.setdefault("reusable_strategies", {})
    config.setdefault("sites", {})
    config.setdefault("languages", {})

    for key in ("global", "default_profile", "reusable_strategies", "sites", "languages"):
        if not isinstance(config[key], dict):
            config[key] = {}

    global_cfg = config["global"]

    for key in ("rubric_strategies_order", "author_strategies_order", "category_strategies_order"):
        if key in global_cfg:
            global_cfg[key] = clean_list(global_cfg[key])

    if not isinstance(global_cfg.get("request"), dict):
        global_cfg["request"] = {}
    req = global_cfg["request"]
    req.setdefault("timeout", 25)
    req.setdefault("retries", 3)
    req.setdefault("backoff_base", 1.6)
    req.setdefault("default_delay_seconds", 1.2)
    req.setdefault("rubric_delay_range_seconds", [1.0, 2.2])
    req.setdefault("max_threads", 8)
    req.setdefault("verify_ssl", True)

    if not isinstance(global_cfg.get("limits"), dict):
        global_cfg["limits"] = {}
    limits = global_cfg["limits"]
    limits.setdefault("max_depth", 3)
    limits.setdefault("max_pages", 1000)
    limits.setdefault("max_items", 10000)
    limits.setdefault("excerpt_len", 260)
    limits.setdefault("hash_truncate", 32)

    if not isinstance(global_cfg.get("url_rules"), dict):
        global_cfg["url_rules"] = {}
    url_rules = global_cfg["url_rules"]
    for k in ("article_url_filters_default", "article_url_excludes_default", "noise_url_parts"):
        url_rules[k] = clean_list(url_rules.get(k))

    if not isinstance(global_cfg.get("content_cleanup"), dict):
        global_cfg["content_cleanup"] = {}
    global_cfg["content_cleanup"]["remove_selectors"] = clean_list(
        global_cfg["content_cleanup"].get("remove_selectors")
    )

    if not isinstance(global_cfg.get("meta_extraction"), dict):
        global_cfg["meta_extraction"] = {}
    meta = global_cfg["meta_extraction"]
    for k in (
        "title_selectors_default", "date_selectors_default",
        "author_selectors_default", "category_selectors_default",
        "image_selectors_default",
    ):
        meta[k] = clean_list(meta.get(k))

    global_cfg["text_rules"] = normalize_text_rules(global_cfg.get("text_rules", {}))

    reusable = config["reusable_strategies"]
    for section in ("rubric_sources", "author_sources", "category_sources", "date_locale"):
        if not isinstance(reusable.get(section), dict):
            reusable[section] = {}

    config["default_profile"] = normalize_profile(config["default_profile"])

    config["sites"] = {
        str(site_key): normalize_profile(site_cfg)
        for site_key, site_cfg in config["sites"].items()
        if isinstance(site_cfg, dict)
    }

    return config


# ---------------------------------------------------------------------------
# Site matching
# ---------------------------------------------------------------------------

def get_site_key(url: str, config: Dict[str, Any]) -> str:
    """Return the site key with the longest match for the URL domain."""
    domain = safe_domain(url)
    if not domain:
        return "default"

    sites = config.get("sites", {})
    if not isinstance(sites, dict):
        return "default"

    candidates: List[tuple] = []
    for site_key, site_cfg in sites.items():
        if not isinstance(site_cfg, dict):
            continue
        for pat in normalize_site_match(site_cfg.get("match")):
            if pat and (pat in domain or domain.endswith("." + pat)):
                candidates.append((len(pat), str(site_key)))

    if not candidates:
        return "default"

    candidates.sort(reverse=True)
    return candidates[0][1]


def resolve_auto_family(
    domain: str, html_sample: str, default_profile: Dict[str, Any]
) -> Dict[str, Any]:
    families_cfg = default_profile.get("auto_detect_family", {})
    if not families_cfg.get("enabled") or not html_sample:
        return {}

    best_family = None
    best_score = 0

    for family_name, family_cfg in families_cfg.get("families", {}).items():
        score = 0
        for ind in family_cfg.get("indicators", []):
            weight = ind.get("weight", 1)
            selector = ind.get("selector", "")
            url_pattern = ind.get("url_pattern", "")
            if selector and selector in html_sample:
                score += weight
            if url_pattern and url_pattern in html_sample:
                score += weight

        if score >= family_cfg.get("match_min_score", 1) and score > best_score:
            best_score = score
            best_family = family_name

    return families_cfg["families"][best_family].get("profile", {}) if best_family else {}


# ---------------------------------------------------------------------------
# Resolution of language-dependent data
# ---------------------------------------------------------------------------

def resolve_noise_words(site_cfg: Dict[str, Any], config: Dict[str, Any]) -> List[str]:
    languages = site_cfg.get("noise_language", ["common", "ru"])
    lang_data = config.get("languages", {})

    words: List[str] = []
    for lang in languages:
        if lang in lang_data:
            words.extend(lang_data[lang].get("noise_words", []))

    if not words:
        fallback = config.get("global", {}).get("text_rules", {}).get("noise_words", {})
        for lang in languages:
            words.extend(fallback.get(lang, []))

    return list(set(words))


def resolve_author_regex_patterns(
    site_cfg: Dict[str, Any], config: Dict[str, Any]
) -> List[str]:
    lang_data = config.get("languages", {})
    patterns: List[str] = []

    common = lang_data.get("_common", {})
    if "author_regex_universal" in common:
        patterns.extend(clean_list(common["author_regex_universal"]))

    noise_langs = site_cfg.get("noise_language", ["common", "ru"])
    for lang in noise_langs:
        if lang in lang_data and "author_regex" in lang_data[lang]:
            patterns.extend(clean_list(lang_data[lang]["author_regex"]))

    default_lang = site_cfg.get("default_language")
    if default_lang and default_lang not in noise_langs and default_lang in lang_data:
        if "author_regex" in lang_data[default_lang]:
            patterns.extend(clean_list(lang_data[default_lang]["author_regex"]))

    if not patterns:
        author_src = config.get("reusable_strategies", {}).get("author_sources", {})
        patterns = clean_list(
            author_src.get("regex_in_content", {}).get("patterns", [])
        )

    return patterns


def get_site_config(
    url: str, html_sample: str = "", config: Dict[str, Any] = None
) -> Dict[str, Any]:
    if not isinstance(config, dict):
        raise ValueError("config must be a dict")

    site_key = get_site_key(url, config)
    default_profile = config.get("default_profile", {})
    site_cfg = config.get("sites", {}).get(site_key, {}) if site_key != "default" else {}

    merged = deep_merge(default_profile, site_cfg)
    merged = _prioritize_site_selectors(merged, site_cfg, default_profile)

    if site_key == "default" and html_sample:
        family_profile = resolve_auto_family(
            safe_domain(url), html_sample, default_profile
        )
        if family_profile:
            merged = deep_merge(merged, family_profile)
            # The family profile was selected because no site profile matched,
            # so its selectors should come before the global defaults.
            merged = _prioritize_site_selectors(merged, family_profile, default_profile)

    if merged.get("verify_ssl") is None:
        global_verify = config.get("global", {}).get("request", {}).get("verify_ssl")
        if global_verify is not None:
            merged["verify_ssl"] = bool(global_verify)

    author_patterns = resolve_author_regex_patterns(merged, config)
    noise_words = resolve_noise_words(merged, config)

    merged = normalize_profile(merged)

    merged["_site_key"] = site_key
    merged["_domain"] = safe_domain(url)
    merged["_url"] = url
    merged["_default_author"] = (
        merged.get("default_author") or default_profile.get("default_author")
    )

    for name, key in (
        ("_global_request", "request"),
        ("_global_limits", "limits"),
        ("_global_url_rules", "url_rules"),
        ("_global_content_cleanup", "content_cleanup"),
        ("_global_meta_extraction", "meta_extraction"),
        ("_global_text_rules", "text_rules"),
    ):
        merged[name] = copy.deepcopy(config.get("global", {}).get(key, {}))

    merged["_reusable_strategies"] = copy.deepcopy(config.get("reusable_strategies", {}))
    merged["_noise_words"] = noise_words
    merged["_author_regex_patterns"] = author_patterns

    return merged


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_config(path: str) -> Dict[str, Any]:
    """Load a monolithic YAML config."""
    if not path:
        raise ValueError("Config path is empty")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    if not isinstance(raw, dict):
        raise ValueError("YAML root must be a mapping/object")

    return normalize_config(extract_root_config(raw))


def load_modular_config(universal_path: str) -> Dict[str, Any]:
    """Load universal.yaml plus the files listed under 'includes'."""
    if not os.path.exists(universal_path):
        raise FileNotFoundError(f"Universal config not found: {universal_path}")

    config_dir = os.path.dirname(os.path.abspath(universal_path))

    with open(universal_path, "r", encoding="utf-8") as f:
        universal = yaml.safe_load(f) or {}

    if not isinstance(universal, dict):
        raise ValueError("universal.yaml root must be a mapping")

    includes = clean_list(universal.get("includes", []))
    if not includes:
        raise ValueError("No 'includes' found in universal.yaml — nothing to load")

    result = copy.deepcopy(universal)
    result.pop("includes", None)

    for include_file in includes:
        include_path = os.path.join(config_dir, include_file)
        if not os.path.exists(include_path):
            logger.warning(f"Included file not found, skipping: {include_path}")
            continue

        with open(include_path, "r", encoding="utf-8") as f:
            included_data = yaml.safe_load(f) or {}

        if not isinstance(included_data, dict):
            logger.warning(f"{include_file} is not a mapping, skipping")
            continue

        result = deep_merge(result, included_data)

    return normalize_config(extract_root_config(result))


@dataclass
class ConfigLoader:
    """Reusable loader — auto-detects monolithic vs modular YAML."""

    path: str
    config: Dict[str, Any] = field(default_factory=dict)

    def load(self) -> Dict[str, Any]:
        with open(self.path, "r", encoding="utf-8") as f:
            header = yaml.safe_load(f) or {}
        self.config = (
            load_modular_config(self.path)
            if "includes" in header
            else load_config(self.path)
        )
        return self.config

    def reload(self) -> Dict[str, Any]:
        return self.load()

    def site_key(self, url: str) -> str:
        if not self.config:
            self.load()
        return get_site_key(url, self.config)

    def site_config(self, url: str, html_sample: str = "") -> Dict[str, Any]:
        if not self.config:
            self.load()
        return get_site_config(url, html_sample, self.config)


# ---------------------------------------------------------------------------
# Convenience API
# ---------------------------------------------------------------------------

def load_yaml_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        header = yaml.safe_load(f) or {}
    return load_modular_config(path) if "includes" in header else load_config(path)


def match_site(url: str, config: Dict[str, Any]) -> str:
    return get_site_key(url, config)


def get_config(
    url: str, config: Dict[str, Any], html_sample: str = ""
) -> Dict[str, Any]:
    return get_site_config(url, html_sample, config)


def load_social_config(path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load the modular social-media config.

    Resolution order:
      1. explicit ``path`` argument
      2. SOCIAL_CONFIG_PATH environment variable
      3. <project_root>/social/config/social.yaml
    """
    resolved = (
        path
        or os.environ.get("SOCIAL_CONFIG_PATH")
        or _DEFAULT_SOCIAL_CONFIG
    )
    if not os.path.exists(resolved):
        raise FileNotFoundError(
            f"Social config not found: {resolved} "
            f"(default: {_DEFAULT_SOCIAL_CONFIG})"
        )
    return load_modular_config(resolved)


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

    cfg_path = "config/universal.yaml"
    test_url = "http://khovar.tj/"

    if len(sys.argv) >= 2:
        cfg_path = sys.argv[1]
    if len(sys.argv) >= 3:
        test_url = sys.argv[2]

    print(f"Config: {cfg_path}")
    print(f"URL: {test_url}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        header = yaml.safe_load(f) or {}

    mode = "MODULAR" if "includes" in header else "MONOLITHIC"
    print(f"Mode: {mode}")
    cfg = load_modular_config(cfg_path) if mode == "MODULAR" else load_config(cfg_path)

    site_key = get_site_key(test_url, cfg)
    site_cfg = get_site_config(test_url, "", cfg)

    print(f"\nSITE_KEY: {site_key}")
    summary = {
        "site_key": site_key,
        "domain": site_cfg.get("_domain"),
        "default_language": site_cfg.get("default_language"),
        "default_author": site_cfg.get("_default_author"),
        "start_url": site_cfg.get("start_url"),
        "title_selectors": site_cfg.get("title_selectors", [])[:3],
        "content_selectors": site_cfg.get("content_selectors", [])[:5],
        "author_selectors": site_cfg.get("author_selectors", [])[:5],
        "article_url_regex": site_cfg.get("article_url_regex"),
        "rubric_strategy": site_cfg.get("rubric_strategy"),
        "author_strategy": site_cfg.get("author_strategy"),
        "noise_words_sample": site_cfg.get("_noise_words", [])[:10],
        "author_regex_patterns_count": len(site_cfg.get("_author_regex_patterns", [])),
        "has_text_rules": bool(site_cfg.get("_global_text_rules")),
        "has_reusable_strategies": bool(site_cfg.get("_reusable_strategies")),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    lang = site_cfg.get("default_language", "unknown")
    if lang in cfg.get("languages", {}):
        pkg = cfg["languages"][lang]
        print(f"\nLanguage package '{lang}':")
        print(f"  noise_words: {len(pkg.get('noise_words', []))} words")
        print(f"  stopwords: {len(pkg.get('stopwords', []))} words")
        if pkg.get("date_locale"):
            print(f"  date_locale: {len(pkg['date_locale'])} months")
        if pkg.get("author_regex"):
            print(f"  author_regex: {len(pkg['author_regex'])} patterns")
        if pkg.get("category_url_patterns"):
            print(f"  category_url_patterns: {len(pkg['category_url_patterns'])} patterns")
            