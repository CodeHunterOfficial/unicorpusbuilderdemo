# config/loader.py — Config loader with logging (fixed)
from __future__ import annotations

import copy
import json
import os
import sys
import io
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import yaml

# Fix Windows console encoding for Unicode output
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add project root to sys.path for logger import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from logger_setup import get_file_logger
# Initialize module logger (file only, no console output)
logger = get_file_logger("loader", "logs/loader.log")

# =====================================================
# Utilities
# =====================================================

def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries, extending lists instead of replacing them."""
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
    """Convert any value to a list, safely."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def normalize_site_match(match_value: Any) -> List[str]:
    """Normalize a site's match pattern(s) to a list of lowercase strings."""
    items = clean_list(match_value)
    return [str(x).strip().lower() for x in items if x is not None and str(x).strip()]


def safe_domain(url: str) -> str:
    """Extract the domain (netloc) from a URL, or empty string on failure."""
    try:
        return urlparse(url).netloc.lower().strip()
    except Exception:
        return ""


def extract_root_config(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Allow both raw YAML and wrapped YAML under 'v6_5' key."""
    if not isinstance(raw, dict):
        return {}
    if "v6_5" in raw and isinstance(raw["v6_5"], dict):
        return raw["v6_5"]
    return raw


# =====================================================
# Normalization of text_rules
# =====================================================

def normalize_text_rules(text_rules: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Ensure the text_rules block has the expected structure and types."""
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


# =====================================================
# Profile normalization
# =====================================================

def normalize_profile(profile: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Normalize a single site/default profile."""
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

    selectors_and_filters = [
        "rubric_selectors",
        "card_selectors",
        "article_link_selectors",
        "pagination_selectors",
        "title_selectors",
        "date_selectors",
        "author_selectors",
        "category_selectors",
        "image_selectors",
        "content_selectors",
        "remove_selectors",
        "article_url_filters",
        "article_url_excludes",
    ]
    for key in selectors_and_filters:
        profile[key] = clean_list(profile.get(key))

    strategy_fields = [
        "rubric_strategy",
        "author_strategy",
        "category_strategy",
        "card_exclude_classes",
    ]
    for key in strategy_fields:
        if key in profile:
            profile[key] = clean_list(profile[key])
        else:
            profile.pop(key, None)

    profile["match"] = normalize_site_match(profile.get("match"))

    if "text_rules" in profile:
        profile["text_rules"] = normalize_text_rules(profile["text_rules"])

    return profile


# =====================================================
# Top-level normalization
# =====================================================

def normalize_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize the entire YAML config structure."""
    config = copy.deepcopy(config or {})

    config.setdefault("version", "6.5")
    config.setdefault("name", "optimal_pipeline")
    config.setdefault("global", {})
    config.setdefault("default_profile", {})
    config.setdefault("reusable_strategies", {})
    config.setdefault("sites", {})
    config.setdefault("languages", {})

    if not isinstance(config["global"], dict):
        config["global"] = {}
    if not isinstance(config["default_profile"], dict):
        config["default_profile"] = {}
    if not isinstance(config["reusable_strategies"], dict):
        config["reusable_strategies"] = {}
    if not isinstance(config["sites"], dict):
        config["sites"] = {}
    if not isinstance(config["languages"], dict):
        config["languages"] = {}

    global_cfg = config["global"]

    for key in [
        "rubric_strategies_order",
        "author_strategies_order",
        "category_strategies_order",
    ]:
        if key in global_cfg:
            global_cfg[key] = clean_list(global_cfg[key])

    if "request" not in global_cfg or not isinstance(global_cfg.get("request"), dict):
        global_cfg["request"] = {}
    req = global_cfg["request"]
    req.setdefault("timeout", 25)
    req.setdefault("retries", 3)
    req.setdefault("backoff_base", 1.6)
    req.setdefault("default_delay_seconds", 1.2)
    req.setdefault("rubric_delay_range_seconds", [1.0, 2.2])
    req.setdefault("max_threads", 8)
    req.setdefault("verify_ssl", True)

    if "limits" not in global_cfg or not isinstance(global_cfg.get("limits"), dict):
        global_cfg["limits"] = {}
    limits = global_cfg["limits"]
    limits.setdefault("max_depth", 3)
    limits.setdefault("max_pages", 1000)
    limits.setdefault("max_items", 10000)
    limits.setdefault("excerpt_len", 260)
    limits.setdefault("hash_truncate", 32)

    if "url_rules" not in global_cfg or not isinstance(global_cfg.get("url_rules"), dict):
        global_cfg["url_rules"] = {}
    url_rules = global_cfg["url_rules"]
    for k in ("article_url_filters_default", "article_url_excludes_default", "noise_url_parts"):
        url_rules[k] = clean_list(url_rules.get(k))

    if "content_cleanup" not in global_cfg or not isinstance(global_cfg.get("content_cleanup"), dict):
        global_cfg["content_cleanup"] = {}
    cc = global_cfg["content_cleanup"]
    cc["remove_selectors"] = clean_list(cc.get("remove_selectors"))

    if "meta_extraction" not in global_cfg or not isinstance(global_cfg.get("meta_extraction"), dict):
        global_cfg["meta_extraction"] = {}
    meta = global_cfg["meta_extraction"]
    for k in ("title_selectors_default", "date_selectors_default", "author_selectors_default",
              "category_selectors_default", "image_selectors_default"):
        meta[k] = clean_list(meta.get(k))

    if "text_rules" in global_cfg:
        global_cfg["text_rules"] = normalize_text_rules(global_cfg["text_rules"])
    else:
        global_cfg["text_rules"] = normalize_text_rules({})

    reusable = config["reusable_strategies"]
    for section in ["rubric_sources", "author_sources", "category_sources", "date_locale"]:
        if section not in reusable or not isinstance(reusable.get(section), dict):
            reusable[section] = {}

    config["default_profile"] = normalize_profile(config["default_profile"])

    normalized_sites = {}
    for site_key, site_cfg in config["sites"].items():
        if not isinstance(site_cfg, dict):
            continue
        normalized_sites[str(site_key)] = normalize_profile(site_cfg)
    config["sites"] = normalized_sites

    return config


# =====================================================
# Site matching
# =====================================================

def get_site_key(url: str, config: Dict[str, Any]) -> str:
    """Return the matching site key for the given URL."""
    domain = safe_domain(url)
    if not domain:
        return "default"

    sites = config.get("sites", {})
    if not isinstance(sites, dict):
        return "default"

    candidates = []
    for site_key, site_cfg in sites.items():
        if not isinstance(site_cfg, dict):
            continue
        matches = normalize_site_match(site_cfg.get("match"))
        for pat in matches:
            if pat and (pat in domain or domain.endswith("." + pat)):
                candidates.append((len(pat), str(site_key)))

    if not candidates:
        return "default"

    candidates.sort(reverse=True)
    return candidates[0][1]


# =====================================================
# Auto-detection of site family
# =====================================================

def resolve_auto_family(domain: str, html_sample: str, default_profile: Dict[str, Any]) -> Dict[str, Any]:
    """Try to detect site family from default_profile and return its profile overrides."""
    families_cfg = default_profile.get("auto_detect_family", {})
    if not families_cfg.get("enabled") or not html_sample:
        return {}

    families = families_cfg.get("families", {})
    best_family = None
    best_score = 0

    for family_name, family_cfg in families.items():
        score = 0
        indicators = family_cfg.get("indicators", [])
        for ind in indicators:
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

    if best_family:
        return families[best_family].get("profile", {})
    return {}


# =====================================================
# Configuration resolution
# =====================================================

def resolve_noise_words(site_cfg: Dict[str, Any], config: Dict[str, Any]) -> List[str]:
    """Collect noise words based on noise_language from languages.yaml."""
    languages = site_cfg.get("noise_language", ["common", "ru"])
    
    lang_data = config.get("languages", {})
    words = []
    for lang in languages:
        if lang in lang_data:
            words.extend(lang_data[lang].get("noise_words", []))
    
    if not words:
        noise_dict = config.get("global", {}).get("text_rules", {}).get("noise_words", {})
        for lang in languages:
            words.extend(noise_dict.get(lang, []))
    
    return list(set(words))


def resolve_author_regex_patterns(site_cfg: Dict[str, Any], config: Dict[str, Any]) -> List[str]:
    """Collect author regex patterns from language packages in languages.yaml."""
    patterns = []
    lang_data = config.get("languages", {})
    
    # Always include _common universal patterns first
    common = lang_data.get("_common", {})
    if "author_regex_universal" in common:
        patterns.extend(clean_list(common["author_regex_universal"]))
    
    # Get noise_language list (languages for UI/noise filtering)
    noise_langs = site_cfg.get("noise_language", ["common", "ru"])
    
    # Add author_regex from each noise language
    for lang in noise_langs:
        if lang in lang_data and "author_regex" in lang_data[lang]:
            patterns.extend(clean_list(lang_data[lang]["author_regex"]))
    
    # Also add from default_language if it differs from noise languages
    default_lang = site_cfg.get("default_language")
    if default_lang and default_lang not in noise_langs and default_lang in lang_data:
        if "author_regex" in lang_data[default_lang]:
            patterns.extend(clean_list(lang_data[default_lang]["author_regex"]))
    
    # If still empty, fallback to global reusable_strategies patterns
    if not patterns:
        reusable = config.get("reusable_strategies", {})
        author_src = reusable.get("author_sources", {})
        regex_cfg = author_src.get("regex_in_content", {})
        patterns = clean_list(regex_cfg.get("patterns", []))
    
    return patterns


def get_site_config(url: str, html_sample: str = "", config: Dict[str, Any] = None) -> Dict[str, Any]:
    """Return fully resolved site configuration for a given URL."""
    if not isinstance(config, dict):
        raise ValueError("config must be a dict")

    site_key = get_site_key(url, config)
    default_profile = config.get("default_profile", {})
    site_cfg = config.get("sites", {}).get(site_key, {}) if site_key != "default" else {}

    merged = deep_merge(default_profile, site_cfg)

    if site_key == "default" and html_sample:
        family_profile = resolve_auto_family(
            safe_domain(url), html_sample, default_profile
        )
        if family_profile:
            merged = deep_merge(merged, family_profile)

    if merged.get("verify_ssl") is None:
        global_verify = config.get("global", {}).get("request", {}).get("verify_ssl")
        if global_verify is not None:
            merged["verify_ssl"] = bool(global_verify)

    # Resolve language-dependent data before normalization
    author_patterns = resolve_author_regex_patterns(merged, config)
    noise_words = resolve_noise_words(merged, config)

    merged = normalize_profile(merged)

    # Inject all resolved metadata into the final config
    merged["_site_key"] = site_key
    merged["_domain"] = safe_domain(url)
    merged["_url"] = url
    merged["_default_author"] = merged.get("default_author") or default_profile.get("default_author")
    merged["_global_request"] = copy.deepcopy(config.get("global", {}).get("request", {}))
    merged["_global_limits"] = copy.deepcopy(config.get("global", {}).get("limits", {}))
    merged["_global_url_rules"] = copy.deepcopy(config.get("global", {}).get("url_rules", {}))
    merged["_global_content_cleanup"] = copy.deepcopy(config.get("global", {}).get("content_cleanup", {}))
    merged["_global_meta_extraction"] = copy.deepcopy(config.get("global", {}).get("meta_extraction", {}))
    merged["_global_text_rules"] = copy.deepcopy(config.get("global", {}).get("text_rules", {}))
    merged["_reusable_strategies"] = copy.deepcopy(config.get("reusable_strategies", {}))
    merged["_noise_words"] = noise_words
    merged["_author_regex_patterns"] = author_patterns

    return merged


# =====================================================
# Loader — monolithic YAML
# =====================================================

def load_config(path: str) -> Dict[str, Any]:
    """Load and normalize a monolithic YAML configuration file."""
    if not path:
        logger.error("[ERROR] Config path is empty")
        raise ValueError("Config path is empty")
    if not os.path.exists(path):
        logger.error(f"[ERROR] Config file not found: {path}")
        raise FileNotFoundError(f"Config file not found: {path}")

    logger.info(f"[INFO] Loading monolithic config from {path}")
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    if not isinstance(raw, dict):
        logger.error("[ERROR] YAML root must be a mapping/object")
        raise ValueError("YAML root must be a mapping/object")

    return normalize_config(extract_root_config(raw))


# =====================================================
# Modular config loader — universal.yaml + includes
# =====================================================

def load_modular_config(universal_path: str) -> Dict[str, Any]:
    """
    Load a modular configuration from universal.yaml + included files.
    """
    if not os.path.exists(universal_path):
        logger.error(f"[ERROR] Universal config not found: {universal_path}")
        raise FileNotFoundError(f"Universal config not found: {universal_path}")
    
    config_dir = os.path.dirname(os.path.abspath(universal_path))
    logger.info(f"[INFO] Loading modular config from {universal_path}")
    
    with open(universal_path, "r", encoding="utf-8") as f:
        universal = yaml.safe_load(f) or {}
    
    if not isinstance(universal, dict):
        logger.error("[ERROR] universal.yaml root must be a mapping")
        raise ValueError("universal.yaml root must be a mapping")
    
    includes = clean_list(universal.get("includes", []))
    
    if not includes:
        logger.error("[ERROR] No 'includes' found in universal.yaml")
        raise ValueError("No 'includes' found in universal.yaml — nothing to load")
    
    result = copy.deepcopy(universal)
    result.pop("includes", None)
    
    for include_file in includes:
        include_path = os.path.join(config_dir, include_file)
        
        if not os.path.exists(include_path):
            msg = f"Included file not found, skipping: {include_path}"
            print(f"[WARN] {msg}")
            logger.warning(f"[WARN] {msg}")
            continue
        
        with open(include_path, "r", encoding="utf-8") as f:
            included_data = yaml.safe_load(f) or {}
        
        if not isinstance(included_data, dict):
            msg = f"{include_file} is not a mapping, skipping"
            print(f"[WARN] {msg}")
            logger.warning(f"[WARN] {msg}")
            continue
        
        msg = f"Loaded: {include_file}"
        print(f"  [OK] {msg}")
        logger.info(f"[OK] {msg}")
        result = deep_merge(result, included_data)
    
    normalized = normalize_config(extract_root_config(result))
    logger.info(f"[OK] Modular config loaded successfully: {len(result.get('sites', {}))} sites")
    return normalized


# =====================================================
# Convenience class
# =====================================================

@dataclass
class ConfigLoader:
    """Reusable config loader wrapper — supports both monolithic and modular."""

    path: str
    config: Dict[str, Any] = field(default_factory=dict)

    def load(self) -> Dict[str, Any]:
        with open(self.path, "r", encoding="utf-8") as f:
            header = yaml.safe_load(f) or {}
        
        if "includes" in header:
            logger.info("[INFO] Auto-detected MODULAR config")
            self.config = load_modular_config(self.path)
        else:
            logger.info("[INFO] Auto-detected MONOLITHIC config")
            self.config = load_config(self.path)
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


# =====================================================
# Helper API for engine integration
# =====================================================

def load_yaml_config(path: str) -> Dict[str, Any]:
    """Load config — auto-detects monolithic vs modular."""
    with open(path, "r", encoding="utf-8") as f:
        header = yaml.safe_load(f) or {}
    
    if "includes" in header:
        return load_modular_config(path)
    return load_config(path)


def match_site(url: str, config: Dict[str, Any]) -> str:
    return get_site_key(url, config)


def get_config(url: str, config: Dict[str, Any], html_sample: str = "") -> Dict[str, Any]:
    return get_site_config(url, html_sample, config)


def load_social_config() -> Dict[str, Any]:
    """Load modular social media configuration."""
    return load_modular_config("social/config/social.yaml")

# =====================================================
# CLI smoke test — auto-detects modular vs monolithic
# =====================================================
if __name__ == "__main__":
    cfg_path = 'universal.yaml'
    test_url = 'http://khovar.tj/'

    if len(sys.argv) >= 2:
        cfg_path = sys.argv[1]
    if len(sys.argv) >= 3:
        test_url = sys.argv[2]

    print(f"Config: {cfg_path}")
    print(f"URL: {test_url}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        header = yaml.safe_load(f) or {}

    if "includes" in header:
        print("Mode: MODULAR")
        cfg = load_modular_config(cfg_path)
    else:
        print("Mode: MONOLITHIC")
        cfg = load_config(cfg_path)

    site_key = get_site_key(test_url, cfg)
    site_cfg = get_site_config(test_url, "", cfg)

    print(f"\nSITE_KEY: {site_key}")
    try:
        print(json.dumps({
            "site_key": site_key,
            "domain": site_cfg.get("_domain"),
            "default_language": site_cfg.get("default_language"),
            "default_author": site_cfg.get("_default_author"),
            "start_url": site_cfg.get("start_url"),
            "title_selectors": site_cfg.get("title_selectors", [])[:3],
            "rubric_strategy": site_cfg.get("rubric_strategy"),
            "author_strategy": site_cfg.get("author_strategy"),
            "noise_words_sample": site_cfg.get("_noise_words", [])[:10],
            "author_regex_patterns_count": len(site_cfg.get("_author_regex_patterns", [])),
            "has_text_rules": bool(site_cfg.get("_global_text_rules")),
            "has_reusable_strategies": bool(site_cfg.get("_reusable_strategies")),
        }, ensure_ascii=False, indent=2))
    except UnicodeEncodeError:
        print(json.dumps({
            "site_key": site_key,
            "domain": site_cfg.get("_domain"),
            "default_language": site_cfg.get("default_language"),
            "default_author": site_cfg.get("_default_author"),
            "start_url": site_cfg.get("start_url"),
            "title_selectors": site_cfg.get("title_selectors", [])[:3],
            "rubric_strategy": site_cfg.get("rubric_strategy"),
            "author_strategy": site_cfg.get("author_strategy"),
            "noise_words_sample": site_cfg.get("_noise_words", [])[:10],
            "author_regex_patterns_count": len(site_cfg.get("_author_regex_patterns", [])),
            "has_text_rules": bool(site_cfg.get("_global_text_rules")),
            "has_reusable_strategies": bool(site_cfg.get("_reusable_strategies")),
        }, ensure_ascii=True, indent=2))

    lang = site_cfg.get("default_language", "unknown")
    if lang and lang in cfg.get("languages", {}):
        lang_pkg = cfg["languages"][lang]
        print(f"\nLanguage package '{lang}':")
        print(f"  noise_words: {len(lang_pkg.get('noise_words', []))} words")
        print(f"  stopwords: {len(lang_pkg.get('stopwords', []))} words")
        if lang_pkg.get("date_locale"):
            print(f"  date_locale: {len(lang_pkg['date_locale'])} months")
        if lang_pkg.get("author_regex"):
            print(f"  author_regex: {len(lang_pkg['author_regex'])} patterns")
        if lang_pkg.get("category_url_patterns"):
            print(f"  category_url_patterns: {len(lang_pkg['category_url_patterns'])} patterns")
    
    # Show resolved author regex patterns
    author_patterns = site_cfg.get("_author_regex_patterns", [])
    if author_patterns:
        print(f"\nResolved author_regex patterns: {len(author_patterns)}")
        for i, pat in enumerate(author_patterns, 1):
            print(f"  {i}. {pat[:100]}...")
    else:
        print("\nResolved author_regex patterns: NONE — check languages.yaml!")