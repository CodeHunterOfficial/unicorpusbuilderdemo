# app.py — Universal Multilingual Scraper Dashboard (v10 — dynamic presets, full social params)
import requests
import streamlit as st
import sys
import os
import json
import time
import subprocess
import threading
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import pandas as pd
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.loader import (
    load_modular_config, get_site_key, get_site_config,
    load_social_config
)
from pipeline.pipeline_core import PipelineEngine
from pipeline.pipeline_extraction import ExtractionEngine, run as run_extraction
from logger_setup import get_file_logger

logger = get_file_logger("streamlit_app", "logs/streamlit_app.log")

PROFILES_FILE = "streamlit_profiles.json"
TASKS_LOG_DIR = "task_logs"
os.makedirs(TASKS_LOG_DIR, exist_ok=True)

OUTPUT_BASE = "output"

# Dynamic presets
def build_url_presets():
    presets = {}
    try:
        config = load_modular_config("config/universal.yaml")
        sites = config.get("sites", {})
        for site_key, site_cfg in sites.items():
            match_list = site_cfg.get("match", [])
            if not match_list:
                continue
            domain = match_list[0]
            start_url = site_cfg.get("start_url") or f"https://{domain}"
            lang = site_cfg.get("default_language", "unknown")
            flag = {
                "tg":"TJ","tt":"TT","ru":"RU","ba":"BA","os":"OS","uz":"UZ","en":"EN"
            }.get(lang, "??")
            label = f"{flag} {lang.upper()} — {site_key}"
            presets[label] = start_url
    except Exception as e:
        logger.error(f"Failed to build URL presets: {e}")
    return presets

URL_PRESETS = build_url_presets()


def build_wiki_presets():
    """Build wiki presets by checking actual availability of dumps."""
    standalone = {
        "cvwiki":       "Chuvash Wikipedia",
        "udmwiki":      "Udmurt Wikipedia",
        "mhrwiki":      "Mari Wikipedia (Meadow)",
        "sahwiki":      "Yakut Wikipedia",
        "bawiki":       "Bashkir Wikipedia",
        "ttwiki":       "Tatar Wikipedia",
        "oswiki":       "Ossetian Wikipedia",
        "bawikibooks":  "Bashkir Wikibooks",
        "ttwikibooks":  "Tatar Wikibooks",
        "cvwikibooks":  "Chuvash Wikibooks",
    }

    incubator_prefixes = {
        "Wb/udm/":  "Udmurt Wikibooks",
        "Wb/mhr/":  "Mari Wikibooks",
        "Wb/sah/":  "Yakut Wikibooks",
        "Wb/os/":   "Ossetian Wikibooks",
    }

    presets = {}

    for db, name in standalone.items():
        url = f"https://dumps.wikimedia.org/{db}/latest/{db}-latest-pages-articles.xml.bz2"
        try:
            r = requests.head(url, timeout=5)
            if r.status_code == 200:
                presets[name] = {
                    "url": url,
                    "dir": f"{db}_output",
                    "filter_prefix": None
                }
        except Exception:
            pass

    incubator_url = "https://dumps.wikimedia.org/incubatorwiki/latest/incubatorwiki-latest-pages-articles.xml.bz2"
    try:
        r = requests.head(incubator_url, timeout=5)
        if r.status_code == 200:
            for prefix, desc in incubator_prefixes.items():
                presets[f"{desc} (Incubator, filter {prefix})"] = {
                    "url": incubator_url,
                    "dir": f"incubator_{prefix.replace('/', '_').strip('_')}_output",
                    "filter_prefix": prefix
                }
    except Exception:
        pass

    return presets

if 'wiki_presets' not in st.session_state:
    st.session_state.wiki_presets = build_wiki_presets()


SOCIAL_SCRIPTS = {
    "VK": "social/vk_scraper.py",
    "Telegram": "social/telegram_scraper.py",
    "Rutube": "social/rutube_scraper.py",
}

# Helpers
def load_profiles():
    if os.path.exists(PROFILES_FILE):
        with open(PROFILES_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_profiles(profiles):
    with open(PROFILES_FILE, "w", encoding="utf-8") as f:
        json.dump(profiles, f, indent=2, ensure_ascii=False)

def run_command_in_thread(cmd, task_id, stop_event):
    log_path = os.path.join(TASKS_LOG_DIR, f"{task_id}.log")
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, encoding='utf-8')
    with open(log_path, 'w', encoding='utf-8') as log_file:
        for line in iter(p.stdout.readline, ''):
            if stop_event.is_set():
                p.terminate()
                log_file.write("\n[STOPPED BY USER]\n")
                break
            log_file.write(line)
            log_file.flush()
    p.stdout.close()
    ret = p.wait()
    with open(log_path, 'a', encoding='utf-8') as log_file:
        log_file.write(f"\nProcess exited with code {ret}\n")
    logger.info(f"Task {task_id} finished with code {ret}")
    return ret

def read_log_tail(task_id, lines=100):
    log_path = os.path.join(TASKS_LOG_DIR, f"{task_id}.log")
    if not os.path.exists(log_path):
        return "No log yet."
    with open(log_path, 'r', encoding='utf-8') as f:
        all_lines = f.readlines()
        return ''.join(all_lines[-lines:])

def run_subprocess_live(cmd, placeholder, post_process_json=False):
    logger.info(f"Running: {' '.join(cmd)}")
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, encoding='utf-8')
    full_output = ""
    for line in iter(p.stdout.readline, ''):
        full_output += line
        placeholder.code(full_output[-10000:], language="log")
    p.stdout.close()
    ret = p.wait()
    if post_process_json and ret == 0:
        try:
            json_match = re.search(r'\{.*\}', full_output, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                placeholder.json(data)
        except:
            pass
    logger.info(f"Command finished with code {ret}")
    return ret

# Session state
if 'profiles' not in st.session_state:
    st.session_state.profiles = load_profiles()
if 'running_tasks' not in st.session_state:
    st.session_state.running_tasks = {}
if 'bulk_select_all' not in st.session_state:
    st.session_state.bulk_select_all = True

# Page config
st.set_page_config(page_title="Multilingual Scraper Dashboard", page_icon="🌐", layout="wide")
st.title("🌐 Universal Multilingual Scraper Dashboard")
st.markdown("Data collection, document processing and corpus building for low-resource languages.")

mode = st.sidebar.radio("📋 Select Mode", [
    "📖 Documentation & Examples",
    "📊 Config Overview",
    "⚡ Quick Start",
    "🔍 Config Check",
    "📋 Task Manager",
    "💾 Profiles",
    "📈 System Status",
    "🧪 Test Components",
    "📱 Social Scrapers",
    "🚀 Bulk Pipeline"
])

if st.sidebar.button("Refresh wiki presets"):
    with st.spinner("Checking wiki dumps..."):
        st.session_state.wiki_presets = build_wiki_presets()
    st.success("Wiki presets updated!")

# ===================== Config Overview =====================
if mode == "📊 Config Overview":
    st.header("📊 Configuration Overview")
    try:
        main_config = load_modular_config("config/universal.yaml")
    except Exception as e:
        st.error(f"Failed to load main config: {e}")
        main_config = None

    try:
        social_config = load_social_config()
    except Exception:
        social_config = None

    doc_config = None
    if os.path.exists("documents/doc_config.yaml"):
        try:
            with open("documents/doc_config.yaml", "r", encoding="utf-8") as f:
                doc_config = yaml.safe_load(f)
        except:
            pass

    if main_config:
        st.subheader("🌍 News Scraping Configuration")
        col1, col2, col3 = st.columns(3)
        with col1:
            sites = main_config.get("sites", {})
            st.metric("Total Sites", len(sites))
            lang_counts = {}
            for s, c in sites.items():
                l = c.get("default_language", "unknown")
                lang_counts[l] = lang_counts.get(l, 0) + 1
            st.write("**By language:**")
            for l, n in sorted(lang_counts.items()):
                flag = {"tg":"TJ","tt":"TT","ru":"RU","ba":"BA","os":"OS","uz":"UZ","en":"EN"}.get(l,"??")
                st.write(f"{flag} {l}: {n}")
        with col2:
            langs = {k: v for k, v in main_config.get("languages", {}).items() if not k.startswith("_")}
            st.metric("Supported Languages", len(langs))
            for lc, p in langs.items():
                if isinstance(p, dict):
                    nw = len(p.get("noise_words", []))
                    sw = len(p.get("stopwords", []))
                    loc = "✅" if p.get("date_locale") else "❌"
                    ar = "✅" if p.get("author_regex") else "❌"
                    st.write(f"**{lc}**: {nw}n/{sw}s | locale:{loc} | regex:{ar}")
        with col3:
            g = main_config.get("global", {})
            r = g.get("request", {})
            lm = g.get("limits", {})
            st.metric("⏱ Timeout", f"{r.get('timeout','?')}s")
            st.metric("📏 Max Depth", lm.get("max_depth","?"))
            st.metric("📄 Max Pages", lm.get("max_pages","?"))
            st.metric("📰 Max Items", lm.get("max_items","?"))

        with st.expander("📌 Auto-Detect Families"):
            fam = main_config.get("default_profile", {}).get("auto_detect_family", {}).get("families", {})
            if fam:
                for fname, fcfg in fam.items():
                    score = fcfg.get("match_min_score", "?")
                    indicators = len(fcfg.get("indicators", []))
                    st.write(f"- **{fname}**: min score={score}, {indicators} indicators")
            else:
                st.write("No families configured.")
        
        with st.expander("🛠 Reusable Strategies"):
            reusable = main_config.get("reusable_strategies", {})
            if reusable:
                rubric = reusable.get("rubric_sources", {})
                author = reusable.get("author_sources", {})
                category = reusable.get("category_sources", {})
                st.write("**Rubric sources:**", list(rubric.keys()))
                st.write("**Author sources:**", list(author.keys()))
                st.write("**Category sources:**", list(category.keys()))

    if social_config:
        st.subheader("📱 Social Configuration")
        vk_domains = social_config.get("vk_domains", [])
        tg_channels = social_config.get("telegram", {}).get("channels", [])
        rt_videos = social_config.get("rutube", {}).get("videos", [])
        rt_playlists = social_config.get("rutube", {}).get("playlists", [])
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("VK Domains", len(vk_domains))
        col2.metric("TG Channels", len(tg_channels))
        col3.metric("RT Videos", len(rt_videos))
        col4.metric("RT Playlists", len(rt_playlists))

    if doc_config:
        st.subheader("📄 Document Configuration")
        ext = doc_config.get("supported_extensions", {})
        st.metric("Supported Formats", len(ext))

# ===================== Quick Start =====================
elif mode == "⚡ Quick Start":
    st.header("⚡ Quick Start – Corpus Builder")
    c1, c2 = st.columns(2)
    with c1:
        lang = st.selectbox("🌍 Language", ["tg","tt","ru","ba","en","os","uz"])
        preset = st.selectbox("📂 Preset", [
            "All sources", "News only", "Social only", "Wiki only", "Documents only"
        ])
    with c2:
        max_art = st.number_input("📊 Max articles per site", 10, 5000, 100)
        soc = st.multiselect(
            "📱 Social platforms",
            ["vk", "rutube", "telegram"],
            default=["vk"]
        )
    
    if st.button("🚀 Start Corpus Build", type="primary"):
        m = {
            "All sources": "all",
            "News only": "news",
            "Social only": "social",
            "Wiki only": "wiki",
            "Documents only": "docs"
        }
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = f"corpus_{lang}_{ts}.jsonl"
        cmd = [
            sys.executable, "corpus/build_corpus.py",
            "--lang", lang,
            "--output", out,
            "--sources", m[preset],
            "--max-items-per-site", str(max_art)
        ]
        if soc:
            cmd.extend(["--social-sources", ",".join(soc)])
        
        tid = f"corpus_{ts}"
        logger.info(f"Starting corpus build: {tid}, lang={lang}, sources={m[preset]}")
        ev = threading.Event()
        th = threading.Thread(target=run_command_in_thread, args=(cmd, tid, ev))
        th.start()
        st.session_state.running_tasks[tid] = (th, ev)
        st.success(f"✅ Task {tid} started!")

# ===================== Config Check =====================
elif mode == "🔍 Config Check":
    st.header("🔍 Site Config Check")
    
    col1, col2 = st.columns(2)
    with col1:
        preset_url = st.selectbox("🌐 Site preset", list(URL_PRESETS.keys()), key="cfg_preset")
    with col2:
        custom_url = st.text_input("✏️ Custom URL (overrides preset)", key="cfg_custom")
    
    url = custom_url if custom_url else URL_PRESETS[preset_url]
    
    if st.button("🔍 Check Configuration", type="primary"):
        try:
            config = load_modular_config("config/universal.yaml")
            sk = get_site_key(url, config)
            sc = get_site_config(url, "", config)
            lang = sc.get("default_language", "?")
            logger.info(f"Config check: {url} -> {sk}")
            
            st.success(f"Site key: **{sk}**")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🌐 Domain", sc.get("_domain"))
                st.metric("🗣 Language", lang)
                st.metric("✍️ Author", sc.get("_default_author") or "auto-detect")
            with col2:
                st.metric("📂 Rubric strategy", str(sc.get("rubric_strategy", [])))
                st.metric("👤 Author strategy", str(sc.get("author_strategy", [])))
                st.metric("🔊 Noise words", len(sc.get("_noise_words", [])))
            with col3:
                st.metric("🔒 SSL verify", str(sc.get("verify_ssl")))
                st.metric("📱 AMP mode", sc.get("amp_mode", "none"))
                st.metric("🍪 Cookie lang", sc.get("cookie_lang") or "none")
            
            if lang in config.get("languages", {}):
                pkg = config["languages"][lang]
                st.write(f"**Stopwords:** {len(pkg.get('stopwords',[]))}")
                if pkg.get("date_locale"):
                    st.write(f"**Date locale:** {len(pkg['date_locale'])} months")
                if pkg.get("author_regex"):
                    st.write(f"**Author regex:** {len(pkg['author_regex'])} patterns")
            
            noise_sample = sc.get('_noise_words', [])[:15]
            if noise_sample:
                st.write("**Noise words sample:**")
                st.code('\n'.join(noise_sample), language=None)
            
            author_pats = sc.get("_author_regex_patterns", [])
            if author_pats:
                with st.expander(f"📝 Author Regex Patterns ({len(author_pats)})"):
                    for i, pat in enumerate(author_pats, 1):
                        st.code(f"{i}. {pat}", language=None)
            
        except Exception as e:
            logger.error(f"Config check failed for {url}: {e}")
            st.error(f"Error: {e}")

# ===================== Task Manager =====================
elif mode == "📋 Task Manager":
    st.header("📋 Task Manager")
    tab1, tab2 = st.tabs(["🔄 Running Tasks", "📄 Logs"])
    
    with tab1:
        if not st.session_state.running_tasks:
            st.info("No active tasks.")
        else:
            for tid, (t, ev) in list(st.session_state.running_tasks.items()):
                cols = st.columns([3,1])
                with cols[0]:
                    status = "🔵 Running" if t.is_alive() else "✅ Done"
                    st.write(f"{status} **{tid}**")
                with cols[1]:
                    if t.is_alive() and st.button("🛑 Stop", key=f"stop_{tid}"):
                        ev.set()
                        logger.info(f"Task {tid} stopped by user")
                        st.warning("Stopping...")
                with st.expander("📄 Last 30 lines"):
                    st.code(read_log_tail(tid, 30), language="log")
    
    with tab2:
        log_dirs = [Path(TASKS_LOG_DIR), Path("logs")]
        all_logs = []
        for d in log_dirs:
            if d.exists():
                all_logs.extend(list(d.glob("*.log")))
        if all_logs:
            sel = st.selectbox("📄 Select log", sorted([l.stem for l in all_logs]))
            if sel:
                log_path = next((l for l in all_logs if l.stem == sel), None)
                if log_path:
                    with open(log_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        st.code(content[-10000:], language="log")
                        st.download_button("⬇️ Download log", content, f"{sel}.log")
        else:
            st.info("No logs yet.")

# ===================== Profiles =====================
elif mode == "💾 Profiles":
    st.header("💾 Task Profiles")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        pname = st.text_input("📛 Profile name")
    with col2:
        lang = st.selectbox("🌍 Language", ["tg","tt","ru","ba","en","os","uz"])
    with col3:
        src = st.multiselect("📂 Sources", ["news","social","wiki","docs"], default=["news"])
    with col4:
        soc = st.multiselect("📱 Social", ["vk","rutube","telegram"], default=["vk"])
    
    mi = st.number_input("📊 Max items per site", 10, 5000, 100)
    
    if st.button("💾 Save Profile", type="primary") and pname:
        st.session_state.profiles[pname] = {
            "language": lang,
            "sources": src,
            "social_sources": ",".join(soc),
            "max_items_per_site": mi
        }
        save_profiles(st.session_state.profiles)
        logger.info(f"Profile saved: {pname}")
        st.success(f"Profile '{pname}' saved!")
    
    if st.session_state.profiles:
        st.markdown("---")
        st.subheader("Saved Profiles")
        for pn, pd in st.session_state.profiles.items():
            cols = st.columns([3,1,1])
            with cols[0]:
                st.write(f"**{pn}** — {pd.get('language')} | {','.join(pd.get('sources',[]))}")
            with cols[1]:
                if st.button("⚡ Load", key=f"load_{pn}"):
                    st.success(f"Loaded profile: {pn}")
            with cols[2]:
                if st.button("🗑️ Delete", key=f"del_{pn}"):
                    del st.session_state.profiles[pn]
                    save_profiles(st.session_state.profiles)
                    st.rerun()
                    logger.info(f"Profile deleted: {pn}")

# ===================== System Status =====================
elif mode == "📈 System Status":
    st.header("📈 System Status")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📰 News", "📱 Social", "📚 Wiki", "📄 Docs", "📦 Corpus"])
    
    with tab1:
        st.subheader("News Articles")
        news_files = list(Path(OUTPUT_BASE).glob("*_articles.jsonl"))
        if news_files:
            data = []
            for f in sorted(news_files):
                count = sum(1 for _ in open(f, 'r', encoding='utf-8'))
                size_kb = round(f.stat().st_size / 1024, 1)
                data.append({"File": f.name, "Articles": count, "Size (KB)": size_kb})
            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True)
            st.metric("Total Files", len(news_files))
            st.metric("Total Articles", sum(d["Articles"] for d in data))
        else:
            st.info("No news articles yet.")
    
    with tab2:
        st.subheader("Social Data")
        social_dir = Path(OUTPUT_BASE) / "social"
        if social_dir.exists():
            for platform in ["posts", "comments", ""]:
                pdir = social_dir / platform if platform else social_dir
                if pdir.exists():
                    files = list(pdir.glob("*.jsonl"))
                    total = sum(sum(1 for _ in open(f, 'r', encoding='utf-8')) for f in files)
                    st.write(f"📱 **{platform or 'root'}**: {len(files)} files, {total} items")
        else:
            st.info("No social data yet.")
    
    with tab3:
        st.subheader("Wiki Dumps")
        wiki_dir = Path(OUTPUT_BASE) / "wiki"
        if wiki_dir.exists():
            for d in wiki_dir.iterdir():
                if d.is_dir():
                    for f in d.rglob("*_all.jsonl"):
                        count = sum(1 for _ in open(f, 'r', encoding='utf-8'))
                        st.write(f"📚 {d.name}: {count} articles")
        else:
            st.info("No wiki dumps yet.")
    
    with tab4:
        st.subheader("Documents")
        docs_dir = Path(OUTPUT_BASE) / "documents"
        if docs_dir.exists():
            for f in docs_dir.glob("*.jsonl"):
                count = sum(1 for _ in open(f, 'r', encoding='utf-8'))
                st.write(f"📄 {f.name}: {count} documents")
        else:
            st.info("No documents yet.")
    
    with tab5:
        st.subheader("Corpus Files")
        corpus_dir = Path(OUTPUT_BASE) / "corpus"
        if corpus_dir.exists():
            corpus_files = list(corpus_dir.glob("*.jsonl"))
        else:
            corpus_files = list(Path(".").glob("corpus*.jsonl"))
        if corpus_files:
            data = []
            for f in sorted(corpus_files):
                count = sum(1 for _ in open(f, 'r', encoding='utf-8'))
                size_kb = round(f.stat().st_size / 1024, 1)
                data.append({"File": f.name, "Articles": count, "Size (KB)": size_kb})
            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No corpus built yet.")

# ===================== Test Components =====================
elif mode == "🧪 Test Components":
    st.header("🧪 Test Individual Components")
    
    script = st.selectbox("📦 Select Component", [
        "config/loader.py",
        "pipeline/pipeline_core.py",
        "pipeline/pipeline_extraction.py",
        "social/vk_scraper.py",
        "social/rutube_scraper.py",
        "social/telegram_scraper.py",
        "wiki/wiki_dump_parser.py",
        "documents/universal_doc_parser.py",
        "corpus/build_corpus.py",
        "build_vk_dataset.py",
    ])
    
    st.markdown("---")
    
    def run_script(script_path, extra_args):
        cmd = [sys.executable, script_path] + extra_args
        ph = st.empty()
        rc = run_subprocess_live(cmd, ph)
        if rc == 0:
            st.success(f"✅ {script_path} completed")
        else:
            st.error(f"❌ Failed with code {rc}")

    if script == "config/loader.py":
        st.subheader("🔧 Config Loader Test")
        col1, col2 = st.columns(2)
        with col1:
            config_preset = st.selectbox("📄 Config preset", [
                "config/universal.yaml",
                "config/global.yaml",
                "config/languages.yaml"
            ])
        with col2:
            preset_url = st.selectbox("🌐 Site preset", list(URL_PRESETS.keys()), key="test_cfg_url")
            custom_url = st.text_input("✏️ Custom URL (overrides preset)", key="test_cfg_custom")
        url = custom_url if custom_url else URL_PRESETS[preset_url]
        if st.button("🚀 Run Config Check"):
            run_script(script, [config_preset, url])

    elif script in ("pipeline/pipeline_core.py", "pipeline/pipeline_extraction.py"):
        st.subheader("📰 Pipeline Test")
        col1, col2, col3 = st.columns(3)
        with col1:
            config_preset = st.selectbox("📄 Config", ["config/universal.yaml"], key="pipe_cfg")
        with col2:
            preset_url = st.selectbox("🌐 Site", list(URL_PRESETS.keys()), key="pipe_url")
            custom_url = st.text_input("✏️ Custom URL", key="pipe_custom")
        with col3:
            max_items = st.number_input("📊 Max items (0=global limit)", 0, 10000, 0, key="pipe_max")
        url = custom_url if custom_url else URL_PRESETS[preset_url]
        extra = [config_preset, url]
        if max_items > 0:
            extra.append(str(max_items))
        if st.button("🚀 Run Pipeline"):
            run_script(script, extra)

    elif script in SOCIAL_SCRIPTS.values():
        platform = {v:k for k,v in SOCIAL_SCRIPTS.items()}[script]
        st.subheader(f"📱 {platform} Scraper")
        extra = []

        col1, col2 = st.columns(2)
        with col1:
            lang = st.selectbox("Language filter", ["", "tg","tt","ru","ba","os","uz","en"], key=f"{platform}_lang")
            max_items = st.number_input("Max posts/videos per target (0 = all)", 0, 10000, 50, key=f"{platform}_max")
        with col2:
            pass
        if lang:
            extra.extend(["--lang", lang])
        if max_items > 0:
            extra.extend(["--max-posts", str(max_items)] if platform != "Rutube" else ["--max-comments", str(max_items)])

        if platform == "VK":
            token = st.text_input("Token (overrides config)", type="password", key="vk_token")
            manual = st.text_input("Domains (space-separated, overrides config)", key="vk_domains")
            if token:
                extra.extend(["--token", token])
            if manual:
                extra.extend(["--domains"] + manual.split())
        elif platform == "Telegram":
            api_id = st.text_input("API ID", key="tg_api_id")
            api_hash = st.text_input("API Hash", type="password", key="tg_api_hash")
            manual = st.text_input("Channels (space-separated, overrides config)", key="tg_channels")
            if api_id:
                extra.extend(["--api-id", api_id])
            if api_hash:
                extra.extend(["--api-hash", api_hash])
            if manual:
                extra.extend(["--channels"] + manual.split())
        elif platform == "Rutube":
            video_id = st.text_input("Single video ID", key="rt_video_id")
            playlist_id = st.text_input("Single playlist ID", key="rt_playlist_id")
            max_videos = st.number_input("Max videos per playlist", 1, 1000, 10, key="rt_max_videos")
            if video_id:
                extra.extend(["--video-id", video_id])
            if playlist_id:
                extra.extend(["--playlist-id", playlist_id])
            extra.extend(["--max-videos", str(max_videos)])

        if st.button(f"🚀 Run {platform} Scraper"):
            run_script(script, extra)

    elif script == "wiki/wiki_dump_parser.py":
        st.subheader("Wiki Dump Parser Test")
        preset_list = list(st.session_state.wiki_presets.keys())
        preset_choice = st.selectbox(
            "Wiki preset",
            ["Select preset..."] + preset_list + ["Custom URL"],
            key="wiki_preset"
        )
        if preset_choice == "Select preset...":
            dump_url = ""
            output_dir = "wiki_output"
            filter_prefix = None
        elif preset_choice == "Custom URL":
            dump_url = st.text_input("Dump URL", key="wiki_custom_url")
            output_dir = st.text_input("Output directory", value="wiki_output", key="wiki_custom_dir")
            filter_prefix = None
        else:
            preset = st.session_state.wiki_presets[preset_choice]
            dump_url = preset["url"]
            output_dir = preset["dir"]
            filter_prefix = preset.get("filter_prefix")
            st.info(f"{dump_url[:120]}...")
            if filter_prefix:
                st.caption(f"Filter prefix: {filter_prefix}")
            custom_override = st.text_input("Override URL (optional)", key="wiki_override")
            if custom_override:
                dump_url = custom_override

        col1, col2 = st.columns(2)
        with col1:
            max_articles = st.number_input("Max articles (0=all)", 0, 100000, 100, key="wiki_max")
        with col2:
            min_length = st.number_input("Min text length", 0, 10000, 100, key="wiki_min")
        translit = st.checkbox("Transliterate to Latin", key="wiki_translit")

        extra = [dump_url, output_dir]
        if filter_prefix:
            extra.extend(["--filter-prefix", filter_prefix])
        if max_articles > 0:
            extra.extend(["--max-articles", str(max_articles)])
        if min_length > 0:
            extra.extend(["--min-length", str(min_length)])
        if translit:
            extra.append("--transliterate")

        if st.button("Run Wiki Parser", disabled=not dump_url):
            run_script(script, extra)

    elif script == "documents/universal_doc_parser.py":
        st.subheader("📄 Document Parser Test")
        col1, col2 = st.columns(2)
        with col1:
            input_path = st.text_input("📁 Input file or folder", "documents/samples")
            doc_config = st.text_input("📄 Config", "documents/doc_config.yaml")
        with col2:
            output_dir = st.text_input("📁 Output directory", placeholder="Leave empty for default")
        extra = [input_path, "-c", doc_config]
        if output_dir:
            extra.extend(["-o", output_dir])
        if st.button("🚀 Run Document Parser"):
            run_script(script, extra)

    elif script == "corpus/build_corpus.py":
        st.subheader("📚 Corpus Builder Test")
        col1, col2, col3 = st.columns(3)
        with col1:
            lang = st.selectbox("🌍 Language", ["tg","tt","ru","ba","en","os","uz"])
        with col2:
            sources = st.multiselect("📂 Sources", ["news","social","wiki","docs","all"], default=["news"])
        with col3:
            max_items = st.number_input("📊 Max items per source", 1, 10000, 50)
        soc_sources = st.multiselect("📱 Social platforms", ["vk","rutube","telegram"], default=["vk"])
        output_file = st.text_input("💾 Output file", f"corpus_{lang}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl")
        extra = [
            "--lang", lang,
            "--sources", ",".join(sources),
            "--max-items-per-site", str(max_items),
            "--output", output_file
        ]
        if soc_sources:
            extra.extend(["--social-sources", ",".join(soc_sources)])
        if st.button("🚀 Build Corpus"):
            run_script(script, extra)

    elif script == "build_vk_dataset.py":
        st.subheader("📦 Build Unified VK Dataset")
        st.markdown("""
        Cleans and merges all collected VK posts and comments into a single dataset.
        - Removes emoji
        - Filters short posts (< 10 characters) and short comments
        - Merges comments with their posts
        - Adds category field (domain name)
        - **Optional language filter** – keeps only posts/comments matching the selected language
        """)
        col_lang, col_btn = st.columns([2, 1])
        with col_lang:
            build_lang = st.selectbox(
                "🌍 Language filter",
                ["All languages", "tg", "os", "udm", "ba", "tt"],
                help="Keep only posts and comments with required characters of the chosen language"
            )
        with col_btn:
            st.write("")
            if st.button("🚀 Build VK Dataset"):
                script_path = "social/scripts/build_vk_dataset.py"
                extra = []
                if build_lang != "All languages":
                    extra.extend(["--lang", build_lang])
                run_script(script_path, extra)

# ===================== Social Scrapers =====================
elif mode == "📱 Social Scrapers":
    st.header("📱 Social Scrapers")
    platform = st.selectbox("Platform", list(SOCIAL_SCRIPTS.keys()))
    
    col1, col2 = st.columns(2)
    with col1:
        lang = st.selectbox("Language filter", ["all", "tg", "tt", "ru", "ba", "os", "uz", "en"], key="soc_lang")
    with col2:
        max_items = st.number_input("Max posts/videos per target (0 = all)", 0, 10000, 50, key="soc_max")
    
    extra = []
    if lang != "all":
        extra.extend(["--lang", lang])
    
    if max_items > 0:
        extra.extend(["--max-posts", str(max_items)] if platform != "Rutube" else ["--max-comments", str(max_items)])
    
    if platform == "VK":
        token = st.text_input("Token (overrides config)", type="password", key="soc_vk_token")
        manual = st.text_input("Domains (space-separated, overrides config)", key="soc_vk_domains")
        if token:
            extra.extend(["--token", token])
        if manual:
            extra.extend(["--domains"] + manual.split())
    elif platform == "Telegram":
        api_id = st.text_input("API ID", key="soc_tg_api_id")
        api_hash = st.text_input("API Hash", type="password", key="soc_tg_api_hash")
        manual = st.text_input("Channels (space-separated, overrides config)", key="soc_tg_channels")
        if api_id:
            extra.extend(["--api-id", api_id])
        if api_hash:
            extra.extend(["--api-hash", api_hash])
        if manual:
            extra.extend(["--channels"] + manual.split())
    elif platform == "Rutube":
        video_id = st.text_input("Single video ID", key="soc_rt_video_id")
        playlist_id = st.text_input("Single playlist ID", key="soc_rt_playlist_id")
        max_videos = st.number_input("Max videos per playlist", 1, 1000, 10, key="soc_rt_max_videos")
        if video_id:
            extra.extend(["--video-id", video_id])
        if playlist_id:
            extra.extend(["--playlist-id", playlist_id])
        extra.extend(["--max-videos", str(max_videos)])
    
    if st.button(f"🚀 Run {platform} Scraper"):
        cmd = [sys.executable, SOCIAL_SCRIPTS[platform]] + extra
        ph = st.empty()
        rc = run_subprocess_live(cmd, ph)
        if rc == 0:
            st.success(f"✅ {platform} scraping completed")
        else:
            st.error(f"❌ Failed with code {rc}")

# ===================== Bulk Pipeline =====================
elif mode == "🚀 Bulk Pipeline":
    st.header("🚀 Bulk Pipeline – Run on Multiple Sites")
    try:
        config = load_modular_config("config/universal.yaml")
    except Exception as e:
        st.error(f"Failed to load config: {e}")
        config = None

    if config:
        sites = config.get("sites", {})
        sites_by_lang = {}
        for sk, sc in sites.items():
            sites_by_lang.setdefault(sc.get("default_language","?"), []).append(sk)

        tab1, tab2 = st.tabs(["📋 From Configuration", "✏️ Manual URLs"])

        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                sel_lang = st.selectbox("🌍 Language", ["All languages"] + sorted(sites_by_lang.keys()), key="bulk_lang")
            with col2:
                max_items = st.number_input("📊 Max articles per site", 1, 10000, 50, key="bulk_max")

            if sel_lang == "All languages":
                target_sites = {k: v for k, v in sites.items()}
            else:
                target_sites = {k: sites[k] for k in sites_by_lang.get(sel_lang, [])}

            st.subheader(f"📋 Sites ({len(target_sites)})")

            if "bulk_select_all" not in st.session_state:
                st.session_state.bulk_select_all = True

            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Select All"):
                    st.session_state.bulk_select_all = True
                    st.rerun()
            with col2:
                if st.button("⬜ Deselect All"):
                    st.session_state.bulk_select_all = False
                    st.rerun()

            selected = []
            cols = st.columns(3)
            for i, sk in enumerate(sorted(target_sites.keys())):
                with cols[i % 3]:
                    key = f"bulk_site_{sk}"
                    if key not in st.session_state:
                        st.session_state[key] = st.session_state.bulk_select_all
                    
                    if st.checkbox(
                        f"{sk} ({target_sites[sk].get('default_language','?')})",
                        key=key
                    ):
                        selected.append(sk)

        with tab2:
            manual_urls = [u.strip() for u in st.text_area(
                "🔗 One URL per line", height=200,
                placeholder="https://example.com/news",
                key="manual_urls"
            ).split('\n') if u.strip().startswith('http')]

        all_urls = list(dict.fromkeys(
            [target_sites[s].get("start_url") or f"https://{target_sites[s].get('match',[''])[0]}"
             for s in selected
             if target_sites.get(s,{}).get('match')] + manual_urls
        ))

        if all_urls:
            st.info(f"🔗 Total URLs: {len(all_urls)}")
            
            if not st.session_state.get("bulk_running", False):
                if st.button("🚀 Start Bulk Scraping", type="primary"):
                    st.session_state.bulk_urls = all_urls
                    st.session_state.bulk_max_items = max_items
                    st.session_state.bulk_log = ""
                    st.session_state.bulk_current = 0
                    st.session_state.bulk_total = len(all_urls)
                    st.session_state.bulk_running = True
                    st.session_state.bulk_results = []
                    logger.info(f"Bulk scraping started: {len(all_urls)} URLs")
                    st.rerun()
            
            if st.session_state.get("bulk_running", False):
                pb = st.progress(min(st.session_state.bulk_current / max(st.session_state.bulk_total, 1), 1.0))
                stt = st.empty()
                lp = st.empty()
                
                stt.text(f"[{st.session_state.bulk_current}/{st.session_state.bulk_total}] Processing...")
                lp.code(st.session_state.bulk_log[-5000:], language="log")
                
                if st.button("🛑 Stop Bulk Scraping"):
                    st.session_state.bulk_running = False
                    logger.info("Bulk scraping stopped by user")
                    st.warning("Stopping after current site...")
                
                if st.session_state.bulk_current < st.session_state.bulk_total and st.session_state.bulk_running:
                    url = st.session_state.bulk_urls[st.session_state.bulk_current]
                    cmd = [sys.executable, "main.py", "pipeline", url, str(st.session_state.bulk_max_items)]
                    
                    try:
                        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
                        st.session_state.bulk_log += f"\n--- {url} ---\n{result.stdout}"
                        if result.stderr:
                            st.session_state.bulk_log += f"\n[STDERR]\n{result.stderr}"
                        st.session_state.bulk_results.append({"url": url, "success": result.returncode == 0})
                    except subprocess.TimeoutExpired:
                        st.session_state.bulk_log += f"\n--- {url} ---\n[TIMEOUT]"
                    except Exception as e:
                        st.session_state.bulk_log += f"\n--- {url} ---\n[ERROR: {e}]"
                    
                    st.session_state.bulk_current += 1
                    time.sleep(1)
                    st.rerun()
                
                elif st.session_state.bulk_current >= st.session_state.bulk_total:
                    pb.progress(1.0)
                    stt.text("✅ Finished!")
                    st.balloons()
                    logger.info("Bulk scraping completed")
                    
                    successes = sum(1 for r in st.session_state.bulk_results if r.get("success"))
                    st.success(f"✅ Completed: {successes}/{len(st.session_state.bulk_results)} sites successful")
                    
                    if st.button("🗑️ Clear Results"):
                        for key in ["bulk_running", "bulk_urls", "bulk_log", "bulk_current", "bulk_total", "bulk_results"]:
                            st.session_state.pop(key, None)
                        st.rerun()
        else:
            st.warning("⚠️ No URLs selected.")
    else:
        st.error("Cannot load configuration.")

# ===================== Documentation & Examples =====================
elif mode == "📖 Documentation & Examples":
    st.header("📖 Documentation & Examples")
    st.markdown("""
                ## 📌 About the project

                **Universal Multilingual Scraper** is a universal text data collector
                for low-resource languages, part of a broader NLP ecosystem.
                The tool can build text corpora from various sources: news websites,
                social networks (VK, Telegram, Rutube), Wikipedia dumps and office documents.

                **Key features**
                - 🌐 **Multilingual**: supports Tajik, Tatar, Bashkir, Russian,
                Ossetian, Uzbek and English.
                - 🧩 **Modular configuration**: all settings are stored in YAML files,
                split by language and source.
                - ⚙️ **Flexible collection strategies**: auto-detection of site families,
                metadata extraction priorities, noise-word and stop-word filtering
                for every language.
                - 🧠 **Smart author extraction**: combination of meta-tags, CSS selectors,
                JSON-LD and regex patterns tailored to each language.
                - 📱 **Social media integration**: collects posts and comments from VK,
                Telegram, Rutube with language filtering options.

                All components are managed through centralised YAML configuration files.
                """)

    with st.expander("📰 News Pipeline (pipeline, bulk, config)"):
        st.code("""
                    # Check configuration for a site
                    python main.py config https://khovar.tj/

                    # Full pipeline (discover + extract)
                    python main.py pipeline https://khovar.tj/
                    python main.py pipeline https://khovar.tj/ 50

                    # Bulk pipeline – all sites of a language
                    python main.py bulk tg
                    python main.py bulk tg 100
                    python main.py bulk all
                    python main.py bulk all 30

                    # Direct script usage (if needed)
                    python pipeline/pipeline_core.py config/universal.yaml https://khovar.tj/
                    python pipeline/pipeline_extraction.py config/universal.yaml https://khovar.tj/ "" 50
                    python config/loader.py config/universal.yaml https://khovar.tj/
                            """)

    with st.expander("📱 Social Scrapers (VK, Telegram, Rutube)"):
        st.subheader("VK")
        st.code("""
                    # Scrape specific domains with language and limit
                    python social/vk_scraper.py --domains club1135692 irta_tv --lang ba --max-posts 5

                    # Scrape all domains of a language from config
                    python social/vk_scraper.py --lang tt --max-posts 100

                    # Scrape all domains from config
                    python social/vk_scraper.py --max-posts 50

                    # With explicit token
                    python social/vk_scraper.py --token your_token --domains allahtan --lang tt --max-posts 5

                    # Via main entry
                    python main.py social vk --lang tt --max-posts 100
                    python main.py social vk --domains club1135692 irta_tv --lang ba --max-posts 10
                            """)
        st.subheader("Telegram")
        st.code("""
                # Specific channels
                python social/telegram_scraper.py --channels vatantat --lang tt --max-posts 50

                # All channels of a language
                python social/telegram_scraper.py --lang tt --max-posts 100

                # With custom API credentials
                python social/telegram_scraper.py --api-id your_api_id --api-hash your_api_hash --channels vatantat --lang tt --max-posts 30

                # Via main entry
                python main.py social telegram --lang tt --max-posts 100
                python main.py social telegram --channels vatantat --lang tt --max-posts 50
               """)
        st.subheader("Rutube")
        st.code("""
                # Comments from a single video
                python social/rutube_scraper.py --video-id video_id_1 --max-comments 100

                # Videos and comments from a playlist
                python social/rutube_scraper.py --playlist-id playlist_id_1 --max-videos 10 --max-comments 50

                # All configured playlists/videos with custom limits
                python social/rutube_scraper.py --max-comments 200 --max-videos 20

                # Via main entry
                python main.py social rutube --max-comments 200
                python main.py social rutube --video-id video_id_1 --max-comments 100
                        """)

    with st.expander("📚 Wiki Dump Parser"):
        st.code("""
                # Wikipedia dump with article limit
                python wiki/wiki_dump_parser.py "https://dumps.wikimedia.org/tgwiki/latest/tgwiki-latest-pages-articles.xml.bz2" tgwiki --max-articles 100

                # Tatar Wikipedia with transliteration
                python wiki/wiki_dump_parser.py "https://dumps.wikimedia.org/ttwiki/latest/ttwiki-latest-pages-articles.xml.bz2" ttwiki --max-articles 200 --transliterate

                # Wikibooks
                python wiki/wiki_dump_parser.py "https://dumps.wikimedia.org/ttwikibooks/latest/ttwikibooks-latest-pages-articles.xml.bz2" wikibooks --max-articles 200 --transliterate

                # Via main entry
                python main.py wiki "https://dumps.wikimedia.org/tgwiki/latest/tgwiki-latest-pages-articles.xml.bz2" tgwiki --max-articles 100
                        """)

    with st.expander("📄 Document Parser"):
        st.code("""
                # Process a folder, save to default output directory
                python main.py docs D:/FinSoft

                # Process with explicit output directory
                python main.py docs D:/FinSoft output/documents

                # Direct script usage
                python documents/universal_doc_parser.py D:/FinSoft -c documents/doc_config.yaml -o output/documents
                        """)

    with st.expander("📦 Corpus Builder"):
        st.code("""
                # Full corpus from all sources for a language
                python corpus/build_corpus.py --lang tg --sources all

                # News + Wiki only
                python corpus/build_corpus.py --lang tg --sources news,wiki --max-items-per-site 80

                # Social only (VK + Rutube)
                python corpus/build_corpus.py --lang tg --sources social --social-sources vk,rutube

                # Custom output file
                python corpus/build_corpus.py --lang tg --output corpus_tg.jsonl --sources all

                # Via main entry
                python main.py corpus --lang tg --sources all
                python main.py corpus --lang tg --sources news,social --social-sources vk
                        """)

    with st.expander("📦 Build VK Dataset"):
        st.code("""
                # Build unified VK dataset from all collected posts/comments (no filter)
                python main.py build-vk

                # Build dataset for a specific language (only posts/comments with required letters)
                python main.py build-vk --lang tg
                python main.py build-vk --lang os
                python main.py build-vk --lang udm
                python main.py build-vk --lang ba
                python main.py build-vk --lang tt

                # Direct script usage (also supports --lang)
                python social/scripts/build_vk_dataset.py --lang tg
                python social/scripts/build_vk_dataset.py --lang os
                        """)

    with st.expander("⚙️ Configuration System"):
        st.markdown("""
                    **Modular configuration structure**  
                    - `config/universal.yaml` – entry point, includes all other files  
                    - `config/global.yaml` – shared settings (requests, limits, strategies)  
                    - `config/languages.yaml` – language packages (noise words, stopwords, regex, locales)  
                    - `config/sites_*.yaml` – site profiles grouped by language  

                    **Social networks**  
                    - `social/config/social.yaml` – entry point for social media  
                    - `social/config/vk_*.yaml` – VK community lists by language  

                    **Documents and wiki**  
                    - `documents/doc_config.yaml` – document parser settings  
                    - `wiki/wiki_dump_parser.py` – accepts dump URL and parameters directly  

                    All components use a single loader `config/loader.py` that supports
                    modular loading with `includes` and deep dictionary merging.
                    """)
        
st.sidebar.markdown("---")
st.sidebar.info("Built for multilingual NLP · Version 2.0")