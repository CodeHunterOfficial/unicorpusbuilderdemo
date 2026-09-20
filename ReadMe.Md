# UniCorpusBuilder

**Universal multilingual scraper and corpus builder for low-resource languages.**

This is a **demonstration version** of the full UniCorpusBuilder project. It includes the core infrastructure for collecting text data from news websites, social networks (VK, Telegram, Rutube), Wikipedia dumps, and office documents for low-resource languages.

**Repository:** [https://github.com/CodeHunterOfficial/unicorpusbuilderdemo](https://github.com/CodeHunterOfficial/unicorpusbuilderdemo)

The full version will include additional features:
- Data annotation and labeling tools
- Advanced corpus management
- Comprehensive test suite (unit, integration, E2E)
- Quality assurance pipelines
- Model training utilities

---

## Supported Languages

| Code  | Language  | Sources                            |
|-------|-----------|------------------------------------|
| `tt`  | Tatar     | 14 news sites, VK, Telegram        |
| `tg`  | Tajik     | 14 news sites, VK                  |
| `os`  | Ossetian  | 6 news sites, VK                   |
| `ba`  | Bashkir   | 2 news sites, VK                   |
| `ru`  | Russian   | 2 news sites (ria.ru, mk.ru)       |
| `uz`  | Uzbek     | 1 news site                        |
| `udm` | Udmurt    | VK                                 |
| `en`  | English   | —                                  |

The article accompanying this repository focuses on four languages: Tatar, Tajik, Bashkir, and Ossetian.

---

## Project Structure

```
scraper_project/
├── main.py                       # Universal entry point
├── app.py                        # Streamlit dashboard
├── logger_setup.py               # Logging configuration
├── requirements.txt              # Python dependencies
│
├── config/                       # YAML configuration
│   ├── universal.yaml            # Entry point (includes)
│   ├── global.yaml               # Global settings
│   ├── languages.yaml            # Language packages
│   ├── loader.py                 # Config loader
│   ├── sites_tt.yaml             # Tatar sites
│   ├── sites_tg.yaml             # Tajik sites
│   ├── sites_ba.yaml             # Bashkir sites
│   ├── sites_ru.yaml             # Russian sites
│   ├── sites_os.yaml             # Ossetian sites
│   └── sites_uz.yaml             # Uzbek sites
│
├── pipeline/                     # Core pipeline
│   ├── pipeline_core.py          # Crawling and discovery
│   └── pipeline_extraction.py    # Content extraction
│
├── social/                       # Social network scrapers
│   ├── vk_scraper.py             # VK (synchronous)
│   ├── vk_scraper_async.py       # VK (asynchronous)
│   ├── telegram_scraper.py       # Telegram
│   ├── rutube_scraper.py         # Rutube
│   ├── config/                   # Social configs
│   │   ├── social.yaml           # Entry point
│   │   ├── vk_tt.yaml            # VK Tatar groups
│   │   ├── vk_tg.yaml            # VK Tajik groups
│   │   ├── vk_ba.yaml            # VK Bashkir groups
│   │   ├── vk_os.yaml            # VK Ossetian groups
│   │   └── vk_udm.yaml           # VK Udmurt groups
│   └── scripts/
│       └── build_vk_dataset.py   # VK data unification
│
├── wiki/                         # Wikipedia
│   ├── wiki_dump_parser.py       # Dump parser
│   └── tat_transliterator.py     # Tatar transliteration
│
├── documents/                    # Documents
│   ├── universal_doc_parser.py   # Universal document parser
│   └── doc_config.yaml           # Document configuration
│
├── corpus/                       # Corpus
│   └── build_corpus.py           # Corpus builder
│
├── tools/                        # Analysis and reproducibility scripts
│   ├── analyze_configs.py        # YAML configuration statistics
│   ├── list_sites.py             # Sources per language
│   ├── hf_count.py               # Documents, characters, tokens per dataset
│   ├── hf_quality.py             # Language purity and near-duplicates
│   ├── strategy_hitrate.py       # Strategy distribution and ROUGE-L
│   ├── benchmark_trafilatura.py  # Comparison with Trafilatura
│   ├── inspect_binary_labels.py  # Class distribution in binary datasets
│   ├── add_tokenizer.py          # Add mBERT tokenizer counts
│   ├── diagnose_khovar.py        # Discovery diagnostics (example)
│   └── README.md                 # Instructions for reproduction
│
└── output/                       # Results (gitignored)
    ├── news/                     # News articles
    ├── social/                   # Social media data
    ├── wiki/                     # Wikipedia dumps
    ├── documents/                # Document data
    └── corpus/                   # Unified corpora
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/CodeHunterOfficial/unicorpusbuilderdemo.git
cd unicorpusbuilderdemo
```

### 2. Create virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Playwright (for Rutube)

```bash
playwright install chromium
```

### 5. Install Tesseract (for OCR)

Download from [GitHub Tesseract](https://github.com/UB-Mannheim/tesseract/wiki) and install.

---

## Configuration

### Main configuration files

| File | Purpose |
|------|---------|
| `config/universal.yaml` | Entry point, includes list |
| `config/global.yaml` | Global settings (request, limits, strategies) |
| `config/languages.yaml` | Language packages (noise words, stopwords, regex) |
| `config/sites_*.yaml` | Site profiles by language |
| `social/config/social.yaml` | Social media configuration |

### Configuring VK token

Edit `social/config/social.yaml`:

```yaml
vk:
  token: "your_vk_token"
```

### Configuring Telegram API

```yaml
telegram:
  api_id: "your_api_id"
  api_hash: "your_api_hash"
```

### Configuring Rutube

```yaml
rutube:
  channels:
    - id: "channel_id_1"
  videos:
    - id: "video_id_1"
      max_comments: 500
  playlists:
    - id: "playlist_id_1"
      max_videos: 50
```

---

## Usage

### Check site configuration

```bash
python main.py config https://khovar.tj/
python main.py config https://khovar.tj/ config/universal.yaml
```

### News pipeline

```bash
# Full pipeline (discovery + extraction)
python main.py pipeline https://khovar.tj/

# With article limit
python main.py pipeline https://khovar.tj/ 50

# Direct script usage
python pipeline/pipeline_extraction.py config/universal.yaml https://khovar.tj/ 50
```

### Bulk pipeline

```bash
# All Tajik sites
python main.py bulk tg

# All Tatar sites with limit
python main.py bulk tt 100

# All sites
python main.py bulk all
python main.py bulk all 30
```

### Social network scrapers

**VK:**

```bash
python main.py social vk --lang tt --max-posts 100
python main.py social vk --domains club1135692 irta_tv --lang ba --max-posts 10
python main.py social vk --token your_token --domains allahtan --lang tt --max-posts 5
```

**Telegram:**

```bash
python main.py social telegram --lang tt --max-posts 100
python main.py social telegram --channels vatantat --lang tt --max-posts 50
```

**Rutube:**

```bash
python main.py social rutube --video-id video_id_1 --max-comments 100
python main.py social rutube --playlist-id playlist_id_1 --max-videos 10 --max-comments 50
```

**All social scrapers:**

```bash
python main.py social all --lang tt --max-posts 100
```

### Wikipedia dump parser

```bash
# Tajik Wikipedia
python main.py wiki "https://dumps.wikimedia.org/tgwiki/latest/tgwiki-latest-pages-articles.xml.bz2" tgwiki --max-articles 100

# Tatar Wikipedia with transliteration
python main.py wiki "https://dumps.wikimedia.org/ttwiki/latest/ttwiki-latest-pages-articles.xml.bz2" ttwiki --max-articles 200 --transliterate

# Bashkir Wikipedia
python main.py wiki "https://dumps.wikimedia.org/bawiki/latest/bawiki-latest-pages-articles.xml.bz2" bawiki --max-articles 500
```

### Document parser

```bash
# Process a folder
python main.py docs ./documents_to_parse

# With output directory
python main.py docs ./documents_to_parse output/documents

# Direct script usage
python documents/universal_doc_parser.py ./documents_to_parse -c documents/doc_config.yaml -o output/documents
```

### Corpus builder

```bash
# Full corpus from all sources
python main.py corpus --lang tg --sources all

# News + Wiki only
python main.py corpus --lang tg --sources news,wiki --max-items-per-site 200

# Social networks only
python main.py corpus --lang tg --sources social --social-sources vk,rutube

# Custom output file
python main.py corpus --lang tg --sources all --output corpus_tg.jsonl
```

### Build unified VK dataset

```bash
# No filter
python main.py build-vk

# Filter by language
python main.py build-vk --lang tg
python main.py build-vk --lang os
python main.py build-vk --lang udm
```

### Streamlit dashboard

```bash
streamlit run app.py
```

---

## Analysis scripts (`tools/`)

The `tools/` directory contains scripts for reproducing the numerical results
presented in the accompanying article. See [`tools/README.md`](tools/README.md)
for details.

Quick start:

```bash
python tools/analyze_configs.py
python tools/list_sites.py
python tools/hf_count.py
python tools/hf_quality.py --sample 2000 --near-dups
python tools/strategy_hitrate.py
python tools/benchmark_trafilatura.py
python tools/inspect_binary_labels.py
```

| Script | Purpose | Output |
|--------|---------|--------|
| `analyze_configs.py` | YAML configuration statistics | stdout |
| `list_sites.py` | Sources grouped by language | stdout |
| `hf_count.py` | Documents, characters, tokens per dataset | `hf_count*.csv` |
| `hf_quality.py` | Language purity and near-duplicates | `hf_quality*.csv` |
| `strategy_hitrate.py` | First-matching strategy distribution and ROUGE-L | `strategy_*.csv` |
| `benchmark_trafilatura.py` | Direct comparison with Trafilatura | `trafilatura_*.csv` |
| `inspect_binary_labels.py` | Class distribution in binary datasets | stdout |
| `add_tokenizer.py` | Add mBERT tokenizer counts | `hf_count_v2.csv` |
| `diagnose_khovar.py` | Discovery diagnostics for khovar.tj | stdout |

---

## Output structure

```
output/
├── khovar_2026-08-29_articles.jsonl    # News
├── social/
│   ├── posts/
│   │   └── vk_allahtan_posts.jsonl
│   ├── comments/
│   │   └── vk_allahtan_comments.jsonl
│   └── vk_unified.jsonl
├── wiki/
│   ├── tgwiki/
│   │   └── tgwiki-latest-pages-articles_all.jsonl
│   └── ttwiki/
│       └── ttwiki-latest-pages-articles_all.jsonl
├── documents/
│   └── documents_20260829_120000.jsonl
└── corpus/
    └── corpus_tg_20260829_120000.jsonl
```

---

## Key features

1. **Site engine auto-detection** — np_widget, WordPress, RFE/RL, Nuxt, Drupal
2. **Multi-strategy extraction** — meta tags, CSS selectors, JSON-LD, regex
3. **Language filtering** — noise words, stopwords, localized months
4. **Transliteration** — Tatar Latin → Cyrillic
5. **OCR** — text recognition in images
6. **Archive support** — ZIP, RAR, 7z, TAR
7. **API sources** — JSON API support (president.tj)
8. **Reproducible analysis** — dedicated `tools/` scripts for every reported metric

---

## Important notes

### Sensitive data

This demo version contains **placeholder values** for:

- VK tokens
- Telegram API credentials
- Rutube video/playlist IDs

**Do not commit real credentials to the repository.**

### Configuration files

The configuration files contain:

- Site names and URLs
- VK group/public names in their original languages
- Selector patterns for web scraping

These are **demonstration examples** and may need adjustment for production use.

---

## Data availability

The corpora collected using this system are published on Hugging Face:

- [TatarNLPWorld](https://huggingface.co/TatarNLPWorld)
- [TajikNLPWorld](https://huggingface.co/TajikNLPWorld)
- [BashkirNLPWorld](https://huggingface.co/BashkirNLPWorld)
- [OssetianNLPWorld](https://huggingface.co/OssetianNLPWorld)

All collections follow the practice adopted in corpus linguistics
(HPLT, OSCAR projects). Rights to the texts belong to their authors and
publishers; licenses (MIT, CC BY-SA 4.0, CC BY 4.0) apply only to the data
structure, metadata, and annotations created by the authors. A
notice-and-takedown procedure is in effect.

---

## Citation

If you use this code or the collected corpora in your research, please cite:

```
Arabov M.K. Universal scalable architecture for text collection and annotation
for low-resource languages: a multi-strategy approach to metadata extraction.
Information Processes, 2026.
```

---

## License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

### Third-party licenses

| Library | License |
|---------|---------|
| Python | PSF License |
| Streamlit | Apache 2.0 |
| Requests | Apache 2.0 |
| BeautifulSoup4 | MIT |
| Playwright | Apache 2.0 |
| Telethon | MIT |
| aiohttp | Apache 2.0 |
| tqdm | MIT |
| PyYAML | MIT |
| pandas | BSD 3-Clause |
| PyPDF2 | BSD 3-Clause |
| python-docx | MIT |
| openpyxl | MIT |
| python-pptx | MIT |
| pytesseract | Apache 2.0 |
| Pillow | HPND |
| ebooklib | AGPL-3.0 |
| rarfile | ISC |
| py7zr | LGPL-2.1 |
| dateutil | Apache 2.0 |

---

## Contributing

This is a demo version. The full project will be released separately.

For contributions:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

---

## Contact

- **GitHub:** [CodeHunterOfficial](https://github.com/CodeHunterOfficial)
- **Repository:** [https://github.com/CodeHunterOfficial/unicorpusbuilderdemo](https://github.com/CodeHunterOfficial/unicorpusbuilderdemo)
---

## 🔮 Future Plans (Full Version)

- [ ] Data annotation and labeling tools
- [ ] Advanced corpus management
- [ ] Unit, integration, and E2E tests
- [ ] Quality assurance pipelines
- [ ] Model training utilities
- [ ] REST API for corpus access
- [ ] Distributed scraping
- [ ] Data versioning

---

**Version:** 1.0-demo  
**Last Updated:** 2026-05-01