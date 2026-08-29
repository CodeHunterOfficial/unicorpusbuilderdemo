# 📖 UniCorpusBuilder — README (Demo Version)

**Universal Multilingual Scraper & Corpus Builder — Demo**

This is a **demonstration version** of the full UniCorpusBuilder project. It includes the core scraping infrastructure for collecting text data from news websites, social networks (VK, Telegram, Rutube), Wikipedia dumps, and documents for low-resource languages.

The full project will include additional features such as:
- Data annotation and labeling tools
- Advanced corpus creation and management
- Comprehensive testing suite (unit, integration, E2E)
- Quality assurance pipelines
- Model training utilities

---

## 🌍 Supported Languages

| Code | Language | Sources |
|------|----------|---------|
| `tg` | Tajik | 10+ news sites, VK |
| `tt` | Tatar | 13+ news sites, VK, Telegram |
| `ba` | Bashkir | 2 news sites, VK |
| `ru` | Russian | 2 news sites (ria.ru, mk.ru) |
| `os` | Ossetian | 6 news sites, VK |
| `uz` | Uzbek | 1 news site |
| `udm` | Udmurt | VK |
| `en` | English | — |

---

## 📁 Project Structure

```
scraper_project/
├── main.py                    # Main entry point
├── app.py                     # Streamlit Dashboard
├── logger_setup.py            # Logging configuration
│
├── config/                    # Configuration
│   ├── universal.yaml         # Entry point (includes)
│   ├── global.yaml            # Global settings
│   ├── languages.yaml         # Language packages
│   ├── loader.py              # Config loader
│   ├── sites_tt.yaml          # Tatar sites
│   ├── sites_tg.yaml          # Tajik sites
│   ├── sites_ba.yaml          # Bashkir sites
│   ├── sites_ru.yaml          # Russian sites
│   ├── sites_os.yaml          # Ossetian sites
│   └── sites_uz.yaml          # Uzbek sites
│
├── pipeline/                  # Core pipeline
│   ├── pipeline_core.py       # Crawling and discovery
│   └── pipeline_extraction.py # Content extraction
│
├── social/                    # Social networks
│   ├── vk_scraper.py          # VK synchronous
│   ├── vk_scraper_async.py    # VK asynchronous
│   ├── telegram_scraper.py    # Telegram
│   ├── rutube_scraper.py      # Rutube
│   ├── config/                # Social configs
│   │   ├── social.yaml        # Entry point
│   │   ├── vk_tt.yaml         # VK Tatar groups
│   │   ├── vk_tg.yaml         # VK Tajik groups
│   │   ├── vk_ba.yaml         # VK Bashkir groups
│   │   ├── vk_os.yaml         # VK Ossetian groups
│   │   └── vk_udm.yaml        # VK Udmurt groups
│   └── scripts/
│       └── build_vk_dataset.py # VK data unification
│
├── wiki/                      # Wikipedia
│   ├── wiki_dump_parser.py    # Dump parser
│   └── tat_transliterator.py  # Tatar transliteration
│
├── documents/                 # Documents
│   ├── universal_doc_parser.py # Universal document parser
│   └── doc_config.yaml         # Document configuration
│
├── corpus/                    # Corpus
│   └── build_corpus.py        # Corpus builder
│
└── output/                    # Results
    ├── news/                  # News articles
    ├── social/                # Social media data
    ├── wiki/                  # Wikipedia dumps
    ├── documents/             # Document data
    └── corpus/                # Unified corpora
```

---

## 🔧 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/CodeHunterOfficial/UniCorpusBuilder.git
cd UniCorpusBuilder
```

### 2. Create Virtual Environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

### 3. Install Dependencies

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

## ⚙️ Configuration

### Main Configuration Files

| File | Purpose |
|------|---------|
| `config/universal.yaml` | Entry point, includes list |
| `config/global.yaml` | Global settings (request, limits, strategies) |
| `config/languages.yaml` | Language packages (noise words, stopwords, regex) |
| `config/sites_*.yaml` | Site profiles by language |
| `social/config/social.yaml` | Social media configuration |

### Configuring VK Token

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

## 📋 Commands

### 1. Check Site Configuration

```bash
# Check configuration for a site
python main.py config https://khovar.tj/

# With a different config file
python main.py config https://khovar.tj/ config/universal.yaml
```

---

### 2. News Pipeline

```bash
# Full pipeline (discovery + extraction)
python main.py pipeline https://khovar.tj/

# With article limit
python main.py pipeline https://khovar.tj/ 50

# Direct script usage
python pipeline/pipeline_extraction.py config/universal.yaml https://khovar.tj/ 50
```

---

### 3. Bulk Pipeline

```bash
# All Tajik sites
python main.py bulk tg

# All Tatar sites with limit
python main.py bulk tt 100

# All sites
python main.py bulk all

# All sites with limit
python main.py bulk all 30
```

---

### 4. Social Networks

#### VK

```bash
# All Tatar groups from config
python main.py social vk --lang tt --max-posts 100

# Specific groups
python main.py social vk --domains club1135692 irta_tv --lang ba --max-posts 10

# With token
python main.py social vk --token your_token --domains allahtan --lang tt --max-posts 5
```

#### Telegram

```bash
# All Tatar channels
python main.py social telegram --lang tt --max-posts 100

# Specific channel
python main.py social telegram --channels vatantat --lang tt --max-posts 50
```

#### Rutube

```bash
# Comments from a video
python main.py social rutube --video-id video_id_1 --max-comments 100

# Playlist
python main.py social rutube --playlist-id playlist_id_1 --max-videos 10 --max-comments 50
```

#### All Social Networks

```bash
python main.py social all --lang tt --max-posts 100
```

---

### 5. Wikipedia

```bash
# Tajik Wikipedia
python main.py wiki "https://dumps.wikimedia.org/tgwiki/latest/tgwiki-latest-pages-articles.xml.bz2" tgwiki --max-articles 100

# Tatar Wikipedia with transliteration
python main.py wiki "https://dumps.wikimedia.org/ttwiki/latest/ttwiki-latest-pages-articles.xml.bz2" ttwiki --max-articles 200 --transliterate

# Bashkir Wikipedia
python main.py wiki "https://dumps.wikimedia.org/bawiki/latest/bawiki-latest-pages-articles.xml.bz2" bawiki --max-articles 500
```

---

### 6. Documents

```bash
# Process a folder
python main.py docs D:/FinSoft

# With output directory
python main.py docs D:/FinSoft output/documents

# Direct script usage
python documents/universal_doc_parser.py D:/FinSoft -c documents/doc_config.yaml -o output/documents
```

---

### 7. Corpus Building

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

---

### 8. Build Unified VK Dataset

```bash
# No filter
python main.py build-vk

# Tajik only
python main.py build-vk --lang tg

# Ossetian only
python main.py build-vk --lang os

# Udmurt only
python main.py build-vk --lang udm
```

---

### 9. Streamlit Dashboard

```bash
streamlit run app.py
```

---

## 📊 Output Structure

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

## 🧩 Key Features

1. **Auto-detection of site families** — np_widget, WordPress, RFE/RL, Nuxt, Drupal
2. **Multiple extraction strategies** — meta tags, CSS selectors, JSON-LD, regex
3. **Language filtering** — noise words, stopwords, localized months
4. **Transliteration** — Tatar Latin → Cyrillic
5. **OCR** — text recognition in images
6. **Archive support** — ZIP, RAR, 7z, TAR
7. **API sources** — JSON API support (president.tj)

---

## ⚠️ Important Notes

### Sensitive Data

This demo version contains **placeholder values** for:
- VK tokens
- Telegram API credentials
- Rutube video/playlist IDs

**Do not commit real credentials to the repository.**

### Configuration Files

The configuration files contain:
- Site names and URLs
- VK group/public names in their original languages
- Selector patterns for web scraping

These are **demonstration examples** and may need adjustment for production use.

---

## 📄 License

This project is licensed under the **MIT License**.

### MIT License

```
MIT License

Copyright (c) 2026 CodeHunterOfficial

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### Third-Party Licenses

This project uses the following third-party libraries:

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

## 🤝 Contributing

This is a demo version. The full project will be released separately.

For contributions:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

---

## 📧 Contact

- **GitHub:** [CodeHunterOfficial](https://github.com/CodeHunterOfficial)
- **Repository:** [UniCorpusBuilder](https://github.com/CodeHunterOfficial/UniCorpusBuilder)

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
**Last Updated:** 2026-01-29