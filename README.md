# Lyric Atlas (Streamlit + Python backend)
Go to [lyricatlas.com](lyricatlas.com)
Analyzes the words an artist returns to most using a reliable local lyric-analysis workflow.

UPDATE: Genius has started aggressively removing auto-scrapers (including BeautifulSoup, which this app uses). As such, live updating may be blocked. So, test app capabilities through the Archived Artists section.

To regain functionality, clone this repo and create your own Genius API key.

## Features

- Three input modes:
  - CSV upload mode (recommended default)
  - Official Genius metadata mode (`https://api.genius.com`)
  - Paste lyrics mode (manual song entry)
- Remote lyric corpus storage in Cloudflare R2 (no on-disk artist JSON cache in the deployed app)
- Title normalization and configurable deduplication modes
- Lyric cleaning with stopword removal and lemmatization
- Metrics: vocabulary size, lexical diversity, top words, top bigrams
- Compare multiple artists and their most frequently used words
- Optional OpenAI semantic categorization of top words only (no lyric text sent)
- Streamlit visualizations and CSV/JSON export
- **Lyric Atlas v2 scaffold** for Supabase + Cloudflare R2 storage split
