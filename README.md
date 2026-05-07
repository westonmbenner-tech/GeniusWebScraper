# Lyric Atlas (Streamlit + Python backend)

Analyze the words an artist returns to most using a reliable local lyric-analysis workflow.

## Features

- Three input modes:
  - CSV upload mode (recommended default)
  - Official Genius metadata mode (`https://api.genius.com`)
  - Paste lyrics mode (manual song entry)
- Remote lyric corpus storage in Cloudflare R2 (no on-disk artist JSON cache in the deployed app)
- Title normalization and configurable deduplication modes
- Lyric cleaning with stopword removal and lemmatization
- Metrics: vocabulary size, lexical diversity, top words, top bigrams
- Optional OpenAI semantic categorization of top words only (no lyric text sent)
- Streamlit visualizations and CSV/JSON export
- **Lyric Atlas v2 scaffold** for Supabase + Cloudflare R2 storage split

## Privacy and usage boundaries

- Lyrics are copyrighted. This app is designed for local analysis workflows.
- The app does **not** display or republish full lyrics.
- UI output is aggregate analysis only (counts, charts, metadata).
- OpenAI categorization sends only top words and counts, never full lyrics.

## Official API vs unofficial scraping

- The app's official Genius integration uses only the documented API base: `https://api.genius.com`.
- Official Genius API mode is metadata-focused (song ids, titles, artists, URLs, dates) and may not include full lyrics text.
- Unofficial web scraping endpoints can be blocked by Cloudflare and are intentionally not used for core discovery in this app.
- For reliable lyric token analysis, use **CSV upload mode** or **Paste lyrics mode**.

## Setup

1. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   # Windows PowerShell
   .venv\Scripts\Activate.ps1
   # macOS/Linux
   source .venv/bin/activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Create `.env.local` from the example:

   ```bash
   copy .env.example .env.local
   ```

4. Add API keys in `.env.local`:

   - `GENIUS_ACCESS_TOKEN`: create from [Genius API Clients](https://genius.com/api-clients)
   - `OPENAI_API_KEY`: optional; only needed for semantic categorization
   - `R2_ACCOUNT_ID`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_BUCKET_NAME`, `R2_ENDPOINT`
   - `NEXT_PUBLIC_SUPABASE_URL`, `NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY`, `SUPABASE_PRIVATE_KEY`

5. Run the app:

   ```bash
   streamlit run app.py
   ```

## Project structure

```text
ArtistWordprint/
  app.py
  requirements.txt
  .env.example
  src/
    __init__.py
    analysis.py
    atlas_backend.py
    atlas_env.py
    atlas_types.py
    cache.py  # slugify helper only
    categorize.py
    dedupe.py
    genius_client.py
    lyrics_cleaner.py
    r2_client.py
    r2_json.py
    r2_keys.py
    supabase_client.py
    visualizations.py
  supabase/migrations/
    20260507_lyric_atlas_v2.sql
```

## Notes

- If Genius blocks or fails, the app surfaces clear warnings and does not crash.
- CSV upload is the most reliable mode for local end-to-end lyric analysis.
- NLTK resources are downloaded on first run as needed.

## Lyric Atlas v2 architecture (scaffold)

- **Supabase is the query layer**: artist/song metadata, stats, ranks, summaries, analysis run metadata, and R2 keys.
- **R2 is the artifact layer**: raw lyric corpora, per-song lyrics blobs, full analysis JSON, per-song detailed JSON, debug blobs, exports.
- Full lyrics are **not stored in Supabase** in this v2 architecture.
- Use `src/atlas_backend.py` for server-side scaffolding methods equivalent to:
  - artist import
  - analysis save
  - profile bundle read
  - song lyrics fetch
  - full analysis artifact fetch

## Cache-first workflow

- User searches artist + requested song count.
- Official Genius metadata discovery runs first.
- Supabase archive is checked (`src/db.py`) for prior analysis results.
- If archive is loaded, charts render immediately without scraping.
- If fresh run is selected, R2 corpus is checked (`src/r2_store.py`).
- If R2 already has enough lyrics, analysis runs directly from that stored corpus.
- If lyrics are missing, only missing songs are fetched with `lyricsgenius`, then corpus is re-uploaded to R2.
- Analysis summaries/results are persisted to Supabase; heavy lyric/full-analysis artifacts live in R2.
- The UI does not display full lyric corpora publicly.

## R2 security requirements

- Keep R2 credentials server-side only.
- Never expose R2 secrets as `NEXT_PUBLIC_*`.
- Use an R2 token scoped to **Object Read & Write** on the `lyric-atlas` bucket.
- Do **not** use Cloudflare R2 Admin Read & Write for app runtime.
