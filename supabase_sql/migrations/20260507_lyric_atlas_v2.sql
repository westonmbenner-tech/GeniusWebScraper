create extension if not exists "pgcrypto";

create table if not exists public.artists (
  id uuid primary key default gen_random_uuid(),
  name text not null,
  slug text not null unique,
  image_url text,
  bio_summary text,
  created_at timestamptz default now(),
  updated_at timestamptz default now()
);

create table if not exists public.songs (
  id uuid primary key default gen_random_uuid(),
  artist_id uuid references public.artists(id) on delete cascade,
  title text not null,
  slug text not null,
  album text,
  release_year int,
  track_number int,
  source_url text,
  r2_lyrics_key text not null,
  word_count int,
  unique_word_count int,
  lexical_diversity numeric,
  created_at timestamptz default now(),
  updated_at timestamptz default now(),
  unique(artist_id, slug)
);

create table if not exists public.analysis_runs (
  id uuid primary key default gen_random_uuid(),
  artist_id uuid references public.artists(id) on delete cascade,
  run_id text not null unique,
  status text not null default 'pending',
  model_version text,
  algorithm_version text,
  song_count int,
  total_words int,
  r2_raw_corpus_key text,
  r2_full_analysis_key text,
  r2_per_song_analysis_key text,
  r2_debug_key text,
  error_message text,
  created_at timestamptz default now(),
  completed_at timestamptz
);

create table if not exists public.word_stats (
  id uuid primary key default gen_random_uuid(),
  artist_id uuid references public.artists(id) on delete cascade,
  analysis_run_id uuid references public.analysis_runs(id) on delete cascade,
  word text not null,
  count int not null,
  rank int not null,
  frequency_per_1000 numeric,
  part_of_speech text,
  created_at timestamptz default now(),
  unique(analysis_run_id, word)
);

create table if not exists public.phrase_stats (
  id uuid primary key default gen_random_uuid(),
  artist_id uuid references public.artists(id) on delete cascade,
  analysis_run_id uuid references public.analysis_runs(id) on delete cascade,
  phrase text not null,
  count int not null,
  rank int not null,
  phrase_length int,
  created_at timestamptz default now(),
  unique(analysis_run_id, phrase)
);

create table if not exists public.theme_stats (
  id uuid primary key default gen_random_uuid(),
  artist_id uuid references public.artists(id) on delete cascade,
  analysis_run_id uuid references public.analysis_runs(id) on delete cascade,
  theme text not null,
  score numeric,
  evidence_count int,
  rank int,
  created_at timestamptz default now(),
  unique(analysis_run_id, theme)
);

create table if not exists public.artist_profiles (
  id uuid primary key default gen_random_uuid(),
  artist_id uuid references public.artists(id) on delete cascade,
  analysis_run_id uuid references public.analysis_runs(id) on delete cascade,
  summary text,
  style_notes text,
  common_themes text[],
  signature_words text[],
  signature_phrases text[],
  emotional_palette text[],
  created_at timestamptz default now(),
  unique(analysis_run_id)
);

create table if not exists public.song_analysis_summaries (
  id uuid primary key default gen_random_uuid(),
  song_id uuid references public.songs(id) on delete cascade,
  analysis_run_id uuid references public.analysis_runs(id) on delete cascade,
  short_summary text,
  dominant_themes text[],
  mood text,
  notable_words text[],
  notable_phrases text[],
  r2_detailed_analysis_key text,
  created_at timestamptz default now(),
  unique(song_id, analysis_run_id)
);

create table if not exists public.lyric_analysis_archives (
  id uuid primary key default gen_random_uuid(),
  analysis_run_id text not null unique,
  genius_artist_id text not null,
  artist_name text not null,
  songs_analyzed int,
  total_meaningful_tokens int,
  unique_word_count int,
  lexical_diversity numeric,
  filtering_strictness text,
  dedupe_mode text,
  categories_exist boolean default false,
  r2_corpus_key text,
  settings_json jsonb,
  top_words_json jsonb,
  bigrams_json jsonb,
  categories_json jsonb,
  summary_json jsonb,
  created_at timestamptz default now()
);

grant usage on schema public to service_role;

grant select, insert, update, delete
on all tables in schema public
to service_role;

alter default privileges in schema public
grant select, insert, update, delete on tables to service_role;