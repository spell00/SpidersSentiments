# SpidersSentiments

The project contains data pipelines and tooling to monitor online
conversations about spiders.  It now includes **Spider Guardian**, a
Retrieval Augmented Generation (RAG) bot that advocates for spiders on
X (formerly Twitter) and learns from community feedback.

## Project structure

* `prepare_dataset.py` – scrapes and preprocesses spider-related news
  articles and exports a HuggingFace dataset.
* `spider_guardian_bot.py` – orchestrates the Spider Guardian bot
  (vector indexing, chatbot integrations, X monitoring, feedback
  learning).

## Installation

Create a virtual environment and install the dependencies:

```bash
python -m venv myenv313
source ~/venvs/myenv313/bin/activate
pip install -r requirements.txt
```

Download the required NLTK resources if you have not already:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

## Building the dataset

Use `prepare_dataset.py` to build the document set consumed by the bot.

```bash
python prepare_dataset.py --chemin_csv data/Data_spider_news_global.csv --column Sensationalism --use_preprocess 1
```

This generates a HuggingFace dataset in `data/dataset_Sensationalism.hf`.

## Running the Spider Guardian bot

1. Set the required API keys as environment variables (only providers
   with available keys will be used):

   * `OPENAI_API_KEY`
   * `ANTHROPIC_API_KEY`
   * `COHERE_API_KEY`

   To engage on X you also need:

   * `TWITTER_BEARER_TOKEN`
   * `TWITTER_API_KEY`
   * `TWITTER_API_SECRET`
   * `TWITTER_ACCESS_TOKEN`
   * `TWITTER_ACCESS_SECRET`

2. Build the vector index from the dataset (this only needs to be done
   when the dataset changes):

   ```bash
   python spider_guardian_bot.py --build-index
   ```

3. Reply to a limited number of posts about spiders:

   ```bash
   python spider_guardian_bot.py --respond 3
   ```

   To mix in saved human tweets (for a more natural tone) tweak the context knobs:

   ```powershell
   python spider_guardian_bot.py --respond 3 --human-posts-top-k 3 --human-posts-path data/streamed_posts.json
   ```

   Set `--human-posts-top-k 0` to disable reusing tweet snippets.

   You can also tune the voice and length:

   ```powershell
   python spider_guardian_bot.py --respond 3 --reply-min-words 14 --reply-max-words 28 --human-style-examples 3
   ```

   Higher `--human-style-examples` values feed more real tweets into the prompt so replies feel closer to human chatter.

4. Collect replies to the bot's comments and update the adaptive
   feedback model:

   ```bash
   python spider_guardian_bot.py --collect-feedback
   ```

All interactions are logged in `data/interactions.jsonl` so the bot can
resume learning between runs.

## Dataset updates and LangSmith integration

Use `update_datasets.py` to upload and refresh datasets in LangSmith from your local SQLite databases. It supports trending posts and the normalized SQL tables (interactions, content, flagged). Most functionality is auto-enabled unless you explicitly opt out with `--no-*` flags, keeping the common path simple.

Key flags (default-on where noted, negative flags disable):

- Trending dataset (auto if DB present; disable with `--no-upload` / `--no-update`)
   - `--upload` / `--no-upload`: Upload trending dataset from `--db`
   - `--update` / `--no-update`: Update an existing LangSmith dataset from `--db` (needs `--dataset` & `--match-key`)
   - `--db <path>`: Path to trending SQLite (e.g., `data/spider_trending.sqlite`)
   - `--dataset <name>`: LangSmith dataset name (e.g., `spider-trending-dataset`)
   - `--show-browser`: Show Firefox to resolve final URLs (useful if redirects)
   - `--url-wait-seconds <int>`: Wait time for URL resolution

- SQL (normalized) datasets (uploads & refresh auto; disable with `--no-upload-all-sql` / `--no-refresh-sql`)
  - `--sql-db <path>`: Path to normalized DB (default `data/spider_guardian.sqlite`)
  - `--upload-all-sql` / `--no-upload-all-sql`: Upload interactions, streamed, flagged
  - Fine-grained disables: `--no-upload-sql-interactions`, `--no-upload-sql-streamed`, `--no-upload-sql-flagged`
  - `--refresh-sql` / `--no-refresh-sql`: Refresh metrics
  - Live metrics default on: disable with `--no-scrape-live-metrics`
  - Force cadence bypass: opt-in with `--force-refresh-sql` (not default)
   - `--verify-replies`: After refresh, verify reply visibility; with `--delete-missing-replies` will remove missing replies
   - `--max-examples <int>`: Limit rows/examples processed (use `-1` for no limit)
   - Progress output: `--progress` (default on) or `--hide-progress`

- Refresh interactions directly in the DB (Selenium) (off unless requested)
   - `--update-sql-interactions-db`: Run Selenium to update metrics in the `interactions` table
   - `--max-age-days <int>`: Only refresh rows newer than this many days
   - `--limit <int>`: Cap number of DB rows refreshed
   - `--show-browser`: Run Selenium with visible browser
   - `--force-update-sql-interactions-db`: Ignore `--max-age-days` (includes all rows unless limited by `--limit` or `--max-examples`)

- Operational convenience
   - `--full-update`: Safe sequence that enables uploads, SQL refresh, DB refresh (recent), and reply verification
   - `--loop --interval <s> --jitter <0..1>`: Repeat cycles on a schedule
   - `--debug-plan`: Print the actions that would run and exit (no side effects)
   - `--plan-refresh`: Preview which interaction examples are eligible for refresh (no writes)

Notes:

- `--full-update` auto-enables: trending upload/update (if applicable), all SQL uploads, forced live refresh, reply verification, and DB interactions refresh.
- Verification is scoped to only the examples refreshed in the same run when both refresh and verify are used together.

### Examples

Typical minimal run (auto defaults):

```powershell
python update_datasets.py --db data/spider_trending.sqlite --dataset spider-trending-dataset --sql-db data\spider_guardian.sqlite --max-examples 5 --progress
```

Disable force & live scraping (use DB-only, cadence-restricted):

```powershell
python update_datasets.py --db data/spider_trending.sqlite --dataset spider-trending-dataset --sql-db data\spider_guardian.sqlite --max-examples 10 --no-force-refresh-sql --no-scrape-live-metrics
```

Continuous full update (looped) with smart defaults:

```powershell
python update_datasets.py --full-update --loop --interval 3600 --jitter 0.15 --db data/spider_trending.sqlite --dataset spider-trending-dataset --sql-db data\spider_guardian.sqlite --max-examples 2 --progress
```
