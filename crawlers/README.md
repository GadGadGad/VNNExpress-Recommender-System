# VnExpress Crawler Pipeline

This project contains a 3-step pipeline for scraping articles, comments, and user profiles from VnExpress. The pipeline is orchestrated by `run_pipeline.py`, which runs all steps in order.

## Requirements

You must install the required Python packages:

```bash
pip install requests beautifulsoup4 toml rich selenium playwright
```

You also need to install the browsers for Selenium and Playwright:

```bash
# For Selenium (deep_crawler.py)
# This will install the correct chromedriver
pip install webdriver-manager 

# For Playwright (user_profile_crawler.py)
playwright install
```

## How to Run the Pipeline

Use the main `run_pipeline.py` script to run all 3 steps.

### Basic Example (Single Worker)

This will crawl 3 pages from the 'the-gioi' category and save all data to the `data/` directory. Step 2 will run using one worker.

```bash
python run_pipeline.py --categories the-gioi --pages 3
```

### Parallel Example (Multiple Workers)

This is the recommended way to run the pipeline. It crawls two categories and uses **4 parallel workers** for the slow Step 2.

```bash
python run_pipeline.py --categories the-gioi kinh-doanh --pages 5 --workers 4 --headless
```

### `run_pipeline.py` Arguments

* `-c, --categories`: (Required) One or more categories to scrape.
* `-p, --pages`: (Optional) How many pages to scan *per category*. Default is 2.
* `-o, --output`: (Optional) The main directory to save all files. Default is `data`.
* `-w, --workers`: (Optional) Number of parallel workers for **Step 2**. Default is 1.
* `--headless`: (Optional) Runs browsers in headless mode (no visible window).
* `--no-cache`: (Optional) Disables caching for all steps.
* `--keep-parts`: (Optional) For debugging. Prevents deletion of temporary files from parallel workers.

## The 3-Step Workflow

`run_pipeline.py` automates this process:

### Step 1: Discover Articles (`main_crawler.py`)

* **What it does:** Uses `requests` and `threading` to quickly scan category pages and find new articles.
* **Output:** `data/articles.csv` — a "to-do list" of article URLs and metadata.

### Step 2: Process Comments (`deep_crawler.py`)

* **What it does:** Uses `Selenium` to read `data/articles.csv`, open each article, click all "load more" and "view replies" buttons, and save the complete comment data.
* **Speed:** This script is single-threaded. The `run_pipeline.py` script speeds it up by splitting `articles.csv` and running multiple copies in parallel using the `--workers` flag.
* **Output:** `data/replies.csv` — a complete dataset of parent-reply comment pairs.

### Step 3: Enrich Users (`user_profile_crawler.py`)

* **What it does:** Uses `Playwright` and `threading` to read `data/replies.csv`, find all unique `user_id`s, and scrape their personal profile pages for their entire comment history.
* **Output:**
    * `data/user_profiles.csv`
    * `data/user_comments.csv`

## Caching and Resumability

* **Caching:** All scripts create a `.cache` directory. This saves network requests. To force a re-scrape, delete this directory or use the `--no-cache` flag.
* **Resumability:** Each script also creates a `.seen*.txt` file. This prevents the script from re-processing an article or user it has *already saved*. To re-process everything, delete these `.txt` files.