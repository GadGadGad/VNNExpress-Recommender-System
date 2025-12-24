"""
vnexpress_csv_crawler.py (STEP 1: DISCOVERER)
- Crawls VNExpress categories to discover articles.
- Can crawl by latest pages OR by specific date ranges.
- Uses Threading for concurrent fetching.
- INPUTS: A list of categories (names or IDs) and optional date range.
- OUTPUTS: articles.csv (the "to-do list" for deep_crawler.py)
"""

import argparse
import csv
import json
import logging
import random
import time
import re
import toml
import sys
from pathlib import Path
import requests
from bs4 import BeautifulSoup
import concurrent.futures
from typing import List, Optional, Tuple
from contextlib import nullcontext
from tqdm import tqdm

from datetime import datetime
try:
    from dateutil.relativedelta import relativedelta
except ImportError:
    print("LỖI: Cần cài 'python-dateutil'.")
    print("Hãy chạy: python -m pip install python-dateutil")
    sys.exit(1)


import rich.logging
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

from utils import Cache, sha1, normalize_url, resolve_category_id

console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[rich.logging.RichHandler(console=console, rich_tracebacks=True, show_path=False, markup=True)]
)
log = logging.getLogger(__name__)

try:
    config = toml.load("config.toml")
    crawler_cfg = config.get("crawler", {})
    files_cfg = config.get("files", {})
    log.info("Loaded settings from [bold cyan]config.toml[/bold cyan]")
except Exception:
    log.warning("config.toml not found. Using default values.")
    crawler_cfg = {}
    files_cfg = {}

BASE = "https://vnexpress.net/"
MIN_SLEEP = crawler_cfg.get("min_sleep", 0.8)
MAX_SLEEP = crawler_cfg.get("max_sleep", 1.6)
RETRY_COUNT = crawler_cfg.get("retry_count", 3)
HEADERS = {"User-Agent": crawler_cfg.get("user_agent", "Mozilla/5.0")}
DEFAULT_OUTPUT = files_cfg.get("default_output_dir", "data")
MAX_WORKERS = crawler_cfg.get("max_workers", 10)


class VnExpressCrawler:
    def __init__(self, output_dir: Path, use_cache=True):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()

        # Use imported Cache
        self.cache = Cache(output_dir / ".cache", enabled=use_cache)

        self.start_time = time.time()
        self.stats = {"cache_hits": 0, "articles_saved": 0}

        self.article_csv = output_dir / "articles.csv"

        self.seen_file = self.output_dir / ".seen_articles.txt"
        self.seen_articles = set()

        self._load_seen()
        self._init_csvs()

    def _load_seen(self):
        """Loads previously scraped URLs to enable resuming."""
        if self.seen_file.exists():
            log.info("Loading previously seen URLs for resumability...")
            with open(self.seen_file, "r", encoding="utf-8") as f:
                self.seen_articles = {line.strip() for line in f}
            log.info(f"Loaded {len(self.seen_articles)} seen URLs. Will skip these.")

    def _init_csvs(self):
        """Initializes CSV files with headers if they don't exist."""
        def init(path, header):
            if not path.exists():
                csv.writer(open(path, "w", newline="", encoding="utf-8")).writerow(header)
                log.info(f"Created new file: [bold]{path.name}[/bold]")

        init(self.article_csv, ["article_id", "url", "title", "short_description", "author", "published_at", "content"])

    def safe_get(self, url: str) -> Optional[str]:
        """Cached and retrying HTTP GET request."""
        cached = self.cache.get(f"html:{url}")
        if cached and "html" in cached:
            self.stats["cache_hits"] += 1
            return cached["html"]

        for i in range(RETRY_COUNT):
            try:
                r = self.session.get(url, headers=HEADERS, timeout=15)
                if r.status_code == 200:
                    self.cache.set(f"html:{url}", {"html": r.text})
                    return r.text
                else:
                    log.warning(f"Failed to get {url} (Status: {r.status_code}). Retrying...")
            except Exception as e:
                log.warning(f"Error getting {url}: {e}. Retrying...")
            time.sleep(MIN_SLEEP + (i * MIN_SLEEP)) # Backoff

        log.error(f"Failed to get {url} after {RETRY_COUNT} retries.")
        return None

    def _split_date_ranges(self, start_str: str, end_str: str) -> List[Tuple[datetime, datetime]]:
        """Splits a date range into 1-year (or less) chunks."""
        try:
            start_date = datetime.strptime(start_str, "%d/%m/%Y")
            end_date = datetime.strptime(end_str, "%d/%m/%Y")
        except ValueError as e:
            log.error(f"Invalid date format: {e}. Use D/M/Y.")
            return []

        if start_date >= end_date:
            log.warning("Start date must be before end date.")
            return []

        ranges = []
        current_start = start_date
        while current_start < end_date:
            # VnExpress limit is 1 year
            one_year_later = current_start + relativedelta(years=1)
            current_end = min(one_year_later, end_date)
            ranges.append((current_start, current_end))
            current_start = current_end

        log.info(f"Split date range '{start_str}' to '{end_str}' into {len(ranges)} chunk(s).")
        return ranges


    def discover_articles(self, category_id: str, pages: int = 2, progress_context=None, task_id=None, from_ts: Optional[int] = None, to_ts: Optional[int] = None) -> List[dict]:
        """Discovers articles from category pages (standard or date range)."""
        articles_data = []
        seen_urls_in_session = set()

        # `category_id` is the category name in standard mode,
        # and the category ID (e.g., 1001002) in date range mode.

        for p in range(1, pages + 1):

            if from_ts and to_ts:
                # Date Range Mode
                url = f"{BASE}/category/day/cateid/{category_id}/fromdate/{from_ts}/todate/{to_ts}/allcate/{category_id}/page/{p}"
            else:
                # Normal Mode
                url = f"{BASE}/{category_id}-p{p}"

            html = self.safe_get(url)
            if not html:
                if progress_context and task_id:
                    progress_context.update(task_id, advance=1) # Vẫn đếm trang thất bại
                continue

            soup = BeautifulSoup(html, "lxml")

            found_on_page = 0
            for article_block in soup.select("article.item-news"):
                link_el = article_block.select_one("h3.title-news a, .title-news a, a.article__link")
                desc_el = article_block.select_one("p.description a, p.description")

                if not link_el:
                    continue

                href = link_el.get("href")
                if href:
                    if href.startswith("/"):
                        href = BASE + href

                    url = normalize_url(href)
                    if url not in seen_urls_in_session and url not in self.seen_articles:
                        seen_urls_in_session.add(url)
                        description = desc_el.text.strip() if desc_el else ""
                        articles_data.append({
                            "url": url,
                            "short_description": description,
                            "category_source": category_id
                        })
                        found_on_page += 1

            if progress_context and task_id:
                progress_context.update(task_id, advance=1)

            time.sleep(random.uniform(MIN_SLEEP, MAX_SLEEP))

        return articles_data


    def fetch_article(self, url: str, short_description: str, category_source: str) -> Optional[dict]:
        """Fetches a single article's HTML metadata."""
        if url in self.seen_articles:
            log.warning(f"Skipping already seen URL: {url}")
            return None

        html = self.safe_get(url)
        if not html:
            return None

        soup = BeautifulSoup(html, "lxml")
        title = soup.select_one("h1.title_news_detail, h1.title-detail")
        published = soup.select_one(".date, span.date")
        content_el = soup.select_one(".fck_detail, .sidebar_1 .Normal")

        author_text = ""
        end_span = soup.select_one("span#article-end")
        if end_span:
            preceding_paragraphs = end_span.find_all_previous("p", class_="Normal")
            for p_tag in preceding_paragraphs:
                if p_tag.get("style") == "text-align:right;":
                    strong_tag = p_tag.select_one("strong")
                    author_text = (strong_tag.text.strip() if strong_tag else p_tag.text.strip())
                    break

        if not author_text: # Fallback
            author_el = soup.select_one(".author_mail, .author")
            if author_el:
                author_text = author_el.text.strip()

        article = {
            "article_id": sha1(url),
            "url": url,
            "title": title.text.strip() if title else "",
            "short_description": short_description,
            "author": author_text,
            "published_at": published.text.strip() if published else "",
            "content": content_el.get_text("\n", strip=True) if content_el else "",
        }
        return article

    def save_article(self, article: dict):
        """Saves one article to articles.csv. (Thread-safe)"""
        try:
            writer = csv.writer(open(self.article_csv, "a", newline="", encoding="utf-8"))
            writer.writerow([
                article["article_id"], article["url"], article["title"],
                article["short_description"], article["author"],
                article["published_at"], article["content"]
            ])

            with open(self.seen_file, "a", encoding="utf-8") as f:
                f.write(f"{article['url']}\n")

            self.stats["articles_saved"] += 1
        except Exception as e:
            log.error(f"Failed to save article {article.get('url')}: {e}")

    def print_summary(self):
        """Uses Rich to print a final summary panel."""
        end_time = time.time()
        total_time = end_time - self.start_time

        summary = (
            f"Discovery Complete ✨\n\n"
            f"[bold green]New Articles Found:[/bold green] {self.stats['articles_saved']}\n"
            f"Total Time: {total_time:.2f} seconds\n"
            f"Cache Hits: {self.stats['cache_hits']}\n"
            f"Output 'to-do list': [italic]{self.article_csv.resolve()}[/italic]"
        )
        console.print(Panel(summary, title="Discovery Summary", border_style="bold magenta", padding=(1, 2)))

    def crawl(self, categories: list[str], pages: int = 2, workers: int = MAX_WORKERS, no_progress: bool = False, from_date: Optional[str] = None, to_date: Optional[str] = None, use_tqdm: bool = False):
        """Main crawl orchestration function."""
        # Force no_progress if use_tqdm is True to avoid conflict
        actual_no_progress = no_progress or use_tqdm

        def fetch_and_save_article(article_info):
            try:
                article = self.fetch_article(
                    article_info["url"],
                    article_info["short_description"],
                    article_info["category_source"]
                )
                if article:
                    self.save_article(article)
                    return 1
            except Exception as e:
                log.error(f"Critical error crawling {article_info['url']}: {e}", exc_info=False)
            return 0

        progress_manager = (
            Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                "[progress.percentage]{task.percentage:>3.1f}%",
                "•",
                TimeRemainingColumn(),
                "•",
                TimeElapsedColumn(),
                console=console,
                transient=False,
            )
            if (not no_progress and not use_tqdm)
            else nullcontext()
        )
        if no_progress:
            log.info("Discovering articles (progress bar disabled)...")


        with progress_manager as progress:

            all_articles_data = []
            date_range_mode = from_date and to_date
            
            categories_to_process = []
            if date_range_mode:
                log.info("Date range mode active. Resolving category names to IDs...")
                for cat_input in categories:
                    cat_id = resolve_category_id(cat_input)
                    if cat_id:
                        categories_to_process.append(cat_id)
                    else:
                        log.warning(f" > [bold yellow]Bỏ qua:[/bold yellow] Không thể tìm thấy ID cho category '{cat_input}'.")

                if not categories_to_process:
                    log.error("[bold red]LỖI: Không có category ID hợp lệ nào để xử lý. Dừng lại.[/bold red]")
                    return
            else:
                categories_to_process = categories

            # define total_tasks for tqdm
            total_tasks = 0
            if date_range_mode:
                date_ranges = self._split_date_ranges(from_date, to_date)
                if date_ranges:
                    total_tasks = len(categories_to_process) * len(date_ranges) * pages
            else:
                total_tasks = len(categories_to_process) * pages

            if use_tqdm and not no_progress:
                pbar1 = tqdm(total=total_tasks, desc="Discovering")
            
            if date_range_mode:
                log.info(f"Running in Date Range mode from [yellow]{from_date}[/yellow] to [yellow]{to_date}[/yellow]")
                if not date_ranges:
                    log.error("No valid date ranges to process. Stopping.")
                    return

                total_tasks = len(categories_to_process) * len(date_ranges) * pages
                discover_task_id = None
                if not no_progress and not use_tqdm:
                    discover_task_id = progress.add_task(f"[cyan]Discovering (by date)...", total=total_tasks)

                for category_id in categories_to_process:
                    log.info(f"Processing [cyan]Cate_ID {category_id}[/cyan]...")
                    for start_dt, end_dt in date_ranges:
                        from_ts = int(start_dt.timestamp())
                        to_ts = int(end_dt.timestamp())
                        log.info(f" > Range: {start_dt.strftime('%d/%m/%Y')} to {end_dt.strftime('%d/%m/%Y')} ({pages} pages)...")

                        articles_data = self.discover_articles(
                            category_id,
                            pages,
                            progress_context=None if use_tqdm else progress,
                            task_id=discover_task_id,
                            from_ts=from_ts,
                            to_ts=to_ts
                        )
                        if use_tqdm and not no_progress:
                            pbar1.update(pages)
                            
                        all_articles_data.extend(articles_data)
                        log.info(f" > Found {len(articles_data)} articles in this range.")

            else:
                log.info("Running in Standard mode (latest articles).")
                discover_task_id = None
                if not no_progress and not use_tqdm:
                    discover_task_id = progress.add_task(f"[cyan]Discovering articles...", total=len(categories_to_process) * pages)

                for category in categories_to_process:
                    log.info(f"Discovering articles in [cyan]{category}[/cyan]...")
                    articles_data = self.discover_articles(
                        category,
                        pages,
                        progress_context=None if use_tqdm else progress,
                        task_id=discover_task_id
                    )
                    if use_tqdm and not no_progress:
                        pbar1.update(pages)
                        
                    all_articles_data.extend(articles_data)
                    log.info(f" > Found {len(articles_data)} new articles in [cyan]{category}[/cyan].")

            if use_tqdm and not no_progress:
                pbar1.close()

            if not all_articles_data:
                log.warning("No new articles found to crawl.")
                self.print_summary()
                return

            unique_new_articles = []
            seen_in_this_run = set()
            for article in all_articles_data:
                url = article["url"]
                if url not in self.seen_articles and url not in seen_in_this_run:
                    unique_new_articles.append(article)
                    seen_in_this_run.add(url)

            log.info(f"Total articles discovered: {len(all_articles_data)}. Unique new articles: [bold green]{len(unique_new_articles)}[/bold green]")

            if not unique_new_articles:
                log.warning("All discovered articles have already been processed in previous runs.")
                self.print_summary()
                return

            crawl_task_id = None
            if not no_progress and not use_tqdm:
                crawl_task_id = progress.add_task(f"[green]Saving {len(unique_new_articles)} articles", total=len(unique_new_articles))
            
            pbar2 = None
            if use_tqdm and not no_progress:
                pbar2 = tqdm(total=len(unique_new_articles), desc="Saving Articles")

            if no_progress:
                log.info(f"Saving {len(unique_new_articles)} articles (progress bar disabled)...")

            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                results = executor.map(fetch_and_save_article, unique_new_articles)

                if no_progress:
                    for _ in results:
                        pass
                else:
                    for result in results:
                        if use_tqdm:
                            pbar2.update(1)
                        else:
                            progress.update(crawl_task_id, advance=1)
            
            if use_tqdm and not no_progress:
                pbar2.close()

        log.info("Crawl complete.")
        self.print_summary()


def run_as_import(categories: list[str], pages: int, output_dir_str: str, use_cache: bool, workers: int, from_date: Optional[str] = None, to_date: Optional[str] = None, no_progress: bool = True, use_tqdm: bool = False, console: Console = None):
    """
    Hàm này được gọi bởi script bên ngoài (pipeline)
    để chạy crawler trong cùng một tiến trình.
    """
    log.info(f"Starting discovery for [bold]{len(categories)}[/bold] categories ({pages} pages each)...")
    if from_date:
        log.info(f"Date range: [yellow]{from_date}[/yellow] to [yellow]{to_date}[/yellow]")
    log.info(f"Using cache: {use_cache}")

    c = VnExpressCrawler(Path(output_dir_str), use_cache=use_cache)

    c.crawl(
        categories,
        pages,
        workers=workers,
        no_progress=no_progress,
        from_date=from_date,
        to_date=to_date,
        use_tqdm=use_tqdm
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="VnExpress Article Discoverer (Step 1)",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--category", "-c",
        required=True,
        nargs="+",
        help="One or more categories (e.g., 'the-gioi' or '1001002')"
    )
    parser.add_argument("--pages", "-p", type=int, default=2, help="Number of pages to discover *per category* or *per date range*")
    parser.add_argument("--output", "-o", default=DEFAULT_OUTPUT, help=f"Output directory (default from config: {DEFAULT_OUTPUT})")
    parser.add_argument("--no-cache", action="store_true", help="Disable all caching for this run")

    parser.add_argument(
        "--from-date",
        type=str,
        default=None,
        help="Start date (D/M/Y). Requires --to-date. (e.g., 01/01/2024)"
    )
    parser.add_argument(
        "--to-date",
        type=str,
        default=None,
        help="End date (D/M/Y). Requires --from-date. (e.g., 31/01/2024)"
    )

    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=MAX_WORKERS,
        help=f"Number of parallel workers (default: {MAX_WORKERS})"
    )
    parser.add_argument(
        "--use-tqdm",
        action="store_true",
        help="Use tqdm progress bars instead of rich"
    )
    args = parser.parse_args()

    if (args.from_date and not args.to_date) or (not args.from_date and args.to_date):
        parser.error("Both --from-date and --to-date must be provided together.")

    if args.from_date:
        try:
            datetime.strptime(args.from_date, "%d/%m/%Y")
            datetime.strptime(args.to_date, "%d/%m/%Y")
        except ValueError:
            parser.error("Invalid date format. Use D/M/Y (e.g., 01/01/2024).")

    rule_title = f"[bold]VnExpress Discoverer[/bold]: [cyan]{', '.join(args.category)}[/cyan]"
    if args.from_date:
        rule_title += f" [yellow]({args.from_date} - {args.to_date})[/yellow]"
    console.rule(rule_title)

    log.info(f"Starting discovery for [bold]{len(args.category)}[/bold] categories ({args.pages} pages each)...")
    if args.from_date:
        log.info(f"Date range: [yellow]{args.from_date}[/yellow] to [yellow]{to_date}[/yellow]")

    log.info(f"Using cache: {not args.no_cache}")

    c = VnExpressCrawler(Path(args.output), use_cache=(not args.no_cache))

    c.crawl(
        args.category,
        args.pages,
        workers=args.workers,
        from_date=args.from_date,
        to_date=args.to_date,
        no_progress=False,
        use_tqdm=args.use_tqdm
    )
