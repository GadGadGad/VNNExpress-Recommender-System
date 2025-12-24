"""
vnexpress_user_profile_crawler.py (STEP 3: ENRICHER - METADATA ONLY)
- Crawls VNExpress user profiles for metadata (Join Date) only.
- Uses Playwright to render JS and Threading for concurrent fetching.
- INPUTS: replies.csv (from deep_crawler.py)
- OUTPUTS: user_profiles.csv
"""

import csv
import logging
import random
import time
import threading
import concurrent.futures
from pathlib import Path
import sys
from typing import Optional, Dict, Any, Set
from playwright.sync_api import sync_playwright, Browser
from bs4 import BeautifulSoup
from contextlib import nullcontext
from tqdm import tqdm

try:
    from playwright_stealth import stealth_sync
except ImportError:
    print("LỖI: Không tìm thấy 'playwright_stealth'.")
    print("Hãy chạy: pip install playwright-stealth==1.0.6")
    sys.exit(1)

# Rich imports
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    Progress, SpinnerColumn, BarColumn, TextColumn,
    TimeElapsedColumn, TimeRemainingColumn
)
from rich.table import Table

from utils import Cache

# Global Settings
MIN_SLEEP, MAX_SLEEP, RETRY_COUNT = 0.5, 1.2, 3
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36"
MAX_WORKERS = 5


class UserProfileCrawler:
    def __init__(self, input_dir: Path, console: Console, use_cache=True):
        self.input_dir = input_dir
        self.replies_csv = input_dir / "replies.csv"
        self.user_profile_csv = input_dir / "user_profiles.csv"
        self.console = console

        self.cache = Cache(input_dir / ".cache", enabled=use_cache)
        self.console.log(f"Cache enabled: [cyan]{use_cache}[/]")

        self.seen_file = self.input_dir / ".seen_users.txt"
        self.seen_users = set()
        self._load_seen_users()

        self.csv_lock = threading.Lock()

        self.console.log("[cyan]Playwright UserProfileCrawler initialized (Metadata Mode).[/]")
        self.thread_local = threading.local()
        self.all_playwright_instances = []
        self.playwright_lock = threading.Lock()

        self.user_profile_csv.parent.mkdir(parents=True, exist_ok=True)
        self._init_csvs()


    def _init_csvs(self):
        """Initializes CSV files with headers if they don't exist."""
        # UPDATE: New Schema -> user_id, username, join_date
        if not self.user_profile_csv.exists():
            with open(self.user_profile_csv, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(["user_id", "username", "join_date"])

    def _load_seen_users(self):
        """Loads previously scraped user IDs to enable resuming."""
        if self.seen_file.exists():
            self.console.log("Loading previously seen users for resumability...")
            with open(self.seen_file, "r", encoding="utf-8") as f:
                self.seen_users = {line.strip() for line in f if line.strip()}
            self.console.log(f"Loaded {len(self.seen_users)} seen users.")

    def close(self):
        """Shuts down the browser and Playwright instance."""
        self.console.log(f"[cyan]Closing {len(self.all_playwright_instances)} browser instance(s)...[/]")
        with self.playwright_lock:
            for playwright in self.all_playwright_instances:
                try:
                    playwright.stop()
                except Exception as e:
                    self.console.log(f"[yellow]Error stopping a playwright instance: {e}[/yellow]")
            self.all_playwright_instances.clear()

    def get_browser(self) -> Browser:
        """Gets or creates a Playwright Browser instance for the current thread."""
        if not hasattr(self.thread_local, "browser"):
            # self.console.log(f"[cyan]Initializing browser for thread {threading.current_thread().name}...[/]")
            try:
                playwright = sync_playwright().start()
                browser = playwright.chromium.launch(headless=True)

                self.thread_local.playwright = playwright
                self.thread_local.browser = browser

                with self.playwright_lock:
                    self.all_playwright_instances.append(playwright)
            except Exception as e:
                self.console.log("[bold red]Failed to initialize Playwright for a thread.[/]")
                self.console.log("Please run: [cyan]playwright install[/]")
                raise e

        return self.thread_local.browser

    def safe_get(self, url: str, max_wait: int = 30) -> Optional[str]:
        """Fetches and renders a page using Playwright, with caching."""
        cache_key = f"html_profile_meta:{url}"
        cached = self.cache.get(cache_key)
        if cached and "html" in cached:
            # self.console.log(f"[green]Cache hit[/] for {url}")
            return cached["html"]

        context = None
        for attempt in range(RETRY_COUNT):
            try:
                browser = self.get_browser()
                context = browser.new_context(
                    user_agent=USER_AGENT,
                    viewport={'width': 1920, 'height': 1080}
                )
                page = context.new_page()
                stealth_sync(page)

                # Wait for networkidle to ensure profile data loads
                page.goto(url, timeout=max_wait * 1000, wait_until="networkidle")

                html = page.content()
                self.cache.set(cache_key, {"html": html})
                context.close()
                return html

            except Exception as e:
                # self.console.log(f"[red]Request failed[/] for {url}: {e.__class__.__name__}")
                if context:
                    context.close()
                time.sleep(random.uniform(0.5, 1.0))
        if context:
            context.close()
        return None

    def fetch_user_profile(self, userid: str) -> Optional[Dict[str, Any]]:
        """
        Fetches the user profile page and parses ONLY metadata (Username, Join Date).
        """
        url = f"https://my.vnexpress.net/users/feed/{userid}"
        html = self.safe_get(url)
        if not html:
            self.console.log(f"[yellow]Profile page for {userid} was empty or failed to load.[/yellow]")
            return None

        soup = BeautifulSoup(html, "lxml")

        username = "N/A"
        join_date = "N/A"

        # Logic: "Nguyễn Văn A Tham gia từ 20/10/2021"
        name_el = soup.select_one("span.name_sub")
        if name_el:
            text = name_el.get_text(separator=" ", strip=True)
            if "Tham gia từ" in text:
                parts = text.split("Tham gia từ")
                username = parts[0].strip()
                if len(parts) > 1:
                    join_date = parts[1].strip()
            else:
                username = text

        profile = {
            "userid": userid,
            "username": username,
            "join_date": join_date
        }
        return profile

    def _load_users_from_replies(self) -> Set[str]:
        """Reads replies.csv (from Step 2) and extracts all unique user IDs."""
        user_ids = set()
        self.console.log(f"Reading user IDs from [cyan]{self.replies_csv.name}[/cyan]...")
        try:
            with open(self.replies_csv, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    pid = row.get("parent_user_id")
                    if pid and pid not in ("N/A", "NO_COMMENT", "N/A (Format Error)"):
                        user_ids.add(pid)

                    rid = row.get("reply_user_id")
                    if rid and rid not in ("N/A", "", "N/A (Format Error)"):
                        user_ids.add(rid)

        except FileNotFoundError:
            self.console.log(f"[bold red]Error: Input file not found at {self.replies_csv}[/]")
            return set()
        except Exception as e:
            self.console.log(f"[bold red]Error reading {self.replies_csv.name}: {e}[/]")
            return set()

        self.console.log(f"Found {len(user_ids)} unique user IDs to process.")
        return user_ids

    def _fetch_and_save_profile(self, uid: str) -> int:
        """
        Wrapper function for threading.
        Fetches one profile, saves it, and returns success status (1 or 0).
        """
        PROFILE_RETRY_COUNT = 2
        profile = None
        for attempt in range(PROFILE_RETRY_COUNT):
            profile = self.fetch_user_profile(uid)
            if profile:
                break
            if attempt < PROFILE_RETRY_COUNT - 1:
                time.sleep(random.uniform(1.0, 2.0))

        if not profile:
            return 0

        with self.csv_lock:
            try:
                with open(self.user_profile_csv, "a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow([
                        profile["userid"],
                        profile["username"],
                        profile["join_date"]
                    ])

                with open(self.seen_file, "a", encoding="utf-8") as f:
                    f.write(f"{uid}\n")

                return 1
            except Exception as e:
                self.console.log(f"[red]Failed to write profile for {uid}: {e}[/red]")
                return 0

    def crawl_profiles(self, workers: int = MAX_WORKERS, no_progress: bool = False, use_tqdm: bool = False):
        """Main crawl loop."""
        self.silent = use_tqdm
        unique_users = self._load_users_from_replies()
        if not unique_users:
            self.console.log("[yellow]No users found to process. Exiting.[/yellow]")
            return None

        users = sorted(list(unique_users))

        # Filter seen
        users_to_crawl = []
        if self.seen_users:
            users_to_crawl = [uid for uid in users if uid not in self.seen_users]
            skipped = len(users) - len(users_to_crawl)
            if skipped > 0:
                self.console.log(f"Skipped {skipped} already processed users.")
        else:
            users_to_crawl = users

        if not users_to_crawl:
            self.console.log("[green]All users have already been processed.[/]")
            return str(self.user_profile_csv)

        self.console.log(f"Total users to crawl: [bold cyan]{len(users_to_crawl)}[/]")
        total_to_crawl = len(users_to_crawl)
        processed = 0

        # Progress Bar
        progress_manager = (
            Progress(
                SpinnerColumn(style="cyan"),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(bar_width=None),
                TextColumn("{task.completed}/{task.total}"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=self.console,
                transient=True,
            )
            if (not no_progress and not use_tqdm)
            else nullcontext()
        )

        if no_progress:
            self.console.log("Crawling user metadata (progress bar disabled)...")

        if use_tqdm and not no_progress:
            pbar = tqdm(total=total_to_crawl, desc="Enriching User Nodes", file=sys.stdout, position=0, leave=True, dynamic_ncols=True, ascii=True, mininterval=0.5)

        with progress_manager as progress:
            task = None
            if not no_progress and not use_tqdm:
                task = progress.add_task("Enriching User Nodes", total=total_to_crawl)

            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                results = executor.map(self._fetch_and_save_profile, users_to_crawl)

                if no_progress:
                    for i, status in enumerate(results):
                        processed += status
                        if (i + 1) % 50 == 0 and not self.silent:
                            self.console.log(f"Processed {i+1}/{total_to_crawl} users...")
                else:
                    for status in results:
                        processed += status
                        if use_tqdm:
                            pbar.update(1)
                        else:
                            progress.update(task, advance=1)
                        # Slightly faster sleep since we do less work per page
                        time.sleep(random.uniform(MIN_SLEEP, MAX_SLEEP))
        
        if use_tqdm and not no_progress:
            pbar.close()

        table = Table(title="VNExpress User Enrichment Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        table.add_row("Profiles CSV", str(self.user_profile_csv))
        table.add_row("Users Processed", str(processed))
        self.console.print(table)

        return str(self.user_profile_csv)


def run_as_import(input_dir_str: str, use_cache: bool, workers: int, console: Console, no_progress: bool = True, use_tqdm: bool = False):
    """
    Called by pipeline script.
    """
    crawler = UserProfileCrawler(Path(input_dir_str), console, use_cache=use_cache)
    try:
        profiles_csv = crawler.crawl_profiles(workers=workers, no_progress=no_progress, use_tqdm=use_tqdm)
        if profiles_csv:
            crawler.console.log(f"[green]Wrote User Metadata to[/]: {profiles_csv}")
    finally:
        crawler.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="VnExpress User Enricher (Step 3 - Metadata Only)")
    parser.add_argument("--input", "-i", default="data", help="Directory containing replies.csv")
    parser.add_argument("--no-cache", action="store_true", help="Disable caching")
    parser.add_argument("--workers", "-w", type=int, default=MAX_WORKERS, help=f"Workers (default: {MAX_WORKERS})")
    parser.add_argument("--use-tqdm", action="store_true", help="Use tqdm progress bars instead of rich")
    args = parser.parse_args()

    console = Console()
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True, markup=True, show_path=False)],
    )
    # Silence noisy libs
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("playwright").setLevel(logging.WARNING)

    console.rule(f"[bold]VnExpress User Enricher (Step 3)[/bold]", style="bold blue")

    crawler = UserProfileCrawler(Path(args.input), console, use_cache=(not args.no_cache))

    try:
        crawler.crawl_profiles(workers=args.workers, use_tqdm=args.use_tqdm)
    except Exception as e:
        console.print_exception()
    finally:
        crawler.close()
