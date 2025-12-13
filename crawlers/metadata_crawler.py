"""
metadata_crawler.py
- Lightweight crawler to extract ONLY category and tags from article URLs.
- Uses Selenium for lazy-loaded content.
- Designed to work with existing articles.csv data (no comment scraping).
- INPUTS: articles.csv (or any file with 'url' column)
- OUTPUTS: metadata.csv (article_url, category, tags)
"""

import time
import argparse
import csv
import logging
import json
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Set, Dict

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.firefox.service import Service as FirefoxService
from selenium.webdriver.firefox.options import Options as FirefoxOptions

try:
    from selenium_stealth import stealth
except ImportError:
    print("ERROR: Need 'selenium-stealth'.")
    print("Run: python -m pip install selenium-stealth")
    sys.exit(1)

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.rule import Rule

from utils import Cache

console = Console()


@dataclass
class Selectors:
    """CSS selectors for metadata extraction."""
    USER_AGENT: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36"
    tags: str = "h4.item-tag a"
    category_breadcrumb: str = "ul.breadcrumb a"


class MetadataCrawler:
    """Lightweight crawler for extracting only category and tags."""

    def __init__(self, output_dir: Path, browser: str, is_headless: bool, use_cache: bool):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.browser = browser
        self.is_headless = is_headless
        self.selectors = Selectors()

        # Output file
        self.metadata_path = output_dir / "metadata.csv"
        self._init_metadata_csv()

        # Cache
        self.cache = Cache(output_dir / ".cache", enabled=use_cache)

        # Resumability
        self.seen_file_path = output_dir / ".seen_metadata.txt"
        self.seen_urls = self._load_seen_urls()

        # Stats
        self.total_processed = 0
        self.total_skipped = 0
        self.total_cache_hits = 0

        # Driver
        self.driver = self._setup_driver()

    def _setup_driver(self) -> WebDriver:
        """Initialize browser driver."""
        logging.info(
            f"Initializing [bold]{self.browser}[/bold] driver in "
            f"[bold]{'headless' if self.is_headless else 'headed'}[/bold] mode..."
        )
        if self.browser == "chrome":
            options = ChromeOptions()
            options.page_load_strategy = 'eager'
            options.add_argument(f"user-agent={self.selectors.USER_AGENT}")
            if self.is_headless:
                options.add_argument("--headless=new")
                options.add_argument("--window-size=2560,1440")
                options.add_argument("--no-sandbox")
                options.add_argument("--disable-gpu")
                options.add_argument("--disable-dev-shm-usage")
            else:
                options.add_argument("--start-maximized")

            options.add_experimental_option('excludeSwitches', ['enable-logging'])
            driver = webdriver.Chrome(service=ChromeService(), options=options)

            stealth(driver,
                    languages=["en-US", "en"],
                    vendor="Google Inc.",
                    platform="Win32",
                    webgl_vendor="Intel Inc.",
                    renderer="Intel Iris OpenGL Engine",
                    fix_hairline=True)

            driver.set_page_load_timeout(30)
            
            # Ensure full window size in headless mode
            if self.is_headless:
                driver.set_window_size(2560, 1440)

        elif self.browser == "firefox":
            options = FirefoxOptions()
            options.page_load_strategy = 'eager'
            if self.is_headless:
                options.add_argument("-headless")
                options.add_argument("-width=2560")
                options.add_argument("-height=1440")
            driver = webdriver.Firefox(service=FirefoxService(), options=options)
            driver.set_page_load_timeout(30)
            if not self.is_headless:
                driver.maximize_window()
            else:
                driver.set_window_size(2560, 1440)
        else:
            raise ValueError(f"Invalid browser: {self.browser}")

        return driver

    def _init_metadata_csv(self):
        """Create metadata.csv with headers if it doesn't exist."""
        if not self.metadata_path.exists():
            with open(self.metadata_path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(["article_url", "category", "tags"])
            logging.info(f"Created: [bold]{self.metadata_path.name}[/bold]")

    def _load_seen_urls(self) -> Set[str]:
        """Load previously processed URLs for resumability."""
        if not self.seen_file_path.exists():
            return set()
        with open(self.seen_file_path, 'r', encoding='utf-8') as f:
            seen = {line.strip() for line in f if line.strip()}
        logging.info(f"Loaded [bold]{len(seen)}[/bold] seen URLs. Will skip.")
        return seen

    def _load_empty_tags_urls(self) -> Set[str]:
        """Load URLs that have empty tags in existing metadata.csv."""
        empty_tags_urls = set()
        if not self.metadata_path.exists():
            return empty_tags_urls
        
        try:
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                for row in reader:
                    if row and len(row) >= 3:
                        url = row[0]
                        tags_str = row[2]
                        # Check if tags is empty list
                        if tags_str in ('[]', '', '""'):
                            empty_tags_urls.add(url)
        except Exception as e:
            logging.warning(f"Could not read metadata.csv: {e}")
        
        if empty_tags_urls:
            logging.info(f"Found [bold yellow]{len(empty_tags_urls)}[/bold yellow] URLs with empty tags to retry.")
        return empty_tags_urls

    def _update_metadata_csv(self, url: str, new_metadata: Dict):
        """Update a specific URL's entry in metadata.csv."""
        rows = []
        updated = False
        
        with open(self.metadata_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if row and row[0] == url:
                    # Update this row
                    rows.append([
                        new_metadata["article_url"],
                        new_metadata["category"],
                        json.dumps(new_metadata["tags"], ensure_ascii=False)
                    ])
                    updated = True
                else:
                    rows.append(row)
        
        if updated:
            with open(self.metadata_path, 'w', newline='', encoding='utf-8') as f:
                csv.writer(f).writerows(rows)

    def _log_seen_url(self, url: str):
        """Mark URL as processed."""
        with open(self.seen_file_path, 'a', encoding='utf-8') as f:
            f.write(f"{url}\n")

    def _extract_metadata(self, url: str) -> Optional[Dict]:
        """Extract category and tags from article page."""
        cache_key = f"metadata:{url}"
        cached = self.cache.get(cache_key)
        if cached:
            logging.info(f"[yellow]Cache HIT[/yellow]: {url.split('/')[-1][:40]}")
            self.total_cache_hits += 1
            return cached

        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.driver.get(url)
            except TimeoutException:
                logging.warning(f"Timeout for {url}. Continuing...")
            except WebDriverException as e:
                logging.error(f"WebDriver error: {e.msg}")
                return None

            # Wait longer on retries
            wait_time = 0.8 + (attempt * 0.5)
            time.sleep(wait_time)

            metadata = {"article_url": url, "category": "", "tags": []}

            # Extract category (FIRST breadcrumb = main category) - available immediately
            try:
                category_els = self.driver.find_elements(By.CSS_SELECTOR, self.selectors.category_breadcrumb)
                if category_els:
                    text = category_els[0].get_attribute('textContent')
                    if text:
                        metadata["category"] = text.strip()
            except Exception as e:
                logging.warning(f"Category extraction failed: {e}")

            # Scroll to bottom to trigger lazy loading of tags (more scrolls on retry)
            scroll_attempts = 2 + attempt  # More scrolls on each retry
            try:
                for _ in range(scroll_attempts):
                    self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                    time.sleep(1.0 + (attempt * 0.5))
            except Exception as e:
                logging.warning(f"Scroll failed: {e}")

            # Extract tags (lazy-loaded, need scroll first)
            try:
                tags_els = self.driver.find_elements(By.CSS_SELECTOR, self.selectors.tags)
                if tags_els:
                    for t in tags_els:
                        text = t.get_attribute('textContent')
                        if text and text.strip():
                            metadata["tags"].append(text.strip())
            except Exception as e:
                logging.warning(f"Tags extraction failed: {e}")

            # If we got tags, no need to retry
            if metadata["tags"]:
                break
            elif attempt < max_retries - 1:
                logging.warning(f"[yellow]No tags found, retrying ({attempt + 2}/{max_retries})...[/yellow]")

        self.cache.set(cache_key, metadata)
        return metadata

    def _save_metadata(self, metadata: Dict):
        """Append metadata to CSV."""
        with open(self.metadata_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                metadata["article_url"],
                metadata["category"],
                json.dumps(metadata["tags"], ensure_ascii=False)
            ])

    def run(self, urls: List[str]):
        """Main execution loop."""
        # First, find URLs with empty tags that need retry
        empty_tags_urls = self._load_empty_tags_urls()
        total_empty_retried = 0
        total_empty_fixed = 0
        
        # Process empty tags URLs first (update existing entries)
        if empty_tags_urls:
            logging.info(f"[bold cyan]Phase 1:[/bold cyan] Retrying {len(empty_tags_urls)} URLs with empty tags...")
            try:
                for i, url in enumerate(empty_tags_urls):
                    logging.info(f"[Retry] ({i+1}/{len(empty_tags_urls)}) {url.split('/')[-1][:50]}")
                    
                    # Clear cache for this URL to force re-extraction
                    cache_key = f"metadata:{url}"
                    self.cache.delete(cache_key)
                    
                    metadata = self._extract_metadata(url)
                    if metadata and metadata["tags"]:
                        self._update_metadata_csv(url, metadata)
                        total_empty_fixed += 1
                        logging.info(f" > [green]Fixed![/green] tags={len(metadata['tags'])}")
                    total_empty_retried += 1
            except Exception as e:
                logging.error(f"Error during empty tags retry: {e}")
        
        # Now process new URLs
        urls_to_process = [u for u in urls if u not in self.seen_urls]
        self.total_skipped = len(urls) - len(urls_to_process)

        if self.total_skipped:
            logging.info(f"Skipped [bold]{self.total_skipped}[/bold] already processed URLs.")

        if not urls_to_process and not empty_tags_urls:
            logging.warning("No new URLs to process.")
            self.driver.quit()
            return
        
        if urls_to_process:
            logging.info(f"[bold cyan]Phase 2:[/bold cyan] Processing [bold green]{len(urls_to_process)}[/bold green] new URLs...")

            try:
                for i, url in enumerate(urls_to_process):
                    logging.info(f"({i+1}/{len(urls_to_process)}) {url.split('/')[-1][:50]}")

                    metadata = self._extract_metadata(url)
                    if metadata:
                        self._save_metadata(metadata)
                        self._log_seen_url(url)
                        self.total_processed += 1
            finally:
                pass
        
        self.driver.quit()
        logging.info("Browser closed.")

        self._print_summary(len(urls_to_process), total_empty_retried, total_empty_fixed)

    def _print_summary(self, total_attempted, empty_retried=0, empty_fixed=0):
        """Print final summary."""
        summary_text = (
            f"Total URLs: [bold]{total_attempted + self.total_skipped}[/bold]\n"
            f"Skipped (seen): [bold yellow]{self.total_skipped}[/bold yellow]\n"
            f"Cache hits: [bold yellow]{self.total_cache_hits}[/bold yellow]\n"
            f"Processed: [bold green]{self.total_processed}[/bold green]\n"
        )
        
        if empty_retried > 0:
            summary_text += f"\n[bold cyan]Empty Tags Retry:[/bold cyan]\n"
            summary_text += f"Retried: [bold]{empty_retried}[/bold]\n"
            summary_text += f"Fixed: [bold green]{empty_fixed}[/bold green]\n"
        
        summary_text += f"\nOutput: [bold cyan]{self.metadata_path.resolve()}[/bold cyan]"
        
        console.print(Panel(
            summary_text,
            title="Metadata Extraction Summary",
            border_style="bold magenta",
            padding=(1, 2),
        ))


def run_as_import(input_file_str: str, output_dir_str: str, browser: str = "chrome", 
                  is_headless: bool = True, use_cache: bool = True):
    """Entry point for pipeline integration."""
    input_path = Path(input_file_str)
    if not input_path.exists():
        logging.error(f"Input file not found: {input_path}")
        return

    # Read URLs from CSV
    urls = []
    with open(input_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        try:
            url_idx = header.index("url")
        except ValueError:
            logging.error(f"No 'url' column found in {input_path}")
            return
        
        for row in reader:
            if row and len(row) > url_idx and row[url_idx].startswith("http"):
                urls.append(row[url_idx].strip())

    if not urls:
        logging.error("No valid URLs found.")
        return

    logging.info(f"Found [bold]{len(urls)}[/bold] URLs in {input_path.name}")

    crawler = MetadataCrawler(
        output_dir=Path(output_dir_str),
        browser=browser,
        is_headless=is_headless,
        use_cache=use_cache
    )
    crawler.run(urls)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract metadata (category, tags) from VnExpress articles.")
    parser.add_argument("-i", "--input", default="data/articles.csv", help="Input CSV with 'url' column")
    parser.add_argument("-o", "--output-dir", default="data", help="Output directory for metadata.csv")
    parser.add_argument("-b", "--browser", choices=["chrome", "firefox"], default="chrome")
    parser.add_argument("--headless", action="store_true", help="Run headless")
    parser.add_argument("--no-cache", action="store_true", help="Disable caching")
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True, markup=True, show_path=False)]
    )

    console.print(Rule("[bold]VnExpress Metadata Crawler[/bold]", style="bold blue"))

    run_as_import(
        input_file_str=args.input,
        output_dir_str=args.output_dir,
        browser=args.browser,
        is_headless=args.headless,
        use_cache=(not args.no_cache)
    )
