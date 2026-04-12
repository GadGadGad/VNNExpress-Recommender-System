"""
vnexpress_deep_crawler.py (STEP 2: PROCESSOR)
- Scrapes VnExpress articles for top-level comments AND their replies.
- Uses Selenium to click "View Replies" and "Read More" links.
- This script is single-threaded for stability. Run multiple
  instances of it (via run_pipeline.py) to parallelize.
- INPUTS: articles.csv (from main_crawler.py)
- OUTPUTS: replies.csv
"""

import time
import argparse
import csv
import re
import logging
import json
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Set, Dict
from contextlib import nullcontext

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, TimeoutException, WebDriverException
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.firefox.service import Service as FirefoxService
from selenium.webdriver.firefox.options import Options as FirefoxOptions

try:
    from selenium_stealth import stealth
except ImportError:
    print("ERROR: 'selenium-stealth' is required.")
    print("Please run: python -m pip install selenium-stealth")
    sys.exit(1)


from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.rule import Rule

from utils import Cache, sha1

log = logging.getLogger()


@dataclass
class Selectors:
    """Holds all CSS selectors in one place for easy maintenance."""
    USER_AGENT: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36"

    comment_block: str = "div.comment_item.width_common"
    parent_content: str = "div.content-comment"
    reply_item: str = "div.sub_comment_item"
    view_replies: str = "a.view_all_reply"
    read_more: str = "a.continue-reading"
    author_link: str = "a.nickname"
    text_content_group: str = "p.content_more, p.full_content, p.content_less"
    date: str = "span.time-com"
    reactions: str = "div.reactions-total a.number"
    filter_bar: str = "div.filter_coment.width_common"
    view_more_comments: str = "a#show_more_comet"


class VnExpressDeepCrawler:
    """
    Main class to manage state (driver, cache, seen_urls)
    """

    def __init__(self, output_file: str, browser: str, is_headless: bool, use_cache: bool, console: Console, console_handler: logging.Handler):
        self.output_path = Path(output_file)
        self.output_dir = self.output_path.parent
        self.browser = browser
        self.is_headless = is_headless
        self.selectors = Selectors()
        self.console = console
        self.console_handler = console_handler

        # Setup Cache (uses imported Cache class)
        self.cache = Cache(self.output_dir / ".cache", enabled=use_cache)

        # Setup Resumability (Seen URLs)
        self.seen_file_path = self.output_dir / ".seen_urls_deep.txt"
        self.seen_urls = self._load_seen_urls()

        # Stats
        self.total_rows_saved = 0
        self.total_urls_processed = 0
        self.total_urls_skipped = 0
        self.total_cache_hits = 0

        # Start Driver
        self.driver = self._setup_driver()

    def _setup_driver(self) -> WebDriver:
        """Initializes and returns the requested webdriver."""
        log.info(
            f"Initializing [bold]{self.browser}[/bold] driver in "
            f"[bold]{'headless' if self.is_headless else 'headed'}[/bold] mode..."
        )
        if self.browser == "chrome":
            options = ChromeOptions()
            options.page_load_strategy = 'eager'
            options.add_argument(f"user-agent={self.selectors.USER_AGENT}")
            if self.is_headless:
                options.add_argument("--headless")
                options.add_argument("--window-size=1920,1080")
                options.add_argument("--no-sandbox")
                options.add_argument("--disable-gpu")
                options.add_argument("--disable-dev-shm-usage")
            else:
                options.add_argument("--start-maximized")

            options.add_experimental_option('excludeSwitches', ['enable-logging'])
            service = ChromeService()
            driver = webdriver.Chrome(service=service, options=options)

            stealth(driver,
                    languages=["en-US", "en"],
                    vendor="Google Inc.",
                    platform="Win32",
                    webgl_vendor="Intel Inc.",
                    renderer="Intel Iris OpenGL Engine",
                    fix_hairline=True,
                    )

            driver.set_page_load_timeout(60)

        elif self.browser == "firefox":
            options = FirefoxOptions()
            options.page_load_strategy = 'eager'
            if self.is_headless:
                options.add_argument("-headless")
            service = FirefoxService()
            driver = webdriver.Firefox(service=service, options=options)
            driver.set_page_load_timeout(60)

            if not self.is_headless:
                driver.maximize_window()
        else:
            log.error(f"Invalid browser: {self.browser}")
            raise ValueError(f"Invalid browser: {self.browser}")

        return driver

    def _load_seen_urls(self) -> Set[str]:
        """Load successfully scraped URLs from .seen_urls.txt"""
        if not self.seen_file_path.exists():
            return set()
        try:
            with open(self.seen_file_path, 'r', encoding='utf-8') as f:
                seen = {line.strip() for line in f if line.strip()}
            log.info(f"Loaded [bold]{len(seen)}[/bold] URLs from .seen file. These will be skipped.")
            return seen
        except Exception as e:
            log.warning(f"Could not load seen file: {e}. Starting fresh.")
            return set()

    def _log_seen_url(self, url: str):
        """Append a successfully scraped URL to the seen file."""
        try:
            with open(self.seen_file_path, 'a', encoding='utf-8') as f:
                f.write(f"{url}\n")
        except Exception as e:
            log.error(f"Failed to write to seen file: {e}")

    def _save_data_chunk(self, data: List[Dict], needs_header: bool):
        """Save a chunk of data to CSV (append mode)."""
        if not data:
            return
        try:
            headers = data[0].keys()
            with open(self.output_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                if needs_header:
                    writer.writeheader()
                writer.writerows(data)
            self.total_rows_saved += len(data)
        except Exception as e:
            log.error(f"An error occurred while saving chunk to CSV: {e}", exc_info=True)

    def _extract_user_id(self, href: str) -> str:
        """Extracts the numeric user ID from a profile URL."""
        if not href:
            return "N/A"
        match = re.search(r'(\d+)$', href)
        if match:
            return match.group(1)
        return "N/A (Format Error)"

    def _parse_comment(self, element: WebElement, sel: Selectors) -> dict:
        """Parses a single comment element (parent or reply)."""
        try:
            read_more_link = element.find_element(By.CSS_SELECTOR, sel.read_more)
            self.driver.execute_script("arguments[0].click();", read_more_link)
            WebDriverWait(self.driver, 0.6).until(EC.staleness_of(read_more_link))
        except (NoSuchElementException, TimeoutException):
            pass
        except Exception as e:
            log.warning(f"  > Error clicking 'Đọc tiếp' in comment: {e}")

        author, user_id = "N/A", "N/A"
        try:
            author_el = element.find_element(By.CSS_SELECTOR, sel.author_link)
            author = author_el.get_attribute("textContent").strip()
            user_id = self._extract_user_id(author_el.get_attribute("href"))
        except NoSuchElementException:
            pass

        JS_GET_CLEAN_TEXT = """
            var element = arguments[0];
            var selectorToRemove = ".txt-name";
            var clone = element.cloneNode(true);
            var childToRemove = clone.querySelector(selectorToRemove);
            if (childToRemove) {
                childToRemove.remove();
            }
            return clone.textContent.trim();
        """
        text = "N/A"
        try:
            text_el = element.find_element(By.CSS_SELECTOR, sel.text_content_group)
            text = self.driver.execute_script(JS_GET_CLEAN_TEXT, text_el)
        except NoSuchElementException:
            pass

        date = "N/A"
        
        # List of common date selectors on VnExpress
        date_selectors = [
            sel.date,
            ".time-count",
            "span.time",
            "a.time-com",
            ".txt-time",
            ".time-public"
        ]
        
        # Try each selector until successful
        for selector in date_selectors:
            try:
                el = element.find_element(By.CSS_SELECTOR, selector)
                val = el.text.strip()
                if val: # If text is found
                    date = val
                    break # Stop searching
            except (NoSuchElementException, Exception):
                continue # Try next selector on error

        reactions = "0"
        try:
            reactions = element.find_element(By.CSS_SELECTOR, sel.reactions).text
        except NoSuchElementException:
            pass

        return { "author": author, "user_id": user_id, "text": text, "date": date, "reactions": reactions }

    def _scrape_article_page(self, url: str) -> list[dict]:
        """Scrapes all comments and replies from a single article URL."""
        cache_key = f"vnexpress_deep_scrape:{url}"
        cached_data = self.cache.get(cache_key)
        
        if cached_data:
            log.info(f"[yellow]Cache HIT.[/yellow] Loading data from cache for: {url.split('/')[-1]}")
            self.total_cache_hits += 1
            return cached_data.get("data", [])

        log.info(f"Cache MISS. Scraping: {url.split('/')[-1]}")

        try:
            self.driver.get(url)
        except TimeoutException:
            log.warning(f"[yellow]Page load timed out[/yellow] (60s) for {url}. Continuing scrape attempt...")
            pass # Ignore timeout and continue
        except WebDriverException as e:
            log.error(f"[red]WebDriverException on driver.get()[/red]: {e.msg}. Skipping this article.")
            return []

        time.sleep(1.2)

        try:
            filter_bar = self.driver.find_element(By.CSS_SELECTOR, self.selectors.filter_bar)
            self.driver.execute_script("arguments[0].scrollIntoView(true);", filter_bar)
            time.sleep(0.5)
        except Exception as e:
            log.warning(f"[yellow]Could not perform initial scroll: {e}[/yellow]")

        log.info("Looking for 'Xem thêm ý kiến' button to load all comments...")
        while True:
            try:
                view_more_button = WebDriverWait(self.driver, 1).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, self.selectors.view_more_comments))
                )
                self.driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", view_more_button)
                time.sleep(0.5)
                self.driver.execute_script("arguments[0].click();", view_more_button)
                time.sleep(1.)
            except TimeoutException:
                log.info(" > 'Xem thêm ý kiến' button no longer found. All comments are loaded.")
                break
            except Exception as e:
                log.warning(f" > Error clicking 'Xem thêm ý kiến': {e}. Assuming all comments loaded.")
                break

        all_comment_blocks = self.driver.find_elements(By.CSS_SELECTOR, self.selectors.comment_block)
        log.info(f"Found [bold]{len(all_comment_blocks)}[/bold] total comment blocks after loading all pages.")
        scraped_data = []

        for block in all_comment_blocks:
            try:
                parent_content_div = block.find_element(By.CSS_SELECTOR, self.selectors.parent_content)
                parent_data = self._parse_comment(parent_content_div, self.selectors)
            except NoSuchElementException:
                log.warning("[yellow]Found comment block without content. Skipping.[/yellow]")
                continue

            while True:
                try:
                    view_replies_link = WebDriverWait(block, 1).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, self.selectors.view_replies))
                    )
                    self.driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", view_replies_link)
                    time.sleep(0.5)
                    self.driver.execute_script("arguments[0].click();", view_replies_link)
                    time.sleep(1.)
                except TimeoutException:
                    break
                except NoSuchElementException:
                    break
                except Exception as e:
                    log.error(f"  > Error clicking 'View Replies' repeatedly: {e}")
                    break

            reply_elements = block.find_elements(By.CSS_SELECTOR, self.selectors.reply_item)

            if not reply_elements:
                scraped_data.append({
                    "article_url": url,
                    **{f"parent_{k}": v for k, v in parent_data.items()},
                    "reply_author": "", "reply_user_id": "", "reply_text": "",
                    "reply_date": "", "reply_reactions": "0",
                })
            else:
                for reply_el in reply_elements:
                    try:
                        reply_data = self._parse_comment(reply_el, self.selectors)
                        scraped_data.append({
                            "article_url": url,
                            **{f"parent_{k}": v for k, v in parent_data.items()},
                            **{f"reply_{k}": v for k, v in reply_data.items()},
                        })
                    except Exception as e:
                        log.warning(f"  > [yellow]Failed to parse a reply: {e}[/yellow]")

        if not scraped_data:
            log.info(" > Article has no comments. Creating a 'no_comment' row.")
            scraped_data.append({
                "article_url": url,
                "parent_author": "NO_COMMENT", "parent_user_id": "NO_COMMENT",
                "parent_text": "NO_COMMENT", "parent_date": "NO_COMMENT",
                "parent_reactions": "0",
                "reply_author": "", "reply_user_id": "", "reply_text": "",
                "reply_date": "", "reply_reactions": "0",
            })

        self.cache.set(cache_key, {"data": scraped_data})
        return scraped_data

    def run(self, urls_to_scrape: List[str]):
        """Main coordinator function, integrates Resumability and Instant Save."""
        urls_to_process = []
        for url in urls_to_scrape:
            if url not in self.seen_urls:
                urls_to_process.append(url)
            else:
                self.total_urls_skipped += 1

        if self.total_urls_skipped > 0:
            log.info(f"Skipped [bold]{self.total_urls_skipped}[/bold] URLs found in .seen file.")

        if not urls_to_process:
            log.warning("[yellow]No new URLs to process. Exiting.[/yellow]")
            self.driver.quit()
            return

        log.info(f"Starting scrape for [bold green]{len(urls_to_process)}[/bold green] new URLs.")
        needs_header = (not self.output_path.exists()) or (self.output_path.stat().st_size == 0)
        if needs_header:
            log.info("Output file not found or empty. Will write new header.")
        else:
            log.info(f"Output file [cyan]{self.output_path.name}[/cyan] exists. Will append data.")

        total_urls = len(urls_to_process)

        try:
            for i, article_url in enumerate(urls_to_process):

                url_name = article_url.split("/")[-1].split(".html")[0]

                log.info(f"Progress: ({i+1}/{total_urls}) Scraping: {url_name}")

                try:
                    article_data = self._scrape_article_page(article_url)
                    if article_data:
                        self._save_data_chunk(article_data, needs_header)
                        self._log_seen_url(article_url)
                        self.total_urls_processed += 1
                        needs_header = False
                    else:
                        log.warning(f"No data returned for {article_url}, will retry next run.")

                except Exception as e:
                    log.error(f"A critical error occurred while scraping {article_url}: {e}", exc_info=True)
                    log.error("Moving to the next URL...")

        finally:
            self.driver.quit()
            log.info("Browser closed.")

        self._print_summary(total_urls)

    def _print_summary(self, total_attempted):
        """Print final summary table."""
        summary_panel = Panel(
            f"Total URLs in file: [bold]{total_attempted + self.total_urls_skipped}[/bold]\n"
            f"URLs Skipped (seen): [bold yellow]{self.total_urls_skipped}[/bold yellow]\n"
            f"URLs Skipped (cache): [bold yellow]{self.total_cache_hits}[/bold yellow]\n"
            f"URLs Processed Now: [bold green]{self.total_urls_processed}[/bold green]\n"
            f"Total Rows Saved: [bold green]{self.total_rows_saved}[/bold green]\n\n"
            f"Output File: [bold cyan]{self.output_path.resolve()}[/bold cyan]\n"
            f"Seen File: [bold cyan]{self.seen_file_path.resolve()}[/bold cyan]",
            title="Crawl Summary",
            border_style="bold magenta",
            padding=(1, 2),
        )
        self.console.print(summary_panel)

def run_as_import(input_file_str: str, output_file_str: str, browser: str, is_headless: bool, use_cache: bool, worker_id: str = "Step2", console: Console = None):
    """
    Called by external script (pipeline)
    to run crawler in the same process.
    """

    if not console:
        from rich.console import Console
        console = Console()
        log.warning("run_as_import được gọi mà không có console, tự tạo một cái.")

    console_handler = None
    for handler in logging.getLogger().handlers:
        if isinstance(handler, RichHandler):
            console_handler = handler
            break

    if not console_handler:
        log.error("PANIC: Không tìm thấy RichHandler từ pipeline. Dừng lại.")
        return

    class WorkerIdFilter(logging.Filter):
        def filter(self, record):
            record.worker_id = worker_id
            return True

    if not any(isinstance(f, WorkerIdFilter) for f in log.filters):
        log.addFilter(WorkerIdFilter())

    log.info(f"Reading URLs from [bold cyan]{input_file_str}[/bold cyan]")
    url_file_path = Path(input_file_str)
    if not url_file_path.is_file():
        log.error(f"[bold red]URL file not found: {url_file_path}[/bold red]")
        log.error("Have you run Step 1 (main_crawler.py) first?")
        raise FileNotFoundError(f"{url_file_path} not found")

    urls_to_scrape = []
    try:
        if url_file_path.suffix == ".csv":
            log.info("Detected .csv file. Reading 'url' column...")
            with open(url_file_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader)
                try:
                    url_index = header.index("url")
                except ValueError:
                    log.error(f"[bold red]CSV file has no 'url' column! Header is: {header}[/bold red]")
                    raise ValueError("CSV file has no 'url' column")

                for row in reader:
                    if row and len(row) > url_index:
                        url = row[url_index].strip()
                        if url.startswith("http"):
                            urls_to_scrape.append(url)
        else:
            log.info("Detected .txt file. Reading line by line...")
            with open(url_file_path, 'r', encoding='utf-8') as f:
                urls_to_scrape = [line.strip() for line in f if line.strip() and line.startswith("http")]
    except Exception as e:
        log.error(f"[bold red]Error reading URL file: {e}[/bold red]", exc_info=True)
        raise e

    if not urls_to_scrape:
        log.error("[bold red]No valid URLs found in the file. Exiting.[/bold red]")
        return

    log.info(f"Found [bold green]{len(urls_to_scrape)}[/bold green] total URLs in file.")

    try:
        crawler = VnExpressDeepCrawler(
            output_file=output_file_str,
            browser=browser,
            is_headless=is_headless,
            use_cache=use_cache,
            console=console,
            console_handler=console_handler
        )

        crawler.run(urls_to_scrape)

    except Exception as e:
        log.error(f"A fatal error occurred: {e}", exc_info=True)
        log.error("Crawler stopped prematurely.")
        raise e


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scrape VnExpress comments and replies from a file of URLs."
    )
    parser.add_argument(
        "-i", "--input-file",
        default="data/articles.csv",
        help="Input file from Step 1 (e.g., data/articles.csv)"
    )
    parser.add_argument(
        "-o", "--output",
        default="data/replies.csv",
        help="Output CSV file name (default: data/replies.csv)"
    )
    parser.add_argument(
        "-l", "--logfile",
        default="deep_crawler.log",
        help="Log file name (default: deep_crawler.log)"
    )
    parser.add_argument(
        "-b", "--browser",
        choices=["chrome", "firefox"],
        default="chrome",
        help="Browser to use (default: chrome)"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run the browser in headless mode"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable file caching for this run"
    )
    parser.add_argument(
        "--worker-id",
        default="Main",
        help="ID of the worker for logging"
    )
    args = parser.parse_args()


    LOG_FORMAT = "[%(worker_id)s] %(message)s"
    log_formatter = logging.Formatter(LOG_FORMAT, datefmt="%Y-%m-%d %H:M:%S")
    file_handler = logging.FileHandler(args.logfile, mode='w', encoding='utf-8')
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - [%(worker_id)s] - %(message)s", datefmt="%Y-%m-%d %H:M:%S"))

    from rich.console import Console
    console = Console()

    console_handler = RichHandler(console=console, rich_tracebacks=True, markup=True, show_path=False)
    console_handler.setFormatter(log_formatter)
    logging.basicConfig(level=logging.INFO, handlers=[file_handler, console_handler])
    log = logging.getLogger()

    class WorkerIdFilter(logging.Filter):
        def filter(self, record):
            record.worker_id = args.worker_id
            return True
    log.addFilter(WorkerIdFilter())

    console.print(Rule(f"[bold]VnExpress Deep Crawler (Step 2)[/bold]", style="bold blue"))

    try:
        run_as_import(
            input_file_str=args.input_file,
            output_file_str=args.output,
            browser=args.browser,
            is_headless=args.headless,
            use_cache=(not args.no_cache),
            worker_id=args.worker_id,
            console=console
        )
    except Exception:
        sys.exit(1)
