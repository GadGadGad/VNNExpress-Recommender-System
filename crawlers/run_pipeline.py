"""
Orchestrator script to run the full VnExpress crawler pipeline.

This script imports and calls the other scripts in the correct order:
1. main_crawler.py (Discoverer)
2. deep_crawler.py (Processor)
3. user_profile_crawler.py (Enricher)

If any step fails, the pipeline will stop.
"""

import logging
from pathlib import Path
import sys
import argparse
import os
from datetime import datetime

import rich
from rich.console import Console
from rich.logging import RichHandler
from rich.rule import Rule
from rich.status import Status

try:
    from main_crawler import run_as_import as run_step_1
    from deep_crawler import run_as_import as run_step_2
    from user_profile_crawler import run_as_import as run_step_3
    from metadata_crawler import run_as_import as run_step_4
except ImportError as e:
    print(f"Lỗi Import: {e}")
    print("Vui lòng đảm bảo các file crawler nằm cùng thư mục.")
    sys.exit(1)


console = Console()

# Check for use_tqdm early to setup logging
use_tqdm_flag = "--use-tqdm" in sys.argv

if use_tqdm_flag:
    # Use standard StreamHandler instead of RichHandler to avoid TTY/escape code conflicts with tqdm
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    # Silence secondary libraries aggressively
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
else:
    handler = rich.logging.RichHandler(console=console, rich_tracebacks=True, show_path=False, markup=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[handler]
)
log = logging.getLogger(__name__)


def main(args):
    """
    Runs the full 3-step pipeline based on parsed arguments.
    """
    start_time = Path.cwd()
    console.rule(f"[bold magenta]🚀 Starting VnExpress Crawler Pipeline[/bold magenta]")

    output_dir = Path(args.output)
    log.info(f"All data will be saved in: [cyan]{start_time / output_dir}[/cyan]")
    output_dir.mkdir(parents=True, exist_ok=True)

    articles_csv = output_dir / "articles.csv"
    replies_csv = output_dir / "replies.csv"

    from_date_arg = args.from_date
    to_date_arg = args.to_date
    if (from_date_arg and not to_date_arg) or (not from_date_arg and to_date_arg):
        log.error("Both --from-date and --to-date must be provided together.")
        sys.exit(1)

    if from_date_arg:
        try:
            datetime.strptime(from_date_arg, "%d/%m/%Y")
            datetime.strptime(to_date_arg, "%d/%m/%Y")
            log.info(f"Running in date range mode: [yellow]{from_date_arg}[/yellow] to [yellow]{to_date_arg}[/yellow]")
        except ValueError:
            log.error("Date format is invalid. Please use D/M/Y (e.g., 01/01/2024).")
            sys.exit(1)

    no_progress_value = not (args.show_progress or args.use_tqdm)
    if args.show_progress and not args.use_tqdm:
        log.warning("[bold yellow]--show-progress is active. Console output for Step 1 & 3 may be messy.[/bold yellow]")


    pipeline_completed = False

    try:
        if 1 in args.steps:
            console.rule(f"[bold cyan]Starting Step: 1. Discover Articles ({args.workers} workers)[/bold cyan]")
            try:
                run_step_1(
                    categories=args.categories,
                    pages=args.pages,
                    output_dir_str=str(output_dir),
                    use_cache=(not args.no_cache),
                    workers=args.workers,
                    from_date=from_date_arg,
                    to_date=to_date_arg,
                    no_progress=no_progress_value,
                    use_tqdm=args.use_tqdm,
                    console=console
                )
                console.log(f"[bold green]✅ Step '1. Discover Articles' completed successfully.[/bold green]\n")
            except Exception as e:
                console.log(f"[bold red]❌ ERROR IN STEP: '1. Discover Articles'[/bold red]")
                log.error(f"Pipeline HALTED due to an error in Step 1: {e}", exc_info=True)
                sys.exit(1)
        else:
            log.info("Skipping Step 1 (Discover Articles) as requested.")


        if 2 in args.steps:
            if 1 not in args.steps and not articles_csv.exists():
                log.error(f"[bold red]Cannot run Step 2: Input file '{articles_csv.name}' not found.[/bold red]")
                log.error("Please run Step 1 first (e.g., --steps 1 2) to generate it.")
                sys.exit(1)

            console.rule("[bold cyan]Starting Step: 2. Process Comments (Deep Crawl)[/bold cyan]")
            log.info("Running Step 2 with a single worker for stability.")
            try:
                run_step_2(
                    input_file_str=str(articles_csv),
                    output_file_str=str(replies_csv),
                    browser=args.browser,
                    is_headless=args.headless,
                    use_cache=(not args.no_cache),
                    worker_id="Step2",
                    console=console
                )
                console.log(f"[bold green]✅ Step '2. Process Comments (Deep Crawl)' completed successfully.[/bold green]\n")

            except Exception as e:
                console.log(f"[bold red]❌ ERROR IN STEP: '2. Process Comments (Deep Crawl)'[/bold red]")
                log.error(f"Pipeline HALTED due to an error in Step 2: {e}", exc_info=True)
                sys.exit(1)

        else:
            log.info("Skipping Step 2 (Process Comments) as requested.")


        if 3 in args.steps:
            if 2 not in args.steps and not replies_csv.exists():
                log.error(f"[bold red]Cannot run Step 3: Input file '{replies_csv.name}' not found.[/bold red]")
                log.error("Please run Step 2 first (e.g., --steps 2 3) to generate it.")
                sys.exit(1)

            console.rule(f"[bold cyan]Starting Step: 3. Enrich Users ({args.workers} workers)[/bold cyan]")
            try:
                run_step_3(
                    input_dir_str=str(output_dir),
                    use_cache=(not args.no_cache),
                    workers=args.workers,
                    console=console,
                    no_progress=no_progress_value,
                    use_tqdm=args.use_tqdm
                )
                console.log(f"[bold green]✅ Step '3. Enrich Users' completed successfully.[/bold green]\n")
            except Exception as e:
                console.log(f"[bold red]❌ ERROR IN STEP: '3. Enrich Users'[/bold red]")
                log.error(f"Pipeline HALTED due to an error in Step 3: {e}", exc_info=True)
                sys.exit(1)
        else:
            log.info("Skipping Step 3 (Enrich Users) as requested.")


        if 4 in args.steps:
            if not articles_csv.exists():
                log.error(f"[bold red]Cannot run Step 4: Input file '{articles_csv.name}' not found.[/bold red]")
                log.error("Please run Step 1 first to generate it.")
                sys.exit(1)

            console.rule("[bold cyan]Starting Step: 4. Extract Metadata Only (category, tags)[/bold cyan]")
            try:
                run_step_4(
                    input_file_str=str(articles_csv),
                    output_dir_str=str(output_dir),
                    browser=args.browser,
                    is_headless=args.headless,
                    use_cache=(not args.no_cache)
                )
                console.log(f"[bold green]✅ Step '4. Extract Metadata Only' completed successfully.[/bold green]\n")
            except Exception as e:
                console.log(f"[bold red]❌ ERROR IN STEP: '4. Extract Metadata Only'[/bold red]")
                log.error(f"Pipeline HALTED due to an error in Step 4: {e}", exc_info=True)
                sys.exit(1)
        else:
            log.info("Skipping Step 4 (Extract Metadata Only) as requested.")


        console.rule(f"[bold green]🎉 Pipeline Finished Successfully! 🎉[/bold green]")
        log.info(f"All output is available in the '{output_dir}' directory.")

        pipeline_completed = True

    except KeyboardInterrupt:
        log.warning("\n[bold yellow]Pipeline manually interrupted by user (Ctrl+C).[/bold yellow]")
        sys.exit(1)

    finally:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the full VnExpress crawler pipeline.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "-c", "--categories",
        required=True,
        nargs="+",
        help="One or more categories (e.g., 'the-gioi' or '1001002')"
    )
    parser.add_argument(
        "-p", "--pages",
        type=int,
        default=2,
        help="Number of pages to discover *per category* (or *per date range*)"
    )
    parser.add_argument(
        "-o", "--output",
        default="data",
        help="Main directory to save all output files (default: 'data')"
    )

    parser.add_argument(
        "-w", "--workers",
        type=int,
        default=5,
        help="Number of parallel workers for Step 1 and Step 3. Step 2 always uses 1 worker. (Default: 5)"
    )

    parser.add_argument(
        "-s", "--steps",
        type=int,
        nargs="+",
        choices=[1, 2, 3, 4],
        default=[1, 2, 3],
        help="Which step(s) to run:\n  1: Discover Articles\n  2: Process Comments (Deep Crawl)\n  3: Enrich Users\n  4: Extract Metadata Only (category, tags) - for existing data\n(default: 1 2 3)")

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
        "-b", "--browser",
        choices=["chrome", "firefox"],
        default="chrome",
        help="Browser to use for Step 2 (Selenium). (Default: chrome)"
    )

    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run the Selenium/Playwright browsers in headless mode"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable caching for all steps"
    )

    parser.add_argument(
        "--show-progress",
        action="store_true",
        help="Show individual progress bars for Step 1 and 3 (can be messy)"
    )

    parser.add_argument(
        "--use-tqdm",
        action="store_true",
        help="Use tqdm progress bars instead of rich (better for Kaggle)"
    )

    parsed_args = parser.parse_args()

    if parsed_args.workers < 1:
        log.error("Number of workers must be at least 1.")
        sys.exit(1)

    main(parsed_args)
