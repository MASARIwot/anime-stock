"""Main entry point for Animetrics AI."""

import argparse
import sys


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Animetrics AI - Anime Stock Market Intelligence",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Dashboard command
    dash_parser = subparsers.add_parser("dashboard", help="Run the Streamlit dashboard")
    dash_parser.add_argument("--port", type=int, default=8501, help="Port to run on")
    
    # Collect command
    collect_parser = subparsers.add_parser("collect", help="Collect stock and news data")
    collect_parser.add_argument("--backfill", action="store_true", help="Fetch 2 years of data")
    collect_parser.add_argument("--skip-sentiment", action="store_true", help="Skip sentiment analysis")
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Run sentiment analysis")
    analyze_parser.add_argument("--date", type=str, help="Analyze specific date (YYYY-MM-DD)")
    
    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Generate predictions")
    predict_parser.add_argument("--ticker", type=str, help="Predict for specific ticker")
    
    args = parser.parse_args()
    
    if args.command == "dashboard":
        import subprocess
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            "src/anime_stock/dashboard/app.py",
            f"--server.port={args.port}",
            "--server.headless=true",
        ])
    
    elif args.command == "collect":
        from anime_stock.scripts import run_collection
        run_collection(
            backfill=args.backfill,
            skip_sentiment=args.skip_sentiment,
        )
    
    elif args.command == "analyze":
        from anime_stock.analysis.sentiment import main as analyze_main
        sys.argv = ["sentiment"]
        if args.date:
            sys.argv.extend(["--date", args.date])
        else:
            sys.argv.append("--all")
        analyze_main()
    
    elif args.command == "predict":
        from anime_stock.analysis.predictor import main as predict_main
        sys.argv = ["predictor"]
        if args.ticker:
            sys.argv.extend(["--ticker", args.ticker])
        predict_main()
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
