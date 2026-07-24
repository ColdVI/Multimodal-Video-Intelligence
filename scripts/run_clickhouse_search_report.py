"""Yerel ClickHouse search-lab HTML + JSON raporunu uret."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from common import load_config
from reports.clickhouse_search_report import write_clickhouse_report


def main():
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    cfg = load_config()
    endpoint = f"http://{cfg['clickhouse']['host']}:{cfg['clickhouse']['port']}/"
    html_path, json_path = write_clickhouse_report(repo_root, endpoint)
    print(f"HTML rapor: {html_path}")
    print(f"JSON kanit: {json_path}")


if __name__ == "__main__":
    main()
