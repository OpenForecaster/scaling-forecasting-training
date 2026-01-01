#!/usr/bin/env python3
"""
Manifold Markets Data Scraper and Processor

Purpose:
    Scrapes binary forecasting questions from Manifold Markets API and filters them
    by resolution dates using the decide_dates_real_fq logic.

Main Functions:
    - scrape_manifold_markets(): Fetches market data from Manifold API
    - epoch_to_datetime(): Converts epoch timestamps to datetime objects
    - decide_resolution_date(): Determines appropriate resolution dates
    - too_close_dates(): Filters out questions resolved too quickly

Output Format:
    JSONL with standardized forecasting question format including question text,
    resolution date, probabilities, and metadata.

Usage:
    python manifold.py --start_date 20240101 --end_date 20241231 --num 1000
"""

import requests
import json
import datetime as dt
import argparse
from tqdm import tqdm
from decide_dates_real_fq import decide_resolution_date, too_close_dates
from pathlib import Path


def epoch_to_datetime(epoch):
    if epoch is None:
        print("Epoch is None")
        return None
    timestamp = epoch // 1000
    if dt.datetime(2299, 12, 31).timestamp() > timestamp > 0:
        return dt.datetime.fromtimestamp(timestamp, dt.UTC)
    return None


def scrape_manifold_markets(
    api_url, start_date=None, end_date=None, max_pages=1, num_questions=500
):
    """
    Fetches market data from the Manifold Markets API and extracts relevant information.
    :param api_url: URL of the Manifold Markets API endpoint.
    :param start_date: Start date in 'yyyymmdd' format.
    :param end_date: End date in 'yyyymmdd' format.
    :param max_pages: Maximum number of pages to fetch from the API.
    :param num_questions: Number of questions to fetch.
    :return: A dictionary containing the extracted market names, their URLs, and other relevant data.
    """
    # Convert start_date and end_date to datetime objects for comparison
    start_datetime = (
        dt.datetime.strptime(start_date, "%Y%m%d").replace(tzinfo=dt.UTC)
        if start_date
        else None
    )
    end_datetime = (
        dt.datetime.strptime(end_date, "%Y%m%d").replace(tzinfo=dt.UTC)
        if end_date
        else None
    )

    headers = {"User-Agent": "Mozilla/5.0"}
    questions_info = []
    total_questions = num_questions  # Total number of questions you want to fetch
    page_size = 1000  # Number of questions per page. Adjust based on API's maximum allowed limit

    seen_ids = set()
    before_id = None

    print("Fetching data from Manifold Markets API...")
    for p in range(max_pages):
        print(f"Fetching page {p + 1} of {max_pages}")
        if len(questions_info) >= total_questions:
            break

        params = {"limit": page_size, "before": before_id}
        # Fetch the content from the API URL
        response = requests.get(api_url, headers=headers, params=params)
        if response.status_code != 200:
            raise Exception(f"Failed to fetch the API data: {api_url}")
        # Parse the JSON content
        data = response.json()

        print(f"\nFetched page {p + 1}")
        print(f"Number of markets in this page: {len(data)}")

        # Extract relevant data from each market
        for market in tqdm(data, desc="Markets"):
            if len(questions_info) >= total_questions:
                break

            if market.get("outcomeType", "").lower() != "binary":
                continue

            close_time = epoch_to_datetime(market.get("closeTime"))
            resolution_time = epoch_to_datetime(market.get("resolutionTime"))

            url = market.get("url", "").lower()
            print(f"\nURL: {url}")

            resolution_date = decide_resolution_date(
                close_date=close_time,
                resolve_date=resolution_time,
                min_date=start_datetime,
                max_date=end_datetime,
            )
            if resolution_date is None:
                continue

            print(f"Resolution date: {resolution_date}")

            created_time = epoch_to_datetime(market.get("createdTime"))

            created_date = created_time
            print(f"Created time: {created_time}")

            if too_close_dates(created_date, resolution_date):
                continue

            market_info = {
                "id": market["id"],
                "title": market["question"],
                "body": market.get("description", "N/A"),
                "question_type": market.get("outcomeType", "").lower(),
                "resolution_date": resolution_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "url": market.get("url", "").lower(),
                "data_source": "manifold",
                "created_date": created_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "metadata": {
                    "topics": market.get("tags", []),
                    "close_time": str(epoch_to_datetime(market.get("closeTime"))),
                    "resolve_time": str(
                        epoch_to_datetime(market.get("resolutionTime"))
                    ),
                    "last_updated_time": str(
                        epoch_to_datetime(market.get("lastUpdatedTime"))
                    ),
                    "created_time": str(epoch_to_datetime(market.get("createdTime"))),
                    "last_bet_time": str(epoch_to_datetime(market.get("lastBetTime"))),
                },
                "resolution": market.get("isResolved", None),
            }

            market_info["metadata"][
                "api_url"
            ] = "https://api.manifold.markets/v0/slug/{}".format(
                market_info["url"].split("/")[-1]
            )
            market_info["metadata"]["market_prob"] = None

            if not market_info["resolution"]:
                market_info["resolution"] = None

            else:
                if market["resolution"].lower() == "yes":
                    market_info["resolution"] = True
                elif market["resolution"].lower() == "no":
                    market_info["resolution"] = False
                else:
                    market_info["resolution"] = None

            questions_info.append(market_info)

        before_id = market["id"]
        print(f"Total questions collected so far: {len(questions_info)}")

    print(f"\nFinal number of questions collected: {len(questions_info)}")
    return questions_info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch live questions from Manifold API."
    )
    parser.add_argument("-start", type=str, help="Start date in yyyymmdd format.")
    parser.add_argument("-end", type=str, help="End date in yyyymmdd format.")
    parser.add_argument(
        "-pages", type=str, help="Max pages of 1000 entries to go through.", default=10
    )
    parser.add_argument(
        "-num", type=int, help="Number of questions to fetch.", default=500
    )
    args = parser.parse_args()

    api_url = "https://api.manifold.markets/v0/markets"

    try:
        # Scrape the Manifold Markets
        data = scrape_manifold_markets(
            api_url, args.start, args.end, args.pages, args.num
        )

        print("Total entries:", len(data))
        # Convert the data to JSON and print

        write_dir = "manifold"
        if not Path(write_dir).exists():
            Path(write_dir).mkdir()

        if args.start or args.end:
            s = "" if args.start is None else args.start
            e = "" if args.end is None else args.end
            with open(f"{write_dir}/manifold_{s}_{e}.json", "w") as json_file:
                json.dump(data, json_file, indent=4)
            print(f"Data saved to {write_dir}/manifold_{s}_{e}.json")
        else:
            with open(f"{write_dir}/manifold.json", "w") as json_file:
                json.dump(data, json_file, indent=4)
            print(f"Data saved to {write_dir}/manifold.json")

    except Exception as e:
        print(f"Error: {e}")
        raise e