#!/usr/bin/env python3
"""
Metaculus Question Scraper (Legacy/Copied from consistency-forecasting)

Purpose:
    Legacy Metaculus scraper copied from https://github.com/dpaleka/consistency-forecasting
    Scrapes binary questions from Metaculus with date filtering and resolution criteria.
    Note: This is kept for reference; prefer using metaculus_new.py for new work.

Main Functions:
    - scrape_metaculus_binary(): Scrapes binary questions from Metaculus  
    - fetch_resolution_criteria(): Extracts resolution criteria from question page
    - decide_resolution_date(): Determines valid resolution dates
    - too_close_dates(): Filters questions resolved too quickly

Usage:
    python metaculus.py --start_date 20240101 --end_date 20241231 --num 1000
"""

import requests
import json
from bs4 import BeautifulSoup
import argparse
import datetime as dt
from tqdm import tqdm
from decide_dates_real_fq import decide_resolution_date, too_close_dates
from pathlib import Path


def normalize_date_string(date_str):
    """
    Normalizes a date string by ensuring it ends with 'Z' and removing milliseconds.
    Also handles the datetime conversion.

    :param date_str: The input date string.
    :return: Normalized datetime object or None if conversion fails.
    """
    try:
        if date_str is None:
            return None
        if "." in date_str:
            date_str = date_str.split(".")[0]
        if not date_str.endswith("Z"):
            date_str += "Z"
        return dt.datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%SZ")
    except Exception as e:
        print(f"Error normalizing date string: {date_str}. Error: {e}")
        return None


def fetch_resolution_criteria(question_url):
    """
    Fetches the Resolution Criteria from a Metaculus question page.

    :param question_url: URL of the Metaculus question.
    :return: The Resolution Criteria text or None if not found.
    """
    response = requests.get(question_url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")

        # Assuming the Resolution Criteria is contained within a specific HTML element
        # You'll need to inspect the page and adjust the selector accordingly
        resolution_criteria_element = soup.select_one(
            "div.resolution-criteria"
        )  # Adjust this selector

        if resolution_criteria_element:
            return resolution_criteria_element.text.strip()

    return None


def fetch_live_questions_with_dates(
    api_url, start_date=None, end_date=None, num_questions=500
):
    """
    Fetches questions and their close dates from the Metaculus API.
    :param api_url: Base URL of the Metaculus API.
    :param start_date: Start date in 'yyyymmdd' format.
    :param end_date: End date in 'yyyymmdd' format.
    :return: A list of tuples containing live question titles and their resolution dates.
    """
    # Load Metaculus token from file
    token_file = Path(__file__).parent / "metaculus_token.txt"
    if token_file.exists():
        TOKEN = token_file.read_text().strip()
    else:
        print(f"Warning: metaculus_token.txt not found at {token_file}")
        TOKEN = ""
    
    headers = {"User-Agent": "Mozilla/5.0"}
    headers["Authorization"] = f"Token {TOKEN}"
    questions_info = []
    total_questions = num_questions  # Total number of questions you want to fetch
    page_size = (
        100  # Number of questions per page. Adjust based on API's maximum allowed limit
    )
    page = 1

    seen_ids = set()
    while len(questions_info) < total_questions:
        print(
            f"Fetching page {page}; we have {len(questions_info)} of desired {total_questions} questions "
        )
        params = {
            "limit": page_size,
            "offset": (page - 1)
            * page_size,  # or 'offset': (page-1) * page_size if the API uses offset
            "resolve_time__gt": start_date,
            "resolve_time__lt": end_date,
            "close_time__gt": start_date,
            "close_time__lt": end_date,
        }
        print(f"{api_url}/questions")
        response = requests.get(f"{api_url}/questions", headers=headers, params=params)

        start_datetime = (
            dt.datetime.strptime(start_date, "%Y%m%d") if start_date else None
        )
        end_datetime = dt.datetime.strptime(end_date, "%Y%m%d") if end_date else None

        if response.status_code != 200:
            raise Exception(f"Failed to fetch the API: {api_url}")

        questions_data = response.json()

        if len(questions_data.get("results", [])) == 0:
            break

        for question in tqdm(questions_data.get("results", [])):
            question_type = question.get("possibilities", {}).get("type")
            if not question_type:
                question_type = question.get("type")

            # Remove this filter to allow both binary and multiple choice questions
            # if question_type != "binary":
            #     continue

            # Optional: You might want to skip questions that aren't either binary or multiple choice
            if question_type not in ["binary", "multiple_choice"]:
                continue

            print("\nURL:", question.get("url"))

            # only include resolution times either in range or expires past 30 days and within 10 years
            resolution_date = None

            close_date = normalize_date_string(question.get("close_time"))
            resolve_date = normalize_date_string(question.get("resolve_time"))
            publish_time = normalize_date_string(question.get("publish_time"))
            created_time = normalize_date_string(question.get("created_time"))

            if close_date is None or resolve_date is None or publish_time is None:
                continue  # Skip this question if any date conversion failed

            resolution_date = decide_resolution_date(
                close_date,
                resolve_date,
                min_date=start_datetime,
                max_date=end_datetime,
            )

            print("Resolution date:", resolution_date)

            created_date = publish_time
            print("Question created:", created_date)

            if too_close_dates(created_date, resolution_date):
                continue

            question_info = {
                "id": question.get("id"),
                "title": question.get("title"),
                "body": question.get(
                    "description"
                ),  # Assuming 'description' is the detailed text
                "question_type": question_type,
                "resolution_date": resolution_date.strftime("%Y-%m-%dT%H:%M:%SZ")
                if resolution_date
                else None,  # You might need to format this date
                "url": f"https://www.metaculus.com/questions/{question.get('id')}",
                "data_source": "metaculus",
                "created_date": created_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "metadata": {
                    "topics": question.get(
                        "tags", []
                    ),  # Assuming 'tags' can be used as topics
                    "api_url": f"https://www.metaculus.com/api2/questions/{question.get('id')}",
                    "market_prob": None,
                    "resolve_time": question.get("resolve_time"),
                    "close_time": question.get("close_time"),
                    "effected_close_time": question.get("effected_close_time"),
                    "created_time": question.get("created_time"),
                    "publish_time": question.get("publish_time"),
                },
                "resolution": question.get("resolution"),
            }

            # For multiple choice questions, add the choices to metadata
            if question_type == "multiple_choice":
                question_info["metadata"]["choices"] = question.get("options")

            if (question_info["id"] not in seen_ids) and (
                (question_info["resolution"] is None)
                or (round(float(question_info["resolution"])) != -2)
            ):
                questions_info.append(question_info)
                seen_ids.add(question_info["id"])
                # print(len(questions_info), 'total_qs')
                # print(page, 'page')
                # print(question_info['title'])
                # print(question_info['resolution_date'])
                # print(question_info['url'])
                # print(question_info['resolution'])
            if len(questions_info) >= total_questions:
                break

        page += 1

    print(f"Fetched {len(questions_info)} questions")
    return questions_info[
        :total_questions
    ]  # Ensure only the requested number of questions are returned


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch live questions from Metaculus API."
    )
    parser.add_argument("-start", type=str, help="Start date in yyyymmdd format.")
    parser.add_argument("-end", type=str, help="End date in yyyymmdd format.")
    parser.add_argument(
        "-num", type=int, default=500, help="Number of questions to fetch."
    )
    args = parser.parse_args()

    api_url = "https://www.metaculus.com/api2"

    write_dir = "metaculus"
    if not Path(write_dir).exists():
        Path(write_dir).mkdir()

    try:
        # Scrape the website
        data = fetch_live_questions_with_dates(api_url, args.start, args.end, args.num)

        # Convert the data to JSON and print
        print("total entries:", len(data))

        if args.start or args.end:
            s = "" if args.start is None else args.start
            e = "" if args.end is None else args.end
            with open(f"{write_dir}/metaculus_{s}_{e}.json", "w") as json_file:
                json.dump(data, json_file, indent=4)
        else:
            with open(f"{write_dir}/metaculus.json", "w") as json_file:
                json.dump(data, json_file, indent=4)

    except Exception as e:
        print(f"Error: {e}")
        raise e