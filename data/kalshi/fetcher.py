#!/usr/bin/env python3
"""
Fetch resolved (settled) Kalshi MCQ events from the last ~3 months and save as JSONL,
with LOTS of progress output + incremental saving so it never feels stuck.

Outputs:
  - JSONL file (one line per MCQ event) with resolution criteria, options, winner, timestamps
  - Raw snapshots of each markets page (for auditing/retry)
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone, timedelta
from collections import defaultdict
from typing import Dict, Any, Iterable, List, Optional

import requests

BASE = "https://api.elections.kalshi.com/trade-api/v2"  # covers ALL markets, not just elections

# ---------- Robust HTTP helpers ----------

def _session(retries: int = 5, timeout: float = 20.0) -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": "kalshi-mcq-jsonl/1.0"})
    s.request_timeout = timeout
    s.max_retries = retries
    return s

def _get_json(
    s: requests.Session,
    url: str,
    params: Optional[Dict[str, Any]] = None,
    *,
    timeout: Optional[float] = None,
    max_retries: Optional[int] = None,
    backoff: float = 0.7,
) -> Dict[str, Any]:
    """GET with retries/backoff + progress-safe timeouts."""
    retries = s.max_retries if max_retries is None else max_retries
    t = getattr(s, "request_timeout", None) if timeout is None else timeout
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            resp = s.get(url, params=params, timeout=t)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            last_err = e
            print(f"[warn] GET {url} (attempt {attempt}/{retries}) failed: {e}", flush=True)
            # brief progressive backoff
            time.sleep(backoff * attempt)
    raise RuntimeError(f"GET {url} failed after {retries} attempts: {last_err}")

def unix_ts(dt: datetime) -> int:
    return int(dt.replace(tzinfo=timezone.utc).timestamp())

# ---------- Data pipeline ----------

def fetch_settled_markets_last3m(
    s: requests.Session,
    out_dir: str,
    days: int = 92,
    page_limit: int = 1000,
) -> List[Dict[str, Any]]:
    """Fetch settled markets that closed within ~last `days`. Save each page to disk."""
    os.makedirs(out_dir, exist_ok=True)
    now_utc = datetime.now(timezone.utc)
    min_close = now_utc - timedelta(days=days)
    params = {
        "status": "settled",
        "min_close_ts": unix_ts(min_close),
        "limit": page_limit,
    }
    cursor = None
    all_markets = []
    page = 0

    print(f"[info] Starting GetMarkets with status=settled, min_close_ts={params['min_close_ts']}, limit={page_limit}", flush=True)
    print(f"[info] Time window: {min_close.isoformat()} — {now_utc.isoformat()}", flush=True)

    while True:
        q = dict(params)
        if cursor:
            q["cursor"] = cursor

        data = _get_json(s, f"{BASE}/markets", q)
        markets = data.get("markets", [])
        cursor = data.get("cursor")
        all_markets.extend(markets)
        page += 1

        # Save raw page snapshot
        snap_path = os.path.join(out_dir, f"markets_page_{page}.json")
        with open(snap_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        print(f"[page {page}] got {len(markets)} markets; total so far: {len(all_markets)}  (saved {snap_path})", flush=True)

        if not cursor:
            break

        # tiny pause to be polite
        time.sleep(0.2)

    print(f"[done] Markets fetched: {len(all_markets)}", flush=True)
    return all_markets

def get_event(s: requests.Session, event_ticker: str) -> Dict[str, Any]:
    return _get_json(
        s,
        f"{BASE}/events/{event_ticker}",
        params={"with_nested_markets": "true"},
    ).get("event", {})

def is_mcq_event(ev: Dict[str, Any]) -> bool:
    # Kalshi MCQ indicator = mutually_exclusive == true, and typically >2 markets
    if not ev.get("mutually_exclusive"):
        return False
    markets = ev.get("markets", [])
    return len(markets) > 2

def choose_times(markets: List[Dict[str, Any]]):
    # created_date: earliest open_time
    opens = [m.get("open_time") for m in markets if m.get("open_time")]
    created_date = min(opens) if opens else None
    # resolution_date proxy: latest latest_expiration_time / expiration_time / close_time
    exps = [
        (m.get("latest_expiration_time") or m.get("expiration_time") or m.get("close_time"))
        for m in markets
    ]
    exps = [t for t in exps if t]
    resolution_date = max(exps) if exps else None
    # scheduled/actual close: we take the last close_time as a rough stand-in (public API has no settled_at)
    last_close = max((m.get("close_time") for m in markets if m.get("close_time")), default=None)
    return created_date, resolution_date, last_close

def extract_rules(markets: List[Dict[str, Any]]):
    r_primary = next((m.get("rules_primary") for m in markets if m.get("rules_primary")), None)
    r_secondary = next((m.get("rules_secondary") for m in markets if m.get("rules_secondary")), None)
    return r_primary, r_secondary

def build_jsonl_record(ev: Dict[str, Any], mkts_for_event: List[Dict[str, Any]]) -> Dict[str, Any]:
    nested = ev.get("markets", [])

    # options + winner
    options = []
    winner_title = None
    winning_ticker = None
    for m in nested:
        title = m.get("title") or m.get("subtitle") or m.get("ticker")
        options.append(title)
        # prefer settled slice result when available
        snap = next((sm for sm in mkts_for_event if sm["ticker"] == m["ticker"]), m)
        if snap.get("result") == "yes":
            winner_title = title
            winning_ticker = m.get("ticker")

    created_date, resolution_date, last_close = choose_times(nested)
    r_primary, r_secondary = extract_rules(nested)
    resolution_criteria = None
    if r_primary or r_secondary:
        resolution_criteria = (r_primary or "") + (("\n\n" + r_secondary) if r_secondary else "")

    body_parts = []
    if resolution_criteria:
        body_parts.append(resolution_criteria)
    if ev.get("category"):
        body_parts.append(f"_Category_: {ev['category']}")
    body = "\n\n".join(body_parts) if body_parts else None

    rec = {
        "id": None,  # Kalshi doesn't expose numeric event IDs publicly
        "title": ev.get("title") or ev.get("event_ticker"),
        "body": body,
        "question_type": "multiple_choice",
        "resolution_date": resolution_date,  # proxy (public API lacks settled_at)
        "url": f"https://kalshi.com/markets/{ev.get('event_ticker')}",
        "data_source": "kalshi",
        "created_date": created_date,
        "metadata": {
            "published_at": None,                # not available in public API
            "open_time": created_date,
            "scheduled_resolve_time": None,      # not available
            "actual_resolve_time": resolution_date,
            "scheduled_close_time": last_close,  # best-available close proxy
            "actual_close_time": last_close,
            "nr_forecasters": None,              # not applicable for Kalshi
            "possibilities": None,
            "options": options,
            "winning_option_ticker": winning_ticker,
            "resolution_criteria": resolution_criteria,
            "fine_print": None,                  # not available separately
            "post_id": None,                     # not applicable
            "category": ev.get("category"),
            "event_ticker": ev.get("event_ticker"),
        },
        "resolution": winner_title,
    }
    return rec

def main():
    ap = argparse.ArgumentParser(description="Export settled Kalshi MCQ events (last ~3 months) to JSONL with progress & checkpoints.")
    ap.add_argument("--days", type=int, default=92, help="Window size in days (default: 92)")
    ap.add_argument("--page-limit", type=int, default=1000, help="GetMarkets page size (1..1000)")
    ap.add_argument("--out-jsonl", default="kalshi_mcq_resolved_last3m.jsonl", help="Output JSONL file")
    ap.add_argument("--out-dir", default="kalshi_runs", help="Directory to save raw pages & logs")
    ap.add_argument("--max-retries", type=int, default=5, help="HTTP retries per request")
    ap.add_argument("--timeout", type=float, default=20.0, help="HTTP timeout (seconds)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    s = _session(retries=args.max_retries, timeout=args.timeout)

    # 1) Fetch all settled markets (with progress and page snapshots)
    markets = fetch_settled_markets_last3m(s, out_dir=args.out_dir, days=args.days, page_limit=args.page_limit)

    # 2) Group by event
    by_event = defaultdict(list)
    for m in markets:
        by_event[m["event_ticker"]].append(m)

    print(f"[info] Unique events: {len(by_event)}", flush=True)

    # 3) Stream JSONL (write every event immediately)
    out_path = os.path.join(args.out_dir, args.out_jsonl) if not os.path.isabs(args.out_jsonl) else args.out_jsonl
    n_events = len(by_event)
    n_mcq = 0
    written = 0

    # resume support: if file exists, load seen event_tickers
    seen = set()
    if os.path.exists(out_path):
        print(f"[resume] Existing JSONL found at {out_path}. Will append new records only.", flush=True)
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    et = obj.get("metadata", {}).get("event_ticker")
                    if et:
                        seen.add(et)
                except Exception:
                    pass
        print(f"[resume] Already have {len(seen)} events in JSONL.", flush=True)

    with open(out_path, "a", encoding="utf-8") as out_f:
        for idx, (et, mkts_for_event) in enumerate(by_event.items(), start=1):
            if et in seen:
                if idx % 25 == 0:
                    print(f"[skip {idx}/{n_events}] {et} already written", flush=True)
                continue

            # Fetch the event with nested markets & verbose progress
            t0 = time.time()
            try:
                ev = get_event(s, et)
            except Exception as e:
                print(f"[error] get_event({et}) failed: {e} (continuing)", flush=True)
                continue
            dt = time.time() - t0

            if idx % 10 == 0 or idx <= 5:
                print(f"[event {idx}/{n_events}] {et} fetched in {dt:.2f}s; markets={len(ev.get('markets', []))}", flush=True)

            if not is_mcq_event(ev):
                # occasional heartbeat
                if idx % 50 == 0:
                    print(f"[info] processed {idx}/{n_events}; MCQ so far: {n_mcq}", flush=True)
                continue

            # Build JSONL record and write immediately
            rec = build_jsonl_record(ev, mkts_for_event)
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_f.flush()  # write-through so you always see progress on disk
            os.fsync(out_f.fileno())
            n_mcq += 1
            written += 1

            if written % 10 == 0 or written <= 3:
                print(f"[write] MCQ written #{written}: {et}", flush=True)

            # tiny pause to be polite
            time.sleep(0.1)

    print(f"[summary] Events scanned: {n_events}; MCQ written: {n_mcq}; JSONL: {out_path}", flush=True)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[abort] Interrupted by user. Partial results are saved.", flush=True)
        sys.exit(130)
