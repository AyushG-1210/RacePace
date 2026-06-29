#!/usr/bin/env python3
import fastf1 as ff1
import pandas as pd
import os
import argparse


def main():
    parser = argparse.ArgumentParser(description='Load laps for a driver from FastF1 and save as pickle')
    parser.add_argument('--driver', default='pisatris', help='Driver name or abbreviation to match (case-insensitive)')
    parser.add_argument('--year', type=int, default=2025, help='Year of the event')
    parser.add_argument('--event', default='Zandvoort', help='Event name (e.g., Zandvoort)')
    parser.add_argument('--session', default='R', help='Session type (FP1, FP2, FP3, Q, R, S)')
    parser.add_argument('--cache', default='.ff1_cache', help='FastF1 cache directory')
    parser.add_argument('--out', default='pisatris_zandvoort_2025.pkl', help='Output pickle path')
    args = parser.parse_args()

    ff1.Cache.enable_cache(args.cache)

    print(f"Fetching session: {args.year} {args.event} {args.session}")
    session = ff1.get_session(args.year, args.event, args.session)
    session.load()

    laps = session.laps.reset_index(drop=True)

    # Match driver by case-insensitive containment in the `Driver` column
    driver_query = args.driver.lower()
    if 'Driver' not in laps.columns:
        print('Unexpected laps format; columns:', laps.columns.tolist())
        return

    mask = laps['Driver'].astype(str).str.lower().str.contains(driver_query)
    selected = laps[mask]

    if selected.empty:
        available = sorted(set(laps['Driver'].astype(str).unique()))
        print('No laps matched the driver query:', args.driver)
        print('Available driver codes:', available)
        return

    # Ensure output directory exists
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    selected.to_pickle(args.out)
    print(f'Saved {len(selected)} lap rows to {args.out}')


if __name__ == '__main__':
    main()
