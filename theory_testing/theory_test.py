import pandas as pd
import numpy as np
from datetime import timedelta
import os
from event_specs import EVENT_SPECS

# --- Configuration ---
ALLOCATION_PER_STOCK = 5000.0
POST_EVENT_WINDOW = timedelta(hours=72)
CONFIDENCE_HIGH = 0.55
CONFIDENCE_LOW = 0.45

OUTPUT_PATHS_CSV = "detailed_trade_paths.csv"

STOCK_BIAS = {
    'dovish': {
        # כשהריבית יורדת בגלל פחד ממיתון, קונים חופי מבטחים (Risk-Off)
        'TLT': 1,  # אג"ח ממשלתי ל-20 שנה (המרוויח הראשי)
        'GLD': 1,  # תעודת סל על זהב
        'KO': 1,   # קוקה קולה (הגנתי)
        'PEP': 1,  # פפסיקו (הגנתי)
        'PG': 1,   # פרוקטר אנד גמבל (הגנתי)
        'XLU': 1,  # סקטור התשתיות והחשמל (הגנתי ורגיש לריבית)
        # ועדיין עושים שורט על בנקים כי הם נפגעים מריבית נמוכה ואבטלה
        'JPM': -1, 'BAC': -1, 'WFC': -1
    },
    'hawkish': {
        # כשהריבית נשארת גבוהה בגלל כלכלה חזקה (אינפלציה)
        # עושים שורט על חברות שתלויות בהלוואות ובכסף זול
        'UPST': -1, 'CVNA': -1, 'RUN': -1, 'ENPH': -1, 'TSLA': -1, 'RGTI': -1,
        'DHI': -1, 'LEN': -1, 'VNQ': -1,
        # וקונים בנקים כי הם מרוויחים מריבית גבוהה
        'JPM': 1, 'BAC': 1, 'WFC': 1
    }
}


def parse_timestamp(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, utc=True, errors="coerce")
    return parsed.dt.tz_convert("UTC").dt.tz_localize(None).astype("datetime64[ns]")


def get_pm_confidence_time(pm_csv_path: str) -> pd.Timestamp | None:
    if not os.path.exists(pm_csv_path): return None
    df = pd.read_csv(pm_csv_path)
    col_lower = {c: c.lower() for c in df.columns}
    time_col = next((c for c in df.columns if col_lower[c] in ['timestamp', 'date', 'time', 'datetime']), None)
    prob_col = next((c for c in df.columns if col_lower[c] in ['prob_yes', 'price', 'close', 'last', 'yes_price']),
                    None)

    if not time_col or not prob_col: return None
    df['parsed_time'] = parse_timestamp(df[time_col])
    df = df.dropna(subset=['parsed_time', prob_col]).sort_values('parsed_time')
    df[prob_col] = pd.to_numeric(df[prob_col], errors='coerce')

    confident_rows = df[(df[prob_col] >= CONFIDENCE_HIGH) | (df[prob_col] <= CONFIDENCE_LOW)]
    if confident_rows.empty: return None
    return confident_rows.iloc[0]['parsed_time']


def generate_trade_paths():
    all_paths = []
    dynamic_event_types = {"fomc_jan_2025": "hawkish", "nfp_may_2025": "dovish", "fomc_jul_2025": "dovish"}

    for event_name, specs in EVENT_SPECS.items():
        print(f"Processing event: {event_name}...")

        # 1. פה הייתה חסרה השורה שהגדירה את סוג האירוע!
        event_type = dynamic_event_types.get(event_name, specs.get('event_type'))

        if not specs.get('poly') or not specs['poly'].get('event_csv'): continue
        if event_type not in STOCK_BIAS: continue

        pm_csv = specs['poly']['event_csv']
        yahoo_csv = specs['yahoo']['hourly_csv']

        formation_start = parse_timestamp(pd.Series([specs['poly']['formation_start']])).iloc[0]
        event_ts = parse_timestamp(pd.Series([specs['event_ts']])).iloc[0]

        t_conf = get_pm_confidence_time(pm_csv)
        print(f"    [Debug] PM Confidence Time: {t_conf} | Official Event: {event_ts}")
        if t_conf is None or t_conf >= event_ts:
            print(f"  -> Skipping: PM confidence not reached or not a leading event.")
            continue

        if not os.path.exists(yahoo_csv): continue

        market_df = pd.read_csv(yahoo_csv)
        if 'Datetime' not in market_df.columns: continue
        market_df['parsed_time'] = parse_timestamp(market_df['Datetime'])
        market_df = market_df.sort_values('parsed_time')

        # אנחנו לוקחים נתונים מתחילת השמועה (Formation) ועד 72 שעות אחרי ההכרזה
        t_end = event_ts + POST_EVENT_WINDOW
        window_df = market_df[
            (market_df['parsed_time'] >= formation_start) & (market_df['parsed_time'] <= t_end)].copy()
        if window_df.empty: continue

        for ticker, direction in STOCK_BIAS[event_type].items():
            if ticker not in window_df.columns or 'SPY' not in window_df.columns: continue

            # מוצאים את מחיר הכניסה המדויק ברגע ש-PM הגיע לוודאות
            entry_data = window_df[window_df['parsed_time'] >= t_conf]
            if entry_data.empty: continue

            entry_price = entry_data[ticker].iloc[0]
            spy_entry_price = entry_data['SPY'].iloc[0]
            entry_time = entry_data['parsed_time'].iloc[0]

            if pd.isna(entry_price) or entry_price == 0: continue

            valid_mask = window_df[ticker].notna() & window_df['SPY'].notna()
            clean_times = window_df['parsed_time'][valid_mask].reset_index(drop=True)
            clean_prices = window_df[ticker][valid_mask].reset_index(drop=True)
            spy_prices = window_df['SPY'][valid_mask].reset_index(drop=True)

            # חישוב התשואות (מנורמל ביחס לנקודת הכניסה t_conf)
            hours_from_entry = (clean_times - entry_time).dt.total_seconds() / 3600.0
            stock_return_pct = (clean_prices / entry_price) - 1.0
            spy_return_pct = (spy_prices / spy_entry_price) - 1.0
            strategy_return_pct = stock_return_pct * direction

            shares = ALLOCATION_PER_STOCK / entry_price
            pnl_usd = shares * (clean_prices - entry_price) * direction

            path_df = pd.DataFrame({
                'event': event_name,
                'event_type': event_type,
                'ticker': ticker,
                'trade_direction': "LONG" if direction == 1 else "SHORT",
                'timestamp': clean_times,
                'hours_from_entry': hours_from_entry,
                'stock_price': clean_prices,
                'stock_return_pct': stock_return_pct,
                'spy_return_pct': spy_return_pct,
                'strategy_return_pct': strategy_return_pct,
                'pnl_usd': pnl_usd
            })

            # סימון בוליאני של נקודות הזמן החשובות לגרף
            path_df['is_formation_start'] = clean_times == formation_start
            path_df['is_confidence_hit'] = clean_times == entry_time
            path_df['is_official_event'] = clean_times == event_ts

            all_paths.append(path_df)

    if all_paths:
        final_df = pd.concat(all_paths, ignore_index=True)
        final_df.to_csv(OUTPUT_PATHS_CSV, index=False)
        print(f"\nSuccess! Detailed timeseries paths saved to '{OUTPUT_PATHS_CSV}'.")
        print(f"Total data points: {len(final_df)}")
    else:
        print("\nNo data generated.")


if __name__ == "__main__":
    generate_trade_paths()