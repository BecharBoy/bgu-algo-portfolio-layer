import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def plot_refined_trades():
    df = pd.read_csv("detailed_trade_paths.csv")
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    for (event, ticker), data in df.groupby(['event', 'ticker']):
        plt.figure(figsize=(15, 8))

        # ציר X כהפרש ימים/שעות או תאריכים
        plt.plot(data['timestamp'], data['strategy_return_pct'] * 100, label='הכסף שלך (Strategy)', color='green', lw=2)
        plt.plot(data['timestamp'], data['stock_return_pct'] * 100, label=f'מחיר המניה ({ticker})', color='blue',
                 ls='--', alpha=0.6)
        plt.plot(data['timestamp'], data['spy_return_pct'] * 100, label='מדד השוק (SPY)', color='gray', alpha=0.4)

        # הוספת ציוני דרך עם תאריכים
        for _, row in data.iterrows():
            if row['is_formation_start']:
                plt.axvline(row['timestamp'], color='purple', alpha=0.3, label='תחילת שאילתה')
            if row['is_confidence_hit']:
                plt.axvline(row['timestamp'], color='orange', ls='--', label='כניסה (PM Confidence)')
            if row['is_official_event']:
                plt.axvline(row['timestamp'], color='red', lw=2, label='הכרזה רשמית')

        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        plt.title(f"ניתוח ארביטראז': {event} | {ticker}")
        plt.legend()
        plt.grid(True, alpha=0.2)
        plt.savefig(f"trade_graphs/{event}_{ticker}.png")
        plt.close()