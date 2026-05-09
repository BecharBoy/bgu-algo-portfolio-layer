from datetime import UTC, datetime, timedelta


def utc_dt(
        year: int,
        month: int,
        day: int,
        hour: int = 0,
        minute: int = 0,
        second: int = 0,
) -> datetime:
    return datetime(year, month, day, hour, minute, second, tzinfo=UTC)


EVENT_WINDOW_BEFORE = timedelta(days=5)
EVENT_WINDOW_AFTER = timedelta(days=3)


def event_window(event_ts: datetime) -> tuple[datetime, datetime]:
    return event_ts - EVENT_WINDOW_BEFORE, event_ts + EVENT_WINDOW_AFTER


EVENT_SPECS = {
    # ====== 2024 EVENTS ======

    "cpi_hot_april": {
        "event_ts": utc_dt(2024, 5, 15, 8, 30, 0),
        "event_type": "hawkish",
        "poly": {
            "condition_id": "0x8cd0bb2e47841c153d2cf99c286e7f03a3b8b0e2ecd4bcbed8ec8c8c1a5fd7ff",
            "formation_start": utc_dt(2024, 3, 20, 0, 0, 0),
            "formation_end": utc_dt(2024, 4, 11, 23, 59, 59),
            "formation_csv": "polymarket_cpi_april.csv",
            "event_csv": "polymarket_cpi_april_window.csv",
            "fidelity": 60,
        },
        "yahoo": {
            "hourly_csv": "hourly_cpi_hot_april.csv",
        },
    },

    "cpi_cool_july": {
        "event_ts": utc_dt(2024, 7, 11, 8, 30, 0),
        "event_type": "dovish",
        "poly": None,
        "yahoo": {
            "hourly_csv": "hourly_cpi_cool_july.csv",
        },
    },

    "jobs_aug_unemployment": {
        "event_ts": utc_dt(2024, 9, 6, 8, 30, 0),
        "event_type": "dovish",
        "poly": {
            "condition_id": "0x632e67a2d2855b11c7d3854c8634558f2e1e8d6e72e5e64b07c60f11c5ae9370",
            "formation_start": utc_dt(2024, 9, 4, 0, 0, 0),
            "formation_end": utc_dt(2024, 9, 7, 23, 59, 59),
            "formation_csv": "polymarket_jobs_august.csv",
            "event_csv": "polymarket_jobs_august_window.csv",
            "fidelity": 60,
        },
        "yahoo": {
            "hourly_csv": "hourly_jobs_aug_unemployment.csv",
        },
    },

    "fomc_cut_sep": {
        "event_ts": utc_dt(2024, 9, 18, 14, 0, 0),
        "event_type": "dovish",
        "poly": {
            "condition_id": "0xdd22472e552920b8438158ea7238bfadfa4f736aa4cee91a6b86c39ead110917",
            "formation_start": utc_dt(2024, 8, 1, 0, 0, 0),
            "formation_end": utc_dt(2024, 9, 21, 23, 59, 59),
            "formation_csv": "polymarket_fed_sep2024.csv",
            "event_csv": "polymarket_fed_sep2024_window.csv",
            "fidelity": 60,
        },
        "yahoo": {
            "hourly_csv": "hourly_fomc_cut_sep.csv",
        },
    },

    # ====== 2025 EVENTS ======

    "fomc_jan_2025": {
        "event_ts": utc_dt(2025, 1, 29, 14, 0, 0),
        "event_type": "neutral",
        "poly": {
            "condition_id": "0x64123306a517e078fa636231a9cc9339a46bcfe3fadf62c92fdb031881c5d0d8",
            "formation_start": utc_dt(2025, 1, 1, 0, 0, 0),
            "formation_end": utc_dt(2025, 1, 31, 23, 59, 59),
            "formation_csv": "polymarket_fomc_jan_2025.csv",
            "event_csv": "polymarket_fomc_jan_2025_window.csv",
            "fidelity": 60,
        },
        "yahoo": {
            "hourly_csv": "hourly_fomc_jan_2025.csv",
        },
    },

    "cpi_mar_2025": {
        "event_ts": utc_dt(2025, 4, 10, 8, 30, 0),
        "event_type": "dovish",
        "poly": {
            "condition_id": "0xe85ecaffd8541f8c74088f671cbdfb97eaaf91f92e1194fb673e8abc3974c75c",
            "formation_start": utc_dt(2025, 3, 20, 0, 0, 0),
            "formation_end": utc_dt(2025, 4, 11, 23, 59, 59),
            "formation_csv": "polymarket_cpi_mar_2025.csv",
            "event_csv": "polymarket_cpi_mar_2025_window.csv",
            "fidelity": 60,
        },
        "yahoo": {
            "hourly_csv": "hourly_cpi_mar_2025.csv",
        },
    },

    "nfp_may_2025": {
        "event_ts": utc_dt(2025, 6, 6, 8, 30, 0),
        "event_type": "neutral",
        "poly": {
            "condition_id": "0xd079e212d412aba9cb264874514b5de9821fbb7c9ed1df22fef9894a9774212f",
            "formation_start": utc_dt(2025, 5, 20, 0, 0, 0),
            "formation_end": utc_dt(2025, 6, 7, 23, 59, 59),
            "formation_csv": "polymarket_nfp_may_2025.csv",
            "event_csv": "polymarket_nfp_may_2025_window.csv",
            "fidelity": 60,
        },
        "yahoo": {
            "hourly_csv": "hourly_nfp_may_2025.csv",
        },
    },

    "fomc_jul_2025": {
        "event_ts": utc_dt(2025, 7, 30, 14, 0, 0),
        "event_type": "neutral",
        "poly": {
            "condition_id": "0x0d09f8cb4ac601074c02828328a893fdab030ece3ca682faaef89d82c43daec2",
            "formation_start": utc_dt(2025, 7, 1, 0, 0, 0),
            "formation_end": utc_dt(2025, 7, 31, 23, 59, 59),
            "formation_csv": "polymarket_fomc_jul_2025.csv",
            "event_csv": "polymarket_fomc_jul_2025_window.csv",
            "fidelity": 60,
        },
        "yahoo": {
            "hourly_csv": "hourly_fomc_jul_2025.csv",
        },
    },
}