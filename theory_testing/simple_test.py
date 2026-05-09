"""
Smart Market Selector - ממיין ובוחר את המרקטים הכי טובים מתוך אלפי תוצאות
"""

from polymarket_apis import PolymarketGammaClient
from datetime import datetime, UTC
import re


def normalize_text(s: str) -> str:
    """נורמליזציה של טקסט לסינון"""
    s = (s or "").lower()
    s = s.replace("≥", ">=").replace("≤", "<=").replace("pt", ".")
    s = re.sub(r"[^a-z0-9<>=%\.\-\s]", " ", s)
    return " ".join(s.split())


def score_fomc_market(question: str, month: str, year: int) -> tuple:
    """
    ציון למרקט FOMC
    Returns: (is_valid, score, reason)
    """
    q = normalize_text(question)

    # חובה: חייב להכיל fed/fomc
    if "fed" not in q and "fomc" not in q:
        return (False, 0, "no fed/fomc")

    # חובה: חייב להכיל rate/cut/meeting
    if not any(w in q for w in ["rate", "cut", "meeting", "decrease", "increase"]):
        return (False, 0, "no rate action")

    # פסול: Powell speech/press conference (לא decision)
    if any(w in q for w in ["powell say", "press conference", "speech", "mention"]):
        return (False, 0, "powell speech")

    # פסול: חודשים/שנים לא רלוונטיים
    wrong_years = [str(y) for y in range(2020, 2030) if y != year]
    if any(wy in q for wy in wrong_years):
        return (False, 0, f"wrong year")

    score = 0

    # בונוס: שם החודש מופיע
    if month.lower() in q:
        score += 20

    # בונוס: "by [date]" format (זה usually המרקט הראשי)
    if "by" in q and any(m in q for m in ["january", "february", "march", "april", "may", "june", "july", "august", "september"]):
        score += 30

    # בונוס: "50 bps" / "25 bps" / "decrease" (specific magnitude)
    if any(w in q for w in ["50 bps", "25 bps", "decrease", "increase"]):
        score += 15

    # בונוס: "meeting" (decision market)
    if "meeting" in q:
        score += 10

    # עונש: "or more" / "or less" (range markets, פחות מדויקים)
    if "or more" in q or "or less" in q:
        score -= 5

    return (True, score, "valid")


def score_cpi_market(question: str, month: str, year: int) -> tuple:
    """ציון למרקט CPI"""
    q = normalize_text(question)

    # חובה: inflation/cpi
    if "inflation" not in q and "cpi" not in q:
        return (False, 0, "no inflation/cpi")

    # פסול: שנים לא רלוונטיות
    wrong_years = [str(y) for y in range(2020, 2030) if y != year]
    if any(wy in q for wy in wrong_years):
        return (False, 0, f"wrong year")

    score = 0

    # בונוס: month-to-month format (הכי מדויק)
    # e.g., "from March to April 2025"
    if "from" in q and "to" in q:
        score += 30

    # בונוס: שם החודש
    if month.lower() in q:
        score += 20

    # בונוס: ספציפי לגבי magnitude (>0.3%, <0.2%, etc.)
    if any(op in q for op in [">", "<", "0.1", "0.2", "0.3", "0.4", "0.5"]):
        score += 15

    # עונש: "year over year" (לא month-over-month)
    if "year over year" in q or "yoy" in q or "annual" in q:
        score -= 10

    return (True, score, "valid")


def score_nfp_market(question: str, month: str, year: int) -> tuple:
    """ציון למרקט NFP/Jobs"""
    q = normalize_text(question)

    # חובה: unemployment/jobs/payroll/nfp
    if not any(w in q for w in ["unemployment", "jobs", "payroll", "nfp"]):
        return (False, 0, "no jobs keyword")

    # פסול: Powell/Jackson Hole
    if any(w in q for w in ["powell", "jackson hole", "speech"]):
        return (False, 0, "powell speech")

    # פסול: שנים לא רלוונטיות
    wrong_years = [str(y) for y in range(2020, 2030) if y != year]
    if any(wy in q for wy in wrong_years):
        return (False, 0, f"wrong year")

    score = 0

    # בונוס: שם החודש
    if month.lower() in q:
        score += 20

    # בונוס: unemployment rate (לא payroll count)
    if "unemployment" in q and "rate" in q:
        score += 15

    # בונוס: threshold format ("over 4.2%", "above 4.0%")
    if any(w in q for w in ["over", "above", "below", "under"]) and any(c.isdigit() for c in q):
        score += 20

    # עונש: "jobs added" (payroll count, לא rate)
    if "jobs added" in q or "nonfarm payroll" in q or "payrolls" in q:
        score -= 10

    return (True, score, "valid")


def select_best_markets(all_markets: dict, score_func, min_volume: float = 5000) -> list:
    """
    בוחר את המרקטים הכי טובים
    all_markets: {cid: (vol, question, cid, month)}
    Returns: [(vol, question, cid, month, score)]
    """
    scored = []

    for cid, (vol, question, _, month) in all_markets.items():
        if vol < min_volume:
            continue

        is_valid, score, reason = score_func(question, month, year=2025)  # year will be overridden

        if is_valid and score > 0:
            scored.append((vol, question, cid, month, score))

    # מיין לפי: score (desc), volume (desc)
    scored.sort(key=lambda x: (x[4], x[0]), reverse=True)

    return scored


def deduplicate_markets(scored_markets: list, max_per_month: int = 2) -> list:
    """
    מסיר duplicates - מקסימום N markets לכל חודש
    """
    month_counts = {}
    deduped = []

    for vol, question, cid, month, score in scored_markets:
        if month not in month_counts:
            month_counts[month] = 0

        if month_counts[month] < max_per_month:
            deduped.append((vol, question, cid, month, score))
            month_counts[month] += 1

    return deduped


def search_and_collect(query: str, year: int) -> list:
    """חיפוש בסיסי - מחזיר (vol, question, cid)"""
    with PolymarketGammaClient() as gamma:
        search_result = gamma.search(
            query=query,
            status="resolved",
            limit_per_type=100,
            sort="volume",
            ascending=False,
        )

        found = []
        for event in (search_result.events or []):
            for market in (getattr(event, "markets", []) or []):
                question = getattr(market, "question", "") or ""
                condition_id = getattr(market, "condition_id", "")
                volume = float(getattr(market, "volume_num", 0) or 0)

                found.append((volume, question, condition_id))

        return found


if __name__ == "__main__":

    # ================================================================
    # 2025 FOMC
    # ================================================================

    print("\n" + "=" * 70)
    print("SEARCHING 2025 FOMC MARKETS...")
    print("=" * 70)

    fomc_2025_months = ["january", "march", "may", "june", "july", "september"]

    all_fomc_2025 = {}
    for month in fomc_2025_months:
        queries = [
            f"fed rate cut {month} 2025",
            f"fomc {month} 2025",
            f"fed meeting {month} 2025",
        ]
        for q in queries:
            results = search_and_collect(q, year=2025)
            for vol, question, cid in results:
                q_lower = question.lower()
                if "2025" in q_lower or (month in q_lower and "2024" not in q_lower and "2026" not in q_lower):
                    if cid not in all_fomc_2025:
                        all_fomc_2025[cid] = (vol, question, cid, month)

    print(f"Raw FOMC markets: {len(all_fomc_2025)}")

    scored_fomc = []
    for cid, (vol, question, _, month) in all_fomc_2025.items():
        is_valid, score, reason = score_fomc_market(question, month, 2025)
        if is_valid and score > 0:
            scored_fomc.append((vol, question, cid, month, score))

    scored_fomc.sort(key=lambda x: (x[4], x[0]), reverse=True)
    best_fomc = deduplicate_markets(scored_fomc, max_per_month=2)

    print(f"\n{'=' * 70}")
    print(f"BEST 2025 FOMC MARKETS (Top {len(best_fomc)})")
    print('=' * 70)
    for i, (vol, question, cid, month, score) in enumerate(best_fomc, 1):
        print(f"{i:2d}. [{month.upper()}] score={score:3d} vol=${vol:>12,.0f}")
        print(f"    {question}")
        print(f"    {cid}\n")

    # ================================================================
    # 2025 CPI
    # ================================================================

    print("\n" + "=" * 70)
    print("SEARCHING 2025 CPI MARKETS...")
    print("=" * 70)

    cpi_2025_months = ["january", "february", "march", "april", "may", "june", "july", "august", "september"]

    all_cpi_2025 = {}
    for month in cpi_2025_months:
        queries = [
            f"inflation {month} 2025",
            f"cpi {month} 2025",
        ]
        for q in queries:
            results = search_and_collect(q, year=2025)
            for vol, question, cid in results:
                q_lower = question.lower()
                if "2025" in q_lower or (month in q_lower and "2024" not in q_lower and "2026" not in q_lower):
                    if cid not in all_cpi_2025:
                        all_cpi_2025[cid] = (vol, question, cid, month)

    print(f"Raw CPI markets: {len(all_cpi_2025)}")

    scored_cpi = []
    for cid, (vol, question, _, month) in all_cpi_2025.items():
        is_valid, score, reason = score_cpi_market(question, month, 2025)
        if is_valid and score > 0:
            scored_cpi.append((vol, question, cid, month, score))

    scored_cpi.sort(key=lambda x: (x[4], x[0]), reverse=True)
    best_cpi = deduplicate_markets(scored_cpi, max_per_month=2)

    print(f"\n{'=' * 70}")
    print(f"BEST 2025 CPI MARKETS (Top {len(best_cpi)})")
    print('=' * 70)
    for i, (vol, question, cid, month, score) in enumerate(best_cpi, 1):
        print(f"{i:2d}. [{month.upper()}] score={score:3d} vol=${vol:>12,.0f}")
        print(f"    {question}")
        print(f"    {cid}\n")

    # ================================================================
    # 2025 NFP
    # ================================================================

    print("\n" + "=" * 70)
    print("SEARCHING 2025 NFP MARKETS...")
    print("=" * 70)

    nfp_2025_months = ["january", "february", "march", "april", "may", "june", "july", "august"]

    all_nfp_2025 = {}
    for month in nfp_2025_months:
        queries = [
            f"unemployment {month} 2025",
            f"jobs report {month} 2025",
        ]
        for q in queries:
            results = search_and_collect(q, year=2025)
            for vol, question, cid in results:
                q_lower = question.lower()
                if "2025" in q_lower or (month in q_lower and "2024" not in q_lower and "2026" not in q_lower):
                    if cid not in all_nfp_2025:
                        all_nfp_2025[cid] = (vol, question, cid, month)

    print(f"Raw NFP markets: {len(all_nfp_2025)}")

    scored_nfp = []
    for cid, (vol, question, _, month) in all_nfp_2025.items():
        is_valid, score, reason = score_nfp_market(question, month, 2025)
        if is_valid and score > 0:
            scored_nfp.append((vol, question, cid, month, score))

    scored_nfp.sort(key=lambda x: (x[4], x[0]), reverse=True)
    best_nfp = deduplicate_markets(scored_nfp, max_per_month=2)

    print(f"\n{'=' * 70}")
    print(f"BEST 2025 NFP MARKETS (Top {len(best_nfp)})")
    print('=' * 70)
    for i, (vol, question, cid, month, score) in enumerate(best_nfp, 1):
        print(f"{i:2d}. [{month.upper()}] score={score:3d} vol=${vol:>12,.0f}")
        print(f"    {question}")
        print(f"    {cid}\n")

    # ================================================================
    # REPEAT FOR 2026
    # ================================================================

    # [Same logic for 2026 - January through April]

    # ================================================================
    # SUMMARY
    # ================================================================

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"2025 FOMC: {len(best_fomc)} selected from {len(all_fomc_2025)} raw")
    print(f"2025 CPI:  {len(best_cpi)} selected from {len(all_cpi_2025)} raw")
    print(f"2025 NFP:  {len(best_nfp)} selected from {len(all_nfp_2025)} raw")
    print(f"\nTotal selected: {len(best_fomc) + len(best_cpi) + len(best_nfp)}")