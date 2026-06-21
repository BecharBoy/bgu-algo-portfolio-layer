from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import httpx

NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"

HISTORICAL_SYMBOL_ALIASES: dict[str, str] = {
    "FB": "META",
}

INVALID_SYMBOL_PATTERN = re.compile(r"^\d+:\d+$")
PAREN_TICKER_PATTERN = re.compile(r"\(([A-Z][A-Z0-9.\-]{0,9})\)?")
EXCHANGE_SUFFIX_PATTERN = re.compile(r"^([A-Z][A-Z0-9.\-]{0,9})\.[A-Z]{1,2}$")


@dataclass(frozen=True)
class SecurityMasterEntry:
    official_symbol: str
    yfinance_symbol: str
    security_name: str
    exchange: str
    is_etf: bool
    source: str


@dataclass(frozen=True)
class AssetResolution:
    original_symbol: str
    entry: SecurityMasterEntry | None
    resolved_symbol: str | None = None
    rejection_reason: str | None = None
    match_method: str | None = None

    @property
    def official_symbol(self) -> str | None:
        return self.entry.official_symbol if self.entry else None

    @property
    def security_name(self) -> str | None:
        return self.entry.security_name if self.entry else None

    @property
    def exchange(self) -> str | None:
        return self.entry.exchange if self.entry else None

    @property
    def is_etf(self) -> bool | None:
        return self.entry.is_etf if self.entry else None


def yfinance_symbol(symbol: str) -> str:
    normalized = symbol.strip().upper()
    return normalized.replace(".", "-").replace("^", "-P")


def entries_from_text(nasdaq_text: str, other_text: str) -> list[SecurityMasterEntry]:
    entries: list[SecurityMasterEntry] = []
    entries.extend(_parse_nasdaq_listed(nasdaq_text))
    entries.extend(_parse_other_listed(other_text))
    return entries


def _parse_nasdaq_listed(text: str) -> list[SecurityMasterEntry]:
    entries: list[SecurityMasterEntry] = []
    for line in text.splitlines():
        if not line or line.startswith("Symbol|") or line.startswith("File Creation Time"):
            continue
        parts = line.split("|")
        if len(parts) < 7:
            continue
        symbol = parts[0].strip().upper()
        if not symbol or parts[3].strip().upper() == "Y":
            continue
        entries.append(
            SecurityMasterEntry(
                official_symbol=symbol,
                yfinance_symbol=yfinance_symbol(symbol),
                security_name=parts[1].strip(),
                exchange="NASDAQ",
                is_etf=parts[6].strip().upper() == "Y",
                source="nasdaqlisted",
            )
        )
    return entries


def _parse_other_listed(text: str) -> list[SecurityMasterEntry]:
    entries: list[SecurityMasterEntry] = []
    for line in text.splitlines():
        if not line or line.startswith("ACT Symbol|") or line.startswith("File Creation Time"):
            continue
        parts = line.split("|")
        if len(parts) < 8:
            continue
        symbol = parts[0].strip().upper()
        if not symbol or parts[6].strip().upper() == "Y":
            continue
        exchange = parts[2].strip() or "OTHER"
        entries.append(
            SecurityMasterEntry(
                official_symbol=symbol,
                yfinance_symbol=yfinance_symbol(symbol),
                security_name=parts[1].strip(),
                exchange=exchange,
                is_etf=parts[4].strip().upper() == "Y",
                source="otherlisted",
            )
        )
    return entries


async def download_security_master_entries() -> list[SecurityMasterEntry]:
    async with httpx.AsyncClient(timeout=60.0) as client:
        nasdaq_response, other_response = await client.get(NASDAQ_LISTED_URL), await client.get(
            OTHER_LISTED_URL
        )
        nasdaq_response.raise_for_status()
        other_response.raise_for_status()
    return entries_from_text(nasdaq_response.text, other_response.text)


class SecurityMaster:
    def __init__(self, entries: Iterable[SecurityMasterEntry]) -> None:
        self._by_official: dict[str, SecurityMasterEntry] = {}
        self._by_yfinance: dict[str, SecurityMasterEntry] = {}
        self._by_name_prefix: dict[str, SecurityMasterEntry] = {}
        for entry in entries:
            self._by_official[entry.official_symbol.upper()] = entry
            self._by_yfinance[entry.yfinance_symbol.upper()] = entry
            prefix = _name_prefix(entry.security_name)
            if prefix and prefix not in self._by_name_prefix:
                self._by_name_prefix[prefix] = entry

    def resolve(
        self,
        original_symbol: str,
        *,
        asset_names: list[str] | None = None,
    ) -> AssetResolution:
        raw = original_symbol.strip()
        upper = raw.upper()
        if INVALID_SYMBOL_PATTERN.match(raw):
            return AssetResolution(
                original_symbol=raw,
                entry=None,
                rejection_reason="invalid_symbol_format",
            )
        if "/" in raw:
            return AssetResolution(
                original_symbol=raw,
                entry=None,
                rejection_reason="not_in_us_security_master",
            )

        alias_target = _historical_alias(upper, asset_names or [])
        if alias_target:
            entry = self._by_official.get(alias_target) or self._by_yfinance.get(alias_target)
            if entry:
                return AssetResolution(
                    original_symbol=raw,
                    entry=entry,
                    resolved_symbol=entry.yfinance_symbol,
                    match_method="historical_symbol_alias",
                )

        suffix_match = EXCHANGE_SUFFIX_PATTERN.match(upper)
        if suffix_match:
            entry = self._lookup_symbol(suffix_match.group(1))
            if entry:
                return _accepted(raw, entry, "exchange_suffix")

        paren_match = PAREN_TICKER_PATTERN.search(upper)
        if paren_match:
            entry = self._lookup_symbol(paren_match.group(1))
            if entry:
                return _accepted(raw, entry, "parenthetical_ticker")

        entry = self._lookup_symbol(upper)
        if entry:
            return _accepted(raw, entry, "direct_symbol")

        for candidate in _name_candidates(raw, asset_names or []):
            entry = self._by_name_prefix.get(candidate)
            if entry:
                return _accepted(raw, entry, "security_name_prefix")

        if not _looks_like_ticker_attempt(raw):
            for entry in self._by_official.values():
                if _names_match(entry.security_name, asset_names or []):
                    return _accepted(raw, entry, "asset_name_match")

        return AssetResolution(
            original_symbol=raw,
            entry=None,
            rejection_reason="not_in_us_security_master",
        )

    def _lookup_symbol(self, symbol: str) -> SecurityMasterEntry | None:
        normalized = yfinance_symbol(symbol.upper())
        return self._by_official.get(symbol.upper()) or self._by_yfinance.get(normalized)


def _accepted(original: str, entry: SecurityMasterEntry, method: str) -> AssetResolution:
    return AssetResolution(
        original_symbol=original,
        entry=entry,
        resolved_symbol=entry.yfinance_symbol,
        match_method=method,
    )


def _name_prefix(security_name: str) -> str | None:
    cleaned = security_name.strip()
    if not cleaned:
        return None
    token = cleaned.split()[0].upper().strip(".,")
    return token or None


def _name_candidates(raw: str, asset_names: list[str]) -> list[str]:
    candidates: list[str] = []
    token = raw.split()[0].upper().strip(".,") if raw.split() else raw.upper()
    if token:
        candidates.append(token)
    if not _looks_like_ticker_attempt(raw):
        for name in asset_names:
            prefix = _name_prefix(name)
            if prefix:
                candidates.append(prefix)
    return candidates


def _names_match(security_name: str, asset_names: list[str]) -> bool:
    normalized_name = security_name.lower()
    for asset_name in asset_names:
        cleaned = asset_name.lower().strip()
        if cleaned and cleaned in normalized_name:
            return True
    return False


def _looks_like_ticker_attempt(raw: str) -> bool:
    cleaned = raw.strip().upper()
    if not cleaned or any(ch in cleaned for ch in " ,("):
        return False
    if INVALID_SYMBOL_PATTERN.match(raw) or "/" in raw:
        return False
    return bool(re.match(r"^[A-Z0-9.\-]{1,9}$", cleaned))


def _historical_alias(symbol: str, asset_names: list[str]) -> str | None:
    target = HISTORICAL_SYMBOL_ALIASES.get(symbol)
    if not target:
        return None
    if symbol == "FB":
        if any("facebook" in name.lower() for name in asset_names):
            return target
    return target
