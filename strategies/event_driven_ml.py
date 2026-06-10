from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

import numpy as np

from main_backtesting.models import (
    MLModelSnapshot,
    MLObservation,
    MLPrediction,
    PriceBar,
)

FEATURE_NAMES = [
    "asset_ytd_change",
    "sector_one_month_trend",
    "spy_two_week_trend",
    "asset_two_week_trend",
]
ALPHAS = [0.01, 0.1, 1.0, 10.0]
DAILY_SESSION_LENGTH = timedelta(hours=6, minutes=30)


def close_as_of(bars: list[PriceBar], timestamp: datetime) -> float | None:
    value = None
    for bar in bars:
        if bar.timestamp + DAILY_SESSION_LENGTH > timestamp:
            break
        value = bar.close
    return value


def return_between(bars: list[PriceBar], start: datetime, end: datetime) -> float | None:
    start_price = close_as_of(bars, start)
    end_price = close_as_of(bars, end)
    if start_price is None or end_price is None or start_price <= 0:
        return None
    return end_price / start_price - 1.0


def observation_targets(
    bars: list[PriceBar],
    *,
    event_start: datetime,
    peak_window_start: datetime,
    end: datetime,
) -> tuple[int | None, float | None, dict[str, Any]]:
    event_window = [
        bar
        for bar in bars
        if event_start <= bar.timestamp and bar.timestamp + DAILY_SESSION_LENGTH <= end
    ]
    peak_window = [
        bar
        for bar in event_window
        if peak_window_start <= bar.timestamp
    ]
    if not event_window or event_window[0].open <= 0:
        return None, None, {"reason": "missing_event_price_path"}
    if not peak_window:
        return None, None, {"reason": "missing_post_threshold_price_path"}
    opening = event_window[0].open
    above = sum(bar.close > opening for bar in event_window)
    below_or_equal = len(event_window) - above
    direction = 1 if above > below_or_equal else -1
    maximum_change = max((bar.high / opening) - 1.0 for bar in peak_window)
    minimum_change = min((bar.low / opening) - 1.0 for bar in peak_window)
    signed_peak = maximum_change if direction == 1 else minimum_change
    return direction, signed_peak, {
        "event_open_price": opening,
        "event_path_rows": len(event_window),
        "post_threshold_path_rows": len(peak_window),
        "maximum_change": maximum_change,
        "minimum_change": minimum_change,
        "majority_above_count": above,
        "majority_below_or_equal_count": below_or_equal,
    }


def build_observation(
    *,
    run_id: UUID,
    event_id: str,
    market_id: str,
    first_pass_number: int,
    first_pass_at: datetime,
    event_created_at: datetime,
    event_end_at: datetime,
    label_data_cutoff: datetime,
    symbol: str,
    event_archetype: str,
    resolution: str,
    asset_daily: list[PriceBar],
    sector_daily: list[PriceBar],
    spy_daily: list[PriceBar],
    research_data: dict[str, Any],
) -> MLObservation:
    from datetime import timedelta, timezone

    year_start = datetime(first_pass_at.year, 1, 1, tzinfo=timezone.utc)
    features = {
        "asset_ytd_change": return_between(asset_daily, year_start, first_pass_at),
        "sector_one_month_trend": return_between(
            sector_daily, first_pass_at - timedelta(days=30), first_pass_at
        ),
        "spy_two_week_trend": return_between(
            spy_daily, first_pass_at - timedelta(days=14), first_pass_at
        ),
        "asset_two_week_trend": return_between(
            asset_daily, first_pass_at - timedelta(days=14), first_pass_at
        ),
    }
    known_opening_bar = next(
        (
            bar
            for bar in asset_daily
            if event_created_at <= bar.timestamp <= label_data_cutoff
        ),
        None,
    )
    if event_end_at > label_data_cutoff:
        target_direction, target_magnitude = None, None
        target_data = {
            "reason": "event_unresolved_at_historical_data_cutoff",
            "historical_data_cutoff": label_data_cutoff,
            "event_open_price": known_opening_bar.open if known_opening_bar else None,
        }
    else:
        target_direction, target_magnitude, target_data = observation_targets(
            asset_daily,
            event_start=event_created_at,
            peak_window_start=first_pass_at,
            end=event_end_at,
        )
    missing = [name for name, value in features.items() if value is None]
    valid = not missing and target_direction is not None and target_magnitude is not None
    exclusion = None
    if missing:
        exclusion = f"missing_required_feature:{','.join(missing)}"
    elif target_direction is None:
        exclusion = str(target_data.get("reason") or "missing_target")
    return MLObservation(
        observation_id=uuid4(),
        run_id=run_id,
        event_id=event_id,
        market_id=market_id,
        first_pass_number=first_pass_number,
        first_pass_at=first_pass_at,
        label_available_at=event_end_at,
        symbol=symbol.upper(),
        event_archetype=event_archetype,
        resolution=resolution,  # type: ignore[arg-type]
        features=features,
        research_data={**research_data, **target_data},
        classification_target=target_direction,
        regression_target=target_magnitude,
        valid_for_training=valid,
        exclusion_reason=exclusion,
    )


def _matrix(observations: list[MLObservation]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.array(
        [[float(item.features[name]) for name in FEATURE_NAMES] for item in observations],
        dtype=float,
    )
    y_class = np.array([int(item.classification_target) for item in observations], dtype=float)
    y_class = (y_class + 1.0) / 2.0
    y_reg = np.array([float(item.regression_target) for item in observations], dtype=float)
    return x, y_class, y_reg


def _standardize(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means = x.mean(axis=0)
    scales = x.std(axis=0)
    scales[scales == 0] = 1.0
    return (x - means) / scales, means, scales


def _sigmoid(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(value, -35, 35)
    return 1.0 / (1.0 + np.exp(-clipped))


def _fit_logistic(x: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, float]:
    weights = np.zeros(x.shape[1], dtype=float)
    intercept = 0.0
    learning_rate = 0.1
    for _ in range(2_000):
        predictions = _sigmoid(x @ weights + intercept)
        error = predictions - y
        gradient = (x.T @ error) / len(x) + alpha * weights / len(x)
        intercept_gradient = float(error.mean())
        weights -= learning_rate * gradient
        intercept -= learning_rate * intercept_gradient
    return weights, intercept


def _fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, float]:
    intercept = float(y.mean())
    centered = y - intercept
    identity = np.eye(x.shape[1])
    weights = np.linalg.solve(x.T @ x + alpha * identity, x.T @ centered)
    return weights, intercept


def _walk_forward_alpha(x: np.ndarray, y: np.ndarray, *, classification: bool) -> tuple[float, dict[str, float]]:
    if len(x) < 5:
        return 1.0, {}
    scores: dict[float, list[float]] = {alpha: [] for alpha in ALPHAS}
    start = max(3, len(x) // 2)
    for index in range(start, len(x)):
        train_x, train_y = x[:index], y[:index]
        test_x, test_y = x[index : index + 1], y[index : index + 1]
        for alpha in ALPHAS:
            if classification:
                if len(set(train_y.tolist())) < 2:
                    continue
                weights, intercept = _fit_logistic(train_x, train_y, alpha)
                probability = float(_sigmoid(test_x @ weights + intercept)[0])
                scores[alpha].append(-(float(test_y[0]) - probability) ** 2)
            else:
                weights, intercept = _fit_ridge(train_x, train_y, alpha)
                prediction = float((test_x @ weights + intercept)[0])
                scores[alpha].append(-abs(float(test_y[0]) - prediction))
    means = {alpha: float(np.mean(values)) for alpha, values in scores.items() if values}
    selected = max(means, key=means.get) if means else 1.0
    return selected, {str(key): value for key, value in means.items()}


def train_snapshot(
    *,
    run_id: UUID,
    symbol: str,
    event_archetype: str,
    training_cutoff: datetime,
    observations: list[MLObservation],
    minimum_prior_observations: int,
    prediction_features: dict[str, float | None] | None = None,
) -> MLModelSnapshot:
    valid = [
        item
        for item in observations
        if item.valid_for_training
        and item.classification_target is not None
        and item.regression_target is not None
    ]
    base = dict(
        snapshot_id=uuid4(),
        run_id=run_id,
        symbol=symbol.upper(),
        event_archetype=event_archetype,
        training_cutoff=training_cutoff,
        training_event_ids=[item.event_id for item in valid],
        training_sample_count=len(valid),
        feature_names=FEATURE_NAMES,
        feature_means={},
        feature_scales={},
        classifier_coefficients=None,
        classifier_intercept=None,
        ridge_coefficients=None,
        ridge_intercept=None,
        hyperparameters={},
        validation_metrics={},
    )
    if len(valid) < minimum_prior_observations:
        return MLModelSnapshot(status="insufficient_history", **base)
    classes = {item.classification_target for item in valid}
    if len(classes) < 2:
        return MLModelSnapshot(status="insufficient_class_diversity", **base)
    if prediction_features is not None and any(
        prediction_features.get(name) is None for name in FEATURE_NAMES
    ):
        return MLModelSnapshot(status="missing_required_feature", **base)

    x, y_class, y_reg = _matrix(valid)
    standardized, means, scales = _standardize(x)
    classifier_alpha, classifier_cv = _walk_forward_alpha(
        standardized, y_class, classification=True
    )
    ridge_alpha, ridge_cv = _walk_forward_alpha(standardized, y_reg, classification=False)
    class_weights, class_intercept = _fit_logistic(standardized, y_class, classifier_alpha)
    ridge_weights, ridge_intercept = _fit_ridge(standardized, y_reg, ridge_alpha)
    class_predictions = (_sigmoid(standardized @ class_weights + class_intercept) >= 0.5).astype(float)
    ridge_predictions = standardized @ ridge_weights + ridge_intercept
    metrics = {
        "training_classification_accuracy": float((class_predictions == y_class).mean()),
        "training_regression_mae": float(np.abs(ridge_predictions - y_reg).mean()),
        "training_regression_rmse": float(np.sqrt(((ridge_predictions - y_reg) ** 2).mean())),
        "classifier_walk_forward_scores": classifier_cv,
        "ridge_walk_forward_scores": ridge_cv,
    }
    return MLModelSnapshot(
        status="trained",
        feature_means=dict(zip(FEATURE_NAMES, means.tolist())),
        feature_scales=dict(zip(FEATURE_NAMES, scales.tolist())),
        classifier_coefficients=dict(zip(FEATURE_NAMES, class_weights.tolist())),
        classifier_intercept=class_intercept,
        ridge_coefficients=dict(zip(FEATURE_NAMES, ridge_weights.tolist())),
        ridge_intercept=ridge_intercept,
        hyperparameters={
            "classifier_l2_alpha": classifier_alpha,
            "ridge_alpha": ridge_alpha,
        },
        validation_metrics=metrics,
        **{key: value for key, value in base.items() if key not in {
            "feature_means", "feature_scales", "classifier_coefficients",
            "classifier_intercept", "ridge_coefficients", "ridge_intercept",
            "hyperparameters", "validation_metrics"
        }},
    )


def predict(
    snapshot: MLModelSnapshot,
    *,
    run_id: UUID,
    market_id: str,
    event_id: str,
    pass_number: int,
    symbol: str,
    features: dict[str, float | None],
    event_open_price: float,
    realized_price_at_entry: float,
) -> MLPrediction | None:
    if snapshot.status != "trained":
        return None
    if any(features.get(name) is None for name in FEATURE_NAMES):
        return None
    vector = np.array([float(features[name]) for name in FEATURE_NAMES], dtype=float)
    means = np.array([snapshot.feature_means[name] for name in FEATURE_NAMES])
    scales = np.array([snapshot.feature_scales[name] for name in FEATURE_NAMES])
    standardized = (vector - means) / scales
    class_weights = np.array(
        [snapshot.classifier_coefficients[name] for name in FEATURE_NAMES], dtype=float
    )
    ridge_weights = np.array(
        [snapshot.ridge_coefficients[name] for name in FEATURE_NAMES], dtype=float
    )
    probability = float(_sigmoid(np.array([standardized @ class_weights + snapshot.classifier_intercept]))[0])
    direction = "long" if probability >= 0.5 else "short"
    predicted_peak = float(standardized @ ridge_weights + snapshot.ridge_intercept)
    directions_agree = (direction == "long" and predicted_peak > 0) or (
        direction == "short" and predicted_peak < 0
    )
    realized = realized_price_at_entry / event_open_price - 1.0
    directional_realized = realized if direction == "long" else -realized
    gap = abs(predicted_peak) - directional_realized
    target_price = event_open_price * (1.0 + predicted_peak)
    return MLPrediction(
        prediction_id=uuid4(),
        run_id=run_id,
        snapshot_id=snapshot.snapshot_id,
        market_id=market_id,
        event_id=event_id,
        pass_number=pass_number,
        symbol=symbol.upper(),
        direction=direction,
        classification_probability=probability,
        predicted_peak_percent=predicted_peak,
        predicted_target_price=target_price,
        realized_move_at_entry=realized,
        remaining_gap=gap,
        directions_agree=directions_agree,
    )


def evaluate_prediction(
    prediction: MLPrediction,
    *,
    event_open_price: float,
    bars_until_event_end: list[PriceBar],
) -> MLPrediction:
    if not bars_until_event_end:
        return prediction
    changes = [bar.close / event_open_price - 1.0 for bar in bars_until_event_end]
    favorable = max(changes) if prediction.direction == "long" else -min(changes)
    adverse = min(changes) if prediction.direction == "long" else -max(changes)
    if prediction.direction == "long":
        hit = next(
            (bar for bar in bars_until_event_end if bar.high >= prediction.predicted_target_price),
            None,
        )
        actual_direction = "long" if max(changes) >= abs(min(changes)) else "short"
    else:
        hit = next(
            (bar for bar in bars_until_event_end if bar.low <= prediction.predicted_target_price),
            None,
        )
        actual_direction = "short" if abs(min(changes)) > max(changes) else "long"
    prediction.target_reached = hit is not None
    prediction.target_reached_at = hit.timestamp if hit else None
    prediction.actual_max_favorable = favorable
    prediction.actual_max_adverse = adverse
    prediction.actual_direction = actual_direction
    prediction.classification_correct = actual_direction == prediction.direction
    actual_signed_peak = max(changes) if actual_direction == "long" else min(changes)
    prediction.regression_error = actual_signed_peak - prediction.predicted_peak_percent
    return prediction
