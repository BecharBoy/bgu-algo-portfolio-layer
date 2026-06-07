from __future__ import annotations

from datetime import datetime
from uuid import UUID
import asyncpg
from main_backtesting.models import MLModelSnapshot, MLObservation, MLPrediction

from database.backtesting.repositories._shared import SCHEMA, json_text, json_value


async def save_ml_observation(conn: asyncpg.Connection, item: MLObservation) -> None:
    await conn.execute(
        f"""
        INSERT INTO {SCHEMA}.historical_ml_observations (
            observation_id, run_id, event_id, market_id, first_pass_number,
            first_pass_at, label_available_at, symbol, event_archetype, resolution, features,
            research_data, classification_target, regression_target,
            valid_for_training, exclusion_reason
        )
        VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11::JSONB,$12::JSONB,$13,$14,$15,$16)
        ON CONFLICT (run_id, event_id, symbol) DO NOTHING
        """,
        item.observation_id,
        item.run_id,
        item.event_id,
        item.market_id,
        item.first_pass_number,
        item.first_pass_at,
        item.label_available_at,
        item.symbol,
        item.event_archetype,
        item.resolution,
        json_text(item.features),
        json_text(item.research_data),
        item.classification_target,
        item.regression_target,
        item.valid_for_training,
        item.exclusion_reason,
    )


async def prior_ml_observations(
    conn: asyncpg.Connection,
    *,
    run_id: UUID,
    symbol: str,
    event_archetype: str,
    before: datetime,
) -> list[MLObservation]:
    rows = await conn.fetch(
        f"""
        SELECT * FROM {SCHEMA}.historical_ml_observations
        WHERE run_id = $1 AND symbol = $2 AND event_archetype = $3
          AND valid_for_training AND label_available_at IS NOT NULL
          AND label_available_at < $4
        ORDER BY label_available_at, first_pass_at
        """,
        run_id,
        symbol.upper(),
        event_archetype,
        before,
    )
    return [
        MLObservation(
            observation_id=row["observation_id"],
            run_id=row["run_id"],
            event_id=row["event_id"],
            market_id=row["market_id"],
            first_pass_number=row["first_pass_number"],
            first_pass_at=row["first_pass_at"],
            label_available_at=row["label_available_at"],
            symbol=row["symbol"],
            event_archetype=row["event_archetype"],
            resolution=row["resolution"],
            features=json_value(row["features"]),
            research_data=json_value(row["research_data"]),
            classification_target=row["classification_target"],
            regression_target=row["regression_target"],
            valid_for_training=row["valid_for_training"],
            exclusion_reason=row["exclusion_reason"],
        )
        for row in rows
    ]


async def run_ml_observations(conn: asyncpg.Connection, run_id: UUID) -> list[MLObservation]:
    rows = await conn.fetch(
        f"""
        SELECT * FROM {SCHEMA}.historical_ml_observations
        WHERE run_id = $1 ORDER BY first_pass_at, event_id, symbol
        """,
        run_id,
    )
    return [
        MLObservation(
            observation_id=row["observation_id"],
            run_id=row["run_id"],
            event_id=row["event_id"],
            market_id=row["market_id"],
            first_pass_number=row["first_pass_number"],
            first_pass_at=row["first_pass_at"],
            label_available_at=row["label_available_at"],
            symbol=row["symbol"],
            event_archetype=row["event_archetype"],
            resolution=row["resolution"],
            features=json_value(row["features"]),
            research_data=json_value(row["research_data"]),
            classification_target=row["classification_target"],
            regression_target=row["regression_target"],
            valid_for_training=row["valid_for_training"],
            exclusion_reason=row["exclusion_reason"],
        )
        for row in rows
    ]


async def save_model_snapshot(conn: asyncpg.Connection, item: MLModelSnapshot) -> None:
    await conn.execute(
        f"""
        INSERT INTO {SCHEMA}.historical_ml_model_snapshots (
            snapshot_id, run_id, symbol, event_archetype, training_cutoff,
            training_event_ids, training_sample_count, status, feature_names,
            feature_means, feature_scales, classifier_coefficients,
            classifier_intercept, ridge_coefficients, ridge_intercept,
            hyperparameters, validation_metrics
        )
        VALUES (
            $1,$2,$3,$4,$5,$6,$7,$8,$9,$10::JSONB,$11::JSONB,$12::JSONB,
            $13,$14::JSONB,$15,$16::JSONB,$17::JSONB
        )
        ON CONFLICT (run_id, symbol, event_archetype, training_cutoff) DO NOTHING
        """,
        item.snapshot_id,
        item.run_id,
        item.symbol,
        item.event_archetype,
        item.training_cutoff,
        item.training_event_ids,
        item.training_sample_count,
        item.status,
        item.feature_names,
        json_text(item.feature_means),
        json_text(item.feature_scales),
        json_text(item.classifier_coefficients),
        item.classifier_intercept,
        json_text(item.ridge_coefficients),
        item.ridge_intercept,
        json_text(item.hyperparameters),
        json_text(item.validation_metrics),
    )


async def save_ml_prediction(conn: asyncpg.Connection, item: MLPrediction) -> None:
    await conn.execute(
        f"""
        INSERT INTO {SCHEMA}.historical_ml_predictions (
            prediction_id, run_id, snapshot_id, market_id, event_id, pass_number,
            symbol, direction, classification_probability, predicted_peak_percent,
            predicted_target_price, realized_move_at_entry, remaining_gap,
            directions_agree, target_reached, target_reached_at, actual_max_favorable,
            actual_max_adverse, actual_direction, classification_correct, regression_error
        )
        VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19,$20,$21)
        """,
        item.prediction_id,
        item.run_id,
        item.snapshot_id,
        item.market_id,
        item.event_id,
        item.pass_number,
        item.symbol,
        item.direction,
        item.classification_probability,
        item.predicted_peak_percent,
        item.predicted_target_price,
        item.realized_move_at_entry,
        item.remaining_gap,
        item.directions_agree,
        item.target_reached,
        item.target_reached_at,
        item.actual_max_favorable,
        item.actual_max_adverse,
        item.actual_direction,
        item.classification_correct,
        item.regression_error,
    )
