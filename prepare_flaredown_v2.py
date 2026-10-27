"""
prepare_flaredown_v2.py
=======================
This retained v2 filename implements the v3 candidate-preparation methodology.
It rebuilds the RA flare dataset with two genuine fixes over the original v2 method:

FIX 1 - COHORT.  v2 selected "RA patients" with
            str.contains('rheumatoid|arthritis|RA', case=False)
        The pattern 'RA' with case=False matches ANY condition containing the letters
        "ra" (Mi-gra-ine, anxiety -> "generalized anxiety", Ulce-ra-tive colitis,
        Postural ..., Osteoarthritis, Psoriatic arthritis ...). Result: 14,319 "RA" users,
        of whom only ~1,264 report rheumatoid arthritis.
        v3 keeps only users whose condition list contains 'rheumatoid' (or exactly 'RA').

FIX 2 - LOOK-AHEAD BASELINE.  v2 computed personal_baseline as the median severity over the
        patient's WHOLE history (including days after the checkpoint), and used it both as a
        feature and to define the label. v3 uses an EXPANDING median: only days <= t.

Memory-safe: reads data/ra.csv in chunks (the file is ~700 MB).
Output: data/ra_processed_v3_candidate.csv

Checkpoint schema:
    user_id, checkpoint_date, past_mean_severity, past_std_severity,
    past_trend, past_last_severity, past_engagement, personal_baseline,
    flare_next_3d, age, sex

At checkpoint date t, past features use calendar days t-6 through t. The
flare_next_3d proxy is 1 when mean severity over observed days t+1 through
t+3 is strictly greater than 1.15 times the expanding median through t.
At least one future day must be observed; all three are not required.
Missing daily severity remains missing; the past window requires at least
three observed days. Missing age remains missing for train-fold imputation.
Sex retains its source category, with missing/blank values left missing.
"""
import csv
import hashlib
import os
import sqlite3
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

RAW = "data/ra.csv"
OUT = "data/ra_processed_v3_candidate.csv"
MIN_DAYS, PAST, FUT, STEP, MIN_BASE_OBS, FLARE_MULT = 30, 7, 3, 3, 10, 1.15
CHUNK = 2_000_000

STAGED_COLUMNS = (
    "user_id",
    "age",
    "sex",
    "checkin_date",
    "trackable_type",
    "trackable_value",
)
OUTPUT_COLUMNS = (
    "user_id",
    "checkpoint_date",
    "past_mean_severity",
    "past_std_severity",
    "past_trend",
    "past_last_severity",
    "past_engagement",
    "personal_baseline",
    "flare_next_3d",
    "age",
    "sex",
)


def is_ra(names: pd.Series) -> pd.Series:
    n = names.fillna("").str.strip().str.lower()
    return n.str.contains("rheumatoid", regex=False) | n.eq("ra")


def build_patient_checkpoints(uid: str, patient_daily: pd.DataFrame) -> list:
    """Build past-only features and the unchanged future-worsening target."""
    g = patient_daily.set_index("dt").sort_index()
    g = g.reindex(pd.date_range(g.index.min(), g.index.max(), freq="D"))
    g["cnt"] = g["cnt"].fillna(0)
    base = g["sev"].expanding(min_periods=1).median()
    nobs = g["sev"].expanding().count()

    rows = []
    for t in range(PAST - 1, len(g) - FUT, STEP):
        past = g["sev"].iloc[t - PAST + 1: t + 1]
        fut = g["sev"].iloc[t + 1: t + 1 + FUT]
        if past.notna().sum() < 3 or fut.notna().sum() < 1 or nobs.iloc[t] < MIN_BASE_OBS:
            continue
        pm = past.mean()
        rows.append({
            "user_id": uid,
            "checkpoint_date": g.index[t].strftime("%Y-%m-%d"),
            "past_mean_severity": pm,
            "past_std_severity": past.std(),
            "past_trend": np.polyfit(range(len(past)), past.fillna(pm), 1)[0],
            "past_last_severity": past.dropna().iloc[-1],
            "past_engagement": g["cnt"].iloc[t - PAST + 1: t + 1].mean(),
            "personal_baseline": base.iloc[t],
            "flare_next_3d": int(fut.mean() > base.iloc[t] * FLARE_MULT),
        })
    return rows


def _store_ids(connection, table: str, values) -> None:
    connection.executemany(
        f"INSERT OR IGNORE INTO {table} (user_id) VALUES (?)",
        ((str(value),) for value in values if pd.notna(value)),
    )


def _validate_user_ids(chunk: pd.DataFrame) -> None:
    if any(
        pd.notna(value) and not isinstance(value, str)
        for value in chunk["user_id"]
    ):
        raise TypeError(
            "user_id values must be strings; read IDs as strings to preserve leading zeros."
        )


def _matching_ids(connection, table: str, values) -> set:
    user_ids = pd.Series(values, dtype="string").dropna().drop_duplicates().tolist()
    matches = set()
    for start in range(0, len(user_ids), 500):
        batch = user_ids[start:start + 500]
        placeholders = ",".join("?" for _ in batch)
        matches.update(
            row[0]
            for row in connection.execute(
                f"SELECT user_id FROM {table} WHERE user_id IN ({placeholders})",
                batch,
            )
        )
    return matches


def _sqlite_value(value):
    return None if pd.isna(value) else str(value)


def _csv_value(value):
    return "" if pd.isna(value) else value


def _stage_raw_rows(connection, chunks_factory) -> tuple[int, int]:
    """Stage rows from a repeatable factory yielding fresh iterators over identical data.

    Non-null patient IDs must be strings so their source representation is preserved.
    """
    connection.executescript(
        """
        CREATE TABLE old_users (user_id TEXT PRIMARY KEY);
        CREATE TABLE ra_users (user_id TEXT PRIMARY KEY);
        CREATE TABLE source_rows (
            source_order INTEGER PRIMARY KEY,
            user_id TEXT NOT NULL,
            age TEXT,
            sex TEXT,
            checkin_date TEXT,
            trackable_type TEXT,
            trackable_value TEXT
        );
        CREATE INDEX source_rows_user_order
            ON source_rows(user_id, source_order);
        """
    )

    print("Pass 1/3: finding true RA users (condition rows only)...")
    first_iterator = iter(chunks_factory())
    source_row_count = 0
    source_digest = hashlib.sha256()
    for chunk in first_iterator:
        _validate_user_ids(chunk)
        source_row_count += len(chunk)
        source_digest.update(
            pd.util.hash_pandas_object(chunk, index=False).to_numpy().tobytes()
        )
        conditions = chunk[chunk["trackable_type"] == "Condition"]
        old_mask = conditions["trackable_name"].str.contains(
            "rheumatoid|arthritis|RA", case=False, na=False
        )
        old_users = conditions.loc[old_mask, "user_id"].dropna().drop_duplicates()
        ra_users = conditions.loc[is_ra(conditions["trackable_name"]), "user_id"]
        _store_ids(connection, "old_users", old_users)
        _store_ids(connection, "ra_users", ra_users)
        connection.commit()

    old_count = connection.execute("SELECT COUNT(*) FROM old_users").fetchone()[0]
    ra_count = connection.execute("SELECT COUNT(*) FROM ra_users").fetchone()[0]
    print(f"  v2 filter would have selected : {old_count:>6} users")
    print(f"  v3 strict RA filter selects   : {ra_count:>6} users")

    print("Pass 2/3: extracting their rows...")
    second_iterator = iter(chunks_factory())
    if second_iterator is first_iterator:
        raise ValueError(
            "chunks_factory must return a fresh iterator over the same source rows on each call."
        )
    source_offset = 0
    second_digest = hashlib.sha256()
    for chunk in second_iterator:
        _validate_user_ids(chunk)
        matching = _matching_ids(connection, "ra_users", chunk["user_id"])
        second_digest.update(
            pd.util.hash_pandas_object(chunk, index=False).to_numpy().tobytes()
        )
        if matching:
            selected = chunk.loc[
                chunk["user_id"].isin(matching), list(STAGED_COLUMNS)
            ]
            staged = (
                (source_offset + position, *(_sqlite_value(value) for value in row))
                for position, row in enumerate(
                    selected.itertuples(index=False, name=None)
                )
            )
            connection.executemany(
                """
                INSERT INTO source_rows
                    (source_order, user_id, age, sex, checkin_date,
                     trackable_type, trackable_value)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                staged,
            )
        source_offset += len(chunk)
        connection.commit()
    if source_offset != source_row_count:
        raise ValueError(
            "chunks_factory yielded a different number of source rows on its second call."
        )
    if second_digest.digest() != source_digest.digest():
        raise ValueError(
            "chunks_factory yielded different source rows on its second call."
        )
    return old_count, ra_count


def _patient_checkpoints(connection, uid: str) -> list:
    records = connection.execute(
        """
        SELECT user_id, age, sex, checkin_date, trackable_type, trackable_value
        FROM source_rows
        WHERE user_id = ?
        ORDER BY source_order
        """,
        (uid,),
    ).fetchall()
    patient = pd.DataFrame.from_records(records, columns=STAGED_COLUMNS)

    sym = patient[patient["trackable_type"] == "Symptom"].copy()
    sym["v"] = pd.to_numeric(sym["trackable_value"], errors="coerce")
    sym["dt"] = pd.to_datetime(sym["checkin_date"], errors="coerce")
    sym = sym.dropna(subset=["v", "dt"])
    if sym.empty:
        return []

    daily = (
        sym.groupby(["user_id", "dt"])
        .agg(sev=("v", "mean"), cnt=("v", "count"))
        .reset_index()
    )
    if daily["dt"].nunique() < MIN_DAYS:
        return []

    profile = patient[["age", "sex"]].iloc[0]
    age = pd.to_numeric(pd.Series([profile["age"]]), errors="coerce").iloc[0]
    sex = pd.Series([profile["sex"]], dtype="string").str.strip().replace("", pd.NA).iloc[0]

    checkpoints = build_patient_checkpoints(uid, daily.drop(columns="user_id"))
    for row in checkpoints:
        row["age"] = age
        row["sex"] = sex
    return checkpoints


def process_flaredown_chunks(
    chunks_factory,
    output_path=OUT,
    staging_directory=None,
) -> dict:
    """Stage and process chunks from a repeatable factory returning fresh iterators.

    Each call to chunks_factory must yield the same source rows in the same order.
    Non-null user_id values must be strings; use dtype=str when reading CSV data.
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.resolve() == Path(RAW).resolve():
        raise ValueError("Candidate output must not overwrite the raw input.")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing candidate: {output}")

    temporary_output = None
    staging_path = None
    connection = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix="flaredown_stage_",
            suffix=".sqlite",
            dir=staging_directory,
            delete=False,
        ) as staging_file:
            staging_path = Path(staging_file.name)
        with tempfile.NamedTemporaryFile(
            prefix=f".{output.name}.",
            suffix=".partial",
            dir=output.parent,
            delete=False,
            mode="w",
            encoding="utf-8",
            newline="",
        ) as output_file:
            temporary_output = Path(output_file.name)

        connection = sqlite3.connect(staging_path)
        _stage_raw_rows(connection, chunks_factory)
        print("Pass 3/3: building checkpoints...")
        row_count = 0
        patient_count = 0
        positive_count = 0
        with temporary_output.open("w", encoding="utf-8", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(OUTPUT_COLUMNS)
            user_cursor = connection.execute(
                "SELECT user_id FROM ra_users ORDER BY user_id COLLATE BINARY"
            )
            for (uid,) in user_cursor:
                checkpoints = _patient_checkpoints(connection, uid)
                if not checkpoints:
                    continue
                patient_count += 1
                for checkpoint in checkpoints:
                    writer.writerow(
                        [_csv_value(checkpoint[column]) for column in OUTPUT_COLUMNS]
                    )
                    row_count += 1
                    positive_count += checkpoint["flare_next_3d"]
        connection.close()
        connection = None

        if output.exists():
            raise FileExistsError(f"Refusing to overwrite existing candidate: {output}")
        os.link(temporary_output, output)
        temporary_output.unlink()
        temporary_output = None
        return {
            "rows": row_count,
            "patients": patient_count,
            "positive_count": positive_count,
            "output": str(output),
        }
    finally:
        active_exception = sys.exc_info()[0] is not None
        cleanup_errors = []
        staging_cleanup_safe = True
        if connection is not None:
            try:
                connection.close()
            except BaseException as error:
                cleanup_errors.append(("SQLite connection", error))
                staging_cleanup_safe = False
                print(
                    f"Cleanup failed to close SQLite connection; "
                    f"staging file may remain at {staging_path}: {error}",
                    file=sys.stderr,
                )
        if temporary_output is not None:
            try:
                temporary_output.unlink(missing_ok=True)
            except BaseException as error:
                cleanup_errors.append((str(temporary_output), error))
        if staging_path is not None and staging_cleanup_safe:
            try:
                staging_path.unlink(missing_ok=True)
            except BaseException as error:
                cleanup_errors.append((str(staging_path), error))
        if cleanup_errors:
            if active_exception:
                for resource, error in cleanup_errors:
                    if resource == "SQLite connection":
                        continue
                    print(
                        f"Cleanup failed for {resource}: {error}",
                        file=sys.stderr,
                    )
            else:
                raise cleanup_errors[0][1]


def main():
    def chunks_factory():
        return pd.read_csv(RAW, chunksize=CHUNK, dtype=str)

    result = process_flaredown_chunks(chunks_factory)
    positive_rate = result["positive_count"] / result["rows"] if result["rows"] else float("nan")
    print(
        f"\nSaved {result['output']}: {result['rows']} rows, "
        f"{result['patients']} patients, flare rate {positive_rate:.3f}"
    )


if __name__ == "__main__":
    main()
