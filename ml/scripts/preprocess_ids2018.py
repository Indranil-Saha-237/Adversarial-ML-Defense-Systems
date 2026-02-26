import pandas as pd
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Config
ARCHIVE_DIR = "data/archive"
OUTPUT_DIR = "data"
SAMPLE_FRACTION = 0.5
CHUNKSIZE = 200_000  # Rows per chunk — tune down if still OOMing


def process_chunk(chunk, expected_columns):
    """Clean a single chunk: fix headers, encode labels, coerce to numeric."""
    chunk.columns = chunk.columns.str.strip()

    # Drop repeated header rows baked into the data (a known CIC-IDS-2018 issue).
    # These show up as rows where the value of the first column equals its column name.
    first_col = chunk.columns[0]
    chunk = chunk[chunk[first_col].astype(str).str.strip() != first_col]

    # Align columns — skip chunks that don't match the expected schema
    
    # Encode label before coercing everything to numeric
    if 'Label' in chunk.columns:
        chunk['Label'] = chunk['Label'].apply(
            lambda x: 0 if str(x).strip().lower() == 'benign' else 1
        )
    if list(chunk.columns) != expected_columns:
        return None
    
    ids_to_drop = ['Flow ID', 'Source IP', 'Src IP', 'Destination IP','Dst IP', 'Timestamp']
    chunk.drop(columns=[c for c in ids_to_drop if c in chunk.columns],errors='ignore', inplace=True)

    chunk = chunk.apply(pd.to_numeric, errors='coerce')
    chunk.replace([np.inf, -np.inf], np.nan, inplace=True)
    chunk.dropna(inplace=True)

    return chunk


def clean_and_merge():
    all_files = glob.glob(os.path.join(ARCHIVE_DIR, "*.csv"))
    print(f"Found {len(all_files)} files to process.")

    dataframes = []

    for file in all_files:
        basename = os.path.basename(file)
        print(f"Processing {basename}...")

        # Read the header row first to get expected columns
        try:
            header_df = pd.read_csv(file, nrows=0)
        except Exception as e:
            print(f" - ERROR reading header of {basename}: {e}. Skipping.")
            continue

        expected_columns = [c.strip() for c in header_df.columns.tolist()]

        file_chunks = []
        try:
            for chunk in pd.read_csv(
                file,
                chunksize=CHUNKSIZE,
                low_memory=False,
                on_bad_lines='skip',
            ):
                cleaned = process_chunk(chunk, expected_columns)
                if cleaned is not None and len(cleaned) > 0:
                    file_chunks.append(cleaned)
        except Exception as e:
            print(f" - ERROR processing {basename}: {e}. Skipping.")
            continue

        if not file_chunks:
            print(f" - WARNING: No valid records found in {basename}. Skipping.")
            continue

        file_df = pd.concat(file_chunks, ignore_index=True)
        del file_chunks

        # Sample after combining all chunks for the file
        if SAMPLE_FRACTION < 1.0:
            file_df = file_df.sample(frac=SAMPLE_FRACTION, random_state=42)

        print(f" - Added {len(file_df):,} records (sampled).")
        dataframes.append(file_df)

    if not dataframes:
        print("ERROR: No data was loaded. Check your archive directory and file contents.")
        return

    print("Merging all dataframes...")
    full_df = pd.concat(dataframes, axis=0, ignore_index=True)
    del dataframes

    # Final cleanup after concat (column mismatches can introduce new NaNs)
    print("Performing final data cleanup...")
    full_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    full_df.dropna(inplace=True)

    if 'Label' not in full_df.columns:
        print("ERROR: 'Label' column missing from merged dataset.")
        return

    numeric_cols = full_df.columns.drop("Label")
    full_df[numeric_cols] = full_df[numeric_cols].clip(lower=-1e9, upper=1e9)

    print(f"Total master dataset size: {len(full_df):,} records.")
    label_counts = full_df['Label'].value_counts()
    print(f"Label distribution: Benign={label_counts.get(0, 0):,}, Attack={label_counts.get(1, 0):,}")

    X = full_df.drop('Label', axis=1)
    y = full_df['Label']
    del full_df

    # 1. Remove non-finite rows
    is_finite = np.isfinite(X.values).all(axis=1)
    if not is_finite.all():
        print(f"Found {(~is_finite).sum()} non-finite rows. Removing them...")
        X = X[is_finite]
        y = y[is_finite]

    # 2. Drop constant columns (StandardScaler divides by std — zero variance = NaN)
    constant_cols = [col for col in X.columns if X[col].nunique() <= 1]
    if constant_cols:
        print(f"Dropping {len(constant_cols)} constant columns: {constant_cols}")
        X.drop(columns=constant_cols, inplace=True)

    print(f"Final dataset size: {len(X):,} records, {len(X.columns)} features.")

    # 3. Cast to float32 (~50% RAM saving vs float64), then scale once
    print("Scaling features...")
    X = X.astype(np.float32)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)
    del X

    print("Splitting data (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y.values, test_size=0.2, random_state=42, stratify=y.values
    )
    del X_scaled

    print(f"Saving processed data to {OUTPUT_DIR}/...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    np.save(f"{OUTPUT_DIR}/X_train.npy", X_train)
    np.save(f"{OUTPUT_DIR}/y_train.npy", y_train)
    np.save(f"{OUTPUT_DIR}/X_test.npy", X_test)
    np.save(f"{OUTPUT_DIR}/y_test.npy", y_test)

    print("Success! Master training set ready.")
    print(f"  X_train: {X_train.shape}, X_test: {X_test.shape}")


if __name__ == "__main__":
    if os.path.exists(ARCHIVE_DIR):
        clean_and_merge()
    else:
        print(f"Archive directory '{ARCHIVE_DIR}' not found.")