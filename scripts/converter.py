import pandas as pd

# -------------------------------
# 1. Define CSV file paths
# -------------------------------
FILES = [
    "../AAIPL/Logical_Reasoning.csv",
    "../AAIPL/Puzzles.csv",
    "../AAIPL/Family_Tree.csv",
    "../AAIPL/Mixed_Series.csv",
]

OUTPUT_PATH = "../AAIPL/questions_training.json"


def load_and_validate(files: list[str]) -> list[pd.DataFrame]:
    """
    Load CSV files and validate that all have identical column structures.
    """
    dataframes = []

    for file_path in files:
        df = pd.read_csv(file_path)

        print(f"{file_path} → shape: {df.shape}")
        print(f"{file_path} → columns: {list(df.columns)}")

        dataframes.append(df)

    # Validate column consistency
    reference_columns = dataframes[0].columns

    for idx, df in enumerate(dataframes):
        if not df.columns.equals(reference_columns):
            raise ValueError(f"Column mismatch detected in {files[idx]}")

    print("All files share identical column structure.")
    return dataframes


def combine_and_clean(dataframes: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Concatenate dataframes and remove duplicate rows.
    """
    combined = pd.concat(dataframes, ignore_index=True)
    combined = combined.drop_duplicates()

    print(f"Final dataset shape after deduplication: {combined.shape}")
    return combined


def save_as_json(df: pd.DataFrame, output_path: str) -> None:
    """
    Save DataFrame to JSON file.
    """
    df.to_json(output_path, orient="records", indent=4)
    print(f"Saved dataset to: {output_path}")


def main():
    dfs = load_and_validate(FILES)
    combined_df = combine_and_clean(dfs)
    save_as_json(combined_df, OUTPUT_PATH)


if __name__ == "__main__":
    main()