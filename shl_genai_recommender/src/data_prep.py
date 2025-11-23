import os
import pandas as pd
import re
from typing import Dict, List


def clean_text(text: str) -> str:
    """Normalize text by removing newlines, extra spaces, and converting to lowercase."""
    if isinstance(text, str):
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces/newlines
        return text.strip().lower()
    return ""


def prepare_dataset(file_path: str, output_dir: str = "data/processed") -> Dict[str, List[str]]:
    """Load, clean, and structure dataset into usable mappings."""
    # Load Excel file
    excel_data = pd.ExcelFile(file_path)
    train_df = pd.read_excel(excel_data, sheet_name="Train-Set")
    test_df = pd.read_excel(excel_data, sheet_name="Test-Set")

    # Clean text columns
    train_df["Query_clean"] = train_df["Query"].apply(clean_text)
    test_df["Query_clean"] = test_df["Query"].apply(clean_text)

    # Create mapping: query -> list of relevant assessments
    query_to_assessments = (
        train_df.groupby("Query_clean")["Assessment_url"].apply(list).to_dict()
    )

    # Extract unique assessments
    unique_assessments = train_df["Assessment_url"].dropna().unique().tolist()

    # Create output folder if not exists
    os.makedirs(output_dir, exist_ok=True)

    # Save cleaned data for future steps
    train_df.to_csv(os.path.join(output_dir, "train_clean.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test_clean.csv"), index=False)
    pd.DataFrame(unique_assessments, columns=["Assessment_url"]).to_csv(
        os.path.join(output_dir, "unique_assessments.csv"), index=False
    )

    print(f"Data preparation complete.")
    print(f"- Train samples: {len(train_df)}")
    print(f"- Unique queries: {len(query_to_assessments)}")
    print(f"- Unique assessments: {len(unique_assessments)}")

    # Return useful mappings for direct usage
    return query_to_assessments


if __name__ == "__main__":
    # Example usage
    dataset_path = "data/raw/Gen_AI Dataset.xlsx"
    mappings = prepare_dataset(dataset_path)
    print(f"Sample query and its mapped assessments:\n")
    first_item = list(mappings.items())[0]
    print(f"Query:\n{first_item[0][:250]}...")
    print(f"\nAssessments:\n{first_item[1]}")
