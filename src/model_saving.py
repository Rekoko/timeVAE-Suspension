import pandas as pd


def append_row_to_csv(row_df, file_path):
    # Load the existing CSV into a DataFrame
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        df = pd.DataFrame()  # In case the file doesn't exist, create a new DataFrame
    
    # Columns that are in the file but not in the row
    missing_in_row = set(df.columns) - set(row_df.columns)
    # Add missing columns to the row_df with None values
    for col in missing_in_row:
        row_df[col] = None

    # Columns that are in the row but not in the file
    missing_in_file = set(row_df.columns) - set(df.columns)
    # Add missing columns to the main DataFrame with None values
    for col in missing_in_file:
        df[col] = None

    # Reorder columns to match the original CSV structure
    row_df = row_df[df.columns]

    # Append the row to the DataFrame
    df = pd.concat([row_df, df], ignore_index=True)

    # Save the updated DataFrame back to the CSV file
    df.to_csv(file_path, index=False)