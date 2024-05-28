import pandas as pd
import os

def merge_excel_files(input_directory, output_file):
    # List to hold dataframes from each Excel file
    dataframes = []

    # Check if the input directory exists
    if not os.path.isdir(input_directory):
        raise ValueError(f"Input directory '{input_directory}' does not exist")

    # Walk through the directory and its subdirectories
    for root, _, files in os.walk(input_directory):
        for filename in files:
            if filename.endswith(".xlsx") or filename.endswith(".xls"):
                # Construct the full file path
                file_path = os.path.join(root, filename)
                print(f"Processing file: {file_path}")  # Debug statement
                try:
                    # Read the Excel file
                    excel_data = pd.read_excel(file_path, sheet_name=None)
                    # Iterate over each sheet
                    for sheet_name, df in excel_data.items():
                        print(f"Processing sheet: {sheet_name} from file: {filename}")  # Debug statement
                        df['Source File'] = filename  # Add a column to track the source file
                        df['Sheet Name'] = sheet_name  # Add a column to track the sheet name
                        dataframes.append(df)
                except Exception as e:
                    print(f"Failed to read file {file_path}: {e}")  # Debug statement

    if not dataframes:
        raise ValueError("No dataframes to concatenate. Ensure the input directory and subdirectories contain Excel files with data.")

    # Concatenate all dataframes
    merged_df = pd.concat(dataframes, ignore_index=True)

    # Write the merged dataframe to an Excel file
    merged_df.to_excel(output_file, index=False)
    print(f"Merged file saved as: {output_file}")  # Debug statement

# Example usage
input_directory = 'C:/Users/Dell/Downloads/NSMQ Past Questions'  # Replace with the path to your directory
output_file = 'merged_output.xlsx'  # Replace with your desired output file name
merge_excel_files(input_directory, output_file)
