# import pandas as pd
# import json

# xls = pd.ExcelFile('2009 NSMQ contest 1.xlsx')

# for sheet_name in xls.sheet_names:
#     df = pd.read_excel(xls, sheet_name=sheet_name)
#     data_dict = df.to_dict(orient='records')
    

#     json_filename = f'{sheet_name}.json'
#     with open(json_filename, 'w') as f:
#         json.dump(data_dict, f, indent=4)
        
#     print(f'Converted {sheet_name} sheet to {json_filename}')

import pandas as pd

# Load the Excel file
xls = pd.ExcelFile('2013 NSMQ contest 31.xlsx')

# Initialize an empty list to hold DataFrames
all_sheets_df = []

# Iterate over each sheet in the Excel file
for sheet_name in xls.sheet_names:
    # Read the sheet into a DataFrame
    df = pd.read_excel(xls, sheet_name=sheet_name)
    
    # Add a new column to identify the sheet name
    df['SheetName'] = sheet_name
    
    # Append the DataFrame to the list
    all_sheets_df.append(df)

# Concatenate all DataFrames in the list into a single DataFrame
combined_df = pd.concat(all_sheets_df, ignore_index=True)

# Define the CSV file name
csv_filename = 'combined_sheets.csv'

# Write the combined DataFrame to a CSV file
combined_df.to_csv(csv_filename, index=False)

# Print confirmation message
print(f'Converted all sheets to {csv_filename}')



import pandas as pd

# Load the Excel file
xls = pd.ExcelFile('2013 NSMQ contest 31.xlsx')

all_sheets_df = []

for sheet_name in xls.sheet_names:
    df = pd.read_excel(gsheet, sheet_name=sheet_name)
    
    df['SheetName'] = sheet_name
    
    all_sheets_df.append(df)

combined_df = pd.concat(all_sheets_df, ignore_index=True)

csv_filename = 'combined_sheets.csv'

combined_df.to_csv(csv_filename, index=False)

