
# from google.colab import drive
# drive.mount('/content/drive')

# !pip install -q gspread oauth2client

# from oauth2client.service_account import ServiceAccountCredentials
# import gspread
# from googleapiclient.discovery import build
# from google.colab import auth
# import pandas as pd


# # Authenticate and create the PyDrive client
# auth.authenticate_user()

# # Path to service account key file
# SERVICE_ACCOUNT_FILE = '/content/drive/My Drive/secrets/sublime-command-414712.json'

# # Define the scopes
# SCOPES = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']

# # Authenticate using a service account
# credentials = ServiceAccountCredentials.from_json_keyfile_name(SERVICE_ACCOUNT_FILE, SCOPES)
# gc = gspread.authorize(credentials)
# drive_service = build('drive', 'v3', credentials=credentials)

# # folder IDs for the years to access
# folder_ids = {
#     '2016': '1EL42UWE11bass0yMDz6eIE4vOweX5uXl',
#     '2017': '11scjTp305ltJL96YV5e1IvAFB1LONvDJ',
#     '2018': '1p7y6Q7GxCaRUmmdNVA58VdgS3T55jD8G',
#     '2019': '1wuVYGil8TJ13qiewT21JfkaWVqctqpXT',
#     '2020': '1eo2k0KJwyyVYVjwH2kJaKZsItIZis0EL',
#     '2021': '199X5s1oNKoREjNrZup0ivk7SAtDPaMwN',
# }

# all_sheets_df = []

# import time


# for year, folder_id in folder_ids.items():
#     # List all .xlsx files in the specified folder
#     query = f"'{folder_id}' in parents and mimeType='application/vnd.google-apps.spreadsheet'"
#     results = drive_service.files().list(q=query).execute()
#     items = results.get('files', [])

#     if not items:
#         print(f'No files found in the {year} folder.')
#     else:
#         print(f"Found {len(items)} files in the {year} folder.")

#     for item in items:
#         file_id = item['id']
#         file_name = item['name']
#         try:
#             sheet = gc.open_by_key(file_id)

#             for worksheet in sheet.worksheets():
#                 worksheet_title = worksheet.title
#                 data = worksheet.get_all_values()

#                 # Convert data to a DataFrame
#                 df = pd.DataFrame(data)

#                 # Set the first row as headers if necessary
#                 df.columns = df.iloc[0]
#                 df = df[1:]

#                 # Drop the specified columns
#                 columns_to_remove = ["Answer has a figure", "Has Preamble", "calculations present"]
#                 df = df.drop(columns=[col for col in columns_to_remove if col in df.columns])

#                 df['Quiz Round'] = worksheet_title
#                 df['Year_Round'] = file_name

#                 all_sheets_df.append(df)

#                 time.sleep(2)

#         except Exception as e:
#             print(f"Failed to read file {file_name} with id {file_id}: {e}")

# # Combine all dataframes into one
# combined_df = pd.concat(all_sheets_df, ignore_index=True)

# # Save the combined dataframe to a CSV file
# csv_filename = '/content/combined_sheets.csv'
# combined_df.to_csv(csv_filename, index=False)

# print(f"Combined CSV file created at: {csv_filename}")