# import os
# from google.oauth2 import service_account
# from googleapiclient.discovery import build

# # Path to service account key file
# SERVICE_ACCOUNT_FILE = './sublime-command-414712.json'

# # Define the scopes
# SCOPES = ['https://www.googleapis.com/auth/drive', 'https://www.googleapis.com/auth/spreadsheets']

# def authenticate_service_account():
#     """Authenticate using a service account."""
#     try:
#         credentials = service_account.Credentials.from_service_account_file(
#             SERVICE_ACCOUNT_FILE, scopes=SCOPES)
#         drive_service = build('drive', 'v3', credentials=credentials)
#         return drive_service
#     except Exception as e:
#         print(f"Error during authentication: {e}")
#         return None

# def list_files_in_folder(service, folder_id):
#     """List all .xlsx files in a folder and its subfolders."""
#     query = f"'{folder_id}' in parents and (mimeType = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' or mimeType = 'application/vnd.google-apps.folder')"
#     results = service.files().list(q=query, spaces='drive', fields='files(id, name, mimeType)').execute()
#     return results.get('files', [])

# def convert_excel_to_sheets(service, file_id, file_name):
#     """Convert an Excel file to Google Sheets format."""
#     file_metadata = {
#         'name': file_name.replace('.xlsx', ''),
#         'mimeType': 'application/vnd.google-apps.spreadsheet'
#     }
#     converted_file = service.files().copy(fileId=file_id, body=file_metadata).execute()
#     return converted_file['id']

# def convert_all_excel_files(service, folder_id):
#     files_to_convert = []
#     folders_to_check = [folder_id]

#     while folders_to_check:
#         current_folder_id = folders_to_check.pop()
#         files = list_files_in_folder(service, current_folder_id)
        
#         for file in files:
#             if file['mimeType'] == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
#                 files_to_convert.append(file)
#             elif file['mimeType'] == 'application/vnd.google-apps.folder':
#                 folders_to_check.append(file['id'])

#     for file in files_to_convert:
#         print(f"Converting file: {file['name']} (ID: {file['id']})")
#         converted_sheet_id = convert_excel_to_sheets(service, file['id'], file['name'])
#         print(f"Converted to Google Sheets with ID: {converted_sheet_id}")

# def main():
#     drive_service = authenticate_service_account()
#     if not drive_service:
#         print("Failed to authenticate service account.")
#         return

#     base_folder_id = '1n5xfGxNDTWrQglgd21PhP4o_QCwsc65o'  # Replace with your folder ID
#     convert_all_excel_files(drive_service, base_folder_id)

# if __name__ == '__main__':
#     main()
