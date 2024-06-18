# CONVERTION OF .XLXS DATASET TO GOOGLE SHEET FORMAT FOR EASIER FORMATTING #
# ------------------------------------------------------------------------ #

from google.colab import drive
import gspread
from google.auth.transport.requests import Request
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build

# Mount Google Drive
drive.mount('/content/drive')

# Path to service account key file
SERVICE_ACCOUNT_FILE = '/content/drive/My Drive/secrets/sublime-command-414712.json'

# Define the scopes
SCOPES = ['https://www.googleapis.com/auth/drive', 'https://www.googleapis.com/auth/spreadsheets']

def authenticate_service_account():
    """Authenticate using a service account."""
    try:
        credentials = Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE, scopes=SCOPES)
        gc = gspread.authorize(credentials.with_scopes(SCOPES))
        drive_service = build('drive', 'v3', credentials=credentials)
        return drive_service, gc
    except Exception as e:
        print(f"Error during authentication: {e}")
        return None, None

def list_files_in_folder(service, folder_id):
    """List all .xlsx files and folders in a folder."""
    query = f"'{folder_id}' in parents and (mimeType = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' or mimeType = 'application/vnd.google-apps.folder')"
    results = service.files().list(q=query, spaces='drive', fields='files(id, name, mimeType)').execute()
    return results.get('files', [])

def list_sheets_in_folder(service, folder_id):
    """List all Google Sheets in a folder."""
    query = f"'{folder_id}' in parents and mimeType = 'application/vnd.google-apps.spreadsheet'"
    results = service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
    return results.get('files', [])

def convert_excel_to_sheets(service, file_id, file_name):
    """Convert an Excel file to Google Sheets format."""
    file_metadata = {
        'name': file_name.replace('.xlsx', ''),
        'mimeType': 'application/vnd.google-apps.spreadsheet'
    }
    converted_file = service.files().copy(fileId=file_id, body=file_metadata).execute()
    return converted_file['id']

def convert_all_excel_files(service, folder_id):
    files_to_convert = []
    folders_to_check = [folder_id]

    while folders_to_check:
        current_folder_id = folders_to_check.pop()
        files = list_files_in_folder(service, current_folder_id)
        
        for file in files:
            if file['mimeType'] == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
                files_to_convert.append(file)
            elif file['mimeType'] == 'application/vnd.google-apps.folder':
                folders_to_check.append(file['id'])

    existing_sheets = list_sheets_in_folder(service, folder_id)
    existing_sheets_names = [sheet['name'] for sheet in existing_sheets]

    for file in files_to_convert:
        sheet_name = file['name'].replace('.xlsx', '')
        if sheet_name in existing_sheets_names:
            print(f"Skipping conversion for {file['name']}, Google Sheet already exists.")
        else:
            print(f"Converting file: {file['name']} (ID: {file['id']})")
            converted_sheet_id = convert_excel_to_sheets(service, file['id'], file['name'])
            print(f"Converted to Google Sheets with ID: {converted_sheet_id}")

def main():
    drive_service, gc = authenticate_service_account()
    if not drive_service:
        print("Failed to authenticate service account.")
        return

    base_folder_id = '11scjTp305ltJL96YV5e1IvAFB1LONvDJ'  # Replace with your folder ID
    convert_all_excel_files(drive_service, base_folder_id)

if __name__ == '__main__':
    main()
