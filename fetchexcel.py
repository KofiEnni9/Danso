import os
import pandas as pd
import gspread
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from oauth2client.service_account import ServiceAccountCredentials
import io

# Path to service account key file
SERVICE_ACCOUNT_FILE = './sublime-command-414712.json'

# Define the scopes
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

# Define the base directory where the folders and files are stored
base_dir = 'NSMQ Past Questions/NSMQ QUESTIONS SPREADSHEETS'

def authenticate_service_account():
    """Authenticate using a service account."""
    try:
        credentials = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE, scopes=SCOPES)
        drive_service = build('drive', 'v3', credentials=credentials)

        gspread_credentials = ServiceAccountCredentials.from_json_keyfile_name(SERVICE_ACCOUNT_FILE, SCOPES)
        gspread_client = gspread.authorize(gspread_credentials)
        
        return drive_service, gspread_client
    except Exception as e:
        print(f"Error during authentication: {e}")
        return None



def get_file(service, folder_path, contestNO):
    # Split the folder_path into individual folder names
    parts = folder_path.split('/')
    folder_names = parts
    
    # Start with the specified parent_id
    parent_id = '1YRtrllvZwmpADMW22jEfBnYucGV7eBsu'
    
    # Traverse the folder structure to get to the target folder
    for folder_name in folder_names:
        query = f"'{parent_id}' in parents and name = '{folder_name}' and mimeType = 'application/vnd.google-apps.folder'"
        results = service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
        folders = results.get('files', [])
        
        if not folders:
            raise FileNotFoundError(f"Folder '{folder_name}' not found")
        
        # Assuming there's only one folder with the given name
        parent_id = folders[0]['id']
    
    # Query to find the specific file within the final folder
    query = f"'{parent_id}' in parents and name = '{contestNO}' and mimeType = 'application/vnd.google-apps.spreadsheet'"
    results = service.files().list(q=query, spaces='drive', fields='files(id, name, mimeType)').execute()
    
    # Retrieve the list of files
    files = results.get('files', [])
    
    if not files:
        raise FileNotFoundError(f"File '{contestNO}' not found in folder '{folder_names[-1]}'")
    
    # Assuming there's only one file with the given name
    return files[0]

def read_google_sheet(sheet_id, sheet_name, gspread_client):
    sheet = gspread_client.open_by_key(sheet_id).worksheet(sheet_name)
    data = sheet.get_all_records()
    return pd.DataFrame(data)

def main():
    drive_service, gspread_client = authenticate_service_account()
    folder_name = "2016"             
    contestNum = "2"  
    contestNO = f"{folder_name} NSMQ contest {contestNum}" 
    sheet_name = 'Round 1'

    search_term = "Chemistry"   

    if drive_service and gspread_client:
        file = get_file(drive_service, folder_name, contestNO)
        file_id = file['id']
        
        df = read_google_sheet(file_id, sheet_name, gspread_client)
        print(df)
    else:
        print("Failed to authenticate service account.")


if __name__ == '__main__':
    main()
