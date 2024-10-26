import os
import io
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

# Define the scopes for the access (Google Drive read/write access)
SCOPES = ['https://www.googleapis.com/auth/drive']

# Path to the credentials file you downloaded
CREDENTIALS_FILE = 'token_drive.json'


def authenticate_drive():
    """Authenticate the user and return the Google Drive service."""
    creds = None

    # Token file stores user's access and refresh tokens, and is created automatically
    # when the authorization flow completes for the first time.
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)

    # If no valid credentials, ask user to log in
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)

        # Save the credentials for the next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    # Build the Google Drive service
    return build('drive', 'v3', credentials=creds)


def list_files_and_folders_in_folder(service, folder_id):
    """List all files and folders in a specific Google Drive folder."""
    query = f"'{folder_id}' in parents"
    results = service.files().list(q=query, pageSize=1000, fields="files(id, name, mimeType)").execute()
    items = results.get('files', [])

    if not items:
        print(f'No files or folders found in folder ID {folder_id}.')
        return []
    else:
        return items


def download_file(service, file_id, file_name, output_dir):
    """Download a file from Google Drive by file ID."""
    request = service.files().get_media(fileId=file_id)
    file_path = os.path.join(output_dir, file_name)
    fh = io.FileIO(file_path, 'wb')  # Write the file to disk

    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
        print(f"Downloading {file_name}: {int(status.progress() * 100)}% complete.")

    print(f"File {file_name} downloaded to {file_path}.")


def download_files_recursively(service, folder_id, output_dir):
    """Recursively download all files and folders from a Google Drive folder."""
    # Get all items (files and subfolders) in the current folder
    items = list_files_and_folders_in_folder(service, folder_id)

    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for item in items:
        file_id = item['id']
        file_name = item['name']
        mime_type = item['mimeType']

        if mime_type == 'application/vnd.google-apps.folder':
            # If the item is a folder, recursively download its contents
            print(f"Entering folder: {file_name}")
            new_output_dir = os.path.join(output_dir, file_name)
            download_files_recursively(service, file_id, new_output_dir)
        elif file_name.endswith('.h5'):
            # If the item is a file, download it
            print(f"Downloading file: {file_name}")
            download_file(service, file_id, file_name, output_dir)
