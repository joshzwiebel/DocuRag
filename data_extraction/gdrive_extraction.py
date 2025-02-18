import os
import io
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from googleapiclient.errors import HttpError

class GoogleDriveClient:
    def __init__(self, service_account_file, scopes):
        self.service_account_file = service_account_file
        self.scopes = scopes
        self.service = self.authenticate()

    def authenticate(self):
        print("Authenticating with service account file:", self.service_account_file)
        credentials = service_account.Credentials.from_service_account_file(
            self.service_account_file, scopes=self.scopes)
        return build('drive', 'v3', credentials=credentials)

    def list_files(self, page_size=10):
        print("Listing files with page size:", page_size)
        results = self.service.files().list(pageSize=page_size, fields="nextPageToken, files(id, name)").execute()
        items = results.get('files', [])

        if not items:
            print('No files found.')
        else:
            print('Files:')
            for item in items:
                print(f"{item['name']} ({item['id']})")

    def download_file(self, file_id, file_name):
        print(f"Downloading file with ID: {file_id} to {file_name}")
        request = self.service.files().get_media(fileId=file_id)
        fh = io.FileIO(file_name, 'wb')
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
            print(f"Download {int(status.progress() * 100)}%.")

    def print_file_content(self, file_id):
        try:
            # Attempt to download the file content
            request = self.service.files().get_media(fileId=file_id)
            response = request.execute()
            print(response.decode('utf-8'))
        except HttpError as error:
            if error.resp.status == 403 and 'fileNotDownloadable' in str(error):
                # Handle Google Docs Editors files by exporting them
                print("File is not directly downloadable. Attempting to export.")
                request = self.service.files().export(fileId=file_id, mimeType='text/plain')
                response = request.execute()
                print(response.decode('utf-8'))
            else:
                raise ValueError(f"An error occurred: {error}")
    
    def list_files_in_folder(self, folder_id):
        results = self.service.files().list(q=f"'{folder_id}' in parents", fields="files(id, name)").execute()
        items = results.get('files', [])
        if not items:
            print('No files found.')
        else:
            print('Files:')
            for item in items:
                print(f"{item['name']} ({item['id']})")

if __name__ == '__main__':
    SERVICE_ACCOUNT_FILE = 'config/credentials.json'
    SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

    gdrive_client = GoogleDriveClient(SERVICE_ACCOUNT_FILE, SCOPES)
    gdrive_client.list_files()