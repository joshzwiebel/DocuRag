import os
import io
import logging
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from googleapiclient.errors import HttpError
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GoogleDriveClient:
    def __init__(self, service_account_file, scopes):
        self.service_account_file = service_account_file
        self.scopes = scopes
        self.service = self.authenticate()

    def authenticate(self):
        logger.info("Authenticating with service account file: %s", self.service_account_file)
        credentials = service_account.Credentials.from_service_account_file(
            self.service_account_file, scopes=self.scopes)
        return build('drive', 'v3', credentials=credentials)

    def list_files(self, page_size=100):
        logger.info("Listing files with page size: %d", page_size)
        results = self.service.files().list(pageSize=page_size, fields="nextPageToken, files(id, name, mimeType)").execute()
        items = results.get('files', [])

        if not items:
            logger.info('No files found.')
        else:
            logger.info('Files:')
            for item in items:
                logger.info("%s (%s)", item['name'], item['id'])
        return items

    def download_file(self, file_id, file_name):
        logger.info("Downloading file with ID: %s to %s", file_id, file_name)
        request = self.service.files().get_media(fileId=file_id)
        fh = io.FileIO(file_name, 'wb')
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
            logger.info("Download %d%%.", int(status.progress() * 100))

    def get_file_content(self, file_id):
        try:
            # Attempt to download the file content
            request = self.service.files().get_media(fileId=file_id)
            response = request.execute()
            return response.decode('utf-8')
        except HttpError as error:
            if error.resp.status == 403 and 'fileNotDownloadable' in str(error):
                # Handle Google Docs Editors files by exporting them
                logger.info("File is not directly downloadable. Attempting to export.")
                request = self.service.files().export(fileId=file_id, mimeType='text/plain')
                response = request.execute()
                return response.decode('utf-8')
            else:
                logger.error("An error occurred: %s", error)
                raise ValueError(f"An error occurred: {error}")

    def export_file(self, file_id, mime_type):
        try:
            if mime_type == 'application/vnd.google-apps.document':
                export_mime_type = 'text/plain'
            elif mime_type == 'application/vnd.google-apps.spreadsheet':
                export_mime_type = 'text/csv'
            elif mime_type == 'application/vnd.google-apps.presentation':
                export_mime_type = 'application/pdf'
            else:
                export_mime_type = 'application/pdf'  # Default export type

            request = self.service.files().export(fileId=file_id, mimeType=export_mime_type)
            response = request.execute()
            return response.decode('utf-8')
        except HttpError as error:
            logger.error("An error occurred while exporting file %s: %s", file_id, error)
            return None

    def list_files_in_folder(self, folder_id):
        logger.info("Listing files in folder with ID: %s", folder_id)
        results = self.service.files().list(q=f"'{folder_id}' in parents", fields="files(id, name)").execute()
        items = results.get('files', [])
        if not items:
            logger.info('No files found.')
        else:
            logger.info('Files:')
            for item in items:
                logger.info("%s (%s)", item['name'], item['id'])

if __name__ == '__main__':
    load_dotenv()
    SERVICE_ACCOUNT_FILE = os.getenv("SERVICE_ACCOUNT_FILE")
    SCOPES = os.getenv("SCOPES")
    gdrive_client = GoogleDriveClient(SERVICE_ACCOUNT_FILE, [SCOPES])
    gdrive_client.list_files()  