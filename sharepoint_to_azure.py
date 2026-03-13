"""
SharePoint to Azure File Share Pipeline

This script connects to Microsoft SharePoint using the Microsoft Graph API,
downloads policy documents from a specified SharePoint folder, and uploads
them into Azure File Share storage.

Workflow:
1. Authenticate with Azure AD
2. Access SharePoint site and drive
3. Retrieve files from the specified SharePoint folder
4. Download documents from SharePoint
5. Upload them into Azure File Share storage

Purpose:
This pipeline enables automatic synchronization of policy documents
from SharePoint to Azure storage so they can later be processed
by the RAG ingestion pipeline.
"""

import requests
from msal import ConfidentialClientApplication
from azure.storage.fileshare import ShareServiceClient
from azure.core.exceptions import ResourceExistsError
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

TENANT_ID = os.getenv("TENANT_ID")
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
SHAREPOINT_SITE = os.getenv("SHAREPOINT_SITE")
SHAREPOINT_SITE_NAME = os.getenv("SHAREPOINT_SITE_NAME")
SHAREPOINT_FOLDER_PATH = os.getenv("SHAREPOINT_FOLDER_PATH")
AZURE_STORAGE_ACCOUNT = os.getenv("AZURE_STORAGE_ACCOUNT")
AZURE_FILE_SHARE_NAME = os.getenv("AZURE_FILE_SHARE_NAME")
AZURE_STORAGE_KEY = os.getenv("AZURE_STORAGE_KEY")
AZURE_ROOT_FOLDER = os.getenv("AZURE_ROOT_FOLDER")

# Authenticate with Microsoft Azure AD and obtain access token
def get_access_token():
    app = ConfidentialClientApplication(
        CLIENT_ID,
        authority=f"https://login.microsoftonline.com/{TENANT_ID}",
        client_credential=CLIENT_SECRET
    )
    token = app.acquire_token_for_client(
        scopes=["https://graph.microsoft.com/.default"]
    )
    if "access_token" not in token:
        raise Exception(f"Failed to get token: {token}")
    return token["access_token"]

# Get SharePoint drive ID using Microsoft Graph API
def get_drive_id(headers):
    site_url = f"https://graph.microsoft.com/v1.0/sites/{SHAREPOINT_SITE}:/sites/{SHAREPOINT_SITE_NAME}"
    site_resp = requests.get(site_url, headers=headers)
    site_resp.raise_for_status()
    site_id = site_resp.json()["id"]

    drive_url = f"https://graph.microsoft.com/v1.0/sites/{site_id}/drive"
    drive_resp = requests.get(drive_url, headers=headers)
    drive_resp.raise_for_status()
    return drive_resp.json()["id"]

# Get SharePoint folder ID using Microsoft Graph API
def get_folder_id(headers, drive_id):
    url = f"https://graph.microsoft.com/v1.0/drives/{drive_id}/root:/{SHAREPOINT_FOLDER_PATH}"
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    return resp.json()["id"]

# Recursively list all files inside SharePoint folder and subfolders
def list_files_recursive(headers, drive_id, folder_id, current_path=""):
    files = []
    url = f"https://graph.microsoft.com/v1.0/drives/{drive_id}/items/{folder_id}/children"

    while url:
        resp = requests.get(url, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        for item in data.get("value", []):
            name = item["name"]
            item_id = item["id"]
            relative_path = f"{current_path}/{name}" if current_path else name
            if "folder" in item:
                files.extend(
                    list_files_recursive(headers, drive_id, item_id, relative_path)
                )
            elif "file" in item:
                files.append({
                    "id": item_id,
                    "path": relative_path
                })
        url = data.get("@odata.nextLink")
    return files

# Create directory structure in Azure File Share if it does not exist
def create_directory_if_needed(share_client, path):
    if not path:
        return
    parts = path.split("/")
    current = ""
    for part in parts:
        current = f"{current}/{part}" if current else part
        try:
            share_client.get_directory_client(current).create_directory()
        except ResourceExistsError:
            pass
        except Exception:
            pass

# Main pipeline function that downloads files from SharePoint
# and uploads them to Azure File Share storage
def upload_to_azure_file_share():
    token = get_access_token()
    headers = {"Authorization": f"Bearer {token}"}
    drive_id = get_drive_id(headers)
    folder_id = get_folder_id(headers, drive_id)
    files = list_files_recursive(headers, drive_id, folder_id)

    if not files:
        print("No files found")
        return
    service = ShareServiceClient(
        account_url=f"https://{AZURE_STORAGE_ACCOUNT}.file.core.windows.net",
        credential=AZURE_STORAGE_KEY
    )
    share_client = service.get_share_client(AZURE_FILE_SHARE_NAME)
    try:
        share_client.create_share()
    except ResourceExistsError:
        pass
    create_directory_if_needed(share_client, AZURE_ROOT_FOLDER)
    print(f"Found {len(files)} files")

    for file in files:
        file_id = file["id"]
        relative_path = file["path"]
        azure_path = f"{AZURE_ROOT_FOLDER}/{relative_path}"
        print("Uploading:", azure_path)
        parts = azure_path.split("/")
        directory = "/".join(parts[:-1])
        if directory:
            create_directory_if_needed(share_client, directory)
        download_url = f"https://graph.microsoft.com/v1.0/drives/{drive_id}/items/{file_id}/content"
        r = requests.get(download_url, headers=headers, stream=True)
        r.raise_for_status()
        data = r.content
        file_client = share_client.get_file_client(azure_path)
        file_client.create_file(len(data))
        file_client.upload_file(data)

    print("Upload complete")
if __name__ == "__main__":
    upload_to_azure_file_share()