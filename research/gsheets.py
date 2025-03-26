import os
import re
import json
import ollama
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Define OAuth scopes
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.file",
    "https://www.googleapis.com/auth/drive"
]

def authenticate_google():
    """Authenticate using OAuth and return API services"""
    creds = None
    token_file = "token.json"

    if os.path.exists(token_file):
        creds = Credentials.from_authorized_user_file(token_file, SCOPES)

    if not creds or not creds.valid:
        flow = InstalledAppFlow.from_client_config(
            {
                "installed": {
                    "client_id": os.getenv("GOOGLE_CLIENT_ID"),
                    "client_secret": os.getenv("GOOGLE_CLIENT_SECRET"),
                    "auth_uri": os.getenv("GOOGLE_AUTH_URI"),
                    "token_uri": os.getenv("GOOGLE_TOKEN_URI"),
                    "auth_provider_x509_cert_url": os.getenv("GOOGLE_AUTH_PROVIDER_CERT_URL"),
                    "redirect_uris": [os.getenv("GOOGLE_REDIRECT_URIS")]
                }
            },
            SCOPES
        )
        creds = flow.run_local_server(port=8080)
        with open(token_file, "w") as token:
            token.write(creds.to_json())

    sheets_service = build("sheets", "v4", credentials=creds)
    drive_service = build("drive", "v3", credentials=creds)
    
    return sheets_service, drive_service

def generate_spreadsheet_content(topic):
    """Generate spreadsheet content using LLaMA"""
    prompt = f"""Generate a dataset for a Google Sheets spreadsheet on the topic: {topic}. 
    Provide the data in the following JSON format:
    {{
        "title": "Spreadsheet Title",
        "sheets": [
            {{
                "title": "Sheet1",
                "table": {{
                    "headers": ["Column1", "Column2", "Column3"],
                    "rows": [
                        ["Data1", "Data2", "Data3"],
                        ["Data4", "Data5", "Data6"]
                    ]
                }}
            }}
        ]
    }}
    Ensure the data is relevant to the topic and includes at least one sheet with a table.
    """
    
    response = ollama.chat(model="llama3.2:3b", messages=[{"role": "user", "content": prompt}])

    if 'message' in response:
        return response['message']['content']

    return "Error generating content."

def parse_spreadsheet_data(spreadsheet_data, topic):
    """Parse the generated spreadsheet data with fallback"""
    try:
        json_match = re.search(r'\{.*\}', spreadsheet_data, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(0))
        else:
            raise ValueError("No JSON found in the response")
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error parsing JSON: {e}")
        print("Using fallback spreadsheet data")
        data = {
            "title": topic,
            "sheets": [{
                "title": "Data",
                "table": {
                    "headers": ["Category", "Value"],
                    "rows": [
                        ["Item1", "10"],
                        ["Item2", "20"],
                        ["Item3", "30"]
                    ]
                }
            }]
        }
    return data

def create_spreadsheet(sheets_service, title):
    """Create a new Google Sheets spreadsheet"""
    spreadsheet = {
        'properties': {
            'title': title
        }
    }
    spreadsheet = sheets_service.spreadsheets().create(body=spreadsheet, fields='spreadsheetId').execute()
    return spreadsheet.get('spreadsheetId')

def get_sheet_id(sheets_service, spreadsheet_id, sheet_title):
    """Get the sheet ID by title"""
    spreadsheet = sheets_service.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()
    for sheet in spreadsheet['sheets']:
        if sheet['properties']['title'] == sheet_title:
            return sheet['properties']['sheetId']
    return None

def apply_table_borders(sheets_service, spreadsheet_id, sheet_title, num_rows, num_columns):
    """Apply borders to the table range"""
    sheet_id = get_sheet_id(sheets_service, spreadsheet_id, sheet_title)
    border_style = {
        "style": "SOLID",
        "color": {
            "red": 0.0,
            "green": 0.0,
            "blue": 0.0
        }
    }
    requests = [{
        "updateBorders": {
            "range": {
                "sheetId": sheet_id,
                "startRowIndex": 0,
                "endRowIndex": num_rows,
                "startColumnIndex": 0,
                "endColumnIndex": num_columns
            },
            "top": border_style,
            "bottom": border_style,
            "left": border_style,
            "right": border_style,
            "innerHorizontal": border_style,
            "innerVertical": border_style
        }
    }]
    sheets_service.spreadsheets().batchUpdate(
        spreadsheetId=spreadsheet_id,
        body={"requests": requests}
    ).execute()

def add_sheets_and_data(sheets_service, spreadsheet_id, sheets_data):
    """Add sheets and populate them with data"""
    for i, sheet in enumerate(sheets_data):
        if i == 0:
            # Rename the default sheet to the desired title
            spreadsheet = sheets_service.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()
            default_sheet = spreadsheet['sheets'][0]  # Get the first (default) sheet
            default_sheet_id = default_sheet['properties']['sheetId']
            requests = [{
                "updateSheetProperties": {
                    "properties": {
                        "sheetId": default_sheet_id,
                        "title": sheet['title']  # e.g., "Sheet1: Guest List"
                    },
                    "fields": "title"
                }
            }]
            sheets_service.spreadsheets().batchUpdate(
                spreadsheetId=spreadsheet_id,
                body={"requests": requests}
            ).execute()
        else:
            # Create additional sheets with the specified titles
            sheets_service.spreadsheets().batchUpdate(
                spreadsheetId=spreadsheet_id,
                body={
                    "requests": [{
                        "addSheet": {
                            "properties": {
                                "title": sheet['title']
                            }
                        }
                    }]
                }
            ).execute()
        
        # Populate the sheet with data using a properly formatted range
        range_name = f"'{sheet['title']}'!A1"  # e.g., "'Sheet1: Guest List'!A1"
        values = [sheet['table']['headers']] + sheet['table']['rows']
        body = {
            'values': values
        }
        sheets_service.spreadsheets().values().update(
            spreadsheetId=spreadsheet_id,
            range=range_name,
            valueInputOption="RAW",
            body=body
        ).execute()
        
        # Apply borders to the table
        num_columns = len(sheet['table']['headers'])
        num_rows = len(sheet['table']['rows']) + 1  # Including header
        apply_table_borders(sheets_service, spreadsheet_id, sheet['title'], num_rows, num_columns)

def format_headers(sheets_service, spreadsheet_id, sheets_data):
    """Apply bold formatting to headers"""
    for sheet in sheets_data:
        sheet_id = get_sheet_id(sheets_service, spreadsheet_id, sheet['title'])
        requests = [{
            "repeatCell": {
                "range": {
                    "sheetId": sheet_id,
                    "startRowIndex": 0,
                    "endRowIndex": 1
                },
                "cell": {
                    "userEnteredFormat": {
                        "textFormat": {
                            "bold": True
                        }
                    }
                },
                "fields": "userEnteredFormat.textFormat.bold"
            }
        }]
        sheets_service.spreadsheets().batchUpdate(
            spreadsheetId=spreadsheet_id,
            body={"requests": requests}
        ).execute()

def add_chart(sheets_service, spreadsheet_id, sheet_title, num_columns):
    """Add a basic column chart to the sheet"""
    sheet_id = get_sheet_id(sheets_service, spreadsheet_id, sheet_title)
    requests = [{
        "addChart": {
            "chart": {
                "spec": {
                    "title": "Data Chart",
                    "basicChart": {
                        "chartType": "COLUMN",
                        "legendPosition": "BOTTOM_LEGEND",
                        "axis": [
                            {
                                "position": "BOTTOM_AXIS",
                                "title": "Categories"
                            },
                            {
                                "position": "LEFT_AXIS",
                                "title": "Values"
                            }
                        ],
                        "domains": [
                            {
                                "domain": {
                                    "sourceRange": {
                                        "sources": [
                                            {
                                                "sheetId": sheet_id,
                                                "startRowIndex": 0,
                                                "endRowIndex": 100,
                                                "startColumnIndex": 0,
                                                "endColumnIndex": 1
                                            }
                                        ]
                                    }
                                }
                            }
                        ],
                        "series": [
                            {
                                "series": {
                                    "sourceRange": {
                                        "sources": [
                                            {
                                                "sheetId": sheet_id,
                                                "startRowIndex": 0,
                                                "endRowIndex": 100,
                                                "startColumnIndex": i,
                                                "endColumnIndex": i + 1
                                            }
                                        ]
                                    }
                                },
                                "targetAxis": "LEFT_AXIS"
                            } for i in range(1, num_columns)
                        ]
                    }
                },
                "position": {
                    "overlayPosition": {
                        "anchorCell": {
                            "sheetId": sheet_id,
                            "rowIndex": 0,
                            "columnIndex": num_columns + 1
                        },
                        "offsetXPixels": 0,
                        "offsetYPixels": 0
                    }
                }
            }
        }
    }]
    sheets_service.spreadsheets().batchUpdate(
        spreadsheetId=spreadsheet_id,
        body={"requests": requests}
    ).execute()

def main():
    # Get topic from user
    topic = input("Enter the topic for your spreadsheet: ")

    # Authenticate with Google
    print("Authenticating with Google...")
    sheets_service, drive_service = authenticate_google()

    # Generate spreadsheet content
    print("Generating content using LLaMA 3.2...")
    spreadsheet_data = generate_spreadsheet_content(topic)

    # Parse the JSON content
    spreadsheet_data_parsed = parse_spreadsheet_data(spreadsheet_data, topic)

    # Create the spreadsheet
    print("Creating Google Sheets spreadsheet...")
    spreadsheet_id = create_spreadsheet(sheets_service, spreadsheet_data_parsed['title'])

    # Add sheets and data (including borders)
    add_sheets_and_data(sheets_service, spreadsheet_id, spreadsheet_data_parsed['sheets'])

    # Format headers
    format_headers(sheets_service, spreadsheet_id, spreadsheet_data_parsed['sheets'])

    # Optionally add a chart to the first sheet
    first_sheet = spreadsheet_data_parsed['sheets'][0]
    num_columns = len(first_sheet['table']['headers'])
    if num_columns > 1:
        add_chart(sheets_service, spreadsheet_id, first_sheet['title'], num_columns)

    print(f"Spreadsheet created successfully! View it here: https://docs.google.com/spreadsheets/d/{spreadsheet_id}/edit")

if __name__ == "__main__":
    main()