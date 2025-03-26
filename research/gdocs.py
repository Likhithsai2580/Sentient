import os
import json
import time
import io
import ollama
import requests as req
from dotenv import load_dotenv
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload

# Load environment variables
load_dotenv()

# Define OAuth scopes
SCOPES = [
    "https://www.googleapis.com/auth/documents",
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

    docs_service = build("docs", "v1", credentials=creds)
    drive_service = build("drive", "v3", credentials=creds)
    
    return docs_service, drive_service

def search_unsplash_image(query):
    """Search for an image using Unsplash API"""
    unsplash_access_key = os.getenv("UNSPLASH_ACCESS_KEY")
    try:
        url = "https://api.unsplash.com/search/photos"
        headers = {"Authorization": f"Client-ID {unsplash_access_key}"}
        params = {"query": query, "per_page": 1, "orientation": "landscape"}
        
        response = req.get(url, headers=headers, params=params)
        data = response.json()
        
        if data.get("results") and len(data["results"]) > 0:
            return data["results"][0]["urls"]["regular"]
        return None
    except Exception as e:
        print(f"Error searching for image: {e}")
        return None

def upload_image_to_drive(drive_service, image_url):
    """Upload an image to Google Drive and return its ID"""
    try:
        response = req.get(image_url)
        if response.status_code != 200:
            print(f"Failed to download image from {image_url}")
            return None
        
        file = io.BytesIO(response.content)
        file_metadata = {'name': f'doc_image_{int(time.time())}.jpg'}
        media = MediaIoBaseUpload(file, mimetype='image/jpeg')
        uploaded_file = drive_service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id'
        ).execute()
        file_id = uploaded_file.get('id')
        
        # Set permission to anyone with link
        permission = {'type': 'anyone', 'role': 'reader'}
        drive_service.permissions().create(fileId=file_id, body=permission).execute()
        
        return file_id
    except Exception as e:
        print(f"Error uploading image to Drive: {e}")
        return None

def create_document(docs_service, title):
    """Create a new Google Docs document"""
    document = {"title": title}
    document = docs_service.documents().create(body=document).execute()
    return document["documentId"]

def generate_document_content(topic):
    """Generate document content structure"""
    prompt = f"""Create a structured Google Docs document outline on the topic: {topic}. 
    Provide 4-5 sections with detailed content. For each section, include:
    - A heading (H1 or H2 level)
    - Paragraph text (1-2 paragraphs)
    - Bullet points (3-5 items) with some words in **bold** for emphasis
    - Optional image description for relevant visuals
    Respond in the following JSON format:
    {{
        "title": "Document Title",
        "sections": [
            {{
                "heading": "Section Title",
                "heading_level": "H1" or "H2",
                "paragraphs": ["Paragraph 1 text", "Paragraph 2 text"],
                "bullet_points": ["Bullet 1 with **bold** text", "Bullet 2", "Bullet 3"],
                "image_description": "Descriptive image search query"
            }}
        ]
    }}
    """

    response = ollama.chat(model="llama3.2:3b", messages=[{"role": "user", "content": prompt}])

    if 'message' in response:
        return response['message']['content']

    return "Error generating content."
    
    # # Mock response (replace with actual LLM call if available)
    # mock_response = {
    #     "title": f"{topic} Overview",
    #     "sections": [
    #         {
    #             "heading": "Introduction",
    #             "heading_level": "H1",
    #             "paragraphs": [
    #                 f"This document provides a comprehensive overview of {topic}. It aims to inform and educate readers about key aspects and developments.",
    #                 "We'll explore various dimensions and implications of the topic."
    #             ],
    #             "bullet_points": [
    #                 "**Key** objectives of this document",
    #                 "Scope of **coverage**",
    #                 "Intended **audience**"
    #             ],
    #             "image_description": f"Introduction to {topic}"
    #         },
    #         {
    #             "heading": "Background",
    #             "heading_level": "H2",
    #             "paragraphs": [
    #                 f"The history of {topic} spans several important milestones that have shaped its current state."
    #             ],
    #             "bullet_points": [
    #                 "Major **events** in history",
    #                 "**Influential** figures",
    #                 "Key **developments**"
    #             ]
    #         }
    #     ]
    # }
    # return json.dumps(mock_response)

def add_content_to_document(docs_service, drive_service, document_id, content_data):
    """Add structured content to the Google Doc"""
    requests = []
    current_index = 1  # Start at 1 as 0 is reserved
    
    # Add document title
    title_text = content_data["title"] + "\n\n"
    requests.append({
        "insertText": {
            "location": {"index": current_index},
            "text": title_text
        }
    })
    requests.append({
        "updateParagraphStyle": {
            "range": {"startIndex": current_index, "endIndex": current_index + len(content_data["title"])},
            "paragraphStyle": {"namedStyleType": "HEADING_1"},
            "fields": "namedStyleType"
        }
    })
    current_index += len(title_text)

    # Add sections
    for section in content_data["sections"]:
        # Add heading
        heading_text = section["heading"] + "\n"
        requests.append({
            "insertText": {
                "location": {"index": current_index},
                "text": heading_text
            }
        })
        requests.append({
            "updateParagraphStyle": {
                "range": {"startIndex": current_index, "endIndex": current_index + len(section["heading"])},
                "paragraphStyle": {"namedStyleType": f"HEADING_{'1' if section['heading_level'] == 'H1' else '2'}"},
                "fields": "namedStyleType"
            }
        })
        current_index += len(heading_text)

        # Add paragraphs
        for para in section["paragraphs"]:
            para_text = para + "\n\n"
            requests.append({
                "insertText": {
                    "location": {"index": current_index},
                    "text": para_text
                }
            })
            current_index += len(para_text)

        # Add bullet points
        for bullet in section["bullet_points"]:
            bullet_text = bullet + "\n"
            requests.append({
                "insertText": {
                    "location": {"index": current_index},
                    "text": bullet_text
                }
            })
            # Add bullet formatting
            requests.append({
                "createParagraphBullets": {
                    "range": {"startIndex": current_index, "endIndex": current_index + len(bullet_text) - 1},
                    "bulletPreset": "BULLET_DISC_CIRCLE_SQUARE"
                }
            })
            # Add bold formatting
            bold_matches = [m for m in zip(
                [i for i in range(len(bullet)) if bullet[i:i+2] == "**"],
                [i for i in range(len(bullet)) if bullet[i:i+2] == "**" and i > 0]
            ) if m[1] > m[0]]
            for start, end in bold_matches:
                requests.append({
                    "updateTextStyle": {
                        "range": {
                            "startIndex": current_index + start + 2,
                            "endIndex": current_index + end
                        },
                        "textStyle": {"bold": True},
                        "fields": "bold"
                    }
                })
            current_index += len(bullet_text)

        # Add image if present
        if "image_description" in section:
            image_url = search_unsplash_image(section["image_description"])
            if image_url:
                file_id = upload_image_to_drive(drive_service, image_url)
                if file_id:
                    requests.append({
                        "insertInlineImage": {
                            "location": {"index": current_index},
                            "uri": f"https://drive.google.com/uc?id={file_id}",
                            "objectSize": {
                                "height": {"magnitude": 200, "unit": "PT"},
                                "width": {"magnitude": 300, "unit": "PT"}
                            }
                        }
                    })
                    # Increment index conservatively (image doesn't add text length, but we need space)
                    current_index += 1
                    requests.append({
                        "insertText": {
                            "location": {"index": current_index},
                            "text": "\n"
                        }
                    })
                    current_index += 1

        # Add spacing after section
        requests.append({
            "insertText": {
                "location": {"index": current_index},
                "text": "\n"
            }
        })
        current_index += 1

    # Execute all requests
    if requests:
        docs_service.documents().batchUpdate(
            documentId=document_id,
            body={"requests": requests}
        ).execute()

def main():
    # Get topic from user
    topic = input("Enter the topic for your document: ")

    # Authenticate with Google (assuming you have this function)
    print("Authenticating with Google...")
    docs_service, drive_service = authenticate_google()

    # Generate document content (e.g., from an LLM)
    print("Generating content...")
    document_data = generate_document_content(topic)  # Returns a string with JSON

    # Parse the JSON with fallback
    try:
        import re
        json_match = re.search(r'\{.*\}', document_data, re.DOTALL)
        if json_match:
            content_data = json.loads(json_match.group(0))
        else:
            raise ValueError("No JSON found in the response")
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error parsing JSON: {e}")
        print("Using fallback document data")
        content_data = {
            "title": topic,
            "sections": [{
                "heading": "Introduction",
                "paragraphs": ["This is a fallback document about the topic."],
                "bullet_points": ["Point 1", "Point 2", "Point 3"]
            }]
        }

    # Create and populate the document (assuming these functions exist)
    print("Creating Google Docs document...")
    document_id = create_document(docs_service, content_data["title"])
    add_content_to_document(docs_service, drive_service, document_id, content_data)

    print(f"Document created successfully! View it here: https://docs.google.com/document/d/{document_id}/edit")

if __name__ == "__main__":
    main()