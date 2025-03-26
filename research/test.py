import ollama
import os
import json
import time
import requests
import io
from dotenv import load_dotenv
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload

# Load environment variables
load_dotenv()

# Define OAuth scopes
SCOPES = [
    "https://www.googleapis.com/auth/presentations",
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

    slides_service = build("slides", "v1", credentials=creds)
    drive_service = build("drive", "v3", credentials=creds)
    
    return slides_service, drive_service

def search_and_download_unsplash_image(query):
    """Search for an image on Unsplash and return its content as bytes"""
    try:
        url = f'https://api.unsplash.com/search/photos?query={query}'
        headers = {'Authorization': f'Client-ID {os.getenv("UNSPLASH_ACCESS_KEY")}'}
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            search_results = response.json()
            if search_results['results']:
                first_photo = search_results['results'][0]
                image_url = first_photo['urls']['regular']
                image_response = requests.get(image_url)
                if image_response.status_code == 200:
                    return image_response.content  # Return bytes instead of saving
                else:
                    print(f"Error downloading image: {image_response.status_code}")
            else:
                print("No results found for the query.")
        else:
            print(f"API Error: {response.status_code}")
    except Exception as e:
        print(f"Error in image search and download: {e}")
    return None

def generate_chart_image(chart_type, categories, data):
    """Generate a chart image using Matplotlib and return it as bytes"""
    try:
        import matplotlib.pyplot as plt
        buf = io.BytesIO()
        plt.figure(figsize=(6, 4))
        if chart_type == "bar":
            plt.bar(categories, data)
        elif chart_type == "pie":
            plt.pie(data, labels=categories, autopct='%1.1f%%')
        elif chart_type == "line":
            plt.plot(categories, data)
        else:
            print(f"Unsupported chart type: {chart_type}")
            return None
        plt.title("Chart")
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        return buf.getvalue()  # Return bytes
    except ImportError:
        print("Matplotlib is not installed. Please install it with 'pip install matplotlib'.")
        return None
    except Exception as e:
        print(f"Error generating chart: {e}")
        return None

def upload_image_bytes_to_slide(slides_service, drive_service, presentation_id, slide_id, image_bytes, image_name, translateX, translateY, width, height):
    """Upload image bytes to Google Drive and add it to a slide"""
    try:
        file_metadata = {'name': image_name}
        media = MediaIoBaseUpload(io.BytesIO(image_bytes), mimetype='image/png')
        uploaded_file = drive_service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id'
        ).execute()
        file_id = uploaded_file.get('id')
        permission = {'type': 'anyone', 'role': 'reader'}
        drive_service.permissions().create(fileId=file_id, body=permission).execute()
        image_url = f"https://drive.google.com/uc?id={file_id}"
        create_image_request = {
            "createImage": {
                "url": image_url,
                "elementProperties": {
                    "pageObjectId": slide_id,
                    "size": {
                        "width": {"magnitude": width, "unit": "PT"},
                        "height": {"magnitude": height, "unit": "PT"}
                    },
                    "transform": {
                        "scaleX": 1,
                        "scaleY": 1,
                        "translateX": translateX,
                        "translateY": translateY,
                        "unit": "PT"
                    }
                }
            }
        }
        slides_service.presentations().batchUpdate(
            presentationId=presentation_id,
            body={"requests": [create_image_request]}
        ).execute()
        print(f"Successfully added image {image_name} to slide {slide_id}")
        return True
    except Exception as e:
        print(f"Error adding image to slide: {e}")
        return False

def add_slide(slides_service, drive_service, presentation_id, slide_data, layout="TITLE_AND_BODY"):
    """Add a slide to the presentation with specified layout and content"""
    # Create slide request
    create_slide_request = {
        "createSlide": {
            "slideLayoutReference": {"predefinedLayout": layout}
        }
    }
    
    response = slides_service.presentations().batchUpdate(
        presentationId=presentation_id,
        body={"requests": [create_slide_request]}
    ).execute()
    
    slide_id = response['replies'][0]['createSlide']['objectId']
    
    # Get placeholders
    presentation = slides_service.presentations().get(presentationId=presentation_id).execute()
    title_id = None
    subtitle_id = None
    body_id = None
    
    for slide in presentation['slides']:
        if slide['objectId'] == slide_id:
            for element in slide.get('pageElements', []):
                shape = element.get('shape', {})
                placeholder = shape.get('placeholder', {})
                if layout == "TITLE":
                    if placeholder.get('type') == 'CENTER_TITLE':
                        title_id = element['objectId']
                    elif placeholder.get('type') == 'SUBTITLE':
                        subtitle_id = element['objectId']
                elif layout == "TITLE_AND_BODY":
                    if placeholder.get('type') == 'TITLE':
                        title_id = element['objectId']
                    elif placeholder.get('type') == 'BODY':
                        body_id = element['objectId']
    
    # Prepare text requests
    requests = []
    
    if layout == "TITLE":
        if title_id and 'title' in slide_data:
            requests.append({
                "insertText": {
                    "objectId": title_id,
                    "insertionIndex": 0,
                    "text": slide_data['title']
                }
            })
        if subtitle_id and 'subtitle' in slide_data:
            requests.append({
                "insertText": {
                    "objectId": subtitle_id,
                    "insertionIndex": 0,
                    "text": slide_data['subtitle']
                }
            })
    elif layout == "TITLE_AND_BODY":
        if title_id:
            requests.append({
                "insertText": {
                    "objectId": title_id,
                    "insertionIndex": 0,
                    "text": slide_data.get('title', 'Untitled Slide')
                }
            })
        if body_id:
            content_text = "\n".join([f"• {item}" for item in slide_data.get('content', [])])
            requests.append({
                "insertText": {
                    "objectId": body_id,
                    "insertionIndex": 0,
                    "text": content_text
                }
            })
    
    if requests:
        slides_service.presentations().batchUpdate(
            presentationId=presentation_id,
            body={"requests": requests}
        ).execute()
    
    # Add visual elements for content slides
    if layout == "TITLE_AND_BODY" and 'visual' in slide_data:
        visual = slide_data['visual']
        if 'visual' in slide_data and slide_data['visual'] is not None:
            visual = slide_data['visual']
            if visual['type'] == "image":
                # Handle image logic (e.g., download and upload an image)
                image_bytes = search_and_download_unsplash_image(visual['description'])
                if image_bytes:
                    upload_image_bytes_to_slide(
                        slides_service, drive_service, presentation_id, slide_id,
                        image_bytes, f"image_{slide_id}.jpg",
                        translateX=50, translateY=280, width=600, height=120
                    )
            elif visual['type'] == "chart":
                # Handle chart logic
                chart_bytes = generate_chart_image(
                    visual['chart_type'], visual['categories'], visual['data']
                )
                if chart_bytes:
                    upload_image_bytes_to_slide(
                        slides_service, drive_service, presentation_id, slide_id,
                        chart_bytes, f"chart_{slide_id}.png",
                        translateX=50, translateY=280, width=600, height=120
                    )
            elif visual['type'] == "table":
                # Handle table logic
                table_data = visual
                rows = table_data['rows']
                cols = table_data['cols']
                table_request = {
                    "createTable": {
                        "elementProperties": {
                            "pageObjectId": slide_id,
                            "size": {
                                "width": {"magnitude": 600, "unit": "PT"},
                                "height": {"magnitude": 120, "unit": "PT"}
                            },
                            "transform": {
                                "scaleX": 1,
                                "scaleY": 1,
                                "translateX": 50,
                                "translateY": 280,
                                "unit": "PT"
                            }
                        },
                        "rows": rows,
                        "columns": cols
                    }
                }
                response = slides_service.presentations().batchUpdate(
                    presentationId=presentation_id,
                    body={"requests": [table_request]}
                ).execute()
                table_id = response['replies'][0]['createTable']['objectId']
                
                cell_requests = []
                for i in range(rows):
                    for j in range(cols):
                        cell_requests.append({
                            "insertText": {
                                "objectId": table_id,
                                "cellLocation": {"rowIndex": i, "columnIndex": j},
                                "text": str(table_data['data'][i][j])
                            }
                        })
                if cell_requests:
                    slides_service.presentations().batchUpdate(
                        presentationId=presentation_id,
                        body={"requests": cell_requests}
                    ).execute()
    
    return slide_id

def create_presentation(slides_service, title):
    """Create a new Google Slides presentation"""
    presentation = {"title": title}
    presentation = slides_service.presentations().create(body=presentation).execute()
    return presentation["presentationId"]

def generate_presentation_content(topic):
    """Generate slide content using LLaMA"""
    prompt = f"""Create a structured PowerPoint presentation outline on the topic: {topic}. 
    The presentation should include a title slide followed by 4-5 content slides.
    For the title slide, provide the presentation title and a subtitle (e.g., "Presented by [Your Name]").
    For each content slide, include a title, up to 3 bullet points, and optionally one visual element: either an image, a chart, or a table.
    - For an image, provide a descriptive image search query under "description".
    - For a chart, specify the type (bar, pie, line) under "chart_type", categories, and data series.
    - For a table, specify the number of rows and columns, and the cell contents under "data".
    Respond in the following JSON format:
    {{
        "slides": [
            {{
                "type": "title",
                "title": "Presentation Title",
                "subtitle": "Presentation Subtitle"
            }},
            {{
                "type": "content",
                "title": "Slide Title",
                "content": ["Bullet point 1", "Bullet point 2", "Bullet point 3"],
                "visual": {{
                    "type": "image",
                    "description": "Descriptive image search query"
                }}
            }},
            {{
                "type": "content",
                "title": "Slide Title",
                "content": ["Bullet point 1", "Bullet point 2"],
                "visual": {{
                    "type": "chart",
                    "chart_type": "bar",
                    "categories": ["Category1", "Category2", "Category3"],
                    "data": [10, 20, 30]
                }}
            }},
            {{
                "type": "content",
                "title": "Slide Title",
                "content": ["Bullet point 1"],
                "visual": {{
                    "type": "table",
                    "rows": 3,
                    "cols": 2,
                    "data": [["Cell1", "Cell2"], ["Cell3", "Cell4"], ["Cell5", "Cell6"]]
                }}
            }}
        ]
    }}
    Ensure that each content slide has at most one visual element (image, chart, or table).
    """
    
    response = ollama.chat(model="llama3.2:3b", messages=[{"role": "user", "content": prompt}])
    if 'message' in response:
        return response['message']['content']
    return "Error generating content."

def main():
    # Get topic from user
    topic = input("Enter the topic for your presentation: ")

    # Authenticate with Google
    print("Authenticating with Google...")
    slides_service, drive_service = authenticate_google()

    # Generate presentation content
    print("Generating content using LLaMA 3.2...")
    presentation_data = generate_presentation_content(topic)

    # Parse the JSON content
    try:
        import re
        json_match = re.search(r'\{.*\}', presentation_data, re.DOTALL)
        if json_match:
            slides_data = json.loads(json_match.group(0))
            print(slides_data)
        else:
            raise ValueError("No JSON found in the response")
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error parsing JSON: {e}")
        print("Using fallback presentation data")
        slides_data = {
            "slides": [
                {"type": "title", "title": topic, "subtitle": "Presented by User"},
                {"type": "content", "title": "Introduction", "content": ["Overview"], "visual": {"type": "image", "description": topic}}
            ]
        }

    # Create the presentation
    print("Creating Google Slides presentation...")
    presentation_id = create_presentation(slides_service, slides_data['slides'][0]['title'] if slides_data['slides'] else topic)

    # Add slides
    for slide_info in slides_data.get('slides', []):
        layout = "TITLE" if slide_info.get('type') == "title" else "TITLE_AND_BODY"
        add_slide(slides_service, drive_service, presentation_id, slide_info, layout=layout)
        time.sleep(1)  # Prevent API rate limits

    print(f"Presentation created successfully! View it here: https://docs.google.com/presentation/d/{presentation_id}/edit")

if __name__ == "__main__":
    main()