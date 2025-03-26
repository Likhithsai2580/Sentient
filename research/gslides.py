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
from googleapiclient.http import MediaFileUpload

# Load environment variables
load_dotenv()

# Define OAuth scopes
SCOPES = ["https://www.googleapis.com/auth/presentations", 
          "https://www.googleapis.com/auth/drive.file",
          "https://www.googleapis.com/auth/drive"]

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

def search_unsplash_image(query):
    """Search for an image using Unsplash API"""
    unsplash_access_key = os.getenv("UNSPLASH_ACCESS_KEY")
    
    try:
        url = "https://api.unsplash.com/search/photos"
        headers = {"Authorization": f"Client-ID {unsplash_access_key}"}
        params = {"query": query, "per_page": 1, "orientation": "landscape"}
        
        response = requests.get(url, headers=headers, params=params)
        data = response.json()
        
        if data.get("results") and len(data["results"]) > 0:
            return data["results"][0]["urls"]["regular"]
        else:
            print(f"No image found for query: {query}")
            return None
    except Exception as e:
        print(f"Error searching for image: {e}")
        return None

def search_and_download_unsplash_image(query):
    """Search for an image on Unsplash and download it"""
    try:
        url = f'https://api.unsplash.com/search/photos?query={query}'
        headers = {'Authorization': f'Client-ID {os.getenv("UNSPLASH_ACCESS_KEY")}'}
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            search_results = response.json()
            if search_results['results']:
                first_photo = search_results['results'][0]
                image_url = first_photo['urls']['regular']
                photo_id = first_photo['id']
                image_response = requests.get(image_url)
                if image_response.status_code == 200:
                    os.makedirs('unsplash_downloads', exist_ok=True)
                    image_path = os.path.join('unsplash_downloads', f'{photo_id}.jpg')
                    with open(image_path, 'wb') as file:
                        file.write(image_response.content)
                    return image_path
                else:
                    print(f"Error downloading image: {image_response.status_code}")
            else:
                print("No results found for the query.")
        else:
            print(f"API Error: {response.status_code}")
    except Exception as e:
        print(f"Error in image search and download: {e}")
    return None

def upload_imageURL_to_slide(slides_service, drive_service, presentation_id, slide_id, image_url):
    """Upload an image to a specific slide"""
    try:
        response = requests.get(image_url)
        if response.status_code != 200:
            print(f"Failed to download image from {image_url}")
            return False
        file = io.BytesIO(response.content)
        file_metadata = {'name': f'slide_image_{int(time.time())}.jpg'}
        media = MediaIoBaseUpload(file, mimetype='image/jpeg')
        uploaded_file = drive_service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id'
        ).execute()
        file_id = uploaded_file.get('id')
        create_image_request = {
            "createImage": {
                "url": f"https://drive.google.com/uc?id={file_id}",
                "elementProperties": {
                    "pageObjectId": slide_id,
                    "size": {
                        "width": {"magnitude": 450, "unit": "PT"},
                        "height": {"magnitude": 300, "unit": "PT"}
                    },
                    "transform": {
                        "scaleX": 1,
                        "scaleY": 1,
                        "translateX": 50,
                        "translateY": 150,
                        "unit": "PT"
                    }
                }
            }
        }
        slides_service.presentations().batchUpdate(
            presentationId=presentation_id,
            body={"requests": [create_image_request]}
        ).execute()
        return True
    except Exception as e:
        print(f"Error adding image to slide: {e}")
        return False

def upload_imagePATH_to_slide(slides_service, drive_service, presentation_id, slide_id, image_path):
    """Upload a local image to Google Drive and add it to a slide"""
    try:
        file_metadata = {'name': os.path.basename(image_path)}
        media = MediaFileUpload(image_path, mimetype='image/jpeg')
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
                        "width": {"magnitude": 300, "unit": "PT"},
                        "height": {"magnitude": 200, "unit": "PT"}
                    },
                    "transform": {
                        "scaleX": 1,
                        "scaleY": 1,
                        "translateX": 350,
                        "translateY": 200,
                        "unit": "PT"
                    }
                }
            }
        }
        slides_service.presentations().batchUpdate(
            presentationId=presentation_id,
            body={"requests": [create_image_request]}
        ).execute()
        print(f"Successfully added image from {image_path} to slide {slide_id}")
        return True
    except Exception as e:
        print(f"Error adding image to slide: {e}")
        return False

def generate_chart_image(chart_type, categories, data, output_path):
    """Generate a chart image using Matplotlib and save it to output_path."""
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 4))
        if chart_type == "bar":
            plt.bar(categories, data)
        elif chart_type == "pie":
            plt.pie(data, labels=categories, autopct='%1.1f%%')
        elif chart_type == "line":
            plt.plot(categories, data)
        else:
            print(f"Unsupported chart type: {chart_type}")
            return False
        plt.title("Chart")
        plt.savefig(output_path)
        plt.close()
        return True
    except ImportError:
        print("Matplotlib is not installed. Please install it with 'pip install matplotlib'.")
        return False
    except Exception as e:
        print(f"Error generating chart: {e}")
        return False

def add_slide(slides_service, drive_service, presentation_id, slide_data):
    """
    Add a slide to the presentation with title, content, optional image, chart, and table
    
    Args:
        slides_service: Google Slides API service
        drive_service: Google Drive API service
        presentation_id: ID of the presentation
        slide_data: Dictionary containing slide information
    """
    # Create slide request
    create_slide_request = {
        "createSlide": {
            "slideLayoutReference": {"predefinedLayout": "TITLE_AND_BODY"}
        }
    }
    
    # Execute slide creation
    response = slides_service.presentations().batchUpdate(
        presentationId=presentation_id, 
        body={"requests": [create_slide_request]}
    ).execute()
    
    # Get the new slide's ID
    slide_id = response['replies'][0]['createSlide']['objectId']
    
    # Get the presentation to find placeholders
    presentation = slides_service.presentations().get(
        presentationId=presentation_id
    ).execute()
    
    # Find title and body placeholder IDs
    title_id = None
    body_id = None
    
    for slide in presentation['slides']:
        if slide['objectId'] == slide_id:
            for element in slide.get('pageElements', []):
                shape = element.get('shape', {})
                placeholder = shape.get('placeholder', {})
                if placeholder.get('type') == 'TITLE':
                    title_id = element['objectId']
                elif placeholder.get('type') == 'BODY':
                    body_id = element['objectId']
    
    # Prepare requests for text
    requests = []
    
    # Add title
    if title_id:
        requests.append({
            "insertText": {
                "objectId": title_id,
                "insertionIndex": 0,
                "text": slide_data.get('title', 'Untitled Slide')
            }
        })
    
    # Add content
    if body_id:
        # Create bulleted content
        content_text = "\n".join([f"• {item}" for item in slide_data.get('content', [])])
        requests.append({
            "insertText": {
                "objectId": body_id,
                "insertionIndex": 0,
                "text": content_text
            }
        })
    
    # Execute text requests
    if requests:
        slides_service.presentations().batchUpdate(
            presentationId=presentation_id, 
            body={"requests": requests}
        ).execute()
    
    # Add image if description is provided
    if slide_data.get('image_description'):
        image_path = search_and_download_unsplash_image(slide_data['image_description'])
        if image_path:
            upload_imagePATH_to_slide(slides_service, drive_service, presentation_id, slide_id, image_path)
    
    # Add chart if specified
    if 'chart' in slide_data:
        chart_data = slide_data['chart']
        chart_type = chart_data['type']
        categories = chart_data['categories']
        data = chart_data['data']
        chart_image_path = f"chart_{slide_id}.png"
        if generate_chart_image(chart_type, categories, data, chart_image_path):
            # Upload chart image to Google Drive
            file_metadata = {'name': os.path.basename(chart_image_path)}
            media = MediaFileUpload(chart_image_path, mimetype='image/png')
            uploaded_file = drive_service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id'
            ).execute()
            file_id = uploaded_file.get('id')
            # Set permissions
            permission = {'type': 'anyone', 'role': 'reader'}
            drive_service.permissions().create(fileId=file_id, body=permission).execute()
            # Add image to slide
            image_url = f"https://drive.google.com/uc?id={file_id}"
            create_image_request = {
                "createImage": {
                    "url": image_url,
                    "elementProperties": {
                        "pageObjectId": slide_id,
                        "size": {
                            "width": {"magnitude": 400, "unit": "PT"},
                            "height": {"magnitude": 300, "unit": "PT"}
                        },
                        "transform": {
                            "scaleX": 1,
                            "scaleY": 1,
                            "translateX": 50,
                            "translateY": 100,
                            "unit": "PT"
                        }
                    }
                }
            }
            slides_service.presentations().batchUpdate(
                presentationId=presentation_id,
                body={"requests": [create_image_request]}
            ).execute()
            # Clean up local file
            # os.remove(chart_image_path)
    
    # Add table if specified
    if 'table' in slide_data:
        table_data = slide_data['table']
        rows = table_data['rows']
        cols = table_data['cols']
        table_request = {
            "createTable": {
                "elementProperties": {
                    "pageObjectId": slide_id,
                    "size": {
                        "width": {"magnitude": 400, "unit": "PT"},
                        "height": {"magnitude": 200, "unit": "PT"}
                    },
                    "transform": {
                        "scaleX": 1,
                        "scaleY": 1,
                        "translateX": 50,
                        "translateY": 300,
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
        
        # Populate table cells
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
    Provide 5-6 slides with detailed content. For each slide, include a title, bullet points, and optionally:
    - An image description for relevant visuals.
    - A chart (specify type: bar, pie, line; categories; and data series) if the slide benefits from data visualization.
    - A table (specify rows, columns, and cell contents) if the slide needs to display structured data.
    Respond in the following JSON format:
    {{
        "title": "Presentation Title",
        "slides": [
            {{
                "title": "Slide Title",
                "content": ["Bullet point 1", "Bullet point 2", "Bullet point 3"],
                "image_description": "Descriptive image search query",
                "chart": {{
                    "type": "bar",
                    "categories": ["Category1", "Category2", "Category3"],
                    "data": [10, 20, 30]
                }},
                "table": {{
                    "rows": 3,
                    "cols": 2,
                    "data": [["Cell1", "Cell2"], ["Cell3", "Cell4"], ["Cell5", "Cell6"]]
                }}
            }}
        ]
    }}
    Ensure that not every slide has a chart or table; use them only where appropriate.
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
        else:
            raise ValueError("No JSON found in the response")
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error parsing JSON: {e}")
        print("Using fallback presentation data")
        slides_data = {
            "title": topic,
            "slides": [{
                "title": "Introduction",
                "content": ["Overview of the topic", "Key points", "Importance"],
                "image_description": f"Introduction to {topic}"
            }]
        }

    # Create the presentation
    print("Creating Google Slides presentation...")
    presentation_id = create_presentation(slides_service, slides_data.get('title', topic))

    # Add slides
    for slide_info in slides_data.get('slides', []):
        add_slide(slides_service, drive_service, presentation_id, slide_info)
        time.sleep(1)  # Prevent potential API rate limits

    print(f"Presentation created successfully! View it here: https://docs.google.com/presentation/d/{presentation_id}/edit")

if __name__ == "__main__":
    main()