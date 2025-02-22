from sentence_transformers import SentenceTransformer
import numpy as np
from numpy.linalg import norm
import requests
import json
from helpers import *
from prompts import *
import sqlite3
from datetime import datetime, date, timedelta
import spacy
from typing import List, Dict, Optional
from langchain_community.chat_models import ChatOllama
import os
import ast
from dotenv import load_dotenv

load_dotenv()

# Load the English language model for SpaCy
nlp = spacy.load("en_core_web_sm")

def adapt_date_iso(val: date) -> str:
    """Adapt datetime.date to ISO 8601 date for SQLite storage.

    SQLite natively does not support Date or DateTime datatypes.
    This adapter converts Python's datetime.date object into an ISO 8601 string
    for storing in SQLite TEXT columns.

    Args:
        val (date): The datetime.date object to adapt.

    Returns:
        str: ISO 8601 formatted date string.
    """
    return val.isoformat()

def adapt_datetime_iso(val: datetime) -> str:
    """Adapt datetime.datetime to timezone-naive ISO 8601 date for SQLite storage.

    Similar to `adapt_date_iso`, but for datetime.datetime objects.
    This adapter converts Python's datetime.datetime object into a timezone-naive
    ISO 8601 string for storing in SQLite TEXT columns.

    Args:
        val (datetime): The datetime.datetime object to adapt.

    Returns:
        str: Timezone-naive ISO 8601 formatted datetime string.
    """
    return val.isoformat()

def adapt_datetime_epoch(val: datetime) -> int:
    """Adapt datetime.datetime to Unix timestamp for SQLite storage.

    Converts a datetime.datetime object to a Unix timestamp (seconds since epoch)
    for storing in SQLite INTEGER columns. This can be useful for efficient
    range queries and comparisons in SQLite.

    Args:
        val (datetime): The datetime.datetime object to adapt.

    Returns:
        int: Unix timestamp (integer).
    """
    return int(val.timestamp())

sqlite3.register_adapter(date, adapt_date_iso)
sqlite3.register_adapter(datetime, adapt_datetime_iso)

def convert_date(val: bytes) -> date:
    """Convert ISO 8601 date from SQLite to datetime.date object.

    This converter is registered with SQLite to automatically convert
    ISO 8601 date strings retrieved from the database back into
    Python datetime.date objects.

    Args:
        val (bytes): ISO 8601 date string from SQLite (as bytes).

    Returns:
        date: datetime.date object.
    """
    return datetime.date.fromisoformat(val.decode())

def convert_datetime(val: bytes) -> datetime:
    """Convert ISO 8601 datetime from SQLite to datetime.datetime object.

    This converter is registered with SQLite to automatically convert
    ISO 8601 datetime strings retrieved from the database back into
    Python datetime.datetime objects.

    Args:
        val (bytes): ISO 8601 datetime string from SQLite (as bytes).

    Returns:
        datetime: datetime.datetime object.
    """
    return datetime.datetime.fromisoformat(val.decode())

def convert_timestamp(val: bytes) -> datetime:
    """Convert Unix epoch timestamp from SQLite to datetime.datetime object.

    This converter is registered with SQLite to automatically convert
    Unix epoch timestamps retrieved from the database back into
    Python datetime.datetime objects.

    Args:
        val (bytes): Unix epoch timestamp from SQLite (as bytes).

    Returns:
        datetime: datetime.datetime object.
    """
    return datetime.datetime.fromtimestamp(int(val))

sqlite3.register_converter("date", convert_date)
sqlite3.register_converter("datetime", convert_datetime)
sqlite3.register_converter("timestamp", convert_timestamp)

class CustomRunnable:
    """
    A class to interact with a language model API using custom prompts.

    This class handles communication with a language model API, building prompts
    from templates and input variables, sending requests, and processing responses.
    It is designed to be flexible and reusable for different types of language model interactions.
    """
    def __init__(self, model_url: str, model_name: str, system_prompt_template: str, user_prompt_template: str, input_variables: List[str], response_type: str, required_format: Optional[str] = None):
        """
        Initialize the CustomRunnable instance.

        :param model_url: URL of the model API endpoint.
        :type model_url: str
        :param model_name: Name of the model to use.
        :type model_name: str
        :param system_prompt_template: Template for the system prompt with placeholders.
        :type system_prompt_template: str
        :param user_prompt_template: Template for the user prompt with placeholders for input variables.
        :type user_prompt_template: str
        :param input_variables: List of variables required by the prompt.
        :type input_variables: List[str]
        :param response_type: Expected response type, e.g., "json", "chat".
        :type response_type: str
        :param required_format: Expected output format for the model response (e.g., "json").
        :type required_format: Optional[str]
        """
        self.model_url = model_url
        self.model_name = model_name
        self.system_prompt_template = system_prompt_template
        self.user_prompt_template = user_prompt_template
        self.input_variables = input_variables
        self.required_format = required_format
        self.messages = [] # Initialize message history for conversational models
        self.response_type = response_type

    def build_prompt(self, inputs: Dict[str, str]) -> str:
        """
        Build the prompt by substituting input variables into the user prompt template.

        :param inputs: Dictionary of input variables to replace in the prompt template.
        :type inputs: Dict[str, str]
        :return: The constructed user prompt string.
        :rtype: str
        """

        user_prompt = self.user_prompt_template.format(**inputs)

        # Initialize messages with system prompt and assistant acknowledgement
        new_messages = [
            {"role": "user", "content": self.system_prompt_template},
            {"role": "assistant", "content": "Okay, I am ready to help"}
        ]

        # Add initial messages to the history
        for message in new_messages:
            self.messages.append(message)

        return user_prompt

    def add_to_history(self, chat_history: List[Dict[str, str]]):
        """
        Add previous chat history to the current message history.

        :param chat_history: List of dictionaries representing chat history messages.
                             Each dictionary should have 'role' and 'content' keys.
        :type chat_history: List[Dict[str, str]]
        """
        for chat in chat_history:
            self.messages.append(chat)

    def invoke(self, inputs: Dict[str, str]) -> Optional[Dict]:
        """
        Execute the model call with the constructed prompt and return the processed output.

        :param inputs: Dictionary of input values for the prompt.
        :type inputs: Dict[str, str]
        :return: Processed output from the model response. Returns a dictionary if JSON response, otherwise string.
        :rtype: Optional[Dict]
        :raises ValueError: If the request fails or JSON decoding fails.
        """
        user_prompt = self.build_prompt(inputs)
        self.messages.append({"role": "user", "content": user_prompt})

        payload = {
            "model": self.model_name,
            "messages": self.messages,
            "stream": False, # Set stream to False for non-streaming responses
        }

        if self.response_type == "json":
            payload["format"] = self.required_format # Specify response format if JSON is expected

        headers = {"Content-Type": "application/json"}

        response = requests.post(self.model_url, headers=headers, data=json.dumps(payload))

        if response.status_code == 200:
            try:
                data = response.json().get("message", {}).get("content", "")

                # Attempt to parse and fix JSON if response type is JSON
                if self.response_type == "json":
                    try:
                        parsed_data = extract_and_fix_json(data) # Try to fix common JSON errors
                        return parsed_data
                    except Exception:
                        pass # If fix fails, try literal evaluation

                # Attempt to parse response using literal evaluation for Python structures (list, dict, etc.)
                try:
                    parsed_data = ast.literal_eval(data)
                    return parsed_data
                except (ValueError, SyntaxError):
                    pass # If literal evaluation fails, return raw data

                return data # Return raw string data if no parsing is successful

            except json.JSONDecodeError:
                raise ValueError(f"Failed to decode JSON response: {response.text}")
        else:
            raise ValueError(f"Request failed with status code {response.status_code}: {response.text}")

class MemoryManager:
    """
    Manages user memories using vector embeddings and semantic similarity search.

    This class handles storing, retrieving, and updating user memories in a SQLite database.
    It uses sentence embeddings to enable semantic search and categorizes memories
    for better organization and retrieval.
    """
    def __init__(self, db_path: str = "memory.db", model_name: str = os.environ["BASE_MODEL_REPO_ID"]):
        """
        Initialize the MemoryManager.

        :param db_path: Path to the SQLite database file. Defaults to "memory.db".
        :type db_path: str
        :param model_name: Name of the language model to use with ChatOllama.
        :type model_name: str
        """
        self.db_path = db_path

        # Initialize ChatOllama for language model interactions
        self.llm = ChatOllama(
            model=model_name
        )

        # Initialize sentence transformer model for embedding generation
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

        # Define categories for memory organization and keywords associated with each category
        self.categories = {
            "PERSONAL": ["home", "hobby", "diary", "self", "goals", "habit", "routine", "personal"],
            "WORK": ["office", "business", "client", "report", "presentation", "deadline", "manager", "workplace"],
            "SOCIAL": ["meetup", "gathering", "party", "social", "community", "group", "network"],
            "RELATIONSHIP": ["friend", "family", "partner", "colleague", "neighbor"],
            "FINANCE": ["money", "bank", "loan", "debt", "payment", "buy", "sell"],
            "SPIRITUAL": ["pray", "meditation", "temple", "church", "mosque"],
            "CAREER": ["job", "work", "interview", "meeting", "project"],
            "TECHNOLOGY": ["phone", "computer", "laptop", "device", "software"],
            "HEALTH": ["doctor", "medicine", "exercise", "diet", "hospital"],
            "EDUCATION": ["study", "school", "college", "course", "learn"],
            "TRANSPORTATION": ["car", "bike", "bus", "train", "flight"],
            "ENTERTAINMENT": ["movie", "game", "music", "party", "book"],
            "TASKS": ["todo", "deadline", "appointment", "schedule", "reminder"]
        }
        self.initialize_database() # Initialize database tables upon MemoryManager creation

    def query_classification(self, query: str) -> Dict:
        """
        Classify a query into 'Short Term' or 'Long Term' memory type using a language model.

        :param query: The user query string.
        :type query: str
        :return: Dictionary response from the language model, or an empty dictionary on error.
        :rtype: Dict
        """
        try:
            memories = CustomRunnable(
                model_url="http://localhost:11434/api/chat/",
                model_name="llama3.2:3b",
                system_prompt_template=dual_memory_classification_system_template,
                user_prompt_template=dual_memory_classification_user_template,
                input_variables=["query"],
                response_type="chat",
            )

            response = memories.invoke({"query": query})
            return response
        except Exception as e:
            print(f"Error generating response: {e}")
            return {}

    def create_category_table(self, cursor: sqlite3.Cursor, category: str):
        """Create a table for a specific memory category in the database.

        :param cursor: SQLite database cursor object.
        :type cursor: sqlite3.Cursor
        :param category: Category name for the table (e.g., "PERSONAL", "WORK").
        :type category: str
        """
        cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS {category.lower()} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            original_text TEXT NOT NULL,
            keywords TEXT NOT NULL,
            embedding BLOB NOT NULL,
            entities TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expiry_at TIMESTAMP NOT NULL,
            is_active BOOLEAN DEFAULT 1
        )
        ''')

    def initialize_database(self):
        """Initialize the SQLite database by creating tables for each memory category."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create tables for each category defined in self.categories
        for category in self.categories.keys():
            self.create_category_table(cursor, category)

        conn.commit()
        conn.close()

    def compute_embedding(self, text: str) -> bytes:
        """Compute sentence embedding for the given text using SentenceTransformer.

        :param text: Input text for which to compute the embedding.
        :type text: str
        :return: Byte representation of the computed embedding.
        :rtype: bytes
        """
        embedding = self.embedding_model.encode(text)
        return np.array(embedding).tobytes()

    def bytes_to_array(self, embedding_bytes: bytes) -> np.ndarray:
        """Convert byte representation of embedding back to a numpy array.

        :param embedding_bytes: Byte string representing the embedding.
        :type embedding_bytes: bytes
        :return: Numpy array representing the embedding.
        :rtype: np.ndarray
        """
        return np.frombuffer(embedding_bytes, dtype=np.float32)

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two numpy arrays (embeddings).

        :param a: Numpy array representing the first embedding.
        :type a: np.ndarray
        :param b: Numpy array representing the second embedding.
        :type b: np.ndarray
        :return: Cosine similarity score between the two embeddings.
        :rtype: float
        """
        return np.dot(a, b) / (norm(a) * norm(b))

    def extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords and entities from the input text using SpaCy.

        :param text: Input text from which to extract keywords.
        :type text: str
        :return: List of extracted keywords (lemmatized nouns and verbs, and entities).
        :rtype: List[str]
        """
        doc = nlp(text.lower())

        keywords = []
        keywords.extend([ent.text for ent in doc.ents]) # Extract entities
        keywords.extend([token.lemma_ for token in doc
                        if (token.pos_ in ['NOUN', 'VERB'] and
                            not token.is_stop and
                            len(token.text) > 2)]) # Extract lemmatized nouns and verbs

        return list(set(keywords)) # Return unique keywords

    def determine_category(self, keywords: List[str]) -> str:
        """Determine the most relevant memory category based on extracted keywords.

        :param keywords: List of keywords extracted from the memory text.
        :type keywords: List[str]
        :return: The most relevant memory category (e.g., "PERSONAL", "WORK"). Defaults to "TASKS" if no category matches.
        :rtype: str
        """
        category_scores = {category: 0 for category in self.categories}

        for keyword in keywords:
            for category, category_keywords in self.categories.items():
                if any(cat_keyword in keyword for cat_keyword in category_keywords):
                    category_scores[category] += 1 # Increment score if keyword matches category keywords

        max_score = max(category_scores.values())
        if max_score == 0:
            return "TASKS" # Default category if no keyword matches

        return max(category_scores.items(), key=lambda x: x[1])[0] # Return category with the highest score

    def expiry_date_decision(self, query: str) -> Dict:
        """
        Determine the expiry date for a memory based on its content using a language model.

        :param query: The memory text to determine expiry for.
        :type query: str
        :return: Dictionary response from the language model containing expiry information, or an empty dictionary on error.
        :rtype: Dict
        """

        today = date.today()
        formatted_date = today.strftime("%d %B %Y %A") # Format today's date for prompt

        try:
            memories = CustomRunnable(
                model_url="http://localhost:11434/api/chat/",
                model_name="llama3.2:3b",
                system_prompt_template=system_memory_expiry_template,
                user_prompt_template=user_memory_expiry_template,
                input_variables=["query", "formatted_date"],
                required_format=extract_memory_required_format,
                response_type="chat",
            )

            response = memories.invoke({"query": query, "formatted_date": formatted_date})
            return response
        except Exception as e:
            print(f"Error generating response: {e}")
            return {}

    def extract_and_invoke_memory(self, current_query: str) -> Dict:
        """
        Extract memory details from unstructured text using a language model.

        :param current_query: The user's current query or text input.
        :type current_query: str
        :return: Dictionary response from the language model containing extracted memory details, or an empty dictionary on error.
        :rtype: Dict
        """
        date_today = date.today() # Get today's date for context

        try:
            memories = CustomRunnable(
                model_url="http://localhost:11434/api/chat/",
                model_name="llama3.2:3b",
                system_prompt_template=extract_memory_system_prompt_template,
                user_prompt_template=extract_memory_user_prompt_template,
                input_variables=["current_query", "date_today"],
                required_format=extract_memory_required_format,
                response_type="json",
            )

            response = memories.invoke({"current_query": current_query, "date_today": date_today})
            return response
        except Exception as e:
            print(f"Error generating response: {e}")
            return {}


    def update_memory(self, user_id: str, current_query: str) -> Optional[Dict]:
        """
        Update existing memories based on the current user query.

        This function processes the user query, identifies relevant memories,
        and decides whether to update existing memories or create new ones.

        :param user_id: ID of the user interacting with the memory system.
        :type user_id: str
        :param current_query: The current user query string.
        :type current_query: str
        :return: Dictionary containing details of actions taken (memory_updated, memory_created), or None on error.
        :rtype: Optional[Dict]
        """

        memories = self.extract_and_invoke_memory(current_query) # Extract memory details from query
        print(f"Memories: {memories}")

        for mem in memories['memories']: # Iterate through extracted memories
            memory_type = self.query_classification(mem['text']) # Classify memory type (Short/Long Term)
            category = mem['category'] # Get category of the memory

            print(f"Memory type: {memory_type}")

            if memory_type == "Short Term": # Process short-term memories
                # Get relevant memories from the database based on current memory and category
                relevant_memories = self.get_relevant_memories(user_id, mem['text'], category)
                print(f"Relevant memories: {relevant_memories}")

                if not relevant_memories: # If no relevant memories found, store current memory as new
                    retention_days = self.expiry_date_decision(mem['text']) # Determine retention period
                    # print(f"Retention days: {retention_days}")
                    self.store_memory(user_id, mem['text'], retention_days, category) # Store new memory
                    continue # Move to the next extracted memory

                # Prepare context from relevant memories for update decision
                memory_context = [
                    f"Memory {idx+1}: {memory['text']} (ID: {memory['id']}, Created: {memory['created_at']}, Expires: {memory['expiry_at']})"
                    for idx, memory in enumerate(relevant_memories)
                ]
                print(f"Memory context: {memory_context}")

                def get_processed_json_response_update(mem: str, memory_context: List[str],
                              model_url: str ="http://localhost:11434/api/chat",
                              model_name: str ="llama3.2:3b") -> Dict:
                    """
                    Get a JSON response from the language model to decide on memory update actions.

                    :param mem: Current memory text.
                    :type mem: str
                    :param memory_context: List of relevant memory contexts.
                    :type memory_context: List[str]
                    :param model_url: URL of the language model API endpoint.
                    :type model_url: str
                    :param model_name: Name of the language model.
                    :type model_name: str
                    :return: Dictionary response from the language model, or an empty dictionary on error.
                    :rtype: Dict
                    """
                    try:
                        # Create CustomRunnable instance for update decision
                        runnable = CustomRunnable(
                            model_url=model_url,
                            model_name=model_name,
                            system_prompt_template=update_decision_system_prompt,
                            user_prompt_template=update_user_prompt_template,
                            input_variables=["current_query", "memory_context"],
                            required_format=update_required_format,
                            response_type="json"
                        )

                        # Get and process response from the language model
                        response = runnable.invoke({
                            "current_query": mem['text'],
                            "memory_context": memory_context
                        })

                        return response

                    except Exception as e:
                        print(f"Error generating response: {e}")
                        return {}


                def get_memory_category(cursor: sqlite3.Cursor, memory_id: int) -> Optional[str]:
                    """
                    Retrieves the category of a memory from the database.

                    :param cursor: Database cursor.
                    :type cursor: sqlite3.Cursor
                    :param memory_id: ID of the memory to look up.
                    :type memory_id: int
                    :return: Category name or None if not found.
                    :rtype: Optional[str]
                    """
                    # Query all possible category tables to find the memory
                    category_tables = ['relationship', 'finance', 'spiritual', 'career', 'technology', 'health', 'education', 'transportation', 'entertainment', 'tasks', 'work', 'social']  # Add all your category tables here

                    for category in category_tables:
                        cursor.execute(f'''
                            SELECT 1 FROM {category}
                            WHERE id = ?
                        ''', (memory_id,))

                        if cursor.fetchone():
                            return category

                    return None


                update_details = get_processed_json_response_update(mem, memory_context) # Get update decisions from LLM
                print(update_details)

                # Open database connection for updates
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()

                # Prepare result dictionary to track actions
                result = {
                    'action': [],
                    'memory_updated': [],
                    'memory_created': []
                }

                updates = update_details.get('update', []) # Get update instructions from LLM response

            # Handle memory updates based on LLM decision
                if updates:
                    for update in updates:
                        memory_id = update["id"]
                        updated_text = update["text"]
                        print(f"Updating memory {memory_id}")

                        # Determine category of the memory being updated
                        query_keywords = self.extract_keywords(updated_text)
                        original_category = get_memory_category(cursor, memory_id)
                        print(f"Original category: {original_category}")
                        if not original_category:
                            print(f"Warning: Could not find original category for memory {memory_id}")
                            continue

                        # Compute new embedding for the updated text
                        new_embedding = self.compute_embedding(updated_text)

                        # Update memory in the database with new text, embedding, and expiry
                        cursor.execute(f'''
                        UPDATE {original_category}
                        SET original_text = ?,
                            embedding = ?,
                            keywords = ?,
                            expiry_at = datetime('now', '+{self.expiry_date_decision(updated_text)['retention_days']} days') # Determine new expiry
                        WHERE id = ?
                        ''', (
                            updated_text,
                            new_embedding,
                            ','.join(query_keywords),
                            memory_id
                        ))

                        result['action'].append('memory_updated')
                        result.setdefault('memory_updated', []).append(updated_text)
                        print(f"result: {result}")

                conn.commit()
                conn.close()


    def store_memory(self, user_id: str, text: str, retention_days: Dict, category: str) -> bool:
        """Store a new memory in the database under the appropriate category.

        :param user_id: ID of the user.
        :type user_id: str
        :param text: Memory text to be stored.
        :type text: str
        :param retention_days: Dictionary containing retention days from LLM decision.
        :type retention_days: Dict
        :param category: Category of the memory.
        :type category: str
        :return: True if memory storage is successful, False otherwise.
        :rtype: bool
        """

        print(f"Creating memory: {text}")

        # if memory_type == "Short Term":
        try:
            keywords = self.extract_keywords(text) # Extract keywords from memory text
            embedding = self.compute_embedding(text) # Compute embedding for memory text

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            print(f"Retention days: {retention_days}")

            current_time = datetime.now()
            expiry_time = current_time + timedelta(days=int(retention_days['retention_days'])) # Calculate expiry time

            cursor.execute(f'''
            INSERT INTO {category.lower()}
            (user_id, original_text, keywords, embedding, created_at, expiry_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                user_id,
                text,
                ','.join(keywords),
                embedding,
                current_time,
                expiry_time
            ))

            conn.commit()
            conn.close()
            return True

        except Exception as e:
            print(f"Error storing memory: {e}")
            return False

    def get_relevant_memories(self, user_id: str, query: str, category: str, similarity_threshold: float = 0.5) -> List[Dict]:
        """Get memories relevant to the query from the specified category table based on semantic similarity.

        :param user_id: ID of the user.
        :type user_id: str
        :param query: User query string.
        :type query: str
        :param category: Category to search memories in.
        :type category: str
        :param similarity_threshold: Minimum cosine similarity for a memory to be considered relevant. Defaults to 0.5.
        :type similarity_threshold: float
        :return: List of relevant memories as dictionaries, sorted by similarity in descending order.
        :rtype: List[Dict]
        """
        query_embedding = self.embedding_model.encode(query) # Compute embedding for the query

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Query memories from the specified category table that are active and not expired
        cursor.execute(f'''
        SELECT id, original_text, keywords, embedding, created_at, expiry_at
        FROM {category.lower()}
        WHERE user_id = ?
        AND is_active = 1
        AND datetime('now') < expiry_at
        ''', (user_id,))

        memories = []
        for row in cursor.fetchall(): # Fetch all relevant memory rows
            memory_embedding = self.bytes_to_array(row[3]) # Convert embedding bytes to numpy array
            similarity = self.cosine_similarity(query_embedding, memory_embedding) # Compute cosine similarity

            if similarity >= similarity_threshold: # Check if similarity exceeds threshold
                memories.append({
                    'id': row[0],
                    'text': row[1],
                    'keywords': row[2].split(','),
                    'similarity': similarity,
                    'created_at': row[4],
                    'expiry_at': row[5]
                })

        conn.close()
        memories.sort(key=lambda x: x['similarity'], reverse=True) # Sort memories by similarity descending
        return memories

    def process_user_query(self, user_id: str, query: str) -> str:
        """Process user query by retrieving relevant memories and generating a response using ChatOllama.

        :param user_id: ID of the user.
        :type user_id: str
        :param query: User query string.
        :type query: str
        :return: Natural language response to the user query.
        :rtype: str
        """

        query_keywords = self.extract_keywords(query) # Extract keywords from user query
        category = self.determine_category(query_keywords) # Determine category for the query
        relevant_memories = self.get_relevant_memories(user_id, query, category) # Get relevant memories

        print(f"Relevant memories: {relevant_memories}")

        memory_context = "\n".join([
            f"- {memory['text']}"
            for memory in relevant_memories # Format relevant memories into context string
        ])

        print(f"Memory context: {memory_context}")

        try:

            if not memory_context: # If no relevant memories found, use general prompt
                try:
                    # Create CustomRunnable instance for general response
                    runnable = CustomRunnable(
                        model_url="http://localhost:11434/api/chat",
                        model_name="llama3.2:3b",
                        system_prompt_template=system_general_template,
                        user_prompt_template=user_general_template,
                        input_variables=["query"],
                        response_type="chat"
                    )

                    # Get and process general response
                    response = runnable.invoke({"query": query})
                    return response

                except Exception as e:
                    print(f"Error generating response: {e}")
                    return {}

            else: # If relevant memories found, use memory-aware prompt

                try:
                    # Create CustomRunnable instance for memory-aware response
                    runnable = CustomRunnable(
                        model_url="http://localhost:11434/api/chat",
                        model_name="llama3.2:3b",
                        system_prompt_template=system_memory_response_template,
                        user_prompt_template=user_memory_response_template,
                        input_variables=["query", "memory_context"],
                        response_type="chat"
                    )

                    # Get and process memory-aware response
                    response = runnable.invoke({
                        "query": query,
                        "memory_context":memory_context,
                    })

                    return response

                except Exception as e:
                    print(f"Error generating response: {e}")
                    return {}

        except Exception as e:
            print(f"Error processing query with ChatOllama: {e}")
            return "I'm having trouble processing your question. Please try again."

    def cleanup_expired_memories(self):
        """Clean up expired memories from all category tables in the database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            current_time = datetime.now()
            print(f"Current time: {current_time}")
            new_time = current_time + timedelta(minutes=1) # Example: just to show time comparison works
            print(f"New time: {new_time}")
            print(f"Max time: {max(current_time, new_time)}") # Example: just to show time comparison works

            for category in self.categories.keys(): # Iterate through all categories
                table_name = category.lower()
                cursor.execute(f'''
                DELETE FROM {table_name}
                WHERE expiry_at < ? # Delete memories where expiry time is in the past
                ''', (current_time,))

                rows_deleted = cursor.rowcount
                print(f"Deleted {rows_deleted} expired memories from {table_name}")

            conn.commit()
            conn.close()
            print("Memory cleanup completed successfully")

        except Exception as e:
            print(f"Error during memory cleanup: {e}")

def main():
    """Main function to demonstrate the memory management system."""
    memory_manager = MemoryManager(model_name=os.environ["BASE_MODEL_REPO_ID"]) # Initialize MemoryManager

    user_id = "user123" # Define a user ID

    memory_manager.cleanup_expired_memories() # Clean up expired memories

    user_query = "Yesterday I fought with my dad" # Example user query

    response = memory_manager.process_user_query(user_id, user_query) # Process user query
    memory_manager.update_memory(user_id, user_query) # Update memories based on user query
    print(f"Query: {user_query}")
    print(f"Response: {response}") # Print the response

if __name__ == "__main__":
    main()