# game_logic.py

import os
import google.generativeai as genai
import google.auth
import json
import sys
from PIL import Image
import io
import base64

try:
    import vertexai
    from vertexai.preview.vision_models import ImageGenerationModel, ImageGenerationResponse
except ImportError:
    print("ERROR: google-cloud-aiplatform library not found or failed to import.")
    print("Please install it: pip install google-cloud-aiplatform --upgrade")
    sys.exit(1)

# --- Constants ---
# Default model - can be overridden by user selection
DEFAULT_MODEL_NAME = 'gemini-2.0-flash-lite'  # Use a valid default

# Define the allowed models explicitly for validation (matches app.py)
ALLOWED_MODELS = [
    'gemini-2.0-flash-lite',
    'gemini-2.0-flash',
    'gemini-2.5-pro-preview-03-25'
]

SAVE_DIR = os.path.expanduser("~/rpg_saves")
os.makedirs(SAVE_DIR, exist_ok=True)

HISTORY_LIMIT = 10  # Reduced for testing token limits

RPG_RESPONSE = "rpg_response"
RPG_RESPONSE_LOCATION = "location"
RPG_RESPONSE_INVENTORY = "inventory"
sample_data_structure = {
    RPG_RESPONSE: "description of the scene",
    RPG_RESPONSE_LOCATION: "new location",
    RPG_RESPONSE_INVENTORY: "item1,item2"
}
sample_json_string = json.dumps(sample_data_structure)

RPG_INSTRUCTION = f"""Reminders:
      - Your task is to write an updated version rpg reponse in json format like "'{sample_json_string}'".
      - Update the * existing * sections/properties(no new sections, just update existing properties/lists, and only if needed).
      - The response must be a json format.
      - The response should be in the same language as input.
    """

# --- Authentication and Configuration ---
try:
    credentials, project_id = google.auth.default()
    if not project_id:
        project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
        if not project_id:
            raise ValueError(
                "Could not determine Google Cloud project ID. Set the GOOGLE_CLOUD_PROJECT environment variable or ensure gcloud is configured correctly.")

    VERTEX_AI_PROJECT = project_id
    VERTEX_AI_LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")

    # Configure Gemini - API Key might be needed if Application Default Credentials don't work for Generative Language API
    # genai.configure(api_key=os.environ.get("GEMINI_API_KEY")) # Alternative if needed

    vertexai.init(project=VERTEX_AI_PROJECT,
                  location=VERTEX_AI_LOCATION, credentials=credentials)

    print(
        f"Vertex AI initialized for project '{VERTEX_AI_PROJECT}' in location '{VERTEX_AI_LOCATION}'")

except Exception as e:
    print(
        f"ERROR: Failed during Authentication or Initialization: {e}", file=sys.stderr)
    print("Ensure gcloud CLI is authenticated ('gcloud auth application-default login') and necessary APIs (Vertex AI, Generative Language) are enabled for your project.", file=sys.stderr)
    sys.exit(1)


# --- Model Selection ---
try:
    # Imagen Model Instance (as before)
    imagen_model_name = "imagegeneration@006"  # Using stable identifier
    print(f"Using Imagen model: {imagen_model_name}")
    imagen_model = ImageGenerationModel.from_pretrained(imagen_model_name)

    # We will initialize Gemini models on demand in get_gemini_response

except Exception as e:
    print(f"ERROR: Failed to initialize AI models: {e}", file=sys.stderr)
    sys.exit(1)

# --- Gemini Interaction ---


def get_gemini_response(prompt, model_name):
    """
    Sends a prompt to the specified Gemini model and returns the text response
    along with input and output token counts.
    """
    # Validate model name against allowed list
    if model_name not in ALLOWED_MODELS:
        print(
            f"WARN: Invalid model '{model_name}' requested. Using default '{DEFAULT_MODEL_NAME}'.")
        model_name = DEFAULT_MODEL_NAME

    print(f"\n🤖 *Gemini is thinking ({model_name})...*\n")
    input_tokens = 0
    output_tokens = 0
    try:
        # Initialize the specific model instance
        model_to_use = genai.GenerativeModel(model_name)
        response = model_to_use.generate_content(prompt)

        # --- Retrieve Token Counts ---
        if hasattr(response, 'usage_metadata'):
            input_tokens = response.usage_metadata.prompt_token_count
            output_tokens = response.usage_metadata.candidates_token_count
            print(
                f"   Tokens - Input: {input_tokens}, Output: {output_tokens}")
        else:
            print("   WARN: Token usage metadata not found in response.")
            # Optionally, you could estimate tokens here using model.count_tokens(prompt)
            # but this only gives input tokens before the call.

        # --- Process Response Text ---
        if response.parts:
            text = response.text
            text = text.strip()
            # Improved JSON cleaning
            if text.startswith("```json"):
                text = text[7:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
            # Log raw response before potential JSON errors
            print(f">> Raw Gemini text: '{text}'")
            return text, input_tokens, output_tokens
        else:
            feedback = getattr(response, 'prompt_feedback', 'Unknown reason.')
            print(
                f"WARN: Gemini response was empty or blocked. Feedback: {feedback}")
            # Return counts even on block/empty
            return "The way forward seems blocked by an unseen force.", input_tokens, output_tokens

    except Exception as e:
        print(
            f"Error communicating with Gemini ({model_name}): {e}", file=sys.stderr)
        # Return 0 tokens on error, but could also return None or raise
        return f"An error occurred with text generation ({model_name}): {e}", 0, 0


# --- NEW: Image Generation with Imagen ---
# (Keep generate_image_with_imagen as is, but note it also calls get_gemini_response
#  for the image prompt. Ensure it uses a reasonable default model or pass one in if needed)
def generate_image_with_imagen(scene_description):
    """Generates an image based on the scene using Vertex AI Imagen."""
    print("\n🎨 *Generating image with Imagen...*\n")

    image_prompt_request = f"""
    Create a concise, descriptive, and visually evocative prompt suitable for an AI image generator based on this RPG scene. Focus on key elements, mood, and style (fantasy art). Max 50 words.

    Scene: "{scene_description}"

    Image Prompt:
    """
    # Using default model for image prompt generation for simplicity here
    # If needed, this could accept a model_name too
    image_prompt_text, _, _ = get_gemini_response(
        image_prompt_request, DEFAULT_MODEL_NAME)

    if not image_prompt_text or "error occurred" in image_prompt_text.lower():
        print(f"WARN: Could not generate a suitable image prompt. Using fallback.")
        image_prompt = f"Fantasy RPG scene: {scene_description[:200]}"
    else:
        image_prompt = image_prompt_text.split("Image Prompt:")[-1].strip()

    print(f"Using Image Prompt: {image_prompt}")

    try:
        images: ImageGenerationResponse = imagen_model.generate_images(
            prompt=image_prompt,
            number_of_images=1,
        )

        if images.images:
            image_obj = images.images[0]
            if hasattr(image_obj, '_image_bytes'):
                image_bytes = image_obj._image_bytes
                b64_image_data = base64.b64encode(image_bytes).decode('utf-8')
                print("Successfully generated image and encoded to base64.")
                return b64_image_data, None
            else:
                print("ERROR: Could not access image bytes.")
                return None, "Failed to process generated image data."
        else:
            print("WARN: Imagen generation resulted in no images.",
                  getattr(images, '_prediction_response', ''))
            return None, "Image generation failed or was blocked."

    except Exception as e:
        print(f"Error calling Imagen API: {e}", file=sys.stderr)
        error_message = f"Error generating image: {e}"
        # Add more specific error hints if possible
        return None, error_message


# --- Save/Load Functions ---
def save_game(filename, game_state):
    if ".." in filename or "/" in filename or "\\" in filename:
        return False, f"Invalid filename: '{filename}'."
    filepath = os.path.join(SAVE_DIR, filename + ".json")
    try:
        # Ensure all expected keys, including tokens, are present before saving
        game_state.setdefault('total_input_tokens', 0)
        game_state.setdefault('total_output_tokens', 0)

        with open(filepath, 'w') as f:
            json.dump(game_state, f, indent=4)
        print(f"Game saved successfully to {filepath}!")
        return True, f"Game saved successfully as '{filename}.json'."
    except IOError as e:
        print(f"Error saving game to {filepath}: {e}", file=sys.stderr)
        return False, f"Error saving game: {e}"
    except Exception as e:
        print(
            f"An unexpected error occurred while saving: {e}", file=sys.stderr)
        return False, f"An unexpected error occurred while saving: {e}"


def load_game(filename):
    if ".." in filename or "/" in filename or "\\" in filename:
        return None, f"Invalid filename: '{filename}'."
    filepath = os.path.join(SAVE_DIR, filename + ".json")
    if not os.path.exists(filepath):
        return None, f"Save file '{filename}.json' not found."
    try:
        with open(filepath, 'r') as f:
            game_state = json.load(f)

        # Check for essential keys
        required_keys = ["current_location",
                         "player_inventory", "last_scene", "chat_history"]
        if all(k in game_state for k in required_keys):
            # Add token counts if missing from old save files, default to 0
            game_state.setdefault('total_input_tokens', 0)
            game_state.setdefault('total_output_tokens', 0)
            # Reset last token counts upon loading
            game_state['last_input_tokens'] = 0
            game_state['last_output_tokens'] = 0

            print(f"Game loaded successfully from {filepath}!")
            return game_state, f"Game loaded successfully from '{filename}.json'!"
        else:
            print(
                f"Save file {filepath} seems corrupted (missing keys).", file=sys.stderr)
            return None, f"Save file '{filename}.json' appears corrupted."
    except (IOError, json.JSONDecodeError) as e:
        print(f"Error loading game from {filepath}: {e}", file=sys.stderr)
        return None, f"Error loading game '{filename}.json': {e}"
    except Exception as e:
        print(
            f"An unexpected error occurred while loading: {e}", file=sys.stderr)
        return None, f"An unexpected error occurred while loading: {e}"


# --- Initial Game Setup ---
def start_new_game(initial_player_prompt, model_name):
    # Added model_name parameter
    current_location = "an unknown starting point"
    player_inventory = []
    # Use correct structure
    chat_history = [{"role": "user", "parts": [
        {"text": initial_player_prompt}]}]

    full_initial_prompt = f"""
    Roleplay: You are the Dungeon Master for a text-based fantasy RPG.
    Task: Start a new adventure based on the player's first action or thought. Respond ONLY with the JSON structure.
    Player's input: '{initial_player_prompt}'
    Player's location: '{current_location}'
    {RPG_INSTRUCTION}"""

    # Pass model_name, get response text and token counts
    response_text, input_tokens, output_tokens = get_gemini_response(
        full_initial_prompt, model_name)

    if response_text and "error occurred" not in response_text.lower():
        try:
            scene_json = json.loads(response_text)
            # Validate expected keys in the JSON response
            if not all(k in scene_json for k in [RPG_RESPONSE, RPG_RESPONSE_LOCATION, RPG_RESPONSE_INVENTORY]):
                raise ValueError("JSON response missing required keys.")

            chat_history.append({"role": "model", "parts": [
                                {"text": scene_json[RPG_RESPONSE]}]})  # Use correct structure
            current_location = scene_json[RPG_RESPONSE_LOCATION]
            # Handle potentially empty inventory string
            inventory_str = scene_json.get(RPG_RESPONSE_INVENTORY, "")
            player_inventory = [item.strip()
                                for item in inventory_str.split(',') if item.strip()]

            game_state = {
                "current_location": current_location,
                "player_inventory": player_inventory,
                "last_scene": scene_json[RPG_RESPONSE],
                "chat_history": chat_history,
                "last_input_tokens": input_tokens,  # Store last counts
                "last_output_tokens": output_tokens,
                "total_input_tokens": input_tokens,  # Initialize total counts
                "total_output_tokens": output_tokens
            }
            return game_state, "New adventure started!", input_tokens, output_tokens
        except (json.JSONDecodeError, ValueError) as e:
            print(
                f"Error parsing initial JSON response: {e}\nRaw response: {response_text}", file=sys.stderr)
            error_msg = f"Failed to understand the initial scene description from AI: {e}"
            # Return token counts even on parse error
            return None, error_msg, input_tokens, output_tokens
    else:
        error_msg = response_text if response_text else "Failed to get the initial scene description from AI."
        # Return token counts even on AI error
        return None, error_msg, input_tokens, output_tokens


# --- Process Player Turn ---
def process_player_action(player_action, game_state, model_name):
    # Added model_name parameter
    current_location = game_state["current_location"]
    player_inventory = game_state["player_inventory"]
    chat_history = game_state["chat_history"]

    # Add user action to history (ensure correct format)
    chat_history.append({"role": "user", "parts": [{"text": player_action}]})

    # Limit history context sent to the model
    limited_history = chat_history[-HISTORY_LIMIT:]

    # Construct context string (adapt if using different history structure)
    history_context = ""
    for entry in limited_history[:-1]:  # Exclude the latest user action
        role = entry.get('role', 'unknown')
        # Assuming 'parts' contains a list with one text part
        content = entry.get('parts', [{}])[0].get('text', '')
        history_context += f"\n{role.capitalize()}: {content}"

    prompt = f"""
    Roleplay: You are the Dungeon Master for a text-based fantasy RPG.
    Task: Narrate the outcome of the player's action based on the current situation and history. Respond ONLY with the JSON structure.
    Current Location: {current_location}
    Player Inventory: {', '.join(player_inventory) or 'nothing'}
    Recent History (last {len(limited_history)-1} turns): {history_context.strip()}
    Player's Action: "{player_action}"
    {RPG_INSTRUCTION}"""

    # Pass model_name, get response text and token counts
    response_text, input_tokens, output_tokens = get_gemini_response(
        prompt, model_name)

    feedback_messages = []
    if response_text and "error occurred" not in response_text.lower():
        try:
            scene_json = json.loads(response_text)
            # Validate expected keys in the JSON response
            if not all(k in scene_json for k in [RPG_RESPONSE, RPG_RESPONSE_LOCATION, RPG_RESPONSE_INVENTORY]):
                raise ValueError("JSON response missing required keys.")

            # Add AI response to history
            chat_history.append(
                {"role": "model", "parts": [{"text": scene_json[RPG_RESPONSE]}]})

            # Update game state from JSON
            game_state["current_location"] = scene_json[RPG_RESPONSE_LOCATION]
            inventory_str = scene_json.get(RPG_RESPONSE_INVENTORY, "")
            game_state["player_inventory"] = [item.strip()
                                              for item in inventory_str.split(',') if item.strip()]
            game_state["last_scene"] = scene_json[RPG_RESPONSE]
            # Update history in state
            game_state["chat_history"] = chat_history
            game_state["last_input_tokens"] = input_tokens  # Store last counts
            game_state["last_output_tokens"] = output_tokens

            # Accumulate total tokens (ensure totals exist from start/load)
            game_state['total_input_tokens'] = game_state.get(
                'total_input_tokens', 0) + input_tokens
            game_state['total_output_tokens'] = game_state.get(
                'total_output_tokens', 0) + output_tokens

            print(
                f"Updated game_state (tokens: last={input_tokens}/{output_tokens}, total={game_state['total_input_tokens']}/{game_state['total_output_tokens']})")
            # Return only necessary info for app.py
            return game_state, scene_json[RPG_RESPONSE], feedback_messages, input_tokens, output_tokens

        except (json.JSONDecodeError, ValueError) as e:
            print(
                f"Error parsing action JSON response: {e}\nRaw response: {response_text}", file=sys.stderr)
            error_message = f"The AI's response was unclear: {e}"
            feedback_messages.append(error_message)
            chat_history.pop()  # Remove user action that led to error
            game_state["chat_history"] = chat_history
            # Return counts even on parse error, update totals with 0 for this turn
            game_state["last_input_tokens"] = input_tokens
            game_state["last_output_tokens"] = output_tokens
            game_state['total_input_tokens'] = game_state.get(
                'total_input_tokens', 0) + input_tokens
            game_state['total_output_tokens'] = game_state.get(
                'total_output_tokens', 0) + output_tokens
            return game_state, game_state["last_scene"], feedback_messages, input_tokens, output_tokens
    else:
        error_message = response_text if response_text else "The AI seems unresponsive. Please try again."
        feedback_messages.append(error_message)
        chat_history.pop()  # Remove user action that led to error
        game_state["chat_history"] = chat_history
        # Return counts even on AI error, update totals
        game_state["last_input_tokens"] = input_tokens
        game_state["last_output_tokens"] = output_tokens
        game_state['total_input_tokens'] = game_state.get(
            'total_input_tokens', 0) + input_tokens
        game_state['total_output_tokens'] = game_state.get(
            'total_output_tokens', 0) + output_tokens
        return game_state, game_state["last_scene"], feedback_messages, input_tokens, output_tokens
