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
# Use consistent naming with dashes as provided in the prompt
DEFAULT_MODEL_NAME = 'gemini-2.0-flash-lite'

ALLOWED_MODELS = [
    'gemini-2.0-flash-lite',
    'gemini-2.0-flash',
    'gemini-2.5-pro-preview-03-25'  # Use the exact name if this is the correct ID
]

# --- Cost Data (per 1 Million Tokens) ---
MODEL_COSTS = {
    'gemini-2.0-flash-lite': {'input': 0.075, 'output': 0.30},
    'gemini-2.0-flash': {'input': 0.10, 'output': 0.40},
    # Using the name from ALLOWED_MODELS, ensure pricing is correct
    'gemini-2.5-pro-preview-03-25': {'input': 1.25, 'output': 10.0},
    # Add costs for other models if you use them
}

SAVE_DIR = os.path.expanduser("~/rpg_saves")
os.makedirs(SAVE_DIR, exist_ok=True)

HISTORY_LIMIT = 10

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
    imagen_model_name = "imagegeneration@006"
    print(f"Using Imagen model: {imagen_model_name}")
    imagen_model = ImageGenerationModel.from_pretrained(imagen_model_name)

except Exception as e:
    print(f"ERROR: Failed to initialize AI models: {e}", file=sys.stderr)
    sys.exit(1)

# --- Cost Calculation Helper ---


def calculate_cost(model_name, input_tokens, output_tokens):
    """Calculates the estimated cost based on token counts and model rates."""
    cost_info = MODEL_COSTS.get(model_name)
    if not cost_info:
        print(f"WARN: Cost information not found for model '{model_name}'.")
        return 0.0  # Return 0 cost if model pricing isn't defined

    # Costs are per 1 Million tokens, so divide counts by 1,000,000
    input_cost = (input_tokens / 1_000_000) * cost_info['input']
    output_cost = (output_tokens / 1_000_000) * cost_info['output']
    total_cost = input_cost + output_cost
    return total_cost


# --- Gemini Interaction ---
def get_gemini_response(prompt, model_name):
    """
    Sends a prompt to the specified Gemini model and returns the text response
    along with input and output token counts.
    """
    if model_name not in ALLOWED_MODELS:
        print(
            f"WARN: Invalid model '{model_name}' requested. Using default '{DEFAULT_MODEL_NAME}'.")
        model_name = DEFAULT_MODEL_NAME

    print(f"\n🤖 *Gemini is thinking ({model_name})...*\n")
    input_tokens = 0
    output_tokens = 0
    try:
        model_to_use = genai.GenerativeModel(model_name)
        response = model_to_use.generate_content(prompt)

        if hasattr(response, 'usage_metadata'):
            input_tokens = response.usage_metadata.prompt_token_count
            output_tokens = response.usage_metadata.candidates_token_count
            print(
                f"   Tokens - Input: {input_tokens}, Output: {output_tokens}")
        else:
            print("   WARN: Token usage metadata not found in response.")
            # Estimate input tokens if metadata is missing
            try:
                input_tokens = model_to_use.count_tokens(prompt).total_tokens
                print(f"   Estimated Input Tokens: {input_tokens}")
            except Exception as count_e:
                print(f"   WARN: Could not estimate input tokens: {count_e}")

        if response.parts:
            text = response.text.strip()
            if text.startswith("```json"):
                text = text[7:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
            print(f">> Raw Gemini text: '{text}'")
            return text, input_tokens, output_tokens
        else:
            feedback = getattr(response, 'prompt_feedback', 'Unknown reason.')
            print(
                f"WARN: Gemini response was empty or blocked. Feedback: {feedback}")
            return "The way forward seems blocked by an unseen force.", input_tokens, output_tokens

    except Exception as e:
        print(
            f"Error communicating with Gemini ({model_name}): {e}", file=sys.stderr)
        return f"An error occurred with text generation ({model_name}): {e}", 0, 0


# --- Image Generation with Imagen ---
def generate_image_with_imagen(scene_description, prompt_model_name=DEFAULT_MODEL_NAME):
    """
    Generates an image using Imagen. Also uses Gemini to create the image prompt.
    Returns image data, error message, and token counts for the Gemini prompt generation.
    """
    print("\n🎨 *Generating image prompt with Gemini...*\n")
    image_prompt_request = f"""
    Create a concise, descriptive, and visually evocative prompt suitable for an AI image generator based on this RPG scene. Focus on key elements, mood, and style (fantasy art). Max 50 words.

    Scene: "{scene_description}"

    Image Prompt:
    """

    # Use the specified model for generating the image prompt
    image_prompt_text, input_tokens, output_tokens = get_gemini_response(
        image_prompt_request, prompt_model_name)

    if not image_prompt_text or "error occurred" in image_prompt_text.lower():
        print(f"WARN: Could not generate a suitable image prompt. Using fallback.")
        image_prompt = f"Fantasy RPG scene: {scene_description[:200]}"
        # Can't know tokens if prompt generation failed this way
        input_tokens, output_tokens = 0, 0
    else:
        image_prompt = image_prompt_text.split("Image Prompt:")[-1].strip()

    print(f"Using Image Prompt: {image_prompt}")
    print("\n🎨 *Generating image with Imagen...*\n")

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
                # Return Gemini token counts too
                return b64_image_data, None, input_tokens, output_tokens
            else:
                print("ERROR: Could not access image bytes.")
                # Return Gemini token counts even on Imagen error
                return None, "Failed to process generated image data.", input_tokens, output_tokens
        else:
            print("WARN: Imagen generation resulted in no images.",
                  getattr(images, '_prediction_response', ''))
            # Return Gemini token counts even on Imagen error
            return None, "Image generation failed or was blocked.", input_tokens, output_tokens

    except Exception as e:
        print(f"Error calling Imagen API: {e}", file=sys.stderr)
        error_message = f"Error generating image: {e}"
        # Return Gemini token counts even on Imagen error
        return None, error_message, input_tokens, output_tokens


# --- Save/Load Functions ---
def save_game(filename, game_state):
    if ".." in filename or "/" in filename or "\\" in filename:
        return False, f"Invalid filename: '{filename}'."
    filepath = os.path.join(SAVE_DIR, filename + ".json")
    try:
        # Ensure all expected keys are present before saving
        game_state.setdefault('total_input_tokens', 0)
        game_state.setdefault('total_output_tokens', 0)
        game_state.setdefault('last_input_tokens', 0)
        game_state.setdefault('last_output_tokens', 0)
        game_state.setdefault('last_request_cost', 0.0)
        # Note: estimated_total_cost is calculated on the fly in app.py, no need to save

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

        required_keys = ["current_location",
                         "player_inventory", "last_scene", "chat_history"]
        if all(k in game_state for k in required_keys):
            # Add/reset fields upon loading
            game_state.setdefault('total_input_tokens', 0)
            game_state.setdefault('total_output_tokens', 0)
            game_state['last_input_tokens'] = 0  # Reset last counts
            game_state['last_output_tokens'] = 0
            game_state['last_request_cost'] = 0.0  # Reset last cost

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
    current_location = "an unknown starting point"
    player_inventory = []
    chat_history = [{"role": "user", "parts": [
        {"text": initial_player_prompt}]}]

    full_initial_prompt = f"""
    Roleplay: You are the Dungeon Master for a text-based fantasy RPG.
    Task: Start a new adventure based on the player's first action or thought. Respond ONLY with the JSON structure.
    Player's input: '{initial_player_prompt}'
    Player's location: '{current_location}'
    {RPG_INSTRUCTION}"""

    response_text, input_tokens, output_tokens = get_gemini_response(
        full_initial_prompt, model_name)

    # Calculate cost for this initial request
    last_request_cost = calculate_cost(
        model_name, input_tokens, output_tokens)

    if response_text and "error occurred" not in response_text.lower():
        try:
            scene_json = json.loads(response_text)
            if not all(k in scene_json for k in [RPG_RESPONSE, RPG_RESPONSE_LOCATION, RPG_RESPONSE_INVENTORY]):
                raise ValueError("JSON response missing required keys.")

            chat_history.append({"role": "model", "parts": [
                                {"text": scene_json[RPG_RESPONSE]}]})
            current_location = scene_json[RPG_RESPONSE_LOCATION]
            inventory_str = scene_json.get(RPG_RESPONSE_INVENTORY, "")
            player_inventory = [item.strip()
                                for item in inventory_str.split(',') if item.strip()]

            game_state = {
                "current_location": current_location,
                "player_inventory": player_inventory,
                "last_scene": scene_json[RPG_RESPONSE],
                "chat_history": chat_history,
                "last_input_tokens": input_tokens,
                "last_output_tokens": output_tokens,
                "total_input_tokens": input_tokens,
                "total_output_tokens": output_tokens,
                "last_request_cost": last_request_cost,
                # estimated_total_cost will be calculated in app.py based on current model
            }
            return game_state, "New adventure started!", input_tokens, output_tokens
        except (json.JSONDecodeError, ValueError) as e:
            print(
                f"Error parsing initial JSON response: {e}\nRaw response: {response_text}", file=sys.stderr)
            error_msg = f"Failed to understand the initial scene description from AI: {e}"
            # Return token counts even on parse error (cost is calculated above)
            return None, error_msg, input_tokens, output_tokens
    else:
        error_msg = response_text if response_text else "Failed to get the initial scene description from AI."
        # Return token counts even on AI error
        return None, error_msg, input_tokens, output_tokens


# --- Process Player Turn ---
def process_player_action(player_action, game_state, model_name):
    current_location = game_state["current_location"]
    player_inventory = game_state["player_inventory"]
    chat_history = game_state["chat_history"]

    chat_history.append({"role": "user", "parts": [{"text": player_action}]})
    limited_history = chat_history[-HISTORY_LIMIT:]

    history_context = ""
    for entry in limited_history[:-1]:
        role = entry.get('role', 'unknown')
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

    response_text, input_tokens, output_tokens = get_gemini_response(
        prompt, model_name)

    # Calculate cost for this request
    last_request_cost = calculate_cost(
        model_name, input_tokens, output_tokens)
    feedback_messages = []

    # Ensure totals exist before adding to them
    game_state.setdefault('total_input_tokens', 0)
    game_state.setdefault('total_output_tokens', 0)

    if response_text and "error occurred" not in response_text.lower():
        try:
            scene_json = json.loads(response_text)
            if not all(k in scene_json for k in [RPG_RESPONSE, RPG_RESPONSE_LOCATION, RPG_RESPONSE_INVENTORY]):
                raise ValueError("JSON response missing required keys.")

            chat_history.append(
                {"role": "model", "parts": [{"text": scene_json[RPG_RESPONSE]}]})

            game_state["current_location"] = scene_json[RPG_RESPONSE_LOCATION]
            inventory_str = scene_json.get(RPG_RESPONSE_INVENTORY, "")
            game_state["player_inventory"] = [item.strip()
                                              for item in inventory_str.split(',') if item.strip()]
            game_state["last_scene"] = scene_json[RPG_RESPONSE]
            game_state["chat_history"] = chat_history
            game_state["last_input_tokens"] = input_tokens
            game_state["last_output_tokens"] = output_tokens
            game_state["total_input_tokens"] += input_tokens
            game_state["total_output_tokens"] += output_tokens
            game_state["last_request_cost"] = last_request_cost

            print(
                f"Updated game_state (tokens: last={input_tokens}/{output_tokens}, total={game_state['total_input_tokens']}/{game_state['total_output_tokens']}, last_cost=${last_request_cost:.6f})")
            return game_state, scene_json[RPG_RESPONSE], feedback_messages, input_tokens, output_tokens

        except (json.JSONDecodeError, ValueError) as e:
            print(
                f"Error parsing action JSON response: {e}\nRaw response: {response_text}", file=sys.stderr)
            error_message = f"The AI's response was unclear: {e}"
            feedback_messages.append(error_message)
            chat_history.pop()  # Remove user action
            game_state["chat_history"] = chat_history
            # Update state even on error: record tokens/cost used, but don't update totals
            game_state["last_input_tokens"] = input_tokens
            game_state["last_output_tokens"] = output_tokens
            game_state["last_request_cost"] = last_request_cost
            # Do NOT add tokens to total on parse error, as the turn didn't 'succeed'
            return game_state, game_state["last_scene"], feedback_messages, input_tokens, output_tokens
    else:
        error_message = response_text if response_text else "The AI seems unresponsive. Please try again."
        feedback_messages.append(error_message)
        chat_history.pop()  # Remove user action
        game_state["chat_history"] = chat_history
        # Update state even on error: record tokens/cost used
        game_state["last_input_tokens"] = input_tokens
        game_state["last_output_tokens"] = output_tokens
        game_state["last_request_cost"] = last_request_cost
        # Do NOT add tokens to total on AI error
        return game_state, game_state["last_scene"], feedback_messages, input_tokens, output_tokens