# game_logic.py

import os
import google.generativeai as genai
import google.auth
import json
import sys
from PIL import Image
import io
import base64
import time  # For latency measurement

try:
    import vertexai
    from vertexai.preview.vision_models import ImageGenerationModel, ImageGenerationResponse
except ImportError:
    print("ERROR: google-cloud-aiplatform library not found or failed to import.")
    print("Please install it: pip install google-cloud-aiplatform --upgrade")
    sys.exit(1)

# --- Constants ---
# Use consistent naming with dashes as provided in the prompt
DEFAULT_MODEL_NAME = 'gemini-2.5-flash-preview-04-17'

# --- Cost Data (per 1 Million Tokens) ---
MODEL_COSTS = {
    'gemini-2.0-flash-lite': {'input': 0.075, 'output': 0.30},
    'gemini-2.0-flash': {'input': 0.10, 'output': 0.40},
    'gemini-2.5-pro-preview-03-25': {'input': 1.25, 'output': 10.0},
    'gemini-2.5-flash-preview-04-17': {'input': 0.15, 'output': 0.60}
}

IMAGE_MODEL = "imagen-3.0-generate-002"

SAVE_DIR = os.path.expanduser("~/rpg_saves")
os.makedirs(SAVE_DIR, exist_ok=True)

HISTORY_LIMIT = 10  # Max recent turns to send *in addition* to summaries
SUMMARY_INTERVAL = 10  # Number of user/model conversation pairs before summarizing

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

SUMMARY_INSTRUCTION = """
You are an assistant tasked with summarizing conversation history for a text-based RPG.
Condense the following recent turns into a concise summary (around 50-100 words) capturing the key events, decisions, and changes in location or inventory. Focus on information relevant for future context.
Output *only* the summary text, without any preamble or explanation.
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
    # Initialize Generative AI Client for all Gemini models
    # No need to initialize individual models here, done in get_gemini_response
    genai.configure(credentials=credentials)
    print("Google Generative AI Client configured.")

    imagen_model_name = IMAGE_MODEL
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
def get_gemini_response(prompt, model_name, is_summary_request=False):
    """
    Sends a prompt to the specified Gemini model and returns the text response
    along with input tokens, output tokens, and latency.
    """
    if model_name not in MODEL_COSTS.keys():
        print(
            f"WARN: Invalid model '{model_name}' requested. Using default '{DEFAULT_MODEL_NAME}'.")
        model_name = DEFAULT_MODEL_NAME

    task_type = "Summary" if is_summary_request else "Action"
    print(f"\n🤖 *Gemini thinking ({model_name} - {task_type})...*\n")
    input_tokens = 0
    output_tokens = 0
    latency = 0.0
    # Default error
    text = f"An error occurred during text generation ({model_name} - {task_type})."

    try:
        # Use the correct model name format for the API
        api_model_name = f"models/{model_name}"
        model_to_use = genai.GenerativeModel(api_model_name)

        start_time = time.time()
        response = model_to_use.generate_content(prompt)
        end_time = time.time()
        latency = end_time - start_time

        print(f"   Gemini Latency: {latency:.3f}s")

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
            # Clean up JSON responses specifically for game actions
            if not is_summary_request:
                if text.startswith("```json"):
                    text = text[7:]
                if text.endswith("```"):
                    text = text[:-3]
                text = text.strip()
                print(f">> Raw Gemini JSON text: '{text}'")
            else:
                print(f">> Raw Gemini Summary text: '{text}'")

        else:
            feedback = getattr(response, 'prompt_feedback', 'Unknown reason.')
            print(
                f"WARN: Gemini response was empty or blocked. Feedback: {feedback}")
            text = f"The way forward seems blocked by an unseen force ({task_type}). Feedback: {feedback}"

    except Exception as e:
        print(
            f"Error communicating with Gemini ({model_name} - {task_type}): {e}", file=sys.stderr)
        text = f"An error occurred during text generation ({model_name} - {task_type}): {e}"
        # Latency might be partial or 0 in case of early error
        latency = time.time() - start_time if 'start_time' in locals() else 0.0

    return text, input_tokens, output_tokens, latency


# --- Generate Summary ---
def generate_summary(turns_to_summarize, model_name):
    """
    Generates a summary of the provided conversation turns.
    Returns: summary text, input tokens, output tokens, cost, latency.
    """
    history_text = ""
    for entry in turns_to_summarize:
        role = entry.get('role', 'unknown')
        # Handle potential structure variations more robustly
        part_text = '[No text content]'
        if entry.get('parts'):
            first_part = entry['parts'][0]
            if isinstance(first_part, dict) and 'text' in first_part:
                part_text = first_part['text']
            elif isinstance(first_part, str):
                part_text = first_part
        history_text += f"\n{role.capitalize()}: {part_text}"

    prompt = f"""{SUMMARY_INSTRUCTION}

    Recent Turns:
    {history_text.strip()}

    Concise Summary:
    """

    summary_text, summary_input_tokens, summary_output_tokens, summary_latency = get_gemini_response(
        prompt, model_name, is_summary_request=True)

    # Calculate cost for the summary generation
    summary_cost = calculate_cost(
        model_name, summary_input_tokens, summary_output_tokens)
    print(f"   Summary Cost: ${summary_cost:.6f}")

    if "error occurred" in summary_text.lower() or "unseen force" in summary_text.lower():
        print(
            f"WARN: Failed to generate summary. Raw response: {summary_text}")
        return None, summary_input_tokens, summary_output_tokens, summary_cost, summary_latency

    # Basic check to prevent saving overly short/failed summaries
    if len(summary_text) < 10:
        print(
            f"WARN: Generated summary seems too short. Discarding. Raw: {summary_text}")
        return None, summary_input_tokens, summary_output_tokens, summary_cost, summary_latency

    return summary_text, summary_input_tokens, summary_output_tokens, summary_cost, summary_latency


# --- Image Generation with Imagen ---
def generate_image_with_imagen(scene_description, prompt_model_name=DEFAULT_MODEL_NAME):
    """
    Generates an image using Imagen. Also uses Gemini to create the image prompt.
    Returns image data, error message, Gemini input tokens, Gemini output tokens, and Gemini latency.
    """
    print("\n🎨 *Generating image prompt with Gemini...*\n")
    image_prompt_request = f"""
    Create a concise, descriptive, and visually evocative prompt suitable for an AI image generator based on this RPG scene. Focus on key elements, mood, and style (fantasy art). Max 50 words.

    Scene: "{scene_description}"

    Image Prompt:
    """

    # Use the specified model for generating the image prompt
    image_prompt_text, input_tokens, output_tokens, prompt_latency = get_gemini_response(
        image_prompt_request, prompt_model_name)

    if not image_prompt_text or "error occurred" in image_prompt_text.lower():
        print(f"WARN: Could not generate a suitable image prompt. Using fallback.")
        image_prompt = f"Fantasy RPG scene: {scene_description[:200]}"
        # Can't know tokens/latency if prompt generation failed this way
        input_tokens, output_tokens, prompt_latency = 0, 0, 0.0
    else:
        # Extract prompt cleanly, handling potential extra text
        prompt_marker = "Image Prompt:"
        if prompt_marker in image_prompt_text:
            image_prompt = image_prompt_text.split(
                prompt_marker, 1)[-1].strip()
        else:
            image_prompt = image_prompt_text  # Use the whole response if marker not found
            print(
                f"WARN: 'Image Prompt:' marker not found in response. Using full text: {image_prompt}")

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
                # Return Gemini token counts and latency too
                return b64_image_data, None, input_tokens, output_tokens, prompt_latency
            else:
                print("ERROR: Could not access image bytes.")
                # Return Gemini token counts/latency even on Imagen error
                return None, "Failed to process generated image data.", input_tokens, output_tokens, prompt_latency
        else:
            print("WARN: Imagen generation resulted in no images.",
                  getattr(images, '_prediction_response', ''))
            # Return Gemini token counts/latency even on Imagen error
            return None, "Image generation failed or was blocked.", input_tokens, output_tokens, prompt_latency

    except Exception as e:
        print(f"Error calling Imagen API: {e}", file=sys.stderr)
        error_message = f"Error generating image: {e}"
        # Return Gemini token counts/latency even on Imagen error
        return None, error_message, input_tokens, output_tokens, prompt_latency


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
        game_state.setdefault('last_request_latency', 0.0)  # New
        game_state.setdefault('conversation_count', 0)
        game_state.setdefault('history_summaries', [])
        game_state.setdefault('total_summary_input_tokens', 0)
        game_state.setdefault('total_summary_output_tokens', 0)
        game_state.setdefault('total_summary_cost', 0.0)
        game_state.setdefault('last_summary_latency', 0.0)  # New
        game_state.setdefault('total_summary_latency', 0.0)  # New
        game_state.setdefault('last_summary_message', '')

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
            # Add/reset/default fields upon loading
            game_state.setdefault('total_input_tokens', 0)
            game_state.setdefault('total_output_tokens', 0)
            game_state['last_input_tokens'] = 0  # Reset last counts
            game_state['last_output_tokens'] = 0
            game_state['last_request_cost'] = 0.0  # Reset last cost
            game_state.setdefault('last_request_latency',
                                  0.0)  # Add if missing
            game_state['last_request_latency'] = 0.0  # Reset last latency
            game_state.setdefault('conversation_count', 0)
            game_state.setdefault('history_summaries', [])
            game_state.setdefault('total_summary_input_tokens', 0)
            game_state.setdefault('total_summary_output_tokens', 0)
            game_state.setdefault('total_summary_cost', 0.0)
            game_state.setdefault('last_summary_latency',
                                  0.0)  # Add if missing
            game_state.setdefault('total_summary_latency',
                                  0.0)  # Add if missing
            # Reset last summary latency
            game_state['last_summary_latency'] = 0.0
            # Reset last summary message on load
            game_state['last_summary_message'] = ''
            # Keep total summary cost/tokens/latency from save file

            print(f"Game loaded successfully from {filepath}!")
            return game_state, f"Game loaded successfully from '{filename}.json'!"
        else:
            print(
                f"Save file {filepath} seems corrupted (missing essential keys).", file=sys.stderr)
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
    Roleplay: You are the player for a text-based fantasy RPG.
    Task: Start a new adventure based on the player's first action or thought. Respond ONLY with the JSON structure.
    Player's input: '{initial_player_prompt}'
    Player's location: '{current_location}'
    {RPG_INSTRUCTION}"""

    response_text, input_tokens, output_tokens, latency = get_gemini_response(
        full_initial_prompt, model_name)

    # Calculate cost for this initial request
    last_request_cost = calculate_cost(
        model_name, input_tokens, output_tokens)

    if response_text and "error occurred" not in response_text.lower() and "unseen force" not in response_text.lower():
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
                "last_request_latency": latency,  # Store initial latency
                "conversation_count": 1,  # Started with 1 user + 1 model turn
                "history_summaries": [],
                "total_summary_input_tokens": 0,
                "total_summary_output_tokens": 0,
                "total_summary_cost": 0.0,
                "last_summary_latency": 0.0,  # Initialize
                "total_summary_latency": 0.0,  # Initialize
                "last_summary_message": "",
            }
            return game_state, "New adventure started!", input_tokens, output_tokens
        except (json.JSONDecodeError, ValueError) as e:
            print(
                f"Error parsing initial JSON response: {e}\nRaw response: {response_text}", file=sys.stderr)
            error_msg = f"Failed to understand the initial scene description from AI: {e}"
            # Return token counts even on parse error (cost/latency are calculated above)
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
    history_summaries = game_state.get("history_summaries", [])
    conversation_count = game_state.get("conversation_count", 0)
    summary_info_message = ""  # Message to return if summary happens
    # Reset last summary latency for this turn
    game_state['last_summary_latency'] = 0.0

    # Append user action temporarily
    chat_history.append({"role": "user", "parts": [{"text": player_action}]})

    # --- Build Context for Prompt ---
    # 1. Include previous summaries
    summary_context = ""
    if history_summaries:
        summary_context = "\n---\nPrevious Summaries:\n" + \
            "\n---\n".join(history_summaries) + "\n---"

    # 2. Include recent turns *since* last summary (or up to HISTORY_LIMIT if no summaries yet)
    # Calculate how many entries ago the last summary would have started
    # If count is 0, it means a summary just happened (or it's the start). We want turns since then.
    # If count is 1, we want the last 1*2=2 entries (prev user/model).
    # If count is SUMMARY_INTERVAL-1, we want the last (SUMMARY_INTERVAL-1)*2 entries.
    num_pairs_since_summary = conversation_count
    num_entries_since_summary = num_pairs_since_summary * 2

    # Calculate the start index for recent history, ensuring it doesn't go negative
    # It should be relative to the end of the history *before* the current user action was added
    start_index_recent = max(0, len(chat_history) -
                             1 - num_entries_since_summary)

    # Further limit recent history by HISTORY_LIMIT if num_entries_since_summary is large
    # We take the *minimum* of HISTORY_LIMIT pairs and the pairs since the last summary
    max_recent_entries = min(HISTORY_LIMIT * 2, num_entries_since_summary)
    start_index_recent = max(0, len(chat_history) - 1 - max_recent_entries)

    # Exclude current user action
    recent_history_entries = chat_history[start_index_recent:-1]

    print(
        f"   Building prompt context: {len(history_summaries)} summaries, {len(recent_history_entries)} recent history entries.")

    history_context = ""
    for entry in recent_history_entries:
        role = entry.get('role', 'unknown')
        # Handle potential structure variations
        part_text = '[No text content]'
        if entry.get('parts'):
            first_part = entry['parts'][0]
            if isinstance(first_part, dict) and 'text' in first_part:
                part_text = first_part['text']
            elif isinstance(first_part, str):
                part_text = first_part
        history_context += f"\n{role.capitalize()}: {part_text}"

    prompt = f"""
    Roleplay: You are the player for a text-based fantasy RPG.
    Task: Narrate the outcome of the player's action based on the current situation and history. Respond ONLY with the JSON structure.
    {summary_context}
    Current Location: {current_location}
    Player Inventory: {', '.join(player_inventory) or 'nothing'}
    Recent History (since last summary): {history_context.strip() if history_context else 'None'}
    Player's Action: "{player_action}"
    {RPG_INSTRUCTION}"""

    # --- Get AI Response for Action ---
    response_text, input_tokens, output_tokens, action_latency = get_gemini_response(
        prompt, model_name)

    # Calculate cost for this request
    last_request_cost = calculate_cost(
        model_name, input_tokens, output_tokens)
    feedback_messages = []

    # Update totals (do this regardless of success/failure of *parsing*)
    game_state.setdefault('total_input_tokens', 0)
    game_state.setdefault('total_output_tokens', 0)
    game_state['last_input_tokens'] = input_tokens
    game_state['last_output_tokens'] = output_tokens
    game_state['last_request_cost'] = last_request_cost
    # Store latency for this action
    game_state['last_request_latency'] = action_latency

    if response_text and "error occurred" not in response_text.lower() and "unseen force" not in response_text.lower():
        try:
            scene_json = json.loads(response_text)
            if not all(k in scene_json for k in [RPG_RESPONSE, RPG_RESPONSE_LOCATION, RPG_RESPONSE_INVENTORY]):
                raise ValueError("JSON response missing required keys.")

            # Action succeeded, finalize history and update state
            model_response_entry = {"role": "model", "parts": [
                {"text": scene_json[RPG_RESPONSE]}]}
            # Add successful model response
            chat_history.append(model_response_entry)

            game_state["current_location"] = scene_json[RPG_RESPONSE_LOCATION]
            inventory_str = scene_json.get(RPG_RESPONSE_INVENTORY, "")
            game_state["player_inventory"] = [item.strip()
                                              for item in inventory_str.split(',') if item.strip()]
            game_state["last_scene"] = scene_json[RPG_RESPONSE]
            game_state["chat_history"] = chat_history  # Save updated history
            # Add to total now
            game_state["total_input_tokens"] += input_tokens
            game_state["total_output_tokens"] += output_tokens
            game_state["conversation_count"] = conversation_count + \
                1  # Increment count *after* successful pair

            print(
                f"Updated game_state (tokens: last={input_tokens}/{output_tokens}, total={game_state['total_input_tokens']}/{game_state['total_output_tokens']}, last_cost=${last_request_cost:.6f}, latency={action_latency:.3f}s)")

            # --- Check for Summarization ---
            # Use >= in case count somehow skips the exact value
            if game_state["conversation_count"] >= SUMMARY_INTERVAL:
                print(
                    f"\n🔄 Reached {game_state['conversation_count']} conversations (>= interval {SUMMARY_INTERVAL}), attempting summarization...")
                # Select turns for summary: last SUMMARY_INTERVAL pairs (user + model)
                num_entries_to_summarize = SUMMARY_INTERVAL * 2
                if len(chat_history) < num_entries_to_summarize:
                    print(
                        f"WARN: Not enough history ({len(chat_history)}) for {num_entries_to_summarize} entries needed for summary. Skipping.")
                else:
                    # Summarize the turns that *led up to* this point
                    turns_to_summarize = chat_history[-num_entries_to_summarize:]
                    summary_text, sum_in_tok, sum_out_tok, sum_cost, summary_latency = generate_summary(
                        turns_to_summarize, model_name)

                    if summary_text:
                        game_state["history_summaries"].append(summary_text)
                        game_state["conversation_count"] = 0  # Reset counter
                        # Track summary costs/tokens/latency
                        game_state.setdefault('total_summary_input_tokens', 0)
                        game_state.setdefault('total_summary_output_tokens', 0)
                        game_state.setdefault('total_summary_cost', 0.0)
                        game_state.setdefault('total_summary_latency', 0.0)
                        game_state['total_summary_input_tokens'] += sum_in_tok
                        game_state['total_summary_output_tokens'] += sum_out_tok
                        game_state['total_summary_cost'] += sum_cost
                        # Store last summary latency
                        game_state['last_summary_latency'] = summary_latency
                        game_state['total_summary_latency'] += summary_latency
                        summary_info_message = f"[System: Summarized last {SUMMARY_INTERVAL} turns. Cost: ~${sum_cost:.6f}, Latency: {summary_latency:.3f}s]"
                        # Store for display
                        game_state['last_summary_message'] = summary_info_message
                        print(
                            f"   Summary successful. Total summaries: {len(game_state['history_summaries'])}. Total summary latency: {game_state['total_summary_latency']:.3f}s")
                    else:
                        # Use latency from the failed attempt in the message if available
                        fail_latency_msg = f", Latency: {summary_latency:.3f}s" if summary_latency else ""
                        summary_info_message = f"[System: Summary generation failed{fail_latency_msg}.]"
                        # Store for display
                        game_state['last_summary_message'] = summary_info_message
                        print("   Summary generation failed.")
                        # Keep conversation_count as is, try again next time? Or reset?
                        # Let's reset to avoid constant attempts on persistent failure.
                        game_state["conversation_count"] = 0

            return game_state, scene_json[RPG_RESPONSE], feedback_messages, input_tokens, output_tokens, summary_info_message

        except (json.JSONDecodeError, ValueError) as e:
            # Error parsing the JSON response
            print(
                f"Error parsing action JSON response: {e}\nRaw response: {response_text}", file=sys.stderr)
            error_message = f"The AI's response was unclear: {e}"
            feedback_messages.append(error_message)
            chat_history.pop()  # Remove user action, as the turn failed
            game_state["chat_history"] = chat_history
            # Clear any previous summary message
            game_state['last_summary_message'] = ''
            # Don't add tokens to total on parse error, as the turn didn't 'succeed' fully
            # Don't increment conversation_count
            # Return empty summary msg
            return game_state, game_state["last_scene"], feedback_messages, input_tokens, output_tokens, summary_info_message
    else:
        # Error getting response from AI (blocked, network error, etc.)
        error_message = response_text if response_text else "The AI seems unresponsive. Please try again."
        feedback_messages.append(error_message)
        chat_history.pop()  # Remove user action
        game_state["chat_history"] = chat_history
        # Clear any previous summary message
        game_state['last_summary_message'] = ''
        # Do NOT add tokens to total on AI error
        # Do NOT increment conversation_count
        # Return empty summary msg
        return game_state, game_state["last_scene"], feedback_messages, input_tokens, output_tokens, summary_info_message
