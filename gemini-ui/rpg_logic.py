# rpg_logic.py
import os
import google.generativeai as genai
import google.auth
import json
import sys

# --- Constants ---
SAVE_FILENAME = os.path.expanduser("~/tmp/rpg_save_web.json")

# --- Google Cloud Credentials Setup ---
try:
    credentials, project = google.auth.default()
    # Configure the genai library with the API key (token) from credentials
    genai.configure(api_key=credentials.token)
    print("Gemini API configured successfully.")
except Exception as e:
    print(
        f"FATAL ERROR: Could not initialize Google Cloud credentials or configure Gemini API: {e}", file=sys.stderr)
    print("Ensure you have run 'gcloud auth application-default login'", file=sys.stderr)
    sys.exit(1)  # Exit if basic setup fails

# --- Gemini Interaction (Model is now created per request) ---


def get_gemini_response(prompt, model_name):
    """Sends a prompt to Gemini using the specified model and returns the text response."""
    print(
        f"\n🤖 *Gemini ({model_name}) is thinking...*\n")  # Log for server console
    try:
        # Create the model instance here, based on the selected name
        model = genai.GenerativeModel(model_name)

        # Safety settings (same as before)
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]
        response = model.generate_content(
            prompt, safety_settings=safety_settings)

        # Handle potential blocks or empty responses (same as before)
        if not response.candidates:
            print(
                f"Warning: Gemini ({model_name}) response was empty or blocked (no candidates).")
            return "The world seems hazy... you're unsure what happened. Try something else."
        if response.prompt_feedback.block_reason:
            print(
                f"Warning: Prompt blocked by {model_name}. Reason: {response.prompt_feedback.block_reason}")
            return f"Your action was blocked by safety filters ({response.prompt_feedback.block_reason}). Please try a different approach."

        return response.text
    except Exception as e:
        # Catch errors during model instantiation or generation
        print(
            f"Error communicating with Gemini model '{model_name}': {e}", file=sys.stderr)
        # Provide a generic in-game error message
        return f"A strange interference prevents you from understanding the outcome (Model: {model_name}, Error: {e}). Try again."

# --- Save/Load Functions (No changes needed here) ---


def save_game(filename, game_state):
    # ... (same as before) ...
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as f:
            json.dump(game_state, f, indent=4)
        print(f"💾 Game saved successfully to {filename}!")  # Server log
        return True, f"Game saved successfully!"  # Message for UI flash
    except IOError as e:
        print(f"Error saving game to {filename}: {e}", file=sys.stderr)
        return False, f"Error saving game: {e}"
    except Exception as e:
        print(
            f"An unexpected error occurred while saving: {e}", file=sys.stderr)
        return False, f"An unexpected error occurred while saving: {e}"


def load_game(filename):
    # ... (same as before) ...
    if not os.path.exists(filename):
        print(f"Save file not found: {filename}")  # Server log
        return None, f"No save file found."  # Message for UI

    try:
        with open(filename, 'r') as f:
            game_state = json.load(f)
        # Basic validation - ADD MODEL NAME check
        if not all(k in game_state for k in ["current_location", "player_inventory", "last_scene", "chat_history", "model_name"]):
            print(
                f"Save file {filename} is corrupted or incomplete (missing keys).")
            # If model_name is missing, maybe add a default? Or force new game.
            if "model_name" not in game_state and all(k in game_state for k in ["current_location", "player_inventory", "last_scene", "chat_history"]):
                print("Save file missing model_name, will use default on load.")
                # We'll handle adding the default model in app.py load logic
            else:
                return None, f"Save file {filename} seems corrupted. Starting new game."

        print(f"📖 Game loaded successfully from {filename}!")  # Server log
        return game_state, f"Game loaded successfully!"  # Message for UI flash
    except (IOError, json.JSONDecodeError) as e:
        print(f"Error loading game from {filename}: {e}", file=sys.stderr)
        return None, f"Error loading game: {e}. Starting new game."
    except Exception as e:
        print(
            f"An unexpected error occurred while loading: {e}", file=sys.stderr)
        return None, f"Unexpected error loading game: {e}. Starting new game."


# --- Game State Parsing Logic (No changes needed here) ---
def update_game_state_from_response(response_text, current_location, player_inventory):
    # ... (same as before) ...
    new_location = current_location
    new_inventory = list(player_inventory)  # Create a copy to modify
    feedback_messages = []
    response_lower = response_text.lower()

    # --- Inventory Additions ---
    take_keywords = ["you pick up", "you take the",
                     "add ", "acquire the", "you now have"]
    added_items = []
    for keyword in take_keywords:
        if keyword in response_lower:
            parts = response_lower.split(keyword, 1)
            if len(parts) > 1:
                potential_item_phrase = parts[1].split(
                    '.')[0].split(',')[0].strip()
                potential_item = potential_item_phrase.replace(
                    " a ", " ").replace(" an ", " ").replace(" the ", " ").strip()
                if potential_item and len(potential_item) > 2 and potential_item not in ["it", "them", "nothing", "something"]:
                    if potential_item not in added_items and potential_item not in new_inventory:
                        new_inventory.append(potential_item)
                        added_items.append(potential_item)
                        feedback_messages.append(
                            f"Added '{potential_item}' to inventory.")
                        # Server log
                        print(f"[Parser] Added item: {potential_item}")

    # --- Inventory Removals ---
    remove_keywords = ["you drop the", "you use the", "item disappears",
                       "breaks the", "no longer have the", "you consume the", "you give the"]
    removed_items = []
    items_to_check = list(new_inventory)
    for keyword in remove_keywords:
        if keyword in response_lower:
            parts = response_lower.split(keyword, 1)
            if len(parts) > 1:
                relevant_part = parts[1][:50]
                for item in items_to_check:
                    if item.lower() in relevant_part:
                        if item not in removed_items and item in new_inventory:
                            new_inventory.remove(item)
                            removed_items.append(item)
                            feedback_messages.append(
                                f"Removed '{item}' from inventory.")
                            # Server log
                            print(f"[Parser] Removed item: {item}")

    # --- Location Changes ---
    move_keywords = ["you travel to", "you enter the", "you arrive at",
                     "you are now in", "step into the", "you reach the"]
    location_changed = False
    for keyword in move_keywords:
        if keyword in response_lower and not location_changed:
            parts = response_lower.split(keyword, 1)
            if len(parts) > 1:
                potential_location = parts[1].split(
                    '.')[0].split(',')[0].strip()
                potential_location = potential_location.replace(
                    " a ", " ").replace(" an ", " ").replace(" the ", " ").strip()
                if potential_location and len(potential_location) > 3:
                    new_location = potential_location
                    location_changed = True
                    feedback_messages.append(f"Moved to '{new_location}'.")
                    # Server log
                    print(f"[Parser] Changed location: {new_location}")

    return new_location, new_inventory, feedback_messages

# --- Core Game Turn Logic (Needs model_name parameter) ---


def process_player_action(player_action, game_state, model_name):
    """
    Takes the player action, current game state, and model_name,
    interacts with Gemini, updates the state, and returns the new state and feedback.
    """
    # ... (retrieve state from game_state dictionary as before) ...
    current_location = game_state["current_location"]
    player_inventory = game_state["player_inventory"]
    scene_description = game_state["last_scene"]
    chat_history = game_state["chat_history"]

    chat_history.append({"role": "user", "content": player_action})

    # Construct the prompt (same as before)
    prompt = f"""
    Current situation: The player is at '{current_location}'.
    Player inventory: {', '.join(player_inventory) or 'nothing'}.
    Previous scene description: {scene_description}

    Recent Chat History (for context):
    """
    history_limit = 5
    start_index = max(0, len(chat_history) - (history_limit * 2))
    for entry in chat_history[start_index:]:
        prompt += f"\n{entry['role'].capitalize()}: {entry['content']}"

    prompt += f"""

    Player's latest action: '{player_action}'

    Narrator task: Describe the outcome of the player's action and the updated scene.
    - Maintain a consistent fantasy RPG tone.
    - The description should logically follow the action and previous context.
    - Describe changes in location or inventory naturally within the narrative (e.g., "You pick up the dusty key.", "You cautiously step into the echoing cavern.").
    - Avoid directly stating game mechanics like "Inventory updated" or "Location changed" in the narrative itself. Just describe the scene and actions.
    """

    # Call get_gemini_response WITH the model_name
    response_text = get_gemini_response(prompt, model_name)
    print("==input: ", prompt)
    print("==output: ", response_text)

    chat_history.append({"role": "model", "content": response_text})
    new_scene_description = response_text

    # Parse response for state changes (same as before)
    new_location, new_inventory, parse_feedback = update_game_state_from_response(
        response_text, current_location, player_inventory
    )

    # Update game state dictionary - IMPORTANT: Include model_name
    updated_game_state = {
        "current_location": new_location,
        "player_inventory": new_inventory,
        "last_scene": new_scene_description,
        "chat_history": chat_history,
        "model_name": model_name  # Persist the model used for this state
    }

    return updated_game_state, parse_feedback


# --- Start New Game Logic (Needs model_name parameter) ---
def start_new_game(model_name, initial_player_thought="I wake up in a dimly lit clearing."):
    """Initializes a new game state using the specified model and gets the first scene."""
    print(f"Starting new game with model: {model_name}...")  # Server log
    current_location = "an unknown location"
    player_inventory = []
    chat_history = []

    chat_history.append({"role": "user", "content": initial_player_thought})

    # Construct the initial prompt (same as before)
    full_initial_prompt = f"""
    Start a text-based fantasy RPG adventure.
    The player's first thought or action is: '{initial_player_thought}'.
    Describe the initial scene vividly. Where is the player? What do they see, hear, smell?
    Hint at possible immediate actions or points of interest.
    The player starts with no items in their inventory.
    Conclude with a question prompting the player's first real action (e.g., "What do you do?").
    """

    # Call get_gemini_response WITH the model_name
    scene_description = get_gemini_response(full_initial_prompt, model_name)
    print("init prompt: ", full_initial_prompt)
    print("init out: ", scene_description)

    # Fallback logic (same as before)
    if not scene_description or "Try again" in scene_description or "hazy" in scene_description or "interference" in scene_description:
        scene_description = "You find yourself standing in a quiet forest clearing. Sunlight filters through the leaves. A path leads north. What do you do?"
        current_location = "forest clearing"
        print(
            f"Warning: Failed to get initial scene from Gemini ({model_name}), using fallback.")
    else:
        # Try parsing initial location (same as before)
        parsed_loc, _, _ = update_game_state_from_response(
            scene_description, current_location, player_inventory)
        current_location = parsed_loc if parsed_loc != "an unknown location" else "mysterious starting point"

    chat_history.append({"role": "model", "content": scene_description})

    # Create game state - IMPORTANT: Include model_name
    new_game_state = {
        "current_location": current_location,
        "player_inventory": player_inventory,
        "last_scene": scene_description,
        "chat_history": chat_history,
        "model_name": model_name  # Store the model used to start the game
    }
    return new_game_state
