# rpg_logic.py
import os
import google.generativeai as genai
import google.auth
import json
import sys

# --- Constants ---
# Use a different name to avoid conflicts
SAVE_FILENAME = os.path.expanduser("~/tmp/rpg_save_web.json")
MODEL_NAME = 'gemini-2.0-flash'

# --- Gemini Setup (Consider making the model selectable via UI/config later) ---
try:
    # 1. Automatically find your default gcloud credentials
    credentials, project = google.auth.default()

    # 2. Configure the Gemini API with these credentials
    # Use credentials.token if using user credentials
    genai.configure(api_key=credentials.token)

    # --- Choose a Model (Hardcoding for simplicity in web version) ---
    # You could make this configurable later (e.g., environment variable, UI choice)
    # model_name = "gemini-1.5-flash-latest" # Good balance of speed and capability
    # Another solid option if 1.5 isn't available or needed
    model_name = MODEL_NAME
    print(f"Using Gemini Model: {model_name}")
    model = genai.GenerativeModel(model_name)

except Exception as e:
    print(
        f"FATAL ERROR: Could not initialize Google Cloud credentials or Gemini model: {e}", file=sys.stderr)
    print("Ensure you have run 'gcloud auth application-default login'", file=sys.stderr)
    sys.exit(1)  # Exit if Gemini setup fails


# --- Gemini Interaction ---
def get_gemini_response(prompt):
    """Sends a prompt to Gemini and returns the text response."""
    print("\n🤖 *Gemini is thinking...*\n")  # Log for server console
    try:
        # Safety settings can be adjusted if needed
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

        # Handle potential blocks or empty responses
        if not response.candidates:
            print("Warning: Gemini response was empty or blocked (no candidates).")
            return "The world seems hazy... you're unsure what happened. Try something else."
        if response.prompt_feedback.block_reason:
            print(
                f"Warning: Prompt blocked. Reason: {response.prompt_feedback.block_reason}")
            # Consider providing specific feedback based on the block reason if desired
            return f"Your action was blocked by safety filters ({response.prompt_feedback.block_reason}). Please try a different approach."

        return response.text
    except Exception as e:
        print(f"Error communicating with Gemini: {e}", file=sys.stderr)
        # Provide a generic in-game error message
        return "A strange interference prevents you from understanding the outcome. Try again."

# --- Save/Load Functions ---


def save_game(filename, game_state):
    """Saves the current game state to a JSON file."""
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as f:
            json.dump(game_state, f, indent=4)
        print(f"💾 Game saved successfully to {filename}!")  # Server log
        return True, f"Game saved successfully to {filename}!"
    except IOError as e:
        print(f"Error saving game to {filename}: {e}", file=sys.stderr)
        return False, f"Error saving game: {e}"
    except Exception as e:
        print(
            f"An unexpected error occurred while saving: {e}", file=sys.stderr)
        return False, f"An unexpected error occurred while saving: {e}"


def load_game(filename):
    """Loads the game state from a JSON file."""
    if not os.path.exists(filename):
        print(f"Save file not found: {filename}")  # Server log
        return None, f"No save file found at {filename}."  # Message for UI

    try:
        with open(filename, 'r') as f:
            game_state = json.load(f)
        # Basic validation
        if not all(k in game_state for k in ["current_location", "player_inventory", "last_scene", "chat_history"]):
            print(f"Save file {filename} is corrupted or incomplete.")
            return None, f"Save file {filename} seems corrupted. Starting new game."
        print(f"📖 Game loaded successfully from {filename}!")  # Server log
        return game_state, f"Game loaded successfully from {filename}!"
    except (IOError, json.JSONDecodeError) as e:
        print(f"Error loading game from {filename}: {e}", file=sys.stderr)
        return None, f"Error loading game from {filename}: {e}. Starting new game."
    except Exception as e:
        print(
            f"An unexpected error occurred while loading: {e}", file=sys.stderr)
        return None, f"Unexpected error loading game: {e}. Starting new game."

# --- Game State Parsing Logic ---


def update_game_state_from_response(response_text, current_location, player_inventory):
    """
    Parses Gemini's response to potentially update location and inventory.
    Returns the updated location, inventory, and any feedback messages.
    This is still basic parsing and can be improved.
    """
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
                # Extract text after keyword, take first sentence fragment
                potential_item_phrase = parts[1].split(
                    '.')[0].split(',')[0].strip()
                # Simple filtering for common articles/pronouns
                potential_item = potential_item_phrase.replace(
                    " a ", " ").replace(" an ", " ").replace(" the ", " ").strip()
                # More robust filtering might be needed based on game style
                if potential_item and len(potential_item) > 2 and potential_item not in ["it", "them", "nothing", "something"]:
                    # Avoid adding duplicates if already parsed from this response
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
    # Check against potentially updated inventory
    items_to_check = list(new_inventory)
    for keyword in remove_keywords:
        if keyword in response_lower:
            parts = response_lower.split(keyword, 1)
            if len(parts) > 1:
                # Check next 50 chars after keyword
                relevant_part = parts[1][:50]
                for item in items_to_check:
                    # Check if item name exists after keyword (case-insensitive)
                    if item.lower() in relevant_part:
                        # Check if it hasn't been marked for removal already
                        if item not in removed_items and item in new_inventory:
                            new_inventory.remove(item)
                            removed_items.append(item)
                            feedback_messages.append(
                                f"Removed '{item}' from inventory.")
                            # Server log
                            print(f"[Parser] Removed item: {item}")
                            # No break here, could remove multiple items

    # --- Location Changes ---
    move_keywords = ["you travel to", "you enter the", "you arrive at",
                     "you are now in", "step into the", "you reach the"]
    location_changed = False
    for keyword in move_keywords:
        if keyword in response_lower and not location_changed:  # Only change once per response
            parts = response_lower.split(keyword, 1)
            if len(parts) > 1:
                potential_location = parts[1].split(
                    '.')[0].split(',')[0].strip()
                # Basic filtering
                potential_location = potential_location.replace(
                    " a ", " ").replace(" an ", " ").replace(" the ", " ").strip()
                if potential_location and len(potential_location) > 3:
                    new_location = potential_location
                    location_changed = True
                    feedback_messages.append(f"Moved to '{new_location}'.")
                    # Server log
                    print(f"[Parser] Changed location: {new_location}")
                    # break # Found location change

    return new_location, new_inventory, feedback_messages


# --- Core Game Turn Logic ---
def process_player_action(player_action, game_state):
    """
    Takes the player action and current game state, interacts with Gemini,
    updates the state, and returns the new state and feedback.
    """
    current_location = game_state["current_location"]
    player_inventory = game_state["player_inventory"]
    scene_description = game_state["last_scene"]
    chat_history = game_state["chat_history"]

    # Add player's action to chat history
    chat_history.append({"role": "user", "content": player_action})

    # Construct the prompt for Gemini
    prompt = f"""
    Current situation: The player is at '{current_location}'.
    Player inventory: {', '.join(player_inventory) or 'nothing'}.
    Previous scene description: {scene_description}

    Recent Chat History (for context):
    """
    # Include a limited number of recent turns to keep prompt concise
    history_limit = 5
    start_index = max(0, len(chat_history) -
                      (history_limit * 2))  # Look back ~5 turns
    for entry in chat_history[start_index:]:
        prompt += f"\n{entry['role'].capitalize()}: {entry['content']}"

    prompt += f"""

    Player's latest action: '{player_action}'

    Narrator task: Describe the outcome of the player's action and the updated scene.
    - Maintain a consistent fantasy RPG tone.
    - The description should logically follow the action and previous context.
    - Describe changes in location or inventory naturally within the narrative (e.g., "You pick up the dusty key.", "You cautiously step into the echoing cavern.").
    - Conclude with a sentence that prompts the player for their next move (e.g., "What do you do next?", "The corridor stretches before you.").
    - Avoid directly stating game mechanics like "Inventory updated" or "Location changed" in the narrative itself. Just describe the scene and actions.
    """

    response_text = get_gemini_response(prompt)

    # Add Gemini's response to chat history (even if it's an error message)
    chat_history.append({"role": "model", "content": response_text})

    # Update scene description
    new_scene_description = response_text

    # Parse response for state changes
    new_location, new_inventory, parse_feedback = update_game_state_from_response(
        response_text, current_location, player_inventory
    )

    # Update game state dictionary
    updated_game_state = {
        "current_location": new_location,
        "player_inventory": new_inventory,
        "last_scene": new_scene_description,
        "chat_history": chat_history  # Keep the full history
    }

    return updated_game_state, parse_feedback


def start_new_game(initial_player_thought="I wake up in a dimly lit clearing."):
    """Initializes a new game state and gets the first scene from Gemini."""
    print("Starting new game...")  # Server log
    # Initial state before Gemini describes it
    current_location = "an unknown location"
    player_inventory = []
    chat_history = []

    # Add initial player thought to history
    chat_history.append({"role": "user", "content": initial_player_thought})

    # Construct the initial prompt for Gemini
    full_initial_prompt = f"""
    Start a text-based fantasy RPG adventure.
    The player's first thought or action is: '{initial_player_thought}'.
    Describe the initial scene vividly. Where is the player? What do they see, hear, smell?
    Hint at possible immediate actions or points of interest.
    The player starts with no items in their inventory.
    Conclude with a question prompting the player's first real action (e.g., "What do you do?").
    """

    scene_description = get_gemini_response(full_initial_prompt)

    if not scene_description or "Try again" in scene_description or "hazy" in scene_description:  # Basic check if Gemini failed
        # Fallback initial scene if Gemini fails
        scene_description = "You find yourself standing in a quiet forest clearing. Sunlight filters through the leaves. A path leads north. What do you do?"
        current_location = "forest clearing"  # Set a fallback location
        print("Warning: Failed to get initial scene from Gemini, using fallback.")
    else:
        # Try to parse the initial location from Gemini's first response
        # This is tricky, might need refinement or a specific instruction to Gemini
        parsed_loc, _, _ = update_game_state_from_response(
            scene_description, current_location, player_inventory)
        if parsed_loc != "an unknown location":
            current_location = parsed_loc
        else:
            # If parsing failed, make a guess or ask Gemini specifically later
            # Or parse from scene_description manually
            current_location = "mysterious starting point"

    # Add Gemini's first response to history
    chat_history.append({"role": "model", "content": scene_description})

    new_game_state = {
        "current_location": current_location,
        "player_inventory": player_inventory,
        "last_scene": scene_description,
        "chat_history": chat_history
    }
    return new_game_state
