import os
import google.generativeai as genai
import google.auth
import json  # <-- Import json for saving/loading
import sys  # <-- Import sys for exiting on critical errors

# --- Constants ---
# SAVE_FILENAME = "rpg_save.json"
SAVE_FILENAME = os.path.expanduser("~/tmp/rpg_save.json")

# 1. Automatically find your default gcloud credentials
# scopes = ['https://www.googleapis.com/auth/generative-language']
# credentials, project = google.auth.default(scopes=scopes)


credentials, project = google.auth.default()

# 2. Configure the Gemini API with these credentials
genai.configure(api_key=credentials.token)

model_options = {
    "1": "gemini-2.5-pro-preview-03-25",
    "2": 'gemini-2.0-flash',
    "3": "gemini-2.0-flash-lite",
}

print(model_options)
print("Choose a model")
option = input("> ").strip()

model = genai.GenerativeModel(model_options[option])


# --- Gemini Interaction ---
def get_gemini_response(prompt):
    """Sends a prompt to Gemini and returns the text response."""
    print("\n🤖 *Gemini is thinking...*\n")
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error communicating with Gemini: {e}")
        return None

# --- Game Functions ---


def display_scene(description):
    """Prints the current scene description to the console."""
    print("\n" + "=" * 40)
    print(description)
    print("=" * 40)


def get_player_action():
    """Gets the player's action from the console."""
    return input("> ").strip()

# --- Save/Load Functions ---


def save_game(filename, game_state):
    """Saves the current game state to a JSON file."""
    try:
        with open(filename, 'w') as f:
            json.dump(game_state, f, indent=4)  # Use indent for readability
        print(f"\n💾 Game saved successfully to {filename}!")
        return True
    except IOError as e:
        print(f"\nError saving game to {filename}: {e}")
        return False
    except Exception as e:
        print(f"\nAn unexpected error occurred while saving: {e}")
        return False


def load_game(filename):
    """Loads the game state from a JSON file."""
    if not os.path.exists(filename):
        return None  # Return None if save file doesn't exist

    try:
        with open(filename, 'r') as f:
            game_state = json.load(f)
        print(f"\n📂 Game loaded successfully from {filename}!")
        return game_state
    except (IOError, json.JSONDecodeError) as e:
        print(f"\nError loading game from {filename}: {e}")
        print("Starting a new game instead.")
        return None  # Return None if loading fails
    except Exception as e:
        print(f"\nAn unexpected error occurred while loading: {e}")
        print("Starting a new game instead.")
        return None


# --- Main Game Logic ---
def main():
    """Runs the text-based RPG game with Gemini."""
    print("Welcome to the Gemini-Powered Text RPG!")

    print("Commands: type your action, 'save', 'load' (at start), or 'exit'.")

    # --- Try Loading Game ---
    game_state = load_game(SAVE_FILENAME)
    loaded_game = False

    if game_state:
        # Validate loaded state (basic check)
        if all(k in game_state for k in ["current_location", "player_inventory", "last_scene"]):
            print("Resuming your adventure...")
            current_location = game_state["current_location"]
            player_inventory = game_state["player_inventory"]
            scene_description = game_state["last_scene"]
            loaded_game = True
            display_scene(scene_description)  # Show the loaded scene
        else:
            print("Save file seems corrupted. Starting a new game.")
            game_state = None  # Force new game start

    # --- Start New Game if not loaded ---
    if not loaded_game:
        print("\nStarting a new adventure!")
        current_location = "an unknown location"  # More descriptive start
        player_inventory = []
        scene_description = None  # Initialize scene_description

        # Initial scene generation
        print("\nDescribe the beginning of your adventure (e.g., 'I wake up in a dark forest'):")
        initial_prompt = get_player_action()
        if not initial_prompt or initial_prompt.lower() == 'exit':
            print("No adventure today? Goodbye!")
            return

        # Construct a slightly better initial prompt for Gemini
        full_initial_prompt = f"Start a text adventure. The player's initial thought or action is: '{initial_prompt}'. Describe the initial location ({current_location}) vividly, setting the scene and hinting at possible first actions. The player starts with no items."
        scene_description = get_gemini_response(full_initial_prompt)

        if scene_description:
            display_scene(scene_description)
        else:
            print("Failed to get the initial scene description from Gemini. Exiting.")
            return

    # --- Main Game Loop ---
    while True:
        player_action = get_player_action()

        # --- Handle Meta Commands ---
        if player_action.lower() == "exit":
            print("Thanks for playing!")
            break
        elif player_action.lower() == "save":
            # Create the game state dictionary to save
            current_game_state = {
                "current_location": current_location,
                "player_inventory": player_inventory,
                "last_scene": scene_description  # Save the last description shown
            }
            save_game(SAVE_FILENAME, current_game_state)
            continue  # Continue playing after saving
        elif player_action.lower() == "load":
            print("You can only load a game at the very start.")
            continue

        # --- Process Player Action with Gemini ---
        # Construct the prompt for Gemini based on the current state and player action
        prompt = f"""
        Current situation: The player is at '{current_location}'.
        Player inventory: {', '.join(player_inventory) or 'nothing'}.
        Previous description: {scene_description}

        Player's action: '{player_action}'

        Narrator task: Describe the outcome of this action and the updated scene.
        - Be creative and consistent with a fantasy RPG tone.
        - Update the environment or story based on the action.
        - Mention changes in location or inventory implicitly in the description if they occur (e.g., "You step into the cold cave...", "You pick up the shiny amulet.").
        - End with a sentence prompting the player for their next action (e.g., "What do you do next?", "The path forks ahead.").
        """
        # Important Note: Don't explicitly tell Gemini to change location/inventory in the prompt,
        # let it describe the outcome naturally. We will parse its response below.

        response = get_gemini_response(prompt)
        if response:
            # Update the current scene description *before* parsing state changes
            scene_description = response
            display_scene(scene_description)

            # --- Update Game State Based on Gemini's Response (Simple Parsing) ---
            # This parsing is VERY basic and fragile. A more robust game would need
            # better ways to understand state changes from the AI's narrative.
            # Consider asking Gemini to output structured data alongside narrative,
            # or use keyword spotting more carefully.

            response_lower = response.lower()  # For case-insensitive checks

            # Example: Detect taking items
            # Look for phrases like "you pick up", "you take the", "add ... to your inventory"
            take_keywords = ["you pick up",
                             "you take the", "add ", "acquire the"]
            for keyword in take_keywords:
                if keyword in response_lower:
                    # Try to extract item name after the keyword
                    try:
                        potential_item = response_lower.split(keyword, 1)[1].split('.')[
                            0].split(',')[0].strip()
                        # Basic filtering of common words
                        if potential_item and len(potential_item) > 2 and potential_item not in ["it", "the", "a", "an", "nothing", "something"]:
                            # Avoid adding duplicates if mentioned again
                            if potential_item not in player_inventory:
                                player_inventory.append(potential_item)
                                print(
                                    f"\n[Inventory updated: Added '{potential_item}']")
                                break  # Stop after finding one item per response
                    except IndexError:
                        pass  # Ignore if split fails

            # Example: Detect location changes
            # Look for phrases like "you travel to", "you enter the", "arrive at"
            move_keywords = ["you travel to", "you enter the",
                             "you arrive at", "you are now in", "step into the"]
            for keyword in move_keywords:
                if keyword in response_lower:
                    try:
                        potential_location = response_lower.split(keyword, 1)[1].split('.')[
                            0].split(',')[0].strip()
                        if potential_location and len(potential_location) > 3:
                            current_location = potential_location  # Update location
                            print(
                                f"\n[Location updated: Now at '{current_location}']")
                            break  # Stop after finding one location change
                    except IndexError:
                        pass

            # Example: Detect dropping/using items (Remove from inventory)
            # Look for phrases like "you drop the", "you use the", "is gone from your inventory"
            remove_keywords = ["you drop the", "you use the",
                               "item disappears", "breaks the", "no longer have the"]
            items_to_remove = []
            for keyword in remove_keywords:
                if keyword in response_lower:
                    # Check if any inventory item is mentioned shortly after the keyword
                    # This is very approximate!
                    relevant_part = response_lower.split(
                        keyword, 1)[1][:50]  # Check next 50 chars
                    for item in player_inventory:
                        if item in relevant_part:
                            if item not in items_to_remove:
                                items_to_remove.append(item)

            if items_to_remove:
                for item in items_to_remove:
                    if item in player_inventory:  # Check again, might have been removed by another keyword
                        player_inventory.remove(item)
                        print(f"\n[Inventory updated: Removed '{item}']")

        else:
            print("The AI seems unresponsive. Please try your action again.")
            # Keep the old scene_description if the response failed


if __name__ == "__main__":
    main()
