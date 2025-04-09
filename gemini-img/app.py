import os
import google.generativeai as genai
import google.auth
import json
import sys
from flask import Flask, render_template, request, session, redirect, url_for, flash
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel, ImageGenerationResponse

# --- Constants ---
SAVES_DIR = "saves" # Directory to store save files
if not os.path.exists(SAVES_DIR):
    os.makedirs(SAVES_DIR)
    print(f"Created saves directory at: {os.path.abspath(SAVES_DIR)}")

# --- Flask App Setup ---
app = Flask(__name__)
# CHANGE THIS! Use a real secret key for sessions
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "a_very_insecure_default_key")

# --- Gemini Configuration (Consider doing this once at startup) ---
# It might be better to configure Gemini outside request handlers if possible,
# but credentials might need refreshing. Putting it here for simplicity,
# but be mindful of potential performance hits or API call limits.
def configure_gemini():
    """Configures the Gemini client."""
    credentials, project = google.auth.default()
    genai.configure(api_key=credentials.token) # Use the token from credentials
    # You might want to select the model differently in a web UI
    # Maybe a dropdown in the 'New Game' section?
    # For now, hardcoding Pro.
    model_name = "gemini-2.0-flash" # Using a stable recent model
    image_model_name = 'imagen-3.0-generate-002'
    print(f"Using Gemini model: {model_name} {image_model_name}")
    model = genai.GenerativeModel(model_name)
    image_model = ImageGenerationModel.from_pretrained(image_model_name)
    return model, image_model

# --- Gemini Interaction ---
def get_gemini_response(model, prompt):
    """Sends a prompt to Gemini and returns the text response."""
    if not model:
        return "Error: Gemini model not configured."
    print("\n🤖 *Gemini is thinking...*\n")
    try:
        response = model.generate_content(prompt)
        # print(f"Gemini Response Object: {response}") # Debugging
        # print(f"Prompt Feedback: {response.prompt_feedback}") # Debugging Safety
        # Check for safety blocks or empty candidates
        if not response.candidates:
             if response.prompt_feedback.block_reason:
                  print(f"Prompt blocked: {response.prompt_feedback.block_reason}")
                  print(f"Safety Ratings: {response.prompt_feedback.safety_ratings}")
                  return f"⚠️ The prompt was blocked by safety filters: {response.prompt_feedback.block_reason}. Try rephrasing your action."
             else:
                  print("Warning: No content generated, response.candidates is empty.")
                  return "⚠️ The AI did not generate a response. It might be a content filter issue or an API problem. Try again."

        # Access text safely, checking finish_reason
        candidate = response.candidates[0]
        if candidate.finish_reason == 'STOP':
            if candidate.content and candidate.content.parts:
                 return candidate.content.parts[0].text
            else:
                 print("Warning: Finish reason STOP but no content parts found.")
                 return "⚠️ The AI finished but returned no text content. Please try again."
        elif candidate.finish_reason == 'MAX_TOKENS':
             return candidate.content.parts[0].text + "\n\n[... The story was cut short due to maximum length. Be more concise?]"
        elif candidate.finish_reason == 'SAFETY':
             print(f"Response blocked due to SAFETY. Ratings: {candidate.safety_ratings}")
             return f"⚠️ The AI's response was blocked by safety filters. Safety Ratings: {candidate.safety_ratings}. Try a different action."
        else:
             print(f"Warning: Unexpected finish reason: {candidate.finish_reason}")
             # Try returning text if available, otherwise indicate the issue
             if candidate.content and candidate.content.parts:
                 return candidate.content.parts[0].text + f"\n\n[Warning: AI finished unexpectedly: {candidate.finish_reason}]"
             else:
                 return f"⚠️ The AI finished unexpectedly ({candidate.finish_reason}) and returned no text. Please try again."

    except Exception as e:
        print(f"Error communicating with Gemini: {e}")
        # You might want to check specific error types (e.g., google.api_core.exceptions.PermissionDenied)
        return f"Error communicating with the AI: {e}"


def get_gemini_response_with_img(model,image_model, prompt):
    text_response = get_gemini_response(model, prompt)
    response: ImageGenerationResponse = image_model.generate_images(
        prompt=text_response,
        number_of_images=1,  # Generate one image
        # Add other parameters as needed (e.g., aspect_ratio, seed)
    )
    if response.images:
        # Option 1: Save the image to a file
        image_bytes = response.images[0]._image_bytes # Access raw bytes
        with open("templates/generated_image.png", "wb") as f:
             f.write(image_bytes)
        print("Image saved to generated_image.png")
    else:
        print("Image generation failed or returned no images.")
        # Check response object for error details if available
    return text_response

# --- Game State Parsing (Keep your original parsing logic) ---
def update_game_state_from_response(response_text, current_location, player_inventory):
    """
    Parses Gemini's response to potentially update location and inventory.
    Returns the updated (location, inventory, update_messages).
    This is basic keyword spotting and prone to errors.
    """
    new_location = current_location
    new_inventory = list(player_inventory) # Create a copy to modify
    update_messages = [] # Collect messages for the UI

    response_lower = response_text.lower()

    # Example: Detect taking items
    take_keywords = ["you pick up", "you take the", "add ", "acquire the", "you now have"]
    for keyword in take_keywords:
        if keyword in response_lower:
            try:
                # Extract text after keyword, before sentence end/comma
                potential_item_phrase = response_lower.split(keyword, 1)[1]
                potential_item = potential_item_phrase.split('.')[0].split(',')[0].strip()
                # Basic filtering
                if potential_item and len(potential_item) > 2 and potential_item not in ["it", "the", "a", "an", "nothing", "something", "them"]:
                    if potential_item not in new_inventory:
                        new_inventory.append(potential_item)
                        update_messages.append(f"Inventory updated: Added '{potential_item}'")
                        # Don't break, maybe multiple items were taken? Let's allow it.
            except IndexError:
                pass

    # Example: Detect location changes
    move_keywords = ["you travel to", "you enter the", "you arrive at", "you are now in", "step into the", "you reach"]
    for keyword in move_keywords:
        if keyword in response_lower:
            try:
                potential_location_phrase = response_lower.split(keyword, 1)[1]
                potential_location = potential_location_phrase.split('.')[0].split(' where')[0].split(' which')[0].strip()
                if potential_location and len(potential_location) > 3 and potential_location != new_location:
                    new_location = potential_location # Update location
                    update_messages.append(f"Location updated: Now at '{new_location}'")
                    break # Only one location change per response usually makes sense
            except IndexError:
                pass

    # Example: Detect dropping/using items (Remove from inventory)
    remove_keywords = ["you drop the", "you use the", "item disappears", "breaks the", "no longer have the", "you discard the", "is consumed"]
    items_to_remove = []
    for keyword in remove_keywords:
        if keyword in response_lower:
            relevant_part = response_lower.split(keyword, 1)[1][:60] # Check next ~60 chars
            for item in new_inventory: # Check against current *new* inventory
                if item in relevant_part:
                    if item not in items_to_remove:
                        items_to_remove.append(item)

    if items_to_remove:
        for item in items_to_remove:
            if item in new_inventory:
                new_inventory.remove(item)
                update_messages.append(f"Inventory updated: Removed '{item}'")

    return new_location, new_inventory, update_messages


# --- Save/Load Functions (Modified for Web Context) ---
def save_game_state(filename, game_state):
    """Saves the current game state to a JSON file in the SAVES_DIR."""
    if not filename:
        flash("Save filename cannot be empty.", "error")
        return False
    # Basic sanitization: allow alphanumeric, underscores, hyphens
    safe_filename = "".join(c for c in filename if c.isalnum() or c in ('_', '-'))
    if not safe_filename:
         flash("Invalid characters in filename.", "error")
         return False
    safe_filename += ".json" # Add extension

    filepath = os.path.join(SAVES_DIR, safe_filename)
    try:
        with open(filepath, 'w') as f:
            json.dump(game_state, f, indent=4)
        flash(f"💾 Game saved successfully to {safe_filename}!", "success")
        return True
    except IOError as e:
        print(f"Error saving game to {filepath}: {e}")
        flash(f"Error saving game: {e}", "error")
        return False
    except Exception as e:
        print(f"An unexpected error occurred while saving: {e}")
        flash(f"An unexpected error occurred while saving: {e}", "error")
        return False


def load_game_state(filename):
    """Loads the game state from a JSON file in the SAVES_DIR."""
    if not filename:
        flash("Load filename cannot be empty.", "error")
        return None
    # Basic sanitization (same as save)
    safe_filename = "".join(c for c in filename if c.isalnum() or c in ('_', '-'))
    if not safe_filename:
         flash("Invalid characters in filename.", "error")
         return None
    # Assume it might have .json or not, add if missing
    if not safe_filename.lower().endswith('.json'):
        safe_filename += ".json"

    filepath = os.path.join(SAVES_DIR, safe_filename)

    if not os.path.exists(filepath):
        flash(f"Save file '{safe_filename}' not found in saves directory.", "error")
        return None

    try:
        with open(filepath, 'r') as f:
            game_state = json.load(f)
        # Validate essential keys
        if not all(k in game_state for k in ["current_location", "player_inventory", "last_scene", "chat_history"]):
             flash(f"Save file '{safe_filename}' is corrupted or has missing data. Cannot load.", "error")
             return None

        flash(f"📂 Game loaded successfully from {safe_filename}!", "success")
        return game_state
    except (IOError, json.JSONDecodeError) as e:
        print(f"Error loading game from {filepath}: {e}")
        flash(f"Error loading game from {safe_filename}: {e}", "error")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while loading: {e}")
        flash(f"An unexpected error occurred while loading: {e}", "error")
        return None

# --- Flask Routes ---

@app.route('/', methods=['GET'])
def index():
    """Displays the main game page."""
    # Pass game state from session to template
    game_active = 'last_scene' in session
    return render_template('index.html',
                           game_active=game_active,
                           scene=session.get('last_scene'),
                           location=session.get('current_location'),
                           inventory=session.get('player_inventory', []),
                           chat_history=session.get('chat_history', []),
                           update_messages=session.get('update_messages', [])
                          )

@app.route('/new_game', methods=['POST'])
def new_game():
    """Starts a new game."""
    initial_prompt = request.form.get('initial_prompt', '').strip()
    if not initial_prompt:
        flash("Please describe the start of your adventure.", "warning")
        return redirect(url_for('index'))

    model,image_model = configure_gemini()
    if not model:
         flash("Could not configure the AI model. Please check server logs and credentials.", "error")
         return redirect(url_for('index'))

    print("\nStarting a new adventure!")
    session.clear() # Clear any old game state
    session['current_location'] = "an unknown starting location"
    session['player_inventory'] = []
    session['chat_history'] = []
    session['update_messages'] = []

    # Add initial player prompt to chat history
    session['chat_history'].append({"role": "user", "content": initial_prompt})

    # Construct initial prompt for Gemini
    full_initial_prompt = f"""You are a text adventure game master. Start a fantasy text adventure based on the player's first action or thought.
Player's initial thought/action: '{initial_prompt}'
Describe the initial location vividly, setting the scene and hinting at possible first actions. The player starts with no items in their inventory. End your description by asking 'What do you do?'."""

    response_text = get_gemini_response_with_img(model,image_model, full_initial_prompt)

    if response_text and not response_text.startswith("Error:") and not response_text.startswith("⚠️"):
        session['last_scene'] = response_text
        # Add Gemini's response to chat history
        session['chat_history'].append({"role": "model", "content": session['last_scene']})
        # Try to parse initial location from response (optional refinement)
        new_loc, _, msgs = update_game_state_from_response(response_text, session['current_location'], session['player_inventory'])
        session['current_location'] = new_loc
        session['update_messages'].extend(msgs)
        flash("New game started!", "success")
    else:
        session.clear() # Failed to start, clear session
        flash(f"Failed to get initial scene from AI: {response_text}", "error")

    session.modified = True # Explicitly mark session as modified
    return redirect(url_for('index'))


@app.route('/action', methods=['POST'])
def handle_action():
    """Handles player actions."""
    player_action = request.form.get('action', '').strip()

    if not player_action:
        flash("You need to type an action!", "warning")
        return redirect(url_for('index'))

    # Ensure game is active
    if 'last_scene' not in session:
        flash("No active game. Please start a new game or load one.", "error")
        return redirect(url_for('index'))

    # --- Get model (should ideally be stored/reused) ---
    model,image_model = configure_gemini() # Reconfiguring per action is inefficient but handles credential expiry
    if not model:
         flash("Could not configure the AI model. Please check server logs and credentials.", "error")
         return redirect(url_for('index'))

    # --- Add player action to history ---
    session['chat_history'].append({"role": "user", "content": player_action})

    # --- Construct the prompt for Gemini ---
    # Retrieve current state from session
    current_location = session.get('current_location', 'an unknown place')
    player_inventory = session.get('player_inventory', [])
    last_scene = session.get('last_scene', 'The scene is unclear.')
    chat_history = session.get('chat_history', [])

    # Build prompt with history context
    prompt = f"""You are a text adventure game master continuing a fantasy story.
Current Location: '{current_location}'
Player Inventory: {', '.join(player_inventory) or 'nothing'}

Recent History (last 5 turns):"""
    history_context = ""
    for entry in chat_history[-6:]: # Include current action prompt + 5 previous exchanges
        role = "Player" if entry['role'] == 'user' else "Narrator"
        history_context += f"\n{role}: {entry['content']}"

    prompt += history_context

    prompt += f"""

Narrator Task: Describe the outcome of the Player's last action ('{player_action}').
- Be creative, descriptive, and consistent with the fantasy RPG tone.
- Describe changes to the scene, story, location, or inventory naturally within the narrative.
- If the player's action is unclear or impossible, describe why.
- End your response by prompting the player for their next action (e.g., "What do you do now?", "The passage continues into darkness...")."""

    # --- Get Gemini Response ---
    response_text = get_gemini_response_with_img(model,image_model, prompt)
    session['update_messages'] = [] # Clear previous update messages

    if response_text and not response_text.startswith("Error:") and not response_text.startswith("⚠️"):
        # --- Update Game State ---
        session['last_scene'] = response_text
        session['chat_history'].append({"role": "model", "content": response_text})

        # Parse response for state changes
        new_loc, new_inv, update_msgs = update_game_state_from_response(
            response_text,
            current_location,
            player_inventory
        )
        session['current_location'] = new_loc
        session['player_inventory'] = new_inv
        session['update_messages'] = update_msgs # Store messages for display
    else:
        # Handle error - Don't update last_scene, maybe pop last user action?
        # Popping last user action might be confusing, let's just show error.
        flash(f"AI Response Error: {response_text}", "error")
        # Keep the game state as it was before the failed action

    session.modified = True
    return redirect(url_for('index'))

@app.route('/save', methods=['POST'])
def save_game():
    """Saves the current game state."""
    filename = request.form.get('save_filename', '').strip()

    if 'last_scene' not in session:
        flash("No active game to save.", "warning")
        return redirect(url_for('index'))

    # Create the game state dictionary from session data
    current_game_state = {
        "current_location": session.get('current_location'),
        "player_inventory": session.get('player_inventory'),
        "last_scene": session.get('last_scene'),
        "chat_history": session.get('chat_history')
        # Add any other state variables you might have
    }

    save_game_state(filename, current_game_state) # This function now handles flashing messages

    return redirect(url_for('index'))


@app.route('/load', methods=['POST'])
def load_game():
    """Loads a game state."""
    filename = request.form.get('load_filename', '').strip()

    loaded_state = load_game_state(filename) # This function now handles flashing messages

    if loaded_state:
        # Clear existing session and populate with loaded data
        session.clear()
        session['current_location'] = loaded_state['current_location']
        session['player_inventory'] = loaded_state['player_inventory']
        session['last_scene'] = loaded_state['last_scene']
        session['chat_history'] = loaded_state['chat_history']
        session['update_messages'] = ["Game loaded."] # Add a message indicating load success
        session.modified = True
    # If load failed, load_game_state flashes the error, just redirect

    return redirect(url_for('index'))

@app.route('/quit', methods=['POST'])
def quit_game():
    """Clears the session, effectively ending the game."""
    session.clear()
    flash("Game ended. Start a new game or load a previous one.", "info")
    return redirect(url_for('index'))

# --- Run the App ---
if __name__ == "__main__":
    print("Starting Flask app...")
    print(f"Save files will be stored in: {os.path.abspath(SAVES_DIR)}")
    print("To run:")
    print("1. Make sure you are authenticated with Google Cloud: 'gcloud auth application-default login'")
    print("2. Run the script: python app.py")
    print("3. Open your browser to http://127.0.0.1:5000 (or the address provided)")
    # Consider using environment variables for host/port/debug
    app.run(debug=True) # debug=True is helpful for development, disable for production