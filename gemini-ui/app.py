# app.py
import os
from flask import Flask, render_template, request, session, redirect, url_for, flash
import markdown

# Import the game logic functions
from rpg_logic import (
    SAVE_FILENAME,
    save_game,
    load_game,
    start_new_game,
    process_player_action
)

app = Flask(__name__)
app.secret_key = os.environ.get(
    "FLASK_SECRET_KEY", "a_very_default_and_insecure_secret_key")

# --- Model Configuration ---
# ***** UPDATE THIS SECTION *****
AVAILABLE_MODELS = {
    # Using tuples: (display_name, model_id)
    "Flash (Fastest)": "gemini-2.0-flash-lite",  # Updated
    "Balanced": 'gemini-2.0-flash',             # Updated
    "Pro": "gemini-2.5-pro-preview-03-25",      # Updated
}
# Update the default model ID to one from the new list
# Changed default to the new 'Balanced' model
DEFAULT_MODEL_ID = 'gemini-2.0-flash'
# ***** END OF UPDATE SECTION *****

# Fallback logic in case default is somehow not in the list
if DEFAULT_MODEL_ID not in AVAILABLE_MODELS.values():
    try:
        DEFAULT_MODEL_ID = list(AVAILABLE_MODELS.values())[
            0]  # Use the first available as fallback
        print(
            f"Warning: Default model ID '{DEFAULT_MODEL_ID}' was not found in AVAILABLE_MODELS. Falling back to first option.")
    except IndexError:
        # This should not happen if AVAILABLE_MODELS is defined, but handle defensively
        print("FATAL ERROR: AVAILABLE_MODELS is empty. Cannot set a default model.")
        # In a real app, you might exit or raise a more specific configuration error here
        DEFAULT_MODEL_ID = None  # Or handle error more gracefully


# --- Default Initial Prompt ---
DEFAULT_INITIAL_PROMPT = "I wake up in a dimly lit clearing."


# --- Flask Routes ---
# ... (rest of the app.py code remains exactly the same) ...

@app.route('/', methods=['GET'])
def index():
    """Displays the main game page."""
    # Use the potentially updated DEFAULT_MODEL_ID
    current_model_id = session.get('model_name', DEFAULT_MODEL_ID)

    # If default model somehow became None, handle it
    if not current_model_id and DEFAULT_MODEL_ID:
        current_model_id = DEFAULT_MODEL_ID
    elif not current_model_id and not DEFAULT_MODEL_ID:
        # Critical error state - cannot proceed without a model
        flash("FATAL ERROR: No valid default AI model configured. Application cannot run.", "error")
        # Render a minimal error page or return an error response
        return "Internal Server Error: AI Model Configuration Missing", 500

    if current_model_id not in AVAILABLE_MODELS.values():
        flash(
            f"Model '{current_model_id}' no longer available, resetting to default.", "warning")
        current_model_id = DEFAULT_MODEL_ID
        session['model_name'] = current_model_id
        session.pop('game_state', None)

    if 'game_state' not in session:
        loaded_state, message = load_game(SAVE_FILENAME)
        if loaded_state:
            loaded_model = loaded_state.get("model_name")
            # Check against the *new* AVAILABLE_MODELS
            if loaded_model and loaded_model in AVAILABLE_MODELS.values():
                session['game_state'] = loaded_state
                session['model_name'] = loaded_model
                current_model_id = loaded_model
                flash(f"{message} (Model: {loaded_model})", 'info')
            elif loaded_model:
                flash(
                    f"{message}, but saved model '{loaded_model}' is not available. Starting new game with default.", 'warning')
                session['model_name'] = DEFAULT_MODEL_ID
                session['game_state'] = start_new_game(
                    DEFAULT_MODEL_ID, DEFAULT_INITIAL_PROMPT)
            else:
                flash(
                    f"{message}, but save file is old format. Starting new game with default model.", "warning")
                session['model_name'] = DEFAULT_MODEL_ID
                session['game_state'] = start_new_game(
                    DEFAULT_MODEL_ID, DEFAULT_INITIAL_PROMPT)
        else:
            flash(
                "Choose your model and starting scenario below, then click 'Start New Game'.", 'info')

    game_state = session.get('game_state')
    game_running = (game_state is not None)
    scene_html = "<p>Start a new game below.</p>"
    processed_history = []
    inventory = []
    location = "Not started"
    active_model_id = current_model_id

    if game_running:
        scene_text = game_state.get('last_scene', "Error: Scene not found.")
        scene_html = markdown.markdown(scene_text, extensions=[])
        chat_history = game_state.get('chat_history', [])
        for entry in chat_history:
            processed_entry = entry.copy()
            if entry['role'] == 'model':
                processed_entry['content'] = markdown.markdown(
                    entry['content'], extensions=[])
            processed_history.append(processed_entry)
        inventory = game_state.get('player_inventory', [])
        location = game_state.get('current_location', 'Unknown')
        active_model_id = game_state.get('model_name', current_model_id)

    return render_template('index.html',
                           scene=scene_html,
                           history=processed_history,
                           inventory=inventory,
                           location=location,
                           available_models=AVAILABLE_MODELS,  # Pass the updated list
                           current_model_id=active_model_id,
                           game_running=game_running,
                           DEFAULT_INITIAL_PROMPT=DEFAULT_INITIAL_PROMPT
                           )


# --- Other Routes ---
# (handle_action, save_current_game, start_new_game_post, restart_game)
# No changes needed in these routes as they read AVAILABLE_MODELS dynamically

@app.route('/action', methods=['POST'])
def handle_action():
    # ... (no changes needed) ...
    player_action = request.form.get('action_input', '').strip()
    if not player_action:
        flash("Please enter an action.", 'warning')
        return redirect(url_for('index'))
    if 'game_state' not in session or 'model_name' not in session:
        flash("No active game found. Please start a new game first.", 'error')
        return redirect(url_for('index'))
    current_game_state = session['game_state']
    current_model_id = session['model_name']
    if current_model_id not in AVAILABLE_MODELS.values():
        flash(
            f"Selected model '{current_model_id}' is not available. Please choose a model and start a new game.", 'error')
        session.pop('game_state', None)
        return redirect(url_for('index'))
    updated_game_state, parse_feedback = process_player_action(
        player_action, current_game_state, current_model_id
    )
    session['game_state'] = updated_game_state
    for msg in parse_feedback:
        flash(msg, 'info')
    session.modified = True
    return redirect(url_for('index'))


@app.route('/save', methods=['POST'])
def save_current_game():
    # ... (no changes needed) ...
    if 'game_state' in session:
        if 'model_name' not in session['game_state']:
            session['game_state']['model_name'] = session.get(
                'model_name', DEFAULT_MODEL_ID)
        success, message = save_game(SAVE_FILENAME, session['game_state'])
        if success:
            flash(message, 'success')
        else:
            flash(message, 'error')
    else:
        flash("No game running to save.", 'warning')
    return redirect(url_for('index'))


@app.route('/start_new', methods=['POST'])
def start_new_game_post():
    # ... (no changes needed) ...
    selected_model_id = request.form.get('model_select')
    initial_prompt = request.form.get('initial_scene_prompt', '').strip()
    if not initial_prompt:
        initial_prompt = DEFAULT_INITIAL_PROMPT
        flash(
            f"No initial prompt provided, using default: '{DEFAULT_INITIAL_PROMPT}'", "info")
    # Check against the *new* AVAILABLE_MODELS
    if not selected_model_id or selected_model_id not in AVAILABLE_MODELS.values():
        flash("Invalid model selected. Using default.", "warning")
        selected_model_id = DEFAULT_MODEL_ID
    session.pop('game_state', None)
    session['model_name'] = selected_model_id
    session['game_state'] = start_new_game(selected_model_id, initial_prompt)
    flash(
        f"Started a new adventure with model: {selected_model_id}!", 'success')
    session.modified = True
    return redirect(url_for('index'))


@app.route('/restart', methods=['POST'])
def restart_game():
    # ... (no changes needed) ...
    session.pop('game_state', None)
    flash("Restarting game...", 'info')
    session.modified = True
    return redirect(url_for('index'))


# --- Run the App ---
if __name__ == '__main__':
    # Ensure there's a default model before trying to run
    if not DEFAULT_MODEL_ID:
        print("FATAL ERROR: Cannot run application without a valid DEFAULT_MODEL_ID.")
    else:
        app.run(debug=True, host='0.0.0.0', port=5000)
