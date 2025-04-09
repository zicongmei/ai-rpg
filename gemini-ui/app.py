# app.py
import os
from flask import Flask, render_template, request, session, redirect, url_for, flash

# Import the game logic functions
from rpg_logic import (
    SAVE_FILENAME,
    save_game,
    load_game,
    start_new_game,
    process_player_action
)

app = Flask(__name__)
# Secret key is needed for session management. Change this to a random string!
# You can generate one using: python -c 'import secrets; print(secrets.token_hex())'
app.secret_key = os.environ.get(
    "FLASK_SECRET_KEY", "a_very_default_and_insecure_secret_key")

# --- Flask Routes ---


@app.route('/', methods=['GET'])
def index():
    """Displays the main game page."""
    # Check if game state exists in session
    if 'game_state' not in session:
        # Try loading from file first
        loaded_state, message = load_game(SAVE_FILENAME)
        if loaded_state:
            session['game_state'] = loaded_state
            flash(message, 'info')  # Show load message
        else:
            # Start a new game if no save or load failed
            session['game_state'] = start_new_game()
            flash("No save file found or load failed. Starting a new adventure!", 'info')

    # Get current state for rendering
    game_state = session.get('game_state', {})  # Use .get for safety
    scene = game_state.get('last_scene', "Error: Scene not found.")
    # Format chat history for display (optional, can be enhanced)
    history = game_state.get('chat_history', [])
    inventory = game_state.get('player_inventory', [])
    location = game_state.get('current_location', 'Unknown')

    return render_template('index.html', scene=scene, history=history, inventory=inventory, location=location)


@app.route('/action', methods=['POST'])
def handle_action():
    """Processes player actions submitted via the form."""
    player_action = request.form.get('action_input', '').strip()

    if not player_action:
        flash("Please enter an action.", 'warning')
        return redirect(url_for('index'))

    if 'game_state' not in session:
        flash("Game state lost. Please start a new game.", 'error')
        return redirect(url_for('new_game'))  # Redirect to new game

    current_game_state = session['game_state']

    # Process the action using the imported logic
    updated_game_state, parse_feedback = process_player_action(
        player_action, current_game_state)

    # Store the updated state back in the session
    session['game_state'] = updated_game_state

    # Flash any parsing feedback (e.g., inventory changes)
    for msg in parse_feedback:
        flash(msg, 'info')

    # Ensure the session is saved
    session.modified = True

    # Redirect back to the main page to show results
    return redirect(url_for('index'))


@app.route('/save', methods=['POST'])  # Use POST for actions that change state
def save_current_game():
    """Saves the current game state to the file."""
    if 'game_state' in session:
        success, message = save_game(SAVE_FILENAME, session['game_state'])
        if success:
            flash(message, 'success')
        else:
            flash(message, 'error')
    else:
        flash("No game running to save.", 'warning')

    return redirect(url_for('index'))


# Use POST to initiate the new game action
@app.route('/new_game', methods=['POST'])
def start_new_game_route():
    """Clears the session and starts a completely new game."""
    # Optionally get a starting prompt from the user if desired,
    # but for now just uses the default in start_new_game()
    session.pop('game_state', None)  # Clear existing game state
    session['game_state'] = start_new_game()  # Start fresh
    flash("Started a new adventure!", 'success')
    session.modified = True
    return redirect(url_for('index'))


# --- Run the App ---
if __name__ == '__main__':
    # Use 0.0.0.0 to make it accessible on your network (use with caution)
    # Debug=True automatically reloads when code changes, but disable for production
    app.run(debug=True, host='0.0.0.0', port=5000)
