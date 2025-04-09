# app.py

from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import os
import secrets

# Import functions from your game logic file
import game_logic

app = Flask(__name__)
app.secret_key = secrets.token_hex(16) # Or use environment variable


@app.route('/', methods=['GET', 'POST'])
def index():
    """Main game route."""
    if 'game_state' not in session:
        return render_template('index.html', game_state=None, feedback=None)

    game_state = session['game_state']
    feedback = session.pop('feedback', []) # Get feedback flashed from redirects
    system_feedback = []

    if request.method == 'POST':
        action = request.form.get('action')
        if action:
            updated_state, response_text, system_feedback = game_logic.process_player_action(action, game_state)
            session['game_state'] = updated_state
            game_state = updated_state # Update for current render

    all_feedback = feedback + system_feedback

    return render_template('index.html',
                           game_state=game_state,
                           feedback=all_feedback)


@app.route('/new_game', methods=['POST'])
def new_game():
    """Starts a new game."""
    initial_prompt = request.form.get('initial_prompt')
    if not initial_prompt:
        flash("Please provide an initial thought or action.", "warning")
        return redirect(url_for('index'))

    game_state, feedback_msg = game_logic.start_new_game(initial_prompt)

    if game_state:
        session.clear()
        session['game_state'] = game_state
        session['feedback'] = [feedback_msg] # Use list for feedback consistency
    else:
        flash(feedback_msg, "danger")

    return redirect(url_for('index'))


@app.route('/save_game', methods=['POST'])
def save_game_route():
    """Saves the current game state."""
    filename = request.form.get('filename')
    if not filename:
        flash("Please provide a filename.", "warning")
        return redirect(url_for('index'))
    if 'game_state' not in session:
        flash("No active game to save.", "warning")
        return redirect(url_for('index'))

    success, message = game_logic.save_game(filename, session['game_state'])
    flash(message, "success" if success else "danger")
    return redirect(url_for('index'))


@app.route('/load_game', methods=['POST'])
def load_game_route():
    """Loads a game state."""
    filename = request.form.get('filename')
    if not filename:
        flash("Please provide a filename.", "warning")
        return redirect(url_for('index'))

    game_state, message = game_logic.load_game(filename)

    if game_state:
        session.clear()
        session['game_state'] = game_state
        flash(message, "success")
    else:
        flash(message, "danger")

    return redirect(url_for('index'))

# --- UPDATED ROUTE for Image Generation ---
@app.route('/generate_image', methods=['GET']) # Changed route name
def generate_image_route():
    """Generates an image based on the last scene using Imagen."""
    if 'game_state' not in session or not session['game_state'].get('last_scene'):
        return jsonify({"error": "No scene available to generate image from."}), 400

    last_scene = session['game_state']['last_scene']

    # Call the new Imagen generation function
    b64_image_data, error_message = game_logic.generate_image_with_imagen(last_scene)

    if error_message:
         # Return error as JSON
         return jsonify({"error": error_message}), 500 # Internal server error or specific code
    elif b64_image_data:
         # Return base64 image data as JSON
         return jsonify({"image_data": b64_image_data})
    else:
         # Should not happen if error handling is correct, but just in case
         return jsonify({"error": "Unknown error during image generation."}), 500


@app.route('/quit_game', methods=['POST'])
def quit_game():
    """Clears the session."""
    session.clear()
    flash("Game ended. Start a new adventure or load a previous one!", "info")
    return redirect(url_for('index'))


if __name__ == '__main__':
    # Set environment variables if needed before running:
    # export GOOGLE_CLOUD_PROJECT='your-gcp-project-id'
    # export GOOGLE_CLOUD_LOCATION='us-central1'
    app.run(debug=True) # Keep debug=True for development ONLY