# app.py

from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import os
import secrets

# Import functions from your game logic file
import game_logic

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)  # Or use environment variable

# --- Define Available Models ---
# Ensure these match the models supported by your API key/setup and game_logic.ALLOWED_MODELS
AVAILABLE_MODELS = game_logic.ALLOWED_MODELS
DEFAULT_MODEL = game_logic.DEFAULT_MODEL_NAME


@app.route('/', methods=['GET', 'POST'])
def index():
    """Main game route."""
    game_state = session.get('game_state')  # Use .get for safer access
    # Get feedback flashed from redirects
    feedback = session.pop('feedback', [])
    system_feedback = []
    # Get saved model or default
    selected_model = session.get('selected_model', DEFAULT_MODEL)

    if request.method == 'POST':
        if not game_state:
            # Handle case where POST happens but no game state exists (e.g., browser refresh issue)
            flash("No active game. Start a new one or load.", "warning")
            return redirect(url_for('index'))

        action = request.form.get('action')
        # --- Get Selected Model from Form ---
        selected_model = request.form.get(
            'selected_model', selected_model)  # Update if submitted
        # Save choice for next GET request
        session['selected_model'] = selected_model

        if action:
            # --- Pass selected_model to game_logic ---
            updated_state, response_text, system_feedback, input_tokens, output_tokens = \
                game_logic.process_player_action(
                    action, game_state, selected_model)

            # Update game_state in session directly
            session['game_state'] = updated_state
            game_state = updated_state  # Update for current render
            # No need to manually update totals here, game_logic does it now

    all_feedback = feedback + system_feedback

    return render_template('index.html',
                           game_state=game_state,
                           feedback=all_feedback,
                           available_models=AVAILABLE_MODELS,  # Pass models to template
                           selected_model=selected_model)    # Pass current selection


@app.route('/new_game', methods=['POST'])
def new_game():
    """Starts a new game."""
    initial_prompt = request.form.get('initial_prompt')
    # --- Get Selected Model from Form ---
    selected_model = request.form.get('selected_model', DEFAULT_MODEL)

    if not initial_prompt:
        flash("Please provide an initial thought or action.", "warning")
        # Redirect back, but pass models again for the form
        return render_template('index.html', game_state=None, feedback=None,
                               available_models=AVAILABLE_MODELS, selected_model=selected_model)

    # --- Pass selected_model to game_logic ---
    # Expect game_state, msg, input_tokens, output_tokens
    new_game_state, feedback_msg, _, _ = game_logic.start_new_game(
        initial_prompt, selected_model)

    if new_game_state:
        session.clear()  # Clear old session data completely
        session['game_state'] = new_game_state  # Contains initial token counts
        session['selected_model'] = selected_model  # Store chosen model
        session['feedback'] = [feedback_msg]
        # Optional: confirm model
        flash(f"New adventure started with {selected_model}!", "info")
    else:
        flash(feedback_msg, "danger")  # Show error from game_logic

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

    # game_state in session already includes total token counts
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

    loaded_game_state, message = game_logic.load_game(filename)

    if loaded_game_state:
        session.clear()
        # game_logic.load_game now adds/resets token counts appropriately
        session['game_state'] = loaded_game_state
        # Restore the model selection - maybe save/load this too? For now, reset to default.
        session['selected_model'] = DEFAULT_MODEL
        flash(message, "success")
    else:
        flash(message, "danger")

    return redirect(url_for('index'))


@app.route('/generate_image', methods=['GET'])
def generate_image_route():
    """Generates an image based on the last scene using Imagen."""
    if 'game_state' not in session or not session['game_state'].get('last_scene'):
        return jsonify({"error": "No scene available to generate image from."}), 400

    last_scene = session['game_state']['last_scene']

    b64_image_data, error_message = game_logic.generate_image_with_imagen(
        last_scene)

    if error_message:
        return jsonify({"error": error_message}), 500
    elif b64_image_data:
        return jsonify({"image_data": b64_image_data})
    else:
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
    # export GEMINI_API_KEY='your-api-key' # If needed
    app.run(debug=True)
