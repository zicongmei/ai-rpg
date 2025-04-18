# app.py

from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import os
import secrets

# Import functions and constants from your game logic file
import game_logic

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)  # Or use environment variable

# --- Define Available Models ---
AVAILABLE_MODELS = game_logic.MODEL_COSTS.keys()
DEFAULT_MODEL = game_logic.DEFAULT_MODEL_NAME
MODEL_COSTS = game_logic.MODEL_COSTS  # Get costs from game_logic


@app.route('/', methods=['GET', 'POST'])
def index():
    """Main game route."""
    game_state = session.get('game_state')  # Use .get for safer access
    feedback = session.pop('feedback', [])
    system_feedback = []
    selected_model = session.get('selected_model', DEFAULT_MODEL)
    last_request_cost = 0.0
    estimated_total_cost = 0.0

    if request.method == 'POST':
        if not game_state:
            flash("No active game. Start a new one or load.", "warning")
            return redirect(url_for('index'))

        action = request.form.get('action')
        selected_model = request.form.get(
            'selected_model', selected_model)  # Update if submitted
        session['selected_model'] = selected_model

        if action:
            # process_player_action now returns the game_state which includes cost info
            updated_state, _, system_feedback, _, _ = \
                game_logic.process_player_action(
                    action, game_state, selected_model)

            session['game_state'] = updated_state
            game_state = updated_state  # Update for current render

    all_feedback = feedback + system_feedback

    # Calculate costs for display after potential updates
    if game_state:
        last_request_cost = game_state.get('last_request_cost', 0.0)
        # Estimate total cost based on accumulated tokens and currently selected model
        estimated_total_cost = game_logic.calculate_cost(
            selected_model,
            game_state.get('total_input_tokens', 0),
            game_state.get('total_output_tokens', 0)
        )

    return render_template('index.html',
                           game_state=game_state,
                           feedback=all_feedback,
                           available_models=AVAILABLE_MODELS,
                           selected_model=selected_model,
                           last_request_cost=last_request_cost,
                           estimated_total_cost=estimated_total_cost,
                           model_costs=MODEL_COSTS)  # Pass costs to template


@app.route('/new_game', methods=['POST'])
def new_game():
    """Starts a new game."""
    initial_prompt = request.form.get('initial_prompt')
    selected_model = request.form.get('selected_model', DEFAULT_MODEL)

    if not initial_prompt:
        flash("Please provide an initial thought or action.", "warning")
        # Pass model costs even on validation error
        return render_template('index.html', game_state=None, feedback=None,
                               available_models=AVAILABLE_MODELS, selected_model=selected_model,
                               model_costs=MODEL_COSTS)

    # start_new_game returns the initial state including costs
    new_game_state, feedback_msg, _, _ = game_logic.start_new_game(
        initial_prompt, selected_model)

    if new_game_state:
        session.clear()
        session['game_state'] = new_game_state
        session['selected_model'] = selected_model
        session['feedback'] = [feedback_msg]
        flash(f"New adventure started with {selected_model}!", "info")
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

    # game_state already includes tokens and last request cost
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
        session['game_state'] = loaded_game_state
        # Reset to default model on load, user can change it
        session['selected_model'] = DEFAULT_MODEL
        flash(message, "success")
    else:
        flash(message, "danger")

    return redirect(url_for('index'))


@app.route('/generate_image', methods=['GET'])
def generate_image_route():
    """Generates an image based on the last scene using Imagen."""
    # Note: Image generation cost is separate and not currently tracked here.
    # This route primarily uses Imagen, but calls get_gemini_response for prompt generation.
    # We could potentially track the token usage of that call if needed.
    if 'game_state' not in session or not session['game_state'].get('last_scene'):
        return jsonify({"error": "No scene available to generate image from."}), 400

    last_scene = session['game_state']['last_scene']

    # Pass a default model for the prompt generation part
    # TODO: Consider making this selectable or configurable if cost is a concern
    selected_model_for_prompt = session.get('selected_model', DEFAULT_MODEL)
    b64_image_data, error_message, prompt_input_tokens, prompt_output_tokens = \
        game_logic.generate_image_with_imagen(
            last_scene, selected_model_for_prompt)

    # Optional: Update game state with image prompt token usage if needed
    # This would require updating game_state in session, which GET requests usually don't do.
    # For simplicity, we'll ignore these tokens in the main cost calculation for now.
    # if 'game_state' in session and prompt_input_tokens is not None:
    #     session['game_state']['total_input_tokens'] = session['game_state'].get('total_input_tokens', 0) + prompt_input_tokens
    #     session['game_state']['total_output_tokens'] = session['game_state'].get('total_output_tokens', 0) + prompt_output_tokens
    #     session.modified = True # Important if modifying session directly

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
    app.run(debug=True)
