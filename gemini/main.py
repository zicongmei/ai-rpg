import os
import google.generativeai as genai
import google.auth


# 1. Automatically find your default gcloud credentials
# scopes = ['https://www.googleapis.com/auth/generative-language']
# credentials, project = google.auth.default(scopes=scopes)


credentials, project = google.auth.default()

# 2. Configure the Gemini API with these credentials
genai.configure(api_key=credentials.token)
model = genai.GenerativeModel('gemini-2.5-pro-preview-03-25')

def get_gemini_response(prompt):
    """Sends a prompt to Gemini and returns the text response."""
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error communicating with Gemini: {e}")
        return None

def display_scene(description):
    """Prints the current scene description to the console."""
    print("\n" + "=" * 40)
    print(description)
    print("=" * 40)

def get_player_action():
    """Gets the player's action from the console."""
    return input("> ").strip()

def main():
    """Runs the text-based RPG game with Gemini."""
    print("Welcome to the Gemini-Powered Text RPG!")

    # Initialize the game state
    current_location = "unknown location"
    player_inventory = []

    # Initial scene description
    print("Enter initial prompt")
    initial_prompt = get_player_action()

    scene_description = get_gemini_response(initial_prompt)
    if scene_description:
        display_scene(scene_description)
    else:
        print("Failed to get the initial scene description. Exiting.")
        return

    while True:
        player_action = get_player_action()
        if player_action.lower() == "exit":
            print("Thanks for playing!")
            break

        # Construct the prompt for Gemini based on the current scene and player action
        prompt = f"The player is currently at {current_location} and has the following items: {', '.join(player_inventory) or 'nothing'}. " \
                 f"Their action is: '{player_action}'. Describe the outcome of this action and the new scene, including potential next steps for the player."

        response = get_gemini_response(prompt)
        if response:
            display_scene(response)

            # You'll need to implement logic here to update the game state
            # based on Gemini's response. This could involve:
            # - Changing the current_location
            # - Adding or removing items from the player_inventory
            # - Introducing new characters or challenges

            # Example (very basic) state update based on keywords in the response:
            if "you pick up" in response.lower():
                item = response.lower().split("you pick up")[1].split(".")[0].strip()
                if item and item not in player_inventory:
                    player_inventory.append(item)
                    print(f"\nYou added '{item}' to your inventory.")
            elif "you travel to" in response.lower():
                new_location = response.lower().split("you travel to")[1].split(".")[0].strip()
                if new_location:
                    current_location = new_location
                    print(f"\nYou are now at {current_location}.")

        else:
            print("The AI is unresponsive. Try again.")

if __name__ == "__main__":
    main()