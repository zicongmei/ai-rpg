# --- Prerequisites ---
# 1. Install the library: pip install google-generativeai
# 2. Get an API Key: Create one from Google AI Studio (https://aistudio.google.com/app/apikey)

import tkinter as tk
from tkinter import ttk
import google.auth  # Still used for Vertex AI Image Gen (optional)
from tkinter import scrolledtext
import random
from PIL import Image, ImageTk
import os
import io  # To handle image bytes in memory
import time  # To potentially show delays
import re  # To clean up generated text

# --- Google Cloud Vertex AI Imports (for Image Generation) ---
# (Keep these if you still want Vertex AI Image Generation)
try:
    from google.cloud import aiplatform
    print("Vertex AI SDK base loaded (for Image Generation).")
except ImportError:
    print("WARNING: google-cloud-aiplatform library not found. Vertex AI Image Generation disabled.")
    aiplatform = None

try:
    if aiplatform:
        from vertexai.preview.vision_models import ImageGenerationModel
        print("Vertex AI Image Generation Model class loaded.")
    else:
        ImageGenerationModel = None
except ImportError:
    ImageGenerationModel = None
    print("WARNING: Failed to import ImageGenerationModel. Vertex AI Image Generation disabled.")

# --- NEW: Google Generative AI Import (for Text Generation) ---
try:
    import google.generativeai as genai
    print("Google Generative AI SDK loaded (for Text Generation).")
except ImportError:
    print("ERROR: google-generativeai library not found. Text Generation disabled.")
    print("Install it: pip install google-generativeai")
    genai = None  # Flag that the library is missing


# --- Character Class (No changes needed) ---
class Character:
    """Represents a character in the game (Player or Enemy)."""

    def __init__(self, name, hp, attack, gui_update_callback=None):
        self.name = name
        self.max_hp = hp
        self.hp = hp
        self.attack_power = attack
        self.is_defending = False
        self.gui_update_callback = gui_update_callback  # Callback to update GUI elements

    def is_alive(self):
        """Checks if the character has positive HP."""
        return self.hp > 0

    def attack(self, target, log_callback):
        """Performs an attack on the target character."""
        if not self.is_alive():
            log_callback(f"{self.name} cannot attack, they are defeated!")
            return 0

        damage = random.randint(
            int(self.attack_power * 0.8), int(self.attack_power * 1.2))
        log_callback(f"\n{self.name} attacks {target.name}!")

        reduced_damage = False
        if target.is_defending:
            damage = max(0, damage // 2)
            reduced_damage = True
            target.is_defending = False

        actual_damage = target.take_damage(damage, log_callback)

        if reduced_damage:
            log_callback(f"{target.name} was defending! Damage reduced.")

        return actual_damage

    def take_damage(self, damage, log_callback):
        """Applies damage to the character."""
        actual_damage = min(self.hp, damage)
        self.hp -= actual_damage
        self.hp = max(0, self.hp)

        log_callback(
            f"{self.name} takes {actual_damage} damage. Remaining HP: {self.hp}/{self.max_hp}")

        if not self.is_alive():
            log_callback(f"{self.name} has been defeated!")

        if self.gui_update_callback:
            self.gui_update_callback()

        return actual_damage

    def defend(self, log_callback):
        """Sets the character to a defending state for the next attack."""
        if not self.is_alive():
            return
        log_callback(f"\n{self.name} takes a defensive stance!")
        self.is_defending = True

    def reset_defense(self):
        """Resets the defending state."""
        self.is_defending = False


# --- GUI Application Class ---
class RPG_GUI:
    # --- !!! USER CONFIGURATION REQUIRED !!! ---

    # == Vertex AI Image Generation Config (Optional) ==
    # Set PROJECT_ID only if using Vertex AI Image Generation
    # Otherwise, it can be left as "invalid" or removed.
    PROJECT_ID = "invalid"  # <--- REPLACE if using Vertex AI Image Gen
    LOCATION = "us-central1"  # <--- REPLACE if using Vertex AI Image Gen
    IMAGE_MODEL_NAME = "imagegeneration@006"

    # == NEW: Google Generative AI Text Generation Config ==
    # Get your API Key from Google AI Studio: https://aistudio.google.com/app/apikey
    GEMINI_API_KEY = "YOUR_API_KEY"  # <--- REPLACE with your actual Gemini API Key
    # Or other compatible model like gemini-1.5-flash
    TEXT_MODEL_NAME = "gemini-2.0-flash-lite"

    # --- End User Configuration ---

    def __init__(self, root):
        self.root = root
        root.title("Simple RPG Combat (AI Enhanced - google-generativeai)")
        root.geometry("600x650")

        # --- Initialize Vertex AI for Image Generation (Optional) ---
        self.vertex_ai_initialized = False  # Flag for Vertex AI specific setup
        self.image_generation_available = False
        self.image_model = None  # Initialize image model attribute

        credentials, project_id = google.auth.default()
        if project_id:
            self.PROJECT_ID = project_id
        if credentials:
            self.GEMINI_API_KEY = credentials.token

        # Attempt Vertex AI init only if library exists and Project ID is set
        if aiplatform and self.PROJECT_ID != "invalid":
            try:
                aiplatform.init(project=self.PROJECT_ID,
                                location=self.LOCATION, credentials=credentials)
                self.vertex_ai_initialized = True
                print(
                    f"Vertex AI initialized for project '{self.PROJECT_ID}' (Image Generation).")

                # Try initializing Image Generation Model
                if ImageGenerationModel:
                    try:
                        self.image_model = ImageGenerationModel.from_pretrained(
                            self.IMAGE_MODEL_NAME)
                        self.image_generation_available = True
                        print(
                            f"Vertex AI Image Generation Model '{self.IMAGE_MODEL_NAME}' loaded.")
                    except Exception as e:
                        print(
                            f"ERROR: Failed to load Vertex AI Image Generation Model: {e}")
                else:
                    print(
                        "Skipping Vertex AI Image Generation Model loading (class not available).")

            except google.auth.exceptions.DefaultCredentialsError:
                print("WARNING: Could not find default Google Cloud credentials for Vertex AI. Image Generation might fail if not using API keys elsewhere.")
            except Exception as e:
                print(f"ERROR: Failed to initialize Vertex AI: {e}")
                print(
                    "Check Project ID, Location, Auth, and API enablement if using Vertex Image Gen.")
        elif not aiplatform:
            print("Skipping Vertex AI initialization (library not found).")
        elif self.PROJECT_ID == "invalid":
            print(
                "Skipping Vertex AI initialization (PROJECT_ID not set for Image Generation).")

        # --- NEW: Initialize Google Generative AI for Text Generation ---
        self.text_model = None
        self.text_generation_available = False
        if genai and self.GEMINI_API_KEY != "YOUR_API_KEY":
            try:
                genai.configure(api_key=self.GEMINI_API_KEY)
                # Safety settings can be configured globally or per-request
                # Example global config:
                # safety_settings_global = [
                #     {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                #     {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                # ]
                # genai.configure(api_key=self.GEMINI_API_KEY, safety_settings=safety_settings_global)

                self.text_model = genai.GenerativeModel(self.TEXT_MODEL_NAME)
                # Optional: Perform a quick test generation to verify API key and model
                # try:
                #    self.text_model.generate_content("test", generation_config=genai.types.GenerationConfig(candidate_count=1, max_output_tokens=5))
                #    self.text_generation_available = True
                #    print(f"Google Generative AI configured and model '{self.TEXT_MODEL_NAME}' loaded and tested.")
                # except Exception as test_e:
                #     print(f"ERROR: Google Generative AI model test failed: {test_e}")
                #     print("Check your API Key, model name, and internet connection.")

                # If not testing, assume available after model load:
                self.text_generation_available = True
                print(
                    f"Google Generative AI configured and model '{self.TEXT_MODEL_NAME}' loaded.")

            except ValueError as ve:  # Often indicates bad API key format or model name
                print(
                    f"ERROR: Configuration error with Google Generative AI: {ve}")
                print("Please ensure your GEMINI_API_KEY is correct and valid.")
            except Exception as e:
                print(
                    f"ERROR: Failed to configure Google Generative AI or load model: {e}")
                print(f"Model attempted: {self.TEXT_MODEL_NAME}")
        elif not genai:
            print("Skipping Google Generative AI initialization (library not found).")
        else:  # API Key is the default placeholder
            print(
                "Skipping Google Generative AI initialization (GEMINI_API_KEY not set).")
            print("Please replace 'YOUR_API_KEY' in the script configuration.")
        # --- End NEW Text Init ---

        # --- Game State ---
        self.player = Character("Hero", 100, 15, self.update_display)
        self.enemy = Character("Goblin", 50, 8, self.update_display)
        self.game_over = False

        # --- GUI Elements ---
        main_frame = ttk.Frame(root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)

        log_frame = ttk.LabelFrame(main_frame, text="Game Log", padding="10")
        log_frame.grid(row=2, column=0, columnspan=2,
                       sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        main_frame.rowconfigure(2, weight=1)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        self.message_log = scrolledtext.ScrolledText(
            log_frame, wrap=tk.WORD, width=60, height=15, state='disabled')
        self.message_log.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # --- Image Generation (Uses Vertex AI if available) ---
        self.hero_photo_image = None
        self.goblin_photo_image = None
        if self.image_generation_available:  # This flag is set during Vertex AI init
            self.log_message(
                "Attempting to generate images via Vertex AI Imagen...")
            # ...(Image generation code remains the same as before)...
            self.log_message(
                "WARNING: This may take time and might briefly freeze the app.")
            self.root.update()  # Force GUI update

            hero_prompt = "Pixel art style heroic knight character, facing forward, simple white background, fantasy rpg"
            goblin_prompt = "Pixel art style grumpy green goblin warrior with a small wooden club, facing forward, simple white background, fantasy rpg"

            # Generate Hero Image
            self.log_message(f"Generating Hero image...")  # Simplified log
            self.root.update()
            start_time = time.time()
            hero_bytes = self.generate_image_with_imagen(
                hero_prompt)  # Uses Vertex function
            if hero_bytes:
                duration = time.time() - start_time
                self.log_message(
                    f"Hero image generated ({duration:.2f}s). Loading...")
                self.hero_photo_image = self.load_image_from_bytes(
                    hero_bytes, (120, 120))
            else:
                self.log_message("Failed to generate Hero image.")
            self.root.update()

            # Generate Goblin Image
            self.log_message(f"Generating Goblin image...")  # Simplified log
            self.root.update()
            start_time = time.time()
            goblin_bytes = self.generate_image_with_imagen(
                goblin_prompt)  # Uses Vertex function
            if goblin_bytes:
                duration = time.time() - start_time
                self.log_message(
                    f"Goblin image generated ({duration:.2f}s). Loading...")
                self.goblin_photo_image = self.load_image_from_bytes(
                    goblin_bytes, (120, 120))
            else:
                self.log_message("Failed to generate Goblin image.")
            self.root.update()
        else:
            self.log_message(
                "Image generation skipped (Vertex AI Imagen not available or configured).")

        # --- GUI Setup Continued (Player/Enemy Frames, Buttons) ---
        # ...(Frame and Button setup code remains the same)...
        # == Player Info Frame ==
        player_frame = ttk.LabelFrame(main_frame, text="Player", padding="10")
        player_frame.grid(row=0, column=0, sticky=(
            tk.W, tk.E, tk.N, tk.S), pady=5, padx=5)
        player_frame.columnconfigure(1, weight=1)  # Allow HP bar to stretch
        self.player_image_label = ttk.Label(
            player_frame, text="[No Image]")  # Placeholder text
        if self.hero_photo_image:
            self.player_image_label.config(
                image=self.hero_photo_image, text="")
        self.player_image_label.grid(row=0, column=0, columnspan=3, pady=(
            0, 10), sticky=tk.N)  # Center image
        ttk.Label(player_frame, text=f"{self.player.name}:").grid(
            row=1, column=0, sticky=tk.W)
        self.player_hp_label = ttk.Label(
            player_frame, text=f"{self.player.hp}/{self.player.max_hp} HP")
        # Align HP text to the right
        self.player_hp_label.grid(row=1, column=2, sticky=tk.E)
        self.player_hp_bar = ttk.Progressbar(
            player_frame, orient='horizontal', length=150, mode='determinate', maximum=self.player.max_hp)
        self.player_hp_bar.grid(row=1, column=1, sticky=(
            tk.W, tk.E), padx=5)  # Fill horizontal space

        # == Enemy Info Frame ==
        enemy_frame = ttk.LabelFrame(main_frame, text="Enemy", padding="10")
        enemy_frame.grid(row=0, column=1, sticky=(
            tk.W, tk.E, tk.N, tk.S), pady=5, padx=5)
        enemy_frame.columnconfigure(1, weight=1)  # Allow HP bar to stretch
        self.enemy_image_label = ttk.Label(
            enemy_frame, text="[No Image]")  # Placeholder text
        if self.goblin_photo_image:
            self.enemy_image_label.config(
                image=self.goblin_photo_image, text="")
        self.enemy_image_label.grid(row=0, column=0, columnspan=3, pady=(
            0, 10), sticky=tk.N)  # Center image
        ttk.Label(enemy_frame, text=f"{self.enemy.name}:").grid(
            row=1, column=0, sticky=tk.W)
        self.enemy_hp_label = ttk.Label(
            enemy_frame, text=f"{self.enemy.hp}/{self.enemy.max_hp} HP")
        # Align HP text to the right
        self.enemy_hp_label.grid(row=1, column=2, sticky=tk.E)
        self.enemy_hp_bar = ttk.Progressbar(
            enemy_frame, orient='horizontal', length=150, mode='determinate', maximum=self.enemy.max_hp)
        self.enemy_hp_bar.grid(row=1, column=1, sticky=(
            tk.W, tk.E), padx=5)  # Fill horizontal space

        # == Action Buttons ==
        button_frame = ttk.Frame(main_frame, padding="5")
        button_frame.grid(row=1, column=0, columnspan=2, pady=10)
        self.attack_button = ttk.Button(
            button_frame, text="Attack", command=self.player_attack)
        self.attack_button.grid(row=0, column=0, padx=5)
        self.defend_button = ttk.Button(
            button_frame, text="Defend", command=self.player_defend)
        self.defend_button.grid(row=0, column=1, padx=5)

        # --- Initialize Display & Start Encounter ---
        self.update_display()
        self.log_message(f"--- Encounter Start ---")
        self.log_message(f"A wild {self.enemy.name} appears!")

        # --- Generate introductory dialogue (Uses google-generativeai if available) ---
        if self.text_generation_available:  # Check the flag set during genai init
            prompt = f"Generate a short, simple, taunting phrase a fantasy RPG {self.enemy.name} might shout when first encountering a {self.player.name}. Maximum 10 words, no quotes."
            intro_dialogue = self.generate_dialogue(
                prompt)  # Use the updated function
            if intro_dialogue:
                self.log_message(f'{self.enemy.name}: "{intro_dialogue}"')
            else:
                self.log_message(
                    f"({self.enemy.name} growls menacingly - dialogue generation failed)")
        else:
            # Fallback if text gen unavailable
            self.log_message(f"({self.enemy.name} snarls.)")

    # --- Image Generation Helper (Vertex AI - unchanged) ---
    def generate_image_with_imagen(self, prompt_text):
        """Generates an image using Vertex AI Imagen and returns image bytes."""
        if not self.image_generation_available:  # Check Vertex flag
            self.log_message(
                "Vertex AI Imagen not available, cannot generate image.")
            return None
        # ...(rest of the function is the same as before)...
        try:
            response = self.image_model.generate_images(
                prompt=prompt_text,
                number_of_images=1,
                aspect_ratio="1:1",
            )
            if response and response.images:
                image_obj = response.images[0]
                if hasattr(image_obj, '_image_bytes') and image_obj._image_bytes:
                    return image_obj._image_bytes
                else:
                    self.log_message(
                        "ERROR: Could not directly access image bytes from Imagen response.")
                    print("DEBUG: Imagen Response Structure:", response)
                    return None
            else:
                self.log_message(
                    f"ERROR: Imagen API call failed or returned no images.")
                print("DEBUG: Imagen API Response:", response)
                return None
        except Exception as e:
            self.log_message(f"ERROR during image generation: {e}")
            print(f"DEBUG: Exception details during image generation: {e}")
            return None

    # --- Image Loading from Bytes Helper (Unchanged) ---

    def load_image_from_bytes(self, image_bytes, size):
        """Loads image data from bytes, resizes, returns PhotoImage."""
        # ...(function is the same as before)...
        if not image_bytes:
            return None
        try:
            img_data = io.BytesIO(image_bytes)
            original_image = Image.open(img_data)
            resized_image = original_image.resize(
                size, Image.Resampling.LANCZOS)
            photo_image = ImageTk.PhotoImage(resized_image)
            return photo_image
        except Exception as e:
            self.log_message(
                f"ERROR: Failed to load image from generated bytes: {e}")
            print(f"DEBUG: Pillow/Tkinter loading error: {e}")
            return None

    # --- NEW: Dialogue Generation Helper (using google-generativeai) ---
    def generate_dialogue(self, prompt_text, max_output_tokens=30, temperature=0.8):
        """Generates dialogue using Google Generative AI (genai) and returns the text."""
        if not self.text_generation_available or not self.text_model:  # Check genai flag and model object
            return None  # Return None if text generation is not set up

        try:
            # Configure generation parameters using genai types
            # Note: candidate_count=1 is default but explicit here
            generation_config = genai.types.GenerationConfig(
                candidate_count=1,  # Generate one response candidate
                max_output_tokens=max_output_tokens,
                temperature=temperature
            )

            # Safety settings can be passed per-request as well
            safety_settings_req = [
                {"category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                # Add other categories like DANGEROUS_CONTENT, SEXUALITY if needed
            ]

            # Generate content using the genai model object
            response = self.text_model.generate_content(
                prompt_text,
                generation_config=generation_config,
                safety_settings=safety_settings_req  # Pass request-specific safety settings
                # stream=False # Default is False for google-generativeai
            )

            # Check response structure (google-generativeai)
            if response and response.candidates:
                # Access text directly via response.text (handles simple cases)
                # For more complex scenarios (e.g., blocked response), check response.prompt_feedback
                if response.text:
                    generated_text = response.text.strip()
                    # Simple cleanup (same as before)
                    # Remove surrounding quotes
                    generated_text = re.sub(r'^"|"$', '', generated_text)
                    # Remove markdown bold/italic
                    generated_text = re.sub(r'[\*_]', '', generated_text)
                    return generated_text
                else:
                    # Handle cases where generation might be blocked or empty
                    block_reason = response.prompt_feedback.block_reason if response.prompt_feedback else "Unknown"
                    self.log_message(
                        f"WARNING: Gemini response blocked or empty. Reason: {block_reason}")
                    print(
                        f"DEBUG: Gemini Response (Blocked/Empty): {response}")
                    return None  # Indicate failure due to block/empty
            else:
                # This case might indicate other API issues
                self.log_message(
                    f"ERROR: Gemini API call failed or returned unexpected response structure.")
                print("DEBUG: Gemini API Response:", response)
                return None

        # Catch specific exceptions from google-generativeai if needed
        except Exception as e:
            # Log the error, potentially with more specific details from genai exceptions
            self.log_message(f"ERROR during dialogue generation: {e}")
            # Example: Check for specific API errors like permission denied, quota exceeded, etc.
            # if isinstance(e, google.api_core.exceptions.PermissionDenied):
            #    print("DEBUG: Check your API Key permissions.")
            # elif isinstance(e, google.api_core.exceptions.ResourceExhausted):
            #     print("DEBUG: You might have exceeded your API quota.")
            # else:
            print(f"DEBUG: Exception details during dialogue generation: {e}")
            return None

    # --- Logging Function (Unchanged) ---

    def log_message(self, message):
        """Adds a message to the scrollable text log."""
        # ...(function is the same as before)...
        if hasattr(self, 'message_log'):
            self.message_log.config(state='normal')
            self.message_log.insert(tk.END, message + "\n")
            self.message_log.config(state='disabled')
            self.message_log.see(tk.END)
        else:
            print(message)

    # --- Update Display Function (Unchanged) ---
    def update_display(self):
        """Updates HP labels and progress bars for both characters."""
        # ...(function is the same as before)...
        if hasattr(self, 'player_hp_label'):
            self.player_hp_label.config(
                text=f"{self.player.hp}/{self.player.max_hp} HP")
            self.player_hp_bar['value'] = self.player.hp
        if hasattr(self, 'enemy_hp_label'):
            self.enemy_hp_label.config(
                text=f"{self.enemy.hp}/{self.enemy.max_hp} HP")
            self.enemy_hp_bar['value'] = self.enemy.hp

        if not self.game_over:
            if not self.player.is_alive():
                self.end_game(f"You were defeated by the {self.enemy.name}...")
            elif not self.enemy.is_alive():
                self.end_game(
                    f"Congratulations! You defeated the {self.enemy.name}!")

    # --- Player Action Functions (Dialogue generation calls use the updated generate_dialogue) ---

    def player_attack(self):
        """Handles the player's attack action."""
        if self.game_over:
            return
        self.disable_buttons()
        self.player.reset_defense()

        # --- Generate Hero Attack Dialogue (Uses updated genai function) ---
        if self.text_generation_available:
            prompt = f"Generate a short, heroic battle cry (max 8 words, no quotes) a fantasy RPG {self.player.name} might shout when attacking a {self.enemy.name}."
            hero_dialogue = self.generate_dialogue(prompt)
            if hero_dialogue:
                self.log_message(f'{self.player.name}: "{hero_dialogue}"')

        damage_dealt = self.player.attack(self.enemy, self.log_message)

        # --- Generate Enemy Reaction Dialogue (Uses updated genai function) ---
        if self.enemy.is_alive() and damage_dealt > 0 and self.text_generation_available:
            prompt = f"Generate a short phrase of pain or surprise (max 8 words, no quotes) for a fantasy RPG {self.enemy.name} who just got hit by a {self.player.name}."
            enemy_reaction = self.generate_dialogue(prompt)
            if enemy_reaction:
                self.log_message(f'{self.enemy.name}: "{enemy_reaction}"')

        self.update_display()
        if not self.game_over and self.enemy.is_alive():
            self.root.after(700, self.enemy_turn)

    def player_defend(self):
        """Handles the player's defend action."""
        if self.game_over:
            return
        self.disable_buttons()

        # --- Generate Hero Defend Dialogue (Uses updated genai function) ---
        if self.text_generation_available:
            prompt = f"Generate a short phrase (max 8 words, no quotes) for a fantasy RPG {self.player.name} bracing for an attack or taking a defensive stance."
            hero_dialogue = self.generate_dialogue(prompt)
            if hero_dialogue:
                self.log_message(f'{self.player.name}: "{hero_dialogue}"')

        self.player.defend(self.log_message)

        self.update_display()
        if not self.game_over and self.enemy.is_alive():
            self.root.after(700, self.enemy_turn)

    # --- Enemy Turn (Dialogue generation calls use the updated generate_dialogue) ---
    def enemy_turn(self):
        """Handles the enemy's automated turn."""
        if self.game_over:
            return
        self.log_message(f"\n--- {self.enemy.name}'s Turn ---")
        self.enemy.reset_defense()

        # --- Generate Enemy Attack Dialogue (Uses updated genai function) ---
        if self.text_generation_available:
            prompt = f"Generate a short, aggressive or guttural attack phrase (max 8 words, no quotes) for a fantasy RPG {self.enemy.name} attacking a {self.player.name}."
            enemy_dialogue = self.generate_dialogue(prompt)
            if enemy_dialogue:
                self.log_message(f'{self.enemy.name}: "{enemy_dialogue}"')

        damage_dealt = self.enemy.attack(self.player, self.log_message)

        # --- Generate Hero Reaction Dialogue (Uses updated genai function) ---
        if self.player.is_alive() and damage_dealt > 0 and self.text_generation_available:
            prompt = f"Generate a short phrase of pain, determination, or reaction (max 8 words, no quotes) for a fantasy RPG {self.player.name} who just got hit by a {self.enemy.name}."
            hero_reaction = self.generate_dialogue(prompt)
            if hero_reaction:
                self.log_message(f'{self.player.name}: "{hero_reaction}"')

        self.player.reset_defense()
        self.update_display()
        if not self.game_over:
            self.enable_buttons()
            self.log_message("\n--- Your Turn ---")

    # --- Button State Control (Unchanged) ---
    def disable_buttons(self):
        # ...(function is the same as before)...
        if hasattr(self, 'attack_button'):
            self.attack_button.config(state='disabled')
        if hasattr(self, 'defend_button'):
            self.defend_button.config(state='disabled')

    def enable_buttons(self):
        # ...(function is the same as before)...
        if not self.game_over:
            if hasattr(self, 'attack_button'):
                self.attack_button.config(state='normal')
            if hasattr(self, 'defend_button'):
                self.defend_button.config(state='normal')

    # --- End Game (Dialogue generation calls use the updated generate_dialogue) ---
    def end_game(self, message):
        """Handles the game over sequence."""
        if self.game_over:
            return
        self.game_over = True
        self.disable_buttons()
        self.log_message("\n--- Combat Over ---")

        # --- Generate Final Dialogue (Uses updated genai function) ---
        if self.text_generation_available:
            final_words_winner = None  # Track if winner spoke
            # Player won
            if self.player.is_alive() and not self.enemy.is_alive():
                prompt_winner = f"Generate a short, victorious phrase (max 10 words, no quotes) for a fantasy RPG {self.player.name} after defeating a {self.enemy.name}."
                final_words = self.generate_dialogue(prompt_winner)
                if final_words:
                    self.log_message(f'{self.player.name}: "{final_words}"')
                    final_words_winner = True

                prompt_loser = f"Generate a short final groan or cry of defeat (max 8 words, no quotes) for a fantasy RPG {self.enemy.name} as they are defeated."
                final_words_loser = self.generate_dialogue(prompt_loser)
                if final_words_loser:
                    self.log_message(
                        f'{self.enemy.name}: "{final_words_loser}"')

            # Enemy won
            elif not self.player.is_alive() and self.enemy.is_alive():
                prompt_winner = f"Generate a short, gloating or triumphant phrase (max 10 words, no quotes) for a fantasy RPG {self.enemy.name} after defeating a {self.player.name}."
                final_words = self.generate_dialogue(prompt_winner)
                if final_words:
                    self.log_message(f'{self.enemy.name}: "{final_words}"')
                    final_words_winner = True

            if final_words_winner:
                self.root.update()
                time.sleep(0.5)

        self.log_message(message)  # Log the official result


# --- Main Execution ---
if __name__ == "__main__":
    # Now only checks for genai if Vertex AI is *not* used or fails
    # Image generation is optional, Text generation relies on genai
    libs_ok = True
    if not genai:
        print("\nCRITICAL ERROR: google-generativeai library not found (required for text generation).")
        print("Install it using: pip install google-generativeai")
        libs_ok = False
        # Optionally show Tkinter error and exit

    # You could add a similar check for aiplatform if image gen is critical for you
    # if not aiplatform:
    #    print("\nWARNING: google-cloud-aiplatform library not found (required for image generation).")

    if not libs_ok:  # Exit if essential libs are missing
        import tkinter as tk
        from tkinter import messagebox
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(
            "Fatal Error", "Required Python library 'google-generativeai' not found.\nPlease install it (`pip install google-generativeai`) and try again.")
        exit()

    # Start the Tkinter application
    root = tk.Tk()
    try:
        app = RPG_GUI(root)
        root.mainloop()
    except Exception as main_error:
        print(
            f"FATAL ERROR during GUI initialization or main loop: {main_error}")
        try:
            from tkinter import messagebox
            messagebox.showerror(
                "Fatal Error", f"An unexpected error occurred:\n{main_error}\n\nCheck the console output for details.")
        except:
            pass
