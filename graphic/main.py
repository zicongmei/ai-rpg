# --- Prerequisites ---
# 1. Install the library: pip install google-generativeai google-cloud-aiplatform Pillow
# 2. Get an API Key: Create one from Google AI Studio (https://aistudio.google.com/app/apikey)
# 3. Google Cloud Authentication (for Vertex AI): Ensure your environment is authenticated
#    (e.g., using `gcloud auth application-default login`) if using Vertex AI Image Gen.

import tkinter as tk
from tkinter import ttk
import google.auth
from tkinter import scrolledtext
import random
from PIL import Image, ImageTk
import os
import io
import time
import re
# --- NEW: Imports for Parallel Execution ---
import concurrent.futures
import threading  # To run the executor management in a separate thread

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
    genai = None

# --- Character Class (No changes needed - keep as is) ---


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


# --- GUI Application Class (Modified for Parallel Image Generation) ---
class RPG_GUI:
    # --- !!! USER CONFIGURATION REQUIRED !!! ---

    # == Vertex AI Image Generation Config (Optional) ==
    # <--- REPLACE if using Vertex AI Image Gen (or set via gcloud)
    PROJECT_ID = "invalid"
    LOCATION = "us-central1"  # <--- REPLACE if using Vertex AI Image Gen
    IMAGE_MODEL_NAME = "imagegeneration@006"

    # == NEW: Google Generative AI Text Generation Config ==
    GEMINI_API_KEY = "YOUR_API_KEY"  # <--- REPLACE with your actual Gemini API Key
    # Use a compatible model like 1.5 Flash or Pro
    TEXT_MODEL_NAME = "gemini-1.5-flash"

    # --- End User Configuration ---

    def __init__(self, root):
        self.root = root
        root.title("Simple RPG Combat (Parallel AI Images)")
        root.geometry("600x650")

        # --- State for Image Generation ---
        self.hero_photo_image = None  # Holds the PhotoImage to prevent GC
        self.goblin_photo_image = None  # Holds the PhotoImage to prevent GC
        self._image_generation_executor = None  # ThreadPoolExecutor for images

        # --- Initialize Vertex AI for Image Generation (Optional) ---
        # (This part remains largely the same, just sets up flags/models)
        self.vertex_ai_initialized = False
        self.image_generation_available = False
        self.image_model = None

        credentials, project_id = None, None
        try:
            # Try to get credentials and project ID from the environment
            credentials, project_id_found = google.auth.default(
                scopes=['https://www.googleapis.com/auth/cloud-platform']
            )
            if project_id_found:
                # Use environment's project ID if not explicitly set above
                if self.PROJECT_ID == "invalid":
                    self.PROJECT_ID = project_id_found
                    print(f"Using Project ID from gcloud: {self.PROJECT_ID}")
            else:
                # If PROJECT_ID still 'invalid', Vertex AI won't work
                if self.PROJECT_ID == "invalid":
                    print(
                        "WARNING: Could not determine Google Cloud Project ID for Vertex AI.")

        except google.auth.exceptions.DefaultCredentialsError:
            print("WARNING: Google Cloud Default Credentials not found.")
            print("         Vertex AI Image Generation will likely fail unless PROJECT_ID is explicitly set and relevant auth is configured elsewhere.")
        except Exception as e:
            print(f"WARNING: Error during Google Cloud auth setup: {e}")

        # Attempt Vertex AI init only if library exists and Project ID is valid
        if aiplatform and self.PROJECT_ID and self.PROJECT_ID != "invalid":
            try:
                # Initialize AI Platform with detected or provided credentials
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

            except Exception as e:
                print(f"ERROR: Failed to initialize Vertex AI: {e}")
                print(
                    "Check Project ID, Location, Auth, and API enablement if using Vertex Image Gen.")

        elif not aiplatform:
            print("Skipping Vertex AI initialization (library not found).")
        elif not self.PROJECT_ID or self.PROJECT_ID == "invalid":
            print(
                "Skipping Vertex AI initialization (PROJECT_ID not determined or invalid).")

        # --- NEW: Initialize Google Generative AI for Text Generation ---
        # (This part remains the same)
        self.text_model = None
        self.text_generation_available = False
        if genai:
            try:
                credentials, project_id = google.auth.default()
                genai.configure(api_key=credentials.token)
                self.text_model = genai.GenerativeModel(self.TEXT_MODEL_NAME)
                # Simple test (optional, uncomment to verify key/model early)
                # self.text_model.generate_content("test", generation_config=genai.types.GenerationConfig(candidate_count=1, max_output_tokens=5))
                self.text_generation_available = True
                print(
                    f"Google Generative AI configured with model '{self.TEXT_MODEL_NAME}'.")
            except ValueError as ve:
                print(
                    f"ERROR: Configuration error with Google Generative AI: {ve}")
                print(
                    "Please ensure your GEMINI_API_KEY and TEXT_MODEL_NAME are correct.")
            except Exception as e:
                print(
                    f"ERROR: Failed to configure Google Generative AI or load model: {e}")
        elif not genai:
            print("Skipping Google Generative AI initialization (library not found).")
        else:  # API Key is missing or default
            print(
                "Skipping Google Generative AI initialization (GEMINI_API_KEY not set).")

        # --- Game State ---
        self.player = Character("Hero", 100, 15, self.update_display)
        self.enemy = Character("Goblin", 50, 8, self.update_display)
        self.game_over = False

        # --- GUI Elements ---
        main_frame = ttk.Frame(root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)  # Player frame column
        main_frame.columnconfigure(1, weight=1)  # Enemy frame column

        # -- Log Frame --
        log_frame = ttk.LabelFrame(main_frame, text="Game Log", padding="10")
        log_frame.grid(row=2, column=0, columnspan=2,
                       sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        main_frame.rowconfigure(2, weight=1)  # Allow log to expand vertically
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        self.message_log = scrolledtext.ScrolledText(
            log_frame, wrap=tk.WORD, width=60, height=15, state='disabled')
        self.message_log.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # --- GUI Setup Continued (Player/Enemy Frames, Buttons) ---
        # Create frames and labels first, images will be loaded asynchronously

        # == Player Info Frame ==
        player_frame = ttk.LabelFrame(main_frame, text="Player", padding="10")
        player_frame.grid(row=0, column=0, sticky=(
            tk.W, tk.E, tk.N, tk.S), pady=5, padx=5)
        player_frame.columnconfigure(1, weight=1)  # Allow HP bar to stretch

        # Placeholder for Player Image
        self.player_image_label = ttk.Label(
            player_frame, text="[Generating Image...]" if self.image_generation_available else "[Image N/A]", anchor=tk.CENTER)
        self.player_image_label.grid(row=0, column=0, columnspan=3, pady=(
            0, 10), sticky=tk.N + tk.S + tk.E + tk.W, ipady=50)  # Give it some initial size
        # Allow image label to take space
        player_frame.rowconfigure(0, weight=1)

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

        # Placeholder for Enemy Image
        self.enemy_image_label = ttk.Label(
            enemy_frame, text="[Generating Image...]" if self.image_generation_available else "[Image N/A]", anchor=tk.CENTER)
        self.enemy_image_label.grid(row=0, column=0, columnspan=3, pady=(
            0, 10), sticky=tk.N + tk.S + tk.E + tk.W, ipady=50)  # Give it some initial size
        # Allow image label to take space
        enemy_frame.rowconfigure(0, weight=1)

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

        # --- Initialize Display ---
        self.update_display()  # Set initial HP bars/labels

        # --- Start Encounter ---
        self.log_message(f"--- Encounter Start ---")
        self.log_message(f"A wild {self.enemy.name} appears!")

        # --- *** NEW: Initiate Parallel Image Generation *** ---
        self._initiate_parallel_image_generation()  # Start background generation

        # --- Generate introductory dialogue (Uses google-generativeai if available) ---
        # (This part remains the same)
        if self.text_generation_available:
            prompt = f"Generate a short, simple, taunting phrase a fantasy RPG {self.enemy.name} might shout when first encountering a {self.player.name}. Maximum 10 words, no quotes."
            intro_dialogue = self.generate_dialogue(prompt)
            if intro_dialogue:
                self.log_message(f'{self.enemy.name}: "{intro_dialogue}"')
            else:
                self.log_message(
                    f"({self.enemy.name} growls menacingly - dialogue generation failed)")
        else:
            self.log_message(f"({self.enemy.name} snarls.)")  # Fallback

        # --- Set up cleanup for the executor when the window closes ---
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # --- NEW: Method to Start Parallel Image Generation ---
    def _initiate_parallel_image_generation(self):
        """Starts the hero and goblin image generation in parallel background threads."""
        if not self.image_generation_available:
            self.log_message(
                "Image generation skipped (Vertex AI Imagen not available or configured).")
            # Update labels to show N/A permanently if generation wasn't even attempted
            self.player_image_label.config(text="[Image N/A]")
            self.enemy_image_label.config(text="[Image N/A]")
            return

        self.log_message(
            "Starting parallel image generation via Vertex AI Imagen...")

        # Use a separate thread to manage the ThreadPoolExecutor
        # This prevents the executor setup/submission from potentially blocking the main thread momentarily
        thread = threading.Thread(
            target=self._run_image_generation_in_background, daemon=True)
        thread.start()

    # --- NEW: Method Containing the ThreadPoolExecutor Logic ---
    def _run_image_generation_in_background(self):
        """Runs the image generation tasks using a ThreadPoolExecutor."""
        hero_prompt = "Pixel art style heroic knight character, facing forward, simple white background, fantasy rpg"
        goblin_prompt = "Pixel art style grumpy green goblin warrior with a small wooden club, facing forward, simple white background, fantasy rpg"

        # max_workers=2 since we have 2 tasks
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            # Store reference if needed for shutdown
            self._image_generation_executor = executor
            # Submit tasks to the executor. Each task runs _generate_image_task.
            # We pass the prompt and an identifier ('hero' or 'goblin').
            hero_future = executor.submit(
                self._generate_image_task, hero_prompt, "hero")
            goblin_future = executor.submit(
                self._generate_image_task, goblin_prompt, "goblin")

            # Add a callback function that will run when each future completes.
            # This callback runs in the executor's thread.
            hero_future.add_done_callback(self._handle_image_generation_result)
            goblin_future.add_done_callback(
                self._handle_image_generation_result)

        # Keep the executor reference None after the 'with' block if you don't need it later
        # Or manage its shutdown explicitly in _on_close if needed
        self._image_generation_executor = None
        print("Image generation threads finished submitting tasks.")  # Debug log

    # --- NEW: Wrapper Task for Image Generation ---
    def _generate_image_task(self, prompt, image_type):
        """
        Wrapper function to call the actual image generation and handle timing/logging.
        Returns a tuple: (image_type, image_bytes_or_None)
        Runs within the ThreadPoolExecutor thread.
        """
        start_time = time.time()
        print(f"Thread started for: {image_type}")  # Debug log
        image_bytes = self.generate_image_with_imagen(
            prompt, image_type)  # Pass type for logging
        duration = time.time() - start_time
        if image_bytes:
            # Log success from background thread (logging function should be thread-safe enough)
            self.log_message(
                f"{image_type.capitalize()} image generated ({duration:.2f}s). Preparing display...")
        else:
            # Log failure from background thread
            self.log_message(f"Failed to generate {image_type} image.")
            # Debug log
            print(f"Thread finished with failure for: {image_type}")
        return image_type, image_bytes

    # --- NEW: Callback Handler for Completed Image Generation ---
    def _handle_image_generation_result(self, future):
        """
        Callback function executed when a generation task (Future) finishes.
        Runs in the ThreadPoolExecutor thread.
        Schedules the UI update on the main thread.
        """
        try:
            # Get the result from the future. This re-raises exceptions from the task.
            image_type, image_bytes = future.result()

            # --- IMPORTANT: Schedule UI update on the main thread ---
            # Use root.after(0, ...) to run _update_image_ui safely in the Tkinter event loop
            self.root.after(0, self._update_image_ui, image_type, image_bytes)

        except Exception as e:
            # Log any exception that occurred *during* the _generate_image_task execution
            # Log detailed error to console
            print(f"ERROR in image generation task thread: {e}")
            self.log_message(f"Error during background image generation: {e}")
            # Optionally, try to determine image_type if possible from context or future properties
            # and schedule a UI update to show failure state. For simplicity, we might just log here.
            # If future.result() failed, we don't know the image_type directly here.
            # We could modify _generate_image_task to wrap its result/exception handling
            # But for now, just log the error. The UI might remain in "Generating..." state.

    # --- NEW: UI Update Function (Runs on Main Thread) ---

    def _update_image_ui(self, image_type, image_bytes):
        """
        Updates the corresponding image label in the GUI.
        MUST run on the main Tkinter thread (called via root.after).
        """
        target_label = None
        photo_image_attr = None

        if image_type == "hero":
            target_label = self.player_image_label
            photo_image_attr = "hero_photo_image"
        elif image_type == "goblin":
            target_label = self.enemy_image_label
            photo_image_attr = "goblin_photo_image"
        else:
            self.log_message(
                f"Error: Unknown image type '{image_type}' for UI update.")
            return

        if image_bytes:
            photo_image = self.load_image_from_bytes(image_bytes, (120, 120))
            if photo_image:
                # Store the PhotoImage object as an instance variable
                # to prevent it from being garbage collected.
                setattr(self, photo_image_attr, photo_image)
                # Update the label
                # Remove placeholder text
                target_label.config(image=photo_image, text="")
                self.log_message(
                    f"{image_type.capitalize()} image loaded into UI.")
            else:
                # Loading from bytes failed (already logged in load_image_from_bytes)
                target_label.config(image=None, text="[Load Failed]")
                # Clear any previous image ref
                setattr(self, photo_image_attr, None)
        else:
            # Generation failed (already logged in _generate_image_task)
            target_label.config(image=None, text="[Gen Failed]")
            # Clear any previous image ref
            setattr(self, photo_image_attr, None)

    # --- Image Generation Helper (Vertex AI - Modified Slightly for Logging) ---

    def generate_image_with_imagen(self, prompt_text, image_type_for_log=""):
        """Generates an image using Vertex AI Imagen and returns image bytes."""
        # Add a check here as well, although _initiate checks first
        if not self.image_generation_available or not self.image_model:
            print(f"Imagen not available for {image_type_for_log} generation.")
            return None

        log_prefix = f"({image_type_for_log.capitalize()}) " if image_type_for_log else ""
        self.log_message(f"{log_prefix}Sending request to Vertex AI Imagen...")
        # print(f"DEBUG: Generating image for '{image_type_for_log}' with prompt: {prompt_text}") # Uncomment for detailed debug

        try:
            # Make the API call
            response = self.image_model.generate_images(
                prompt=prompt_text,
                number_of_images=1,
                # seed=random.randint(0, 10000), # Optional: for reproducibility if needed
                aspect_ratio="1:1",  # Keep square for consistency
                # safety_filter_level="block_most", # Adjust safety level if desired
                # person_generation="allow_adult", # Or "block_adult"
            )

            # Process the response
            if response and response.images:
                image_obj = response.images[0]
                # Accessing image bytes might vary slightly based on SDK version
                # Try common attributes
                if hasattr(image_obj, '_image_bytes') and image_obj._image_bytes:
                    self.log_message(
                        f"{log_prefix}Image data received from API.")
                    return image_obj._image_bytes
                # Check alternative attribute name
                elif hasattr(image_obj, 'image_bytes') and image_obj.image_bytes:
                    self.log_message(
                        f"{log_prefix}Image data received from API (alt attr).")
                    return image_obj.image_bytes
                else:
                    # Log failure to extract bytes
                    self.log_message(
                        f"ERROR: {log_prefix}Could not extract image bytes from Imagen response.")
                    # Inspect object
                    print(
                        f"DEBUG: Imagen Response Structure ({image_type_for_log}):", image_obj.__dict__)
                    return None
            else:
                # Log API failure or empty response
                self.log_message(
                    f"ERROR: {log_prefix}Imagen API call failed or returned no images.")
                print(
                    f"DEBUG: Imagen API Response ({image_type_for_log}):", response)
                return None
        except Exception as e:
            # Log any exception during the API call
            self.log_message(
                f"ERROR: {log_prefix}Exception during Vertex AI image generation: {e}")
            print(f"DEBUG: Exception details ({image_type_for_log}): {e}")
            # Consider checking for specific API errors (quota, auth, etc.) if needed
            # Example:
            # if "permission denied" in str(e).lower():
            #     self.log_message("Hint: Check API Key/Permissions.")
            # elif "quota" in str(e).lower():
            #     self.log_message("Hint: Check API Quota.")
            return None

    # --- Image Loading from Bytes Helper (Unchanged) ---

    def load_image_from_bytes(self, image_bytes, size):
        """Loads image data from bytes, resizes, returns PhotoImage."""
        if not image_bytes:
            return None
        try:
            img_data = io.BytesIO(image_bytes)
            original_image = Image.open(img_data)
            # Ensure image has an alpha channel for potentially better resizing/display
            if original_image.mode != 'RGBA':
                original_image = original_image.convert('RGBA')
            # Use LANCZOS (also known as ANTIALIAS) for high-quality downscaling
            resized_image = original_image.resize(
                size, Image.Resampling.LANCZOS)
            photo_image = ImageTk.PhotoImage(resized_image)
            return photo_image
        except Exception as e:
            self.log_message(
                f"ERROR: Failed to load image from generated bytes: {e}")
            print(f"DEBUG: Pillow/Tkinter loading error: {e}")
            return None

    # --- Dialogue Generation Helper (using google-generativeai - unchanged) ---

    def generate_dialogue(self, prompt_text, max_output_tokens=30, temperature=0.8):
        """Generates dialogue using Google Generative AI (genai) and returns the text."""
        if not self.text_generation_available or not self.text_model:
            return None

        try:
            generation_config = genai.types.GenerationConfig(
                candidate_count=1,
                max_output_tokens=max_output_tokens,
                temperature=temperature
            )
            safety_settings_req = [
                # {"category": "HARM_CATEGORY_HARASSMENT",
                #     "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                # {"category": "HARM_CATEGORY_HATE_SPEECH",
                #     "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                # Add others like DANGEROUS_CONTENT, SEXUALITY if needed, aligned with model compatibility
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                 "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                # {"category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                #  "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            ]

            response = self.text_model.generate_content(
                prompt_text,
                generation_config=generation_config,
                safety_settings=safety_settings_req
            )

            if response and response.candidates:
                if response.text:
                    generated_text = response.text.strip()
                    # Remove surrounding quotes
                    generated_text = re.sub(r'^"|"$', '', generated_text)
                    # Remove markdown bold/italic
                    generated_text = re.sub(r'[\*_]', '', generated_text)
                    return generated_text
                else:
                    block_reason = "Unknown"
                    safety_ratings = "N/A"
                    try:  # Safely access feedback attributes
                        if response.prompt_feedback:
                            block_reason = response.prompt_feedback.block_reason
                        if response.candidates[0].safety_ratings:
                            safety_ratings = response.candidates[0].safety_ratings
                    except (AttributeError, IndexError):
                        pass  # Ignore if attributes don't exist
                    self.log_message(
                        f"WARNING: Gemini response blocked or empty. Reason: {block_reason}")
                    print(
                        f"DEBUG: Gemini Response Blocked/Empty. Reason: {block_reason}, Safety Ratings: {safety_ratings}")
                    # print("DEBUG Response Object:", response) # Uncomment for full debug
                    return None
            else:
                self.log_message(
                    "ERROR: Gemini API call failed or returned unexpected structure.")
                print("DEBUG: Gemini API Response:", response)
                return None

        except Exception as e:
            self.log_message(f"ERROR during dialogue generation: {e}")
            print(f"DEBUG: Exception details during dialogue generation: {e}")
            # Add specific exception checks if needed (e.g., API key errors, quota)
            return None

    # --- Logging Function (Unchanged) ---
    def log_message(self, message):
        """Adds a message to the scrollable text log. Thread-safe for basic Tkinter append."""
        # Basic appends to Tkinter text widgets are generally considered thread-safe,
        # but complex manipulations should use root.after. This should be fine.
        if hasattr(self, 'message_log') and self.message_log:
            try:
                self.message_log.config(state='normal')
                self.message_log.insert(tk.END, message + "\n")
                self.message_log.config(state='disabled')
                self.message_log.see(tk.END)  # Auto-scroll
            except tk.TclError as e:
                # Handle cases where the widget might be destroyed during shutdown
                print(f"Log Error (potential race condition on close): {e}")
            except Exception as e:
                # Catch other potential issues
                print(f"Unexpected Log Error: {e}")
        else:
            print(message)  # Fallback if GUI isn't ready or destroyed

    # --- Update Display Function (Unchanged) ---
    def update_display(self):
        """Updates HP labels and progress bars for both characters."""
        # Safe to call directly as it only reads game state and updates UI elements
        # that are already created.
        if hasattr(self, 'player_hp_label'):
            self.player_hp_label.config(
                text=f"{self.player.hp}/{self.player.max_hp} HP")
            self.player_hp_bar['value'] = self.player.hp
        if hasattr(self, 'enemy_hp_label'):
            self.enemy_hp_label.config(
                text=f"{self.enemy.hp}/{self.enemy.max_hp} HP")
            self.enemy_hp_bar['value'] = self.enemy.hp

        # Check for game over condition (if not already over)
        if not self.game_over:
            if not self.player.is_alive():
                self.end_game(f"You were defeated by the {self.enemy.name}...")
            elif not self.enemy.is_alive():
                self.end_game(
                    f"Congratulations! You defeated the {self.enemy.name}!")

    # --- Player Action Functions (Unchanged logic, dialogue calls are fine) ---
    def player_attack(self):
        if self.game_over:
            return
        self.disable_buttons()
        self.player.reset_defense()
        if self.text_generation_available:
            prompt = f"Generate a short, heroic battle cry (max 8 words, no quotes) a fantasy RPG {self.player.name} might shout when attacking a {self.enemy.name}."
            hero_dialogue = self.generate_dialogue(prompt)
            if hero_dialogue:
                self.log_message(f'{self.player.name}: "{hero_dialogue}"')
        damage_dealt = self.player.attack(self.enemy, self.log_message)
        if self.enemy.is_alive() and damage_dealt > 0 and self.text_generation_available:
            prompt = f"Generate a short phrase of pain or surprise (max 8 words, no quotes) for a fantasy RPG {self.enemy.name} who just got hit by a {self.player.name}."
            enemy_reaction = self.generate_dialogue(prompt)
            if enemy_reaction:
                self.log_message(f'{self.enemy.name}: "{enemy_reaction}"')
        self.update_display()
        if not self.game_over and self.enemy.is_alive():
            self.root.after(700, self.enemy_turn)  # Schedule enemy turn

    def player_defend(self):
        if self.game_over:
            return
        self.disable_buttons()
        if self.text_generation_available:
            prompt = f"Generate a short phrase (max 8 words, no quotes) for a fantasy RPG {self.player.name} bracing for an attack or taking a defensive stance."
            hero_dialogue = self.generate_dialogue(prompt)
            if hero_dialogue:
                self.log_message(f'{self.player.name}: "{hero_dialogue}"')
        self.player.defend(self.log_message)
        self.update_display()
        if not self.game_over and self.enemy.is_alive():
            self.root.after(700, self.enemy_turn)  # Schedule enemy turn

    # --- Enemy Turn (Unchanged logic, dialogue calls are fine) ---
    def enemy_turn(self):
        if self.game_over:
            return
        self.log_message(f"\n--- {self.enemy.name}'s Turn ---")
        self.enemy.reset_defense()
        if self.text_generation_available:
            prompt = f"Generate a short, aggressive or guttural attack phrase (max 8 words, no quotes) for a fantasy RPG {self.enemy.name} attacking a {self.player.name}."
            enemy_dialogue = self.generate_dialogue(prompt)
            if enemy_dialogue:
                self.log_message(f'{self.enemy.name}: "{enemy_dialogue}"')
        damage_dealt = self.enemy.attack(self.player, self.log_message)
        if self.player.is_alive() and damage_dealt > 0 and self.text_generation_available:
            prompt = f"Generate a short phrase of pain, determination, or reaction (max 8 words, no quotes) for a fantasy RPG {self.player.name} who just got hit by a {self.enemy.name}."
            hero_reaction = self.generate_dialogue(prompt)
            if hero_reaction:
                self.log_message(f'{self.player.name}: "{hero_reaction}"')
        self.player.reset_defense()  # Player defence resets after enemy attack resolves
        self.update_display()
        if not self.game_over:
            self.enable_buttons()
            self.log_message("\n--- Your Turn ---")

    # --- Button State Control (Unchanged) ---
    def disable_buttons(self):
        if hasattr(self, 'attack_button'):
            self.attack_button.config(state='disabled')
        if hasattr(self, 'defend_button'):
            self.defend_button.config(state='disabled')

    def enable_buttons(self):
        if not self.game_over:
            if hasattr(self, 'attack_button'):
                self.attack_button.config(state='normal')
            if hasattr(self, 'defend_button'):
                self.defend_button.config(state='normal')

    # --- End Game (Unchanged logic, dialogue calls are fine) ---
    def end_game(self, message):
        if self.game_over:
            return
        self.game_over = True
        self.disable_buttons()
        self.log_message("\n--- Combat Over ---")
        # (Dialogue generation logic remains the same)
        if self.text_generation_available:
            final_words_winner = None
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
            elif not self.player.is_alive() and self.enemy.is_alive():
                prompt_winner = f"Generate a short, gloating or triumphant phrase (max 10 words, no quotes) for a fantasy RPG {self.enemy.name} after defeating a {self.player.name}."
                final_words = self.generate_dialogue(prompt_winner)
                if final_words:
                    self.log_message(f'{self.enemy.name}: "{final_words}"')
                    final_words_winner = True
            if final_words_winner:
                self.root.update()
                time.sleep(0.5)  # Short pause for effect
        self.log_message(message)

    # --- NEW: Window Close Handler ---
    def _on_close(self):
        """Handles window closing, attempts to shut down executor."""
        print("Window closing...")
        # Optional: Explicitly shut down the executor if it's still running
        # This might help ensure threads terminate cleanly, though daemon threads
        # should exit when the main program exits.
        # if self._image_generation_executor:
        #     print("Shutting down image generation executor...")
        #     # shutdown(wait=False) tells threads to finish current task but doesn't wait
        #     self._image_generation_executor.shutdown(wait=False, cancel_futures=True)
        self.root.destroy()  # Close the Tkinter window


# --- Main Execution (Ensure google-generativeai is checked, Vertex is optional) ---
if __name__ == "__main__":
    libs_ok = True
    # Text generation is core for dialogue, so genai is crucial
    if not genai:
        print("\nCRITICAL ERROR: google-generativeai library not found (required for text generation).")
        print("Install it using: pip install google-generativeai")
        libs_ok = False

    # Image generation is optional, only warn if aiplatform is missing
    if not aiplatform:
        print("\nWARNING: google-cloud-aiplatform library not found.")
        print("         Vertex AI Image Generation will be disabled.")
        print("         Install 'google-cloud-aiplatform' if needed.")

    if not libs_ok:
        # Show a Tkinter error message if the crucial library is missing
        try:
            import tkinter as tk
            from tkinter import messagebox
            root_err = tk.Tk()
            root_err.withdraw()  # Hide the empty root window
            messagebox.showerror(
                "Fatal Error",
                "Required Python library 'google-generativeai' not found.\n"
                "Please install it (`pip install google-generativeai`) and try again."
            )
            root_err.destroy()
        except ImportError:
            # Fallback if Tkinter itself is missing (less likely)
            print("Tkinter library also seems missing. Cannot show graphical error.")
        exit(1)  # Exit if critical library is missing

    # Start the Tkinter application
    root = tk.Tk()
    try:
        app = RPG_GUI(root)
        root.mainloop()
    except Exception as main_error:
        # Catch errors during GUI initialization or the main loop
        print(
            f"\nFATAL ERROR during GUI initialization or main loop: {main_error}", flush=True)
        import traceback
        traceback.print_exc()  # Print detailed traceback
        try:
            # Attempt to show a final error message box
            from tkinter import messagebox
            messagebox.showerror(
                "Fatal Error",
                f"An unexpected error occurred:\n\n{main_error}\n\n"
                "The application needs to close.\nCheck the console output for details."
            )
        except Exception as msg_err:
            print(f"Could not display final error message box: {msg_err}")
        finally:
            if root:
                root.destroy()  # Ensure window is closed on error
            exit(1)  # Exit with error status
