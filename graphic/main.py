import tkinter as tk
from tkinter import ttk
import google.auth
from tkinter import scrolledtext
import random
from PIL import Image, ImageTk
import os
import io  # To handle image bytes in memory
import time  # To potentially show delays

# --- Google Cloud Vertex AI Imports ---
try:
    from google.cloud import aiplatform
    from google.cloud.aiplatform.gapic.schema import predict
    # Check if this import path is correct for image generation parameters
    # from google.protobuf import struct_pb2 # May not be needed directly for basic generation
    print("Vertex AI SDK loaded.")
except ImportError:
    print("ERROR: google-cloud-aiplatform library not found.")
    print("Please install it: pip install google-cloud-aiplatform")
    # Exit or disable generation if library is missing
    aiplatform = None


# --- NEW: Import Vertex AI ---
try:
    from vertexai.preview.vision_models import ImageGenerationModel, ImageGenerationResponse
except ImportError:
    print("ERROR: google-cloud-aiplatform library not found or failed to import.")
    print("Please install it: pip install google-cloud-aiplatform --upgrade")


# --- Character Class (No changes needed) ---


class Character:
    """Represents a character in the game (Player or Enemy)."""

    def __init__(self, name, hp, attack, gui_update_callback=None):
        self.name = name
        self.max_hp = hp
        self.hp = hp
        self.attack_power = attack
        self.is_defending = False
        self.gui_update_callback = gui_update_callback

    def is_alive(self): return self.hp > 0

    def attack(self, target, log_callback):
        if not self.is_alive():
            log_callback(f"{self.name} cannot attack, they are defeated!")
            return
        damage = random.randint(
            int(self.attack_power * 0.8), int(self.attack_power * 1.2))
        log_callback(f"\n{self.name} attacks {target.name}!")
        reduced_damage = False
        if target.is_defending:
            damage = max(0, damage // 2)
            reduced_damage = True
            target.is_defending = False
        target.take_damage(damage, log_callback)
        if reduced_damage:
            log_callback(f"{target.name} was defending! Damage reduced.")

    def take_damage(self, damage, log_callback):
        self.hp -= damage
        self.hp = max(0, self.hp)
        log_callback(
            f"{self.name} takes {damage} damage. Remaining HP: {self.hp}/{self.max_hp}")
        if not self.is_alive():
            log_callback(f"{self.name} has been defeated!")
        if self.gui_update_callback:
            self.gui_update_callback()

    def defend(self, log_callback):
        if not self.is_alive():
            return
        log_callback(f"\n{self.name} takes a defensive stance!")
        self.is_defending = True

    def reset_defense(self): self.is_defending = False


# --- GUI Application Class ---
class RPG_GUI:
    # --- !!! USER CONFIGURATION REQUIRED !!! ---
    # <--- REPLACE with your Google Cloud Project ID
    PROJECT_ID = "invalid"
    # <--- REPLACE with a valid Vertex AI region (e.g., us-central1)
    LOCATION = "us-central1"
    # Model name might change, check Vertex AI documentation for current Imagen models
    IMAGE_MODEL_NAME = "imagegeneration@006"  # Or imagegeneration@005 etc.
    # --- End User Configuration ---

    def __init__(self, root):
        self.root = root
        root.title("Simple RPG Combat (Live Image Generation)")
        root.geometry("600x600")  # Increased height for logs

        credentials, project_id = google.auth.default()
        self.PROJECT_ID = project_id

        # --- Initialize Vertex AI (if available) ---
        self.vertex_ai_available = False
        if aiplatform and self.PROJECT_ID != "your-gcp-project-id":
            try:
                aiplatform.init(project=self.PROJECT_ID,
                                location=self.LOCATION,
                                credentials=credentials)
                self.vertex_ai_available = True
                print(
                    f"Vertex AI initialized for project '{self.PROJECT_ID}' in location '{self.LOCATION}'")
            except Exception as e:
                print(f"ERROR: Failed to initialize Vertex AI: {e}")
                print(
                    "Check your Project ID, Location, Authentication, and API enablement.")
                self.vertex_ai_available = False
        elif not aiplatform:
            print("Skipping Vertex AI initialization (library not found).")
        else:
            print("Skipping Vertex AI initialization (Project ID not set).")

        # --- Game State ---
        self.player = Character("Hero", 100, 15, self.update_display)
        self.enemy = Character("Goblin", 50, 8, self.update_display)
        self.game_over = False

        # --- GUI Elements ---
        # Setup frames first so log exists even if image gen fails
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
            log_frame, wrap=tk.WORD, width=60, height=15, state='disabled')  # Increased height
        self.message_log.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # --- Image Generation ---
        self.hero_photo_image = None
        self.goblin_photo_image = None

        if self.vertex_ai_available:
            self.log_message(
                "Attempting to generate images via Vertex AI Imagen...")
            self.log_message(
                "WARNING: This will take time and may freeze the application temporarily.")
            self.root.update()  # Force GUI update to show message

            # Define prompts
            hero_prompt = "Pixel art style heroic knight character, facing forward, simple background"
            goblin_prompt = "Pixel art style grumpy green goblin warrior with a wooden club, fantasy rpg"

            # Generate Hero Image
            self.log_message(
                f"Generating Hero image with prompt: '{hero_prompt}'...")
            self.root.update()
            start_time = time.time()
            hero_bytes = self.generate_image_with_imagen(hero_prompt)
            if hero_bytes:
                self.log_message(
                    f"Hero image generated successfully ({time.time() - start_time:.2f}s). Loading...")
                self.hero_photo_image = self.load_image_from_bytes(
                    hero_bytes, (100, 100))
            else:
                self.log_message("Failed to generate Hero image.")
            self.root.update()

            # Generate Goblin Image
            self.log_message(
                f"Generating Goblin image with prompt: '{goblin_prompt}'...")
            self.root.update()
            start_time = time.time()
            goblin_bytes = self.generate_image_with_imagen(goblin_prompt)
            if goblin_bytes:
                self.log_message(
                    f"Goblin image generated successfully ({time.time() - start_time:.2f}s). Loading...")
                self.goblin_photo_image = self.load_image_from_bytes(
                    goblin_bytes, (100, 100))
            else:
                self.log_message("Failed to generate Goblin image.")
            self.root.update()

        else:
            self.log_message(
                "Image generation skipped (Vertex AI not available or configured).")
            # Optionally load fallback placeholder images here if needed

        # --- Continue GUI Setup (Player/Enemy Frames, Buttons) ---
        # (Layout code is the same as the previous version with static images,
        # it just uses self.hero_photo_image and self.goblin_photo_image which
        # might be None if generation failed)

        # == Player Info Frame ==
        player_frame = ttk.LabelFrame(main_frame, text="Player", padding="10")
        player_frame.grid(row=0, column=0, sticky=(
            tk.W, tk.E, tk.N, tk.S), pady=5, padx=5)
        player_frame.columnconfigure(1, weight=1)
        self.player_image_label = ttk.Label(
            player_frame, text="No Image")  # Placeholder text
        if self.hero_photo_image:
            self.player_image_label.config(
                image=self.hero_photo_image, text="")  # Display image if loaded
        self.player_image_label.grid(
            row=0, column=0, columnspan=3, pady=(0, 10), sticky=tk.N)
        ttk.Label(player_frame, text=f"{self.player.name}:").grid(
            row=1, column=0, sticky=tk.W)
        self.player_hp_label = ttk.Label(
            player_frame, text=f"{self.player.hp}/{self.player.max_hp} HP")
        self.player_hp_label.grid(row=1, column=2, sticky=tk.E)
        self.player_hp_bar = ttk.Progressbar(
            player_frame, orient='horizontal', length=150, mode='determinate', maximum=self.player.max_hp)
        self.player_hp_bar.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=5)

        # == Enemy Info Frame ==
        enemy_frame = ttk.LabelFrame(main_frame, text="Enemy", padding="10")
        enemy_frame.grid(row=0, column=1, sticky=(
            tk.W, tk.E, tk.N, tk.S), pady=5, padx=5)
        enemy_frame.columnconfigure(1, weight=1)
        self.enemy_image_label = ttk.Label(
            enemy_frame, text="No Image")  # Placeholder text
        if self.goblin_photo_image:
            self.enemy_image_label.config(
                image=self.goblin_photo_image, text="")  # Display image if loaded
        self.enemy_image_label.grid(
            row=0, column=0, columnspan=3, pady=(0, 10), sticky=tk.N)
        ttk.Label(enemy_frame, text=f"{self.enemy.name}:").grid(
            row=1, column=0, sticky=tk.W)
        self.enemy_hp_label = ttk.Label(
            enemy_frame, text=f"{self.enemy.hp}/{self.enemy.max_hp} HP")
        self.enemy_hp_label.grid(row=1, column=2, sticky=tk.E)
        self.enemy_hp_bar = ttk.Progressbar(
            enemy_frame, orient='horizontal', length=150, mode='determinate', maximum=self.enemy.max_hp)
        self.enemy_hp_bar.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=5)

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
        self.update_display()
        self.log_message(f"--- Encounter Start ---")
        # No longer logging specific loaded files
        # self.log_message(f"A wild {self.enemy.name} appears!") # Already logged if gen skipped

    # --- Image Generation Helper ---
    def generate_image_with_imagen(self, prompt_text):
        """Generates an image using Vertex AI Imagen and returns image bytes."""
        if not self.vertex_ai_available:
            self.log_message("Vertex AI not available, cannot generate image.")
            return None

        try:
            # Ensure model name uses the correct format if needed, e.g. full resource path
            # model = aiplatform.gapic.ModelServiceClient().get_model(name=f"projects/{self.PROJECT_ID}/locations/{self.LOCATION}/models/{self.IMAGE_MODEL_NAME}") # Might not be needed

            model = ImageGenerationModel.from_pretrained(
                self.IMAGE_MODEL_NAME)

            # API parameters might change - check documentation
            # See: https://cloud.google.com/vertex-ai/docs/generative-ai/image/generate-images
            response = model.generate_images(
                prompt=prompt_text,
                number_of_images=1,  # Generate one image
                # Add other parameters like seed, aspect_ratio, style_preset etc. if needed
                # Example: generation_params={"sample_count": 1, "aspect_ratio": "1:1"},
            )

            if response and response.images:
                # Access image bytes directly if available
                # The exact structure of 'response' might vary slightly depending on SDK version
                image_obj = response.images[0]
                # Newer SDK versions might have image_obj._image_bytes
                if hasattr(image_obj, '_image_bytes') and image_obj._image_bytes:
                    return image_obj._image_bytes
                else:
                    # Fallback or older method might involve saving/loading, less ideal
                    # Or accessing a different attribute if the structure changed
                    self.log_message(
                        "ERROR: Could not extract image bytes from response.")
                    print("Response structure:", response)  # Debugging
                    return None
            else:
                self.log_message(
                    f"ERROR: Imagen API call failed or returned no images for prompt: {prompt_text}")
                print("API Response:", response)  # Debugging
                return None

        except Exception as e:
            self.log_message(f"ERROR during image generation: {e}")
            print(f"Exception details: {e}")  # Debugging
            return None

    # --- Image Loading from Bytes Helper ---
    def load_image_from_bytes(self, image_bytes, size):
        """Loads image data from bytes, resizes, returns PhotoImage."""
        if not image_bytes:
            return None
        try:
            img_data = io.BytesIO(image_bytes)  # Wrap bytes
            original_image = Image.open(img_data)
            resized_image = original_image.resize(
                size, Image.Resampling.LANCZOS)
            # Keep a reference!
            return ImageTk.PhotoImage(resized_image)
        except Exception as e:
            self.log_message(
                f"ERROR: Failed to load image from generated bytes: {e}")
            print(f"Pillow/Tkinter loading error: {e}")
            return None

    # --- Logging Function ---
    def log_message(self, message):
        """Adds a message to the scrollable text log."""
        # If message_log hasn't been created yet (e.g., during early init errors), print instead
        if hasattr(self, 'message_log'):
            self.message_log.config(state='normal')
            self.message_log.insert(tk.END, message + "\n")
            self.message_log.config(state='disabled')
            self.message_log.see(tk.END)
            # self.root.update_idletasks() # Avoid too many updates during generation
        else:
            print(message)

    # --- Update Display Function --- (No changes needed)

    def update_display(self):
        # (Same as before)
        self.player_hp_label.config(
            text=f"{self.player.hp}/{self.player.max_hp} HP")
        self.player_hp_bar['value'] = self.player.hp
        self.enemy_hp_label.config(
            text=f"{self.enemy.hp}/{self.enemy.max_hp} HP")
        self.enemy_hp_bar['value'] = self.enemy.hp
        if not self.game_over:
            if not self.player.is_alive():
                self.end_game(f"You were defeated by the {self.enemy.name}...")
            elif not self.enemy.is_alive():
                self.end_game(
                    f"Congratulations! You defeated the {self.enemy.name}!")

    # --- Player Action Functions --- (No changes needed)
    def player_attack(self):
        # (Same as before)
        if self.game_over:
            return
        self.disable_buttons()
        self.player.reset_defense()
        self.player.attack(self.enemy, self.log_message)
        if self.enemy.is_alive() and not self.game_over:
            self.root.after(500, self.enemy_turn)
        else:
            if not self.game_over:
                self.enable_buttons()

    def player_defend(self):
        # (Same as before)
        if self.game_over:
            return
        self.disable_buttons()
        self.player.defend(self.log_message)
        if self.enemy.is_alive() and not self.game_over:
            self.root.after(500, self.enemy_turn)
        else:
            if not self.game_over:
                self.enable_buttons()

    # --- Enemy Turn --- (No changes needed)
    def enemy_turn(self):
        # (Same as before)
        if self.game_over:
            return
        self.log_message(f"\n--- {self.enemy.name}'s Turn ---")
        self.enemy.attack(self.player, self.log_message)
        if not self.game_over:
            self.enable_buttons()
            self.player.reset_defense()

    # --- Button State Control --- (No changes needed)
    def disable_buttons(self):
        self.attack_button.config(state='disabled')
        self.defend_button.config(state='disabled')

    def enable_buttons(self):
        if not self.game_over:
            self.attack_button.config(state='normal')
            self.defend_button.config(state='normal')

    # --- End Game --- (No changes needed)
    def end_game(self, message):
        # (Same as before)
        self.log_message("\n--- Combat Over ---")
        self.log_message(message)
        self.game_over = True
        self.disable_buttons()


# --- Main Execution ---
if __name__ == "__main__":
    # Basic check before starting GUI
    if not aiplatform:
        print("\nCRITICAL ERROR: google-cloud-aiplatform library is required for image generation.")
        print("Install it using: pip install google-cloud-aiplatform")
        # Optionally, show a simple Tkinter error window here
        # exit() # Exit if generation is absolutely required

    root = tk.Tk()
    # Add error handling in case RPG_GUI init fails massively
    try:
        app = RPG_GUI(root)
        root.mainloop()
    except Exception as main_error:
        print(f"FATAL ERROR during GUI initialization: {main_error}")
        # Potentially show a Tkinter error dialog here as well
        # messagebox.showerror("Fatal Error", f"Could not start the application:\n{main_error}")
