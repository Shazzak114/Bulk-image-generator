import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import requests
import threading
import os
import time
from datetime import datetime
import json
import re
import urllib.parse
from PIL import Image, ImageTk, ImageDraw, ImageFilter
import io
import random
import string
import base64
from difflib import SequenceMatcher
import numpy as np
import cv2
from collections import defaultdict

class AutoReferenceSystem:
    def __init__(self, config_file="auto_references.json"):
        self.config_file = config_file
        self.references = {}  # character_name -> reference_data
        self.character_prompts = defaultdict(list)  # character_name -> list of prompts
        self.load_references()
        
    def load_references(self):
        """Load references from file"""
        try:
            with open(self.config_file, 'r') as f:
                data = json.load(f)
                self.references = data.get("references", {})
                self.character_prompts = defaultdict(list, data.get("character_prompts", {}))
        except:
            self.references = {}
            self.character_prompts = defaultdict(list)
    
    def save_references(self):
        """Save references to file"""
        data = {
            "references": self.references,
            "character_prompts": dict(self.character_prompts)
        }
        with open(self.config_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def extract_character_names(self, prompt):
        """Extract potential character names from prompt"""
        # Look for proper nouns (capitalized words not at the beginning of sentences)
        words = prompt.split()
        character_names = []
        
        # Common words to exclude
        exclude = {"The", "A", "An", "And", "But", "Or", "For", "Nor", "As", "At", 
                  "By", "In", "Of", "On", "To", "With", "She", "He", "It", "They",
                  "Her", "His", "Its", "Their", "My", "Your", "Our", "This", "That",
                  "These", "Those", "There", "Here", "When", "Where", "Why", "How",
                  "What", "Which", "Who", "Whom", "Whose", "I", "You", "We", "Us"}
        
        for i, word in enumerate(words):
            # Skip if it's the first word or in exclude list
            if i == 0 or word in exclude or len(word) < 3:
                continue
                
            # Check if it's capitalized (potential proper noun)
            if word[0].isupper() and not word.isupper():
                # Check if it appears multiple times in the prompt
                count = words.count(word)
                if count >= 2 or word.endswith(('a', 'e', 'i', 'o', 'u', 'y')):  # Common name endings
                    character_names.append(word)
        
        # Also look for patterns like "named X" or "called X"
        named_patterns = re.findall(r'(?:named|called|known as)\s+(\w+)', prompt, re.IGNORECASE)
        character_names.extend([name for name in named_patterns if name not in exclude])
        
        return list(set(character_names))  # Remove duplicates
    
    def add_reference(self, image_data, prompt, character_names):
        """Add a new reference for each character"""
        for name in character_names:
            # Convert image to base64
            img_buffer = io.BytesIO()
            image = Image.open(io.BytesIO(image_data))
            image.save(img_buffer, format="PNG")
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
            
            # Store reference
            if name not in self.references:
                self.references[name] = {
                    "name": name,
                    "images": [],
                    "descriptions": [],
                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
            
            self.references[name]["images"].append(img_base64)
            self.references[name]["descriptions"].append(prompt)
            self.character_prompts[name].append(prompt)
            
            # Keep only the last 5 images per character to avoid bloat
            if len(self.references[name]["images"]) > 5:
                self.references[name]["images"] = self.references[name]["images"][-5:]
                self.references[name]["descriptions"] = self.references[name]["descriptions"][-5:]
        
        self.save_references()
    
    def find_references(self, prompt):
        """Find references for characters mentioned in the prompt"""
        character_names = self.extract_character_names(prompt)
        references = {}
        
        for name in character_names:
            if name in self.references:
                references[name] = self.references[name]
        
        return references
    
    def get_reference_prompt(self, character_name):
        """Get a detailed prompt for a character reference"""
        if character_name not in self.references:
            return ""
        
        ref = self.references[character_name]
        if not ref["descriptions"]:
            return ""
        
        # Combine all descriptions for this character
        all_descriptions = " ".join(ref["descriptions"])
        
        # Extract key features
        features = []
        
        # Look for appearance descriptions
        appearance_patterns = [
            r'(\w+)\s+(hair|eyes|skin|face|body|build|figure)',
            r'(wearing|dressed in|has|with)\s+(\w+\s+\w+|\w+)',
            r'(tall|short|slim|athletic|muscular|curvy|slender)',
            r'(long|short|curly|straight|wavy|blonde|brunette|redhead|black)\s+hair',
            r'(blue|green|brown|hazel|gray|black)\s+eyes',
            r'(fair|olive|tan|dark|pale)\s+skin'
        ]
        
        for pattern in appearance_patterns:
            matches = re.findall(pattern, all_descriptions, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    features.extend(match)
                else:
                    features.append(match)
        
        # Create a reference prompt
        ref_prompt = f"Character named {character_name}"
        if features:
            unique_features = list(set(features))
            ref_prompt += f" with {', '.join(unique_features[:10])}"  # Limit to top 10 features
        
        return ref_prompt

class ImageQualityChecker:
    @staticmethod
    def check_image_quality(image_data):
        """Check if the generated image is of good quality"""
        try:
            # Convert to PIL Image
            image = Image.open(io.BytesIO(image_data))
            
            # Check if image is too small
            width, height = image.size
            if width < 256 or height < 256:
                return False, "Image too small"
            
            # Convert to numpy array for analysis
            img_array = np.array(image)
            
            # Check for blurriness using Laplacian variance
            if len(img_array.shape) == 3:  # RGB image
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:  # Already grayscale
                gray = img_array
                
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            if laplacian_var < 100:  # Threshold for blurriness
                return False, "Image too blurry"
            
            # Check for extreme brightness or darkness
            mean_brightness = np.mean(gray)
            if mean_brightness < 40 or mean_brightness > 215:
                return False, "Image too bright or too dark"
            
            # Check for low contrast
            contrast = gray.std()
            if contrast < 40:
                return False, "Image has low contrast"
            
            # Check for incomplete human figures (simplified)
            # This is a basic check and might not be 100% accurate
            try:
                # Use edge detection to check for human-like shapes
                edges = cv2.Canny(gray, 50, 150)
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Filter contours by size
                significant_contours = [c for c in contours if cv2.contourArea(c) > 500]
                
                if len(significant_contours) < 3:
                    return False, "Image may have incomplete figure"
            except:
                # If contour detection fails, skip this check
                pass
            
            return True, "Image quality OK"
            
        except Exception as e:
            print(f"Error checking image quality: {e}")
            return False, f"Error checking quality: {str(e)}"

class AdvancedImageGenerator:
    def __init__(self, root):
        self.root = root
        self.root.title("🚀 Advanced AI Image Generator Pro")
        self.root.geometry("1200x900")
        
        # Gun Metal Color Scheme
        self.bg_color = "#2C2C2C"        # Gun metal background
        self.fg_color = "#E0E0E0"        # Light text
        self.accent_color = "#4A90E2"    # Blue accent
        self.button_bg = "#3C3C3C"        # Button background
        self.hover_bg = "#4A4A4A"         # Hover color
        self.entry_bg = "#3A3A3A"         # Entry background
        self.border_color = "#444444"    # Border color
        
        self.root.configure(bg=self.bg_color)
        
        # Create a scrollable frame for the entire interface
        self.canvas = tk.Canvas(self.root, bg=self.bg_color, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas, bg=self.bg_color)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        # Pack the canvas and scrollbar
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        
        # Configuration
        self.config_file = "image_generator_pro_config.json"
        self.load_config()
        
        # Internal platform (only Pollinations)
        self.platform = "Pollinations.AI (Free)"
        self.platform_info = {
            "api_url": "https://pollinations.ai/p/{prompt}",
            "cost": "Free",
            "max_size": "2048x2048",
            "quality_options": ["Standard", "HD", "Ultra HD"],
            "nologo_supported": True,
            "api_key_required": False
        }
        
        # Auto-enhancement templates
        self.quality_enhancers = [
            "masterpiece, best quality, ultra-detailed, high resolution, 8k",
            "photorealistic, ultra realistic, professional photography, dramatic lighting",
            "highly detailed, intricate details, sharp focus, vibrant colors",
            "cinematic lighting, hyperrealistic, ultra HD, stunning detail",
            "award-winning photography, perfect composition, breathtaking view"
        ]
        
        # Sensitive language conversion dictionary
        self.sensitive_language_dict = {
            # Nudity and body parts
            "nude": "unclothed",
            "naked": "without clothes",
            "bare": "uncovered",
            "topless": "without upper clothing",
            "bottomless": "without lower clothing",
            "boobs": "chest",
            "breasts": "chest",
            "tits": "chest",
            "boob": "chest",
            "nipple": "chest area",
            "nipples": "chest area",
            "butt": "figure",
            "ass": "figure",
            "bum": "figure",
            "booty": "figure",
            "panty": "underwear",
            "panties": "underwear",
            "bra": "underwear",
            "lingerie": "underwear",
            "thong": "underwear",
            "bikini": "swimwear",
            "vagina": "private part",
            "pussy": "private part",
            "cunt": "private part",
            "clit": "sensitive area",
            "clitoris": "sensitive area",
            "labia": "sensitive area",
            "hairy vagina": "natural private part",
            "hairy pussy": "natural private part",
            "shaved pussy": "smooth private part",
            "shaved vagina": "smooth private part",
            
            # Sexual terms
            "sexy": "attractive",
            "hot": "passionate",
            "seductive": "alluring",
            "erotic": "sensual",
            "arousing": "stimulating",
            "horny": "excited",
            "lust": "desire",
            "lustful": "desirous",
            "orgasm": "climax",
            "cum": "finish",
            "semen": "fluid",
            "sperm": "fluid",
            "ejaculate": "release",
            "hardcore": "intense",
            "softcore": "mild",
            "porn": "adult content",
            "porno": "adult content",
            "pornography": "adult content",
            "xxx": "adult content",
            
            # Actions and positions
            "fucking": "intimate",
            "sex": "intimacy",
            "sexual": "intimate",
            "intercourse": "connection",
            "penetration": "entry",
            "oral": "stimulation",
            "blowjob": "stimulation",
            "handjob": "stimulation",
            "masturbation": "self-pleasure",
            "masturbating": "self-pleasuring",
            "riding": "straddling",
            "doggystyle": "from behind",
            "missionary": "facing each other",
            "cowgirl": "on top",
            "reverse cowgirl": "on top facing away",
            "spooning": "cuddling",
            "69": "mutual stimulation",
            
            # Violence and gore
            "blood": "red liquid",
            "bloody": "covered in red liquid",
            "gore": "graphic content",
            "gory": "graphic",
            "kill": "defeat",
            "murder": "eliminate",
            "death": "end of life",
            "die": "cease to live",
            "dead": "lifeless",
            "corpse": "body",
            "stab": "pierce",
            "stabbing": "piercing",
            "shoot": "fire at",
            "shooting": "firing",
            "gun": "weapon",
            "knife": "blade",
            "sword": "weapon",
            "violence": "conflict",
            "violent": "aggressive",
            "torture": "inflict pain",
            "torturing": "inflicting pain",
            "rape": "non-consensual",
            "raping": "non-consensual act",
            "assault": "attack",
            "attacking": "assaulting",
            
            # Other sensitive terms
            "drug": "substance",
            "drugs": "substances",
            "cocaine": "white powder",
            "heroin": "brown substance",
            "meth": "crystal",
            "marijuana": "herb",
            "weed": "plant",
            "pot": "plant",
            "lsd": "acid",
            "ecstasy": "pill",
            "overdose": "excessive use",
            "suicide": "self-harm",
            "suicidal": "self-destructive",
            "self-harm": "self-injury",
            "cutting": "self-injury",
            "bulimia": "eating disorder",
            "anorexia": "eating disorder",
            "self-hatred": "negative self-image",
            
            # General adult terms
            "18+": "mature",
            "adult": "mature",
            "mature": "grown-up",
            "nsfw": "sensitive",
            "not safe for work": "sensitive",
            "explicit": "detailed",
            "taboo": "unconventional",
            "forbidden": "restricted",
            "illegal": "prohibited",
            "underage": "minor",
            "minor": "young person",
            "teen": "young adult",
            "teenager": "young adult"
        }
        
        # Initialize systems
        self.auto_reference_system = AutoReferenceSystem()
        self.quality_checker = ImageQualityChecker()
        
        # Saved prompts
        self.saved_prompts_file = "saved_prompts.json"
        self.load_saved_prompts()
        
        # Image references
        self.image_references_file = "image_references.json"
        self.load_image_references()
        
        self.setup_ui()
        
    def load_config(self):
        """Load configuration from file"""
        default_config = {
            "output_dir": "generated_images",
            "default_quality": "Standard",
            "default_size": "1024x1024",
            "auto_enhance": True,
            "remove_watermark": True,
            "enhancement_template": 0,
            "adult_content": False,
            "retry_attempts": 5,
            "character_similarity_threshold": 0.6,
            "quality_check": True,
            "auto_reference": True
        }
        
        try:
            with open(self.config_file, 'r') as f:
                loaded_config = json.load(f)
                # Update with default values for missing keys
                for key, value in default_config.items():
                    if key not in loaded_config:
                        loaded_config[key] = value
                self.config = loaded_config
        except:
            self.config = default_config
            
        # Create output directory if it doesn't exist
        os.makedirs(self.config["output_dir"], exist_ok=True)
    
    def save_config(self):
        """Save configuration to file"""
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def load_saved_prompts(self):
        """Load saved prompts from file"""
        try:
            with open(self.saved_prompts_file, 'r') as f:
                self.saved_prompts = json.load(f)
        except:
            self.saved_prompts = []
    
    def save_saved_prompts(self):
        """Save prompts to file"""
        with open(self.saved_prompts_file, 'w') as f:
            json.dump(self.saved_prompts, f, indent=2)
    
    def load_image_references(self):
        """Load image references from file"""
        try:
            with open(self.image_references_file, 'r') as f:
                self.image_references = json.load(f)
        except:
            self.image_references = []
    
    def save_image_references(self):
        """Save image references to file"""
        with open(self.image_references_file, 'w') as f:
            json.dump(self.image_references, f, indent=2)
    
    def setup_ui(self):
        """Setup the enhanced user interface"""
        # Main container with modern styling
        main_frame = tk.Frame(self.scrollable_frame, bg=self.bg_color)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Title with gradient effect
        title_frame = tk.Frame(main_frame, bg=self.bg_color)
        title_frame.pack(fill=tk.X, pady=(0, 20))
        
        title_label = tk.Label(title_frame, text="🚀 Advanced AI Image Generator Pro", 
                              font=("Arial", 24, "bold"), bg=self.bg_color, fg=self.fg_color)
        title_label.pack()
        
        subtitle_label = tk.Label(title_frame, text="Powered by SZK Creation", 
                                 font=("Arial", 12), bg=self.bg_color, fg=self.accent_color)
        subtitle_label.pack()
        
        # Main content area
        content_frame = tk.Frame(main_frame, bg=self.bg_color)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left panel - Controls
        control_panel = tk.Frame(content_frame, bg=self.button_bg, relief=tk.RAISED, bd=1)
        control_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 20))
        
        # Panel header
        panel_header = tk.Frame(control_panel, bg=self.accent_color, height=40)
        panel_header.pack(fill=tk.X)
        panel_header.pack_propagate(False)
        
        header_label = tk.Label(panel_header, text="Control Panel", 
                               font=("Arial", 14, "bold"), bg=self.accent_color, fg="white")
        header_label.pack(pady=8)
        
        # Control panel content
        controls_container = tk.Frame(control_panel, bg=self.button_bg, padx=15, pady=15)
        controls_container.pack(fill=tk.BOTH, expand=True)
        
        # Size Selection with visualization
        size_frame = tk.Frame(controls_container, bg=self.button_bg)
        size_frame.pack(fill=tk.X, pady=(0, 15))
        
        tk.Label(size_frame, text="📱 Size:", font=("Arial", 11, "bold"), 
                bg=self.button_bg, fg=self.fg_color).pack(anchor=tk.W)
        
        self.size_var = tk.StringVar(value=self.config["default_size"])
        size_options = ["512x512", "768x768", "1024x1024", "1440x1440", "1080x1920 (Mobile)", "1920x1080", "1792x1024", "1024x1792"]
        size_combo = ttk.Combobox(size_frame, textvariable=self.size_var, values=size_options, 
                                 state="readonly", width=18)
        size_combo.pack(fill=tk.X, pady=(5, 0))
        size_combo.bind('<<ComboboxSelected>>', self.update_size_visualization)
        
        # Size visualization canvas
        self.size_canvas = tk.Canvas(size_frame, width=120, height=80, bg=self.entry_bg, highlightthickness=0)
        self.size_canvas.pack(pady=(10, 0))
        self.update_size_visualization()
        
        # Options
        options_frame = tk.LabelFrame(controls_container, text="Options", 
                                     font=("Arial", 11, "bold"), bg=self.button_bg, 
                                     fg=self.fg_color, bd=1, relief=tk.SOLID)
        options_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Auto-enhance checkbox
        self.auto_enhance_var = tk.BooleanVar(value=self.config["auto_enhance"])
        auto_check = tk.Checkbutton(options_frame, text="Auto-enhance prompts", 
                                    variable=self.auto_enhance_var, bg=self.button_bg, 
                                    fg=self.fg_color, selectcolor=self.button_bg,
                                    activebackground=self.button_bg, activeforeground=self.fg_color)
        auto_check.pack(anchor=tk.W, padx=10, pady=5)
        
        # Remove watermark checkbox
        self.remove_watermark_var = tk.BooleanVar(value=self.config["remove_watermark"])
        watermark_check = tk.Checkbutton(options_frame, text="Remove watermark/logo", 
                                        variable=self.remove_watermark_var, bg=self.button_bg, 
                                        fg=self.fg_color, selectcolor=self.button_bg,
                                        activebackground=self.button_bg, activeforeground=self.fg_color)
        watermark_check.pack(anchor=tk.W, padx=10, pady=(0, 5))
        
        # Adult content checkbox
        self.adult_content_var = tk.BooleanVar(value=self.config.get("adult_content", False))
        adult_check = tk.Checkbutton(options_frame, text="Sensitive content", 
                                    variable=self.adult_content_var, bg=self.button_bg, 
                                    fg=self.fg_color, selectcolor=self.button_bg,
                                    activebackground=self.button_bg, activeforeground=self.fg_color)
        adult_check.pack(anchor=tk.W, padx=10, pady=(0, 5))
        
        # Auto reference checkbox
        self.auto_reference_var = tk.BooleanVar(value=self.config.get("auto_reference", True))
        auto_ref_check = tk.Checkbutton(options_frame, text="Auto character reference", 
                                       variable=self.auto_reference_var, bg=self.button_bg, 
                                       fg=self.fg_color, selectcolor=self.button_bg,
                                       activebackground=self.button_bg, activeforeground=self.fg_color)
        auto_ref_check.pack(anchor=tk.W, padx=10, pady=(0, 5))
        
        # Quality check checkbox
        self.quality_check_var = tk.BooleanVar(value=self.config.get("quality_check", True))
        quality_check = tk.Checkbutton(options_frame, text="Quality check", 
                                      variable=self.quality_check_var, bg=self.button_bg, 
                                      fg=self.fg_color, selectcolor=self.button_bg,
                                      activebackground=self.button_bg, activeforeground=self.fg_color)
        quality_check.pack(anchor=tk.W, padx=10, pady=(0, 5))
        
        # Enhancement template selection
        enhance_frame = tk.Frame(options_frame, bg=self.button_bg)
        enhance_frame.pack(fill=tk.X, padx=10, pady=(0, 5))
        
        tk.Label(enhance_frame, text="Enhancement Style:", bg=self.button_bg, 
                fg=self.fg_color).pack(side=tk.LEFT)
        
        self.enhancement_var = tk.IntVar(value=self.config["enhancement_template"])
        enhancement_combo = ttk.Combobox(enhance_frame, textvariable=self.enhancement_var, 
                                       values=list(range(len(self.quality_enhancers))), 
                                       state="readonly", width=5)
        enhancement_combo.pack(side=tk.LEFT, padx=(5, 0))
        
        # Preview enhancement template
        self.preview_label = tk.Label(options_frame, text="", font=("Arial", 9), 
                                    fg=self.accent_color, bg=self.button_bg, 
                                    wraplength=200, justify=tk.LEFT)
        self.preview_label.pack(anchor=tk.W, padx=10, pady=(0, 5))
        self.update_enhancement_preview()
        enhancement_combo.bind('<<ComboboxSelected>>', lambda e: self.update_enhancement_preview())
        
        # Buttons
        button_frame = tk.Frame(controls_container, bg=self.button_bg)
        button_frame.pack(fill=tk.X, pady=15)
        
        # Generate button with special styling
        self.generate_btn = tk.Button(button_frame, text="🚀 Generate Images", 
                                     command=self.generate_images, bg=self.accent_color, 
                                     fg="white", font=("Arial", 11, "bold"),
                                     relief=tk.FLAT, padx=15, pady=8, cursor="hand2")
        self.generate_btn.pack(fill=tk.X, pady=(0, 10))
        
        # Add hover effect
        self.generate_btn.bind("<Enter>", lambda e: self.generate_btn.config(bg="#5A9FEF"))
        self.generate_btn.bind("<Leave>", lambda e: self.generate_btn.config(bg=self.accent_color))
        
        # Other buttons
        buttons = [
            ("🗑️ Clear Prompts", self.clear_prompts),
            ("📁 Browse Output", self.browse_output),
            ("💾 Save Prompts", self.open_save_prompt_dialog),
            ("📋 Saved Prompts", self.open_prompts_manager),
            ("🖼️ Image References", self.open_image_reference_manager),
            ("👥 Auto References", self.open_auto_reference_manager),
            ("⚙️ Settings", self.open_settings)
        ]
        
        for text, command in buttons:
            btn = tk.Button(button_frame, text=text, command=command, 
                           bg=self.hover_bg, fg=self.fg_color, 
                           font=("Arial", 10), relief=tk.FLAT, 
                           padx=10, pady=6, cursor="hand2")
            btn.pack(fill=tk.X, pady=(0, 5))
            
            # Add hover effect
            btn.bind("<Enter>", lambda e, b=btn: b.config(bg=self.bg_color))
            btn.bind("<Leave>", lambda e, b=btn: b.config(bg=self.hover_bg))
        
        # Right panel - Prompt input and preview
        right_panel = tk.Frame(content_frame, bg=self.bg_color)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Prompt Input
        prompt_frame = tk.LabelFrame(right_panel, text="✍️ Prompts (one per line, empty line = next image)", 
                                    font=("Arial", 12, "bold"), bg=self.bg_color, 
                                    fg=self.fg_color, bd=1, relief=tk.SOLID)
        prompt_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        
        self.prompt_text = scrolledtext.ScrolledText(prompt_frame, height=15, width=70, 
                                                    font=("Arial", 11), bg=self.entry_bg, 
                                                    fg=self.fg_color, relief=tk.FLAT,
                                                    wrap=tk.WORD, padx=10, pady=10)
        self.prompt_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Placeholder info
        placeholder_frame = tk.Frame(prompt_frame, bg=self.bg_color)
        placeholder_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        tk.Label(placeholder_frame, text="🎯 Placeholders: {style}, {mood}, {subject}, {background}, {lighting}", 
                font=("Arial", 10, "italic"), bg=self.bg_color, fg=self.accent_color).pack(anchor=tk.W)
        
        # Reference info
        reference_frame = tk.Frame(prompt_frame, bg=self.bg_color)
        reference_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        self.reference_label = tk.Label(reference_frame, text="🖼️ No image references added", 
                                       font=("Arial", 10, "italic"), bg=self.bg_color, 
                                       fg=self.accent_color)
        self.reference_label.pack(anchor=tk.W)
        
        # Auto reference info
        self.auto_ref_label = tk.Label(reference_frame, text="👥 Auto references: None", 
                                      font=("Arial", 10, "italic"), bg=self.bg_color, 
                                      fg=self.accent_color)
        self.auto_ref_label.pack(anchor=tk.W)
        
        # Preview area
        preview_frame = tk.LabelFrame(right_panel, text="🖼️ Recent Images", 
                                     font=("Arial", 12, "bold"), bg=self.bg_color, 
                                     fg=self.fg_color, bd=1, relief=tk.SOLID)
        preview_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.preview_canvas = tk.Canvas(preview_frame, height=150, bg=self.entry_bg, 
                                       highlightthickness=0)
        self.preview_canvas.pack(fill=tk.X, padx=10, pady=10)
        
        # Status Bar
        status_frame = tk.Frame(right_panel, bg=self.button_bg, relief=tk.SUNKEN, bd=1)
        status_frame.pack(fill=tk.X)
        
        self.status_var = tk.StringVar(value="Ready")
        status_bar = tk.Label(status_frame, textvariable=self.status_var, 
                             bg=self.button_bg, fg=self.fg_color, 
                             font=("Arial", 10), anchor=tk.W, padx=10, pady=5)
        status_bar.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Cost estimator
        self.cost_var = tk.StringVar(value="Cost: Free")
        cost_label = tk.Label(status_frame, textvariable=self.cost_var, 
                             bg=self.button_bg, fg=self.accent_color, 
                             font=("Arial", 10, "bold"), padx=10, pady=5)
        cost_label.pack(side=tk.RIGHT)
        
        # Progress Bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(right_panel, variable=self.progress_var, 
                                           maximum=100, length=400, mode='determinate')
        self.progress_bar.pack(fill=tk.X, pady=(0, 10))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)
        content_frame.columnconfigure(1, weight=1)
        content_frame.rowconfigure(0, weight=1)
    
    def update_size_visualization(self, event=None):
        """Update the size visualization canvas"""
        self.size_canvas.delete("all")
        
        size_str = self.size_var.get()
        if "x" in size_str:
            size_str = size_str.replace(" (Mobile)", "")
            width, height = map(int, size_str.split("x"))
        else:
            width, height = 1024, 1024
        
        # Calculate aspect ratio
        aspect_ratio = width / height
        
        # Canvas dimensions
        canvas_width = 120
        canvas_height = 80
        
        # Calculate rectangle dimensions to fit in canvas
        if aspect_ratio > 1:  # Landscape
            rect_width = canvas_width - 20
            rect_height = rect_width / aspect_ratio
        else:  # Portrait or square
            rect_height = canvas_height - 20
            rect_width = rect_height * aspect_ratio
        
        # Center the rectangle
        x1 = (canvas_width - rect_width) / 2
        y1 = (canvas_height - rect_height) / 2
        x2 = x1 + rect_width
        y2 = y1 + rect_height
        
        # Draw rectangle
        self.size_canvas.create_rectangle(x1, y1, x2, y2, outline=self.accent_color, width=2)
        
        # Add size text
        self.size_canvas.create_text(canvas_width/2, canvas_height/2, 
                                   text=f"{width}x{height}", 
                                   fill=self.fg_color, font=("Arial", 9))
        
        # Add view angle indicator
        if width > height:
            view_text = "Landscape View"
        elif height > width:
            view_text = "Portrait View"
        else:
            view_text = "Square View"
        
        self.size_canvas.create_text(canvas_width/2, 10, 
                                   text=view_text, 
                                   fill=self.accent_color, font=("Arial", 8))
    
    def update_enhancement_preview(self):
        """Update the enhancement template preview"""
        template_idx = self.enhancement_var.get()
        if 0 <= template_idx < len(self.quality_enhancers):
            self.preview_label.config(text=self.quality_enhancers[template_idx])
    
    def update_reference_label(self):
        """Update the reference label with current references"""
        if not self.image_references:
            self.reference_label.config(text="🖼️ No image references added")
        else:
            ref_names = [ref["name"] for ref in self.image_references]
            self.reference_label.config(text=f"🖼️ References: {', '.join(ref_names)}")
    
    def update_auto_ref_label(self):
        """Update the auto reference label with current references"""
        if not self.auto_reference_system.references:
            self.auto_ref_label.config(text="👥 Auto references: None")
        else:
            ref_names = list(self.auto_reference_system.references.keys())
            self.auto_ref_label.config(text=f"👥 Auto references: {', '.join(ref_names[:5])}{'...' if len(ref_names) > 5 else ''}")
    
    def clear_prompts(self):
        """Clear the prompt text area"""
        self.prompt_text.delete(1.0, tk.END)
    
    def browse_output(self):
        """Open the output directory"""
        import subprocess
        import platform
        
        output_dir = self.config["output_dir"]
        try:
            if platform.system() == "Windows":
                subprocess.run(['explorer', output_dir])
            elif platform.system() == "Darwin":  # macOS
                subprocess.run(['open', output_dir])
            else:  # Linux
                subprocess.run(['xdg-open', output_dir])
        except Exception as e:
            messagebox.showerror("Error", f"Could not open output directory: {e}")
    
    def save_current_config(self):
        """Save current configuration"""
        self.config["default_quality"] = "Standard"  # Fixed since we removed platform selection
        self.config["default_size"] = self.size_var.get()
        self.config["auto_enhance"] = self.auto_enhance_var.get()
        self.config["remove_watermark"] = self.remove_watermark_var.get()
        self.config["enhancement_template"] = self.enhancement_var.get()
        self.config["adult_content"] = self.adult_content_var.get()
        self.config["auto_reference"] = self.auto_reference_var.get()
        self.config["quality_check"] = self.quality_check_var.get()
        
        self.save_config()
        messagebox.showinfo("Success", "Configuration saved successfully!")
    
    def open_settings(self):
        """Open enhanced settings dialog"""
        settings_window = tk.Toplevel(self.root)
        settings_window.title("⚙️ Advanced Settings")
        settings_window.geometry("600x500")
        settings_window.configure(bg=self.bg_color)
        settings_window.transient(self.root)
        settings_window.grab_set()
        
        # Header
        header_frame = tk.Frame(settings_window, bg=self.accent_color, height=50)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)
        
        tk.Label(header_frame, text="Application Settings", 
                font=("Arial", 16, "bold"), bg=self.accent_color, 
                fg="white").pack(pady=12)
        
        # Settings content
        content_frame = tk.Frame(settings_window, bg=self.bg_color, padx=20, pady=20)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Output directory
        dir_frame = tk.Frame(content_frame, bg=self.bg_color)
        dir_frame.pack(fill=tk.X, pady=(0, 20))
        
        tk.Label(dir_frame, text="Output Directory:", font=("Arial", 11, "bold"), 
                bg=self.bg_color, fg=self.fg_color).pack(anchor=tk.W)
        
        output_var = tk.StringVar(value=self.config["output_dir"])
        output_entry = tk.Entry(dir_frame, textvariable=output_var, 
                              font=("Arial", 10), bg=self.entry_bg, 
                              fg=self.fg_color, relief=tk.FLAT)
        output_entry.pack(fill=tk.X, pady=(5, 0))
        
        browse_btn = tk.Button(dir_frame, text="Browse", 
                              command=lambda: self.browse_directory(output_var),
                              bg=self.hover_bg, fg=self.fg_color, 
                              font=("Arial", 10), relief=tk.FLAT, 
                              padx=10, pady=5, cursor="hand2")
        browse_btn.pack(anchor=tk.E, pady=(5, 0))
        
        # Retry attempts
        retry_frame = tk.Frame(content_frame, bg=self.bg_color)
        retry_frame.pack(fill=tk.X, pady=(0, 20))
        
        tk.Label(retry_frame, text="Retry Attempts:", font=("Arial", 11, "bold"), 
                bg=self.bg_color, fg=self.fg_color).pack(anchor=tk.W)
        
        retry_var = tk.IntVar(value=self.config.get("retry_attempts", 5))
        retry_spinbox = tk.Spinbox(retry_frame, from_=1, to=10, textvariable=retry_var, 
                                  font=("Arial", 10), bg=self.entry_bg, 
                                  fg=self.fg_color, relief=tk.FLAT)
        retry_spinbox.pack(fill=tk.X, pady=(5, 0))
        
        # Character similarity threshold
        threshold_frame = tk.Frame(content_frame, bg=self.bg_color)
        threshold_frame.pack(fill=tk.X, pady=(0, 20))
        
        tk.Label(threshold_frame, text="Character Similarity Threshold:", font=("Arial", 11, "bold"), 
                bg=self.bg_color, fg=self.fg_color).pack(anchor=tk.W)
        
        threshold_var = tk.DoubleVar(value=self.config.get("character_similarity_threshold", 0.6))
        threshold_spinbox = tk.Spinbox(threshold_frame, from_=0.1, to=1.0, increment=0.1, 
                                      textvariable=threshold_var, 
                                      font=("Arial", 10), bg=self.entry_bg, 
                                      fg=self.fg_color, relief=tk.FLAT)
        threshold_spinbox.pack(fill=tk.X, pady=(5, 0))
        
        # Enhancement Templates
        enhance_frame = tk.Frame(content_frame, bg=self.bg_color)
        enhance_frame.pack(fill=tk.BOTH, expand=True)
        
        tk.Label(enhance_frame, text="Prompt Enhancement Templates", 
                font=("Arial", 11, "bold"), bg=self.bg_color, 
                fg=self.fg_color).pack(anchor=tk.W, pady=(0, 10))
        
        template_text = scrolledtext.ScrolledText(enhance_frame, height=10, width=50, 
                                                 font=("Arial", 10), bg=self.entry_bg, 
                                                 fg=self.fg_color, relief=tk.FLAT)
        template_text.pack(fill=tk.BOTH, expand=True)
        
        # Load current templates
        for i, template in enumerate(self.quality_enhancers):
            template_text.insert(tk.END, f"Template {i+1}: {template}\n\n")
        
        def save_enhancement_templates():
            content = template_text.get(1.0, tk.END).strip()
            lines = [line.strip() for line in content.split('\n') if line.strip()]
            new_templates = []
            for line in lines:
                if line.startswith("Template"):
                    template = line.split(":", 1)[1].strip()
                    new_templates.append(template)
            
            if new_templates:
                self.quality_enhancers = new_templates
                self.update_enhancement_preview()
                messagebox.showinfo("Success", f"Saved {len(new_templates)} enhancement templates!")
        
        save_btn = tk.Button(enhance_frame, text="Save Templates", 
                            command=save_enhancement_templates,
                            bg=self.accent_color, fg="white", 
                            font=("Arial", 10, "bold"), relief=tk.FLAT, 
                            padx=15, pady=8, cursor="hand2")
        save_btn.pack(anchor=tk.E, pady=(10, 0))
        
        # Footer buttons
        footer_frame = tk.Frame(settings_window, bg=self.bg_color)
        footer_frame.pack(fill=tk.X, padx=20, pady=(0, 20))
        
        def save_all_settings():
            self.config["output_dir"] = output_var.get()
            self.config["retry_attempts"] = retry_var.get()
            self.config["character_similarity_threshold"] = threshold_var.get()
            self.save_config()
            settings_window.destroy()
            messagebox.showinfo("Success", "All settings saved successfully!")
        
        save_all_btn = tk.Button(footer_frame, text="💾 Save All Settings", 
                                command=save_all_settings,
                                bg=self.accent_color, fg="white", 
                                font=("Arial", 10, "bold"), relief=tk.FLAT, 
                                padx=15, pady=8, cursor="hand2")
        save_all_btn.pack(side=tk.RIGHT)
        
        cancel_btn = tk.Button(footer_frame, text="Cancel", 
                              command=settings_window.destroy,
                              bg=self.hover_bg, fg=self.fg_color, 
                              font=("Arial", 10), relief=tk.FLAT, 
                              padx=15, pady=8, cursor="hand2")
        cancel_btn.pack(side=tk.RIGHT, padx=(0, 10))
    
    def browse_directory(self, var):
        """Browse for directory"""
        directory = filedialog.askdirectory()
        if directory:
            var.set(directory)
    
    def open_save_prompt_dialog(self):
        """Open dialog to save current prompt"""
        prompt_content = self.prompt_text.get(1.0, tk.END).strip()
        if not prompt_content:
            messagebox.showwarning("Warning", "Please enter a prompt to save!")
            return
        
        save_window = tk.Toplevel(self.root)
        save_window.title("Save Prompt")
        save_window.geometry("500x250")
        save_window.configure(bg=self.bg_color)
        save_window.transient(self.root)
        save_window.grab_set()
        
        # Header
        header_frame = tk.Frame(save_window, bg=self.accent_color, height=50)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)
        
        tk.Label(header_frame, text="Save Prompt", 
                font=("Arial", 16, "bold"), bg=self.accent_color, 
                fg="white").pack(pady=12)
        
        # Content
        content_frame = tk.Frame(save_window, bg=self.bg_color, padx=20, pady=20)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Name field
        tk.Label(content_frame, text="Prompt Name:", font=("Arial", 11, "bold"), 
                bg=self.bg_color, fg=self.fg_color).pack(anchor=tk.W)
        
        name_var = tk.StringVar()
        name_entry = tk.Entry(content_frame, textvariable=name_var, 
                             font=("Arial", 10), bg=self.entry_bg, 
                             fg=self.fg_color, relief=tk.FLAT)
        name_entry.pack(fill=tk.X, pady=(5, 15))
        
        # Preview
        tk.Label(content_frame, text="Prompt Preview:", font=("Arial", 11, "bold"), 
                bg=self.bg_color, fg=self.fg_color).pack(anchor=tk.W)
        
        preview_text = scrolledtext.ScrolledText(content_frame, height=5, width=40, 
                                                font=("Arial", 10), bg=self.entry_bg, 
                                                fg=self.fg_color, relief=tk.FLAT)
        preview_text.pack(fill=tk.BOTH, expand=True, pady=(5, 15))
        preview_text.insert(tk.END, prompt_content)
        preview_text.config(state=tk.DISABLED)
        
        # Buttons
        button_frame = tk.Frame(content_frame, bg=self.bg_color)
        button_frame.pack(fill=tk.X)
        
        def save_prompt():
            name = name_var.get().strip()
            if not name:
                messagebox.showwarning("Warning", "Please enter a name for the prompt!")
                return
            
            # Generate a unique ID
            prompt_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
            
            # Save to list
            self.saved_prompts.append({
                "id": prompt_id,
                "name": name,
                "prompt": prompt_content,
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            
            self.save_saved_prompts()
            messagebox.showinfo("Success", "Prompt saved successfully!")
            save_window.destroy()
        
        save_btn = tk.Button(button_frame, text="💾 Save", 
                            command=save_prompt,
                            bg=self.accent_color, fg="white", 
                            font=("Arial", 10, "bold"), relief=tk.FLAT, 
                            padx=15, pady=8, cursor="hand2")
        save_btn.pack(side=tk.RIGHT)
        
        cancel_btn = tk.Button(button_frame, text="Cancel", 
                              command=save_window.destroy,
                              bg=self.hover_bg, fg=self.fg_color, 
                              font=("Arial", 10), relief=tk.FLAT, 
                              padx=15, pady=8, cursor="hand2")
        cancel_btn.pack(side=tk.RIGHT, padx=(0, 10))
    
    def open_prompts_manager(self):
        """Open prompts manager window"""
        manager_window = tk.Toplevel(self.root)
        manager_window.title("📋 Saved Prompts Manager")
        manager_window.geometry("700x500")
        manager_window.configure(bg=self.bg_color)
        manager_window.transient(self.root)
        
        # Header
        header_frame = tk.Frame(manager_window, bg=self.accent_color, height=50)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)
        
        tk.Label(header_frame, text="Saved Prompts Manager", 
                font=("Arial", 16, "bold"), bg=self.accent_color, 
                fg="white").pack(pady=12)
        
        # Content
        content_frame = tk.Frame(manager_window, bg=self.bg_color)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Search frame
        search_frame = tk.Frame(content_frame, bg=self.bg_color)
        search_frame.pack(fill=tk.X, pady=(0, 15))
        
        tk.Label(search_frame, text="Search:", font=("Arial", 11, "bold"), 
                bg=self.bg_color, fg=self.fg_color).pack(side=tk.LEFT, padx=(0, 10))
        
        search_var = tk.StringVar()
        search_entry = tk.Entry(search_frame, textvariable=search_var, 
                              font=("Arial", 10), bg=self.entry_bg, 
                              fg=self.fg_color, relief=tk.FLAT)
        search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # List frame
        list_frame = tk.Frame(content_frame, bg=self.bg_color)
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create listbox with scrollbar
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.prompts_listbox = tk.Listbox(list_frame, bg=self.entry_bg, 
                                         fg=self.fg_color, relief=tk.FLAT,
                                         selectbackground=self.accent_color,
                                         selectforeground="white",
                                         font=("Arial", 10),
                                         yscrollcommand=scrollbar.set)
        self.prompts_listbox.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.prompts_listbox.yview)
        
        # Populate listbox
        self.refresh_prompts_listbox()
        
        # Bind double-click to insert into main prompt area
        self.prompts_listbox.bind("<Double-Button-1>", self.insert_selected_prompt)
        
        # Buttons
        button_frame = tk.Frame(content_frame, bg=self.bg_color)
        button_frame.pack(fill=tk.X, pady=(15, 0))
        
        # Button functions
        def copy_selected_prompt():
            selection = self.prompts_listbox.curselection()
            if selection:
                idx = selection[0]
                if idx < len(self.saved_prompts):
                    prompt = self.saved_prompts[idx]["prompt"]
                    self.root.clipboard_clear()
                    self.root.clipboard_append(prompt)
                    messagebox.showinfo("Success", "Prompt copied to clipboard!")
        
        def edit_selected_prompt():
            selection = self.prompts_listbox.curselection()
            if selection:
                idx = selection[0]
                if idx < len(self.saved_prompts):
                    prompt_data = self.saved_prompts[idx]
                    self.open_edit_prompt_dialog(prompt_data, idx)
        
        def delete_selected_prompt():
            selection = self.prompts_listbox.curselection()
            if selection:
                idx = selection[0]
                if idx < len(self.saved_prompts):
                    result = messagebox.askyesno("Confirm Delete", 
                                               "Are you sure you want to delete this prompt?")
                    if result:
                        del self.saved_prompts[idx]
                        self.save_saved_prompts()
                        self.refresh_prompts_listbox()
        
        # Create buttons
        buttons = [
            ("📋 Copy", copy_selected_prompt),
            ("✏️ Edit", edit_selected_prompt),
            ("🗑️ Delete", delete_selected_prompt),
            ("🔄 Refresh", self.refresh_prompts_listbox),
            ("❌ Close", manager_window.destroy)
        ]
        
        for text, command in buttons:
            btn = tk.Button(button_frame, text=text, command=command, 
                           bg=self.hover_bg, fg=self.fg_color, 
                           font=("Arial", 10), relief=tk.FLAT, 
                           padx=10, pady=6, cursor="hand2")
            btn.pack(side=tk.LEFT, padx=(0, 5))
            
            # Add hover effect
            btn.bind("<Enter>", lambda e, b=btn: b.config(bg=self.bg_color))
            btn.bind("<Leave>", lambda e, b=btn: b.config(bg=self.hover_bg))
    
    def refresh_prompts_listbox(self):
        """Refresh the prompts listbox"""
        self.prompts_listbox.delete(0, tk.END)
        for prompt in self.saved_prompts:
            display_text = f"{prompt['name']} ({prompt['created_at']})"
            self.prompts_listbox.insert(tk.END, display_text)
    
    def insert_selected_prompt(self, event):
        """Insert selected prompt into main prompt area"""
        selection = self.prompts_listbox.curselection()
        if selection:
            idx = selection[0]
            if idx < len(self.saved_prompts):
                prompt = self.saved_prompts[idx]["prompt"]
                self.prompt_text.delete(1.0, tk.END)
                self.prompt_text.insert(tk.END, prompt)
    
    def open_edit_prompt_dialog(self, prompt_data, idx):
        """Open dialog to edit a saved prompt"""
        edit_window = tk.Toplevel(self.root)
        edit_window.title("Edit Prompt")
        edit_window.geometry("500x300")
        edit_window.configure(bg=self.bg_color)
        edit_window.transient(self.root)
        edit_window.grab_set()
        
        # Header
        header_frame = tk.Frame(edit_window, bg=self.accent_color, height=50)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)
        
        tk.Label(header_frame, text="Edit Prompt", 
                font=("Arial", 16, "bold"), bg=self.accent_color, 
                fg="white").pack(pady=12)
        
        # Content
        content_frame = tk.Frame(edit_window, bg=self.bg_color, padx=20, pady=20)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Name field
        tk.Label(content_frame, text="Prompt Name:", font=("Arial", 11, "bold"), 
                bg=self.bg_color, fg=self.fg_color).pack(anchor=tk.W)
        
        name_var = tk.StringVar(value=prompt_data["name"])
        name_entry = tk.Entry(content_frame, textvariable=name_var, 
                             font=("Arial", 10), bg=self.entry_bg, 
                             fg=self.fg_color, relief=tk.FLAT)
        name_entry.pack(fill=tk.X, pady=(5, 15))
        
        # Prompt text
        tk.Label(content_frame, text="Prompt Text:", font=("Arial", 11, "bold"), 
                bg=self.bg_color, fg=self.fg_color).pack(anchor=tk.W)
        
        prompt_text = scrolledtext.ScrolledText(content_frame, height=7, width=40, 
                                               font=("Arial", 10), bg=self.entry_bg, 
                                               fg=self.fg_color, relief=tk.FLAT)
        prompt_text.pack(fill=tk.BOTH, expand=True, pady=(5, 15))
        prompt_text.insert(tk.END, prompt_data["prompt"])
        
        # Buttons
        button_frame = tk.Frame(content_frame, bg=self.bg_color)
        button_frame.pack(fill=tk.X)
        
        def save_changes():
            name = name_var.get().strip()
            if not name:
                messagebox.showwarning("Warning", "Please enter a name for the prompt!")
                return
            
            prompt = prompt_text.get(1.0, tk.END).strip()
            if not prompt:
                messagebox.showwarning("Warning", "Prompt text cannot be empty!")
                return
            
            # Update the prompt
            self.saved_prompts[idx] = {
                "id": prompt_data["id"],
                "name": name,
                "prompt": prompt,
                "created_at": prompt_data["created_at"]
            }
            
            self.save_saved_prompts()
            self.refresh_prompts_listbox()
            messagebox.showinfo("Success", "Prompt updated successfully!")
            edit_window.destroy()
        
        save_btn = tk.Button(button_frame, text="💾 Save Changes", 
                            command=save_changes,
                            bg=self.accent_color, fg="white", 
                            font=("Arial", 10, "bold"), relief=tk.FLAT, 
                            padx=15, pady=8, cursor="hand2")
        save_btn.pack(side=tk.RIGHT)
        
        cancel_btn = tk.Button(button_frame, text="Cancel", 
                              command=edit_window.destroy,
                              bg=self.hover_bg, fg=self.fg_color, 
                              font=("Arial", 10), relief=tk.FLAT, 
                              padx=15, pady=8, cursor="hand2")
        cancel_btn.pack(side=tk.RIGHT, padx=(0, 10))
    
    def open_image_reference_manager(self):
        """Open image reference manager window"""
        manager_window = tk.Toplevel(self.root)
        manager_window.title("🖼️ Image Reference Manager")
        manager_window.geometry("800x600")
        manager_window.configure(bg=self.bg_color)
        manager_window.transient(self.root)
        
        # Header
        header_frame = tk.Frame(manager_window, bg=self.accent_color, height=50)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)
        
        tk.Label(header_frame, text="Image Reference Manager", 
                font=("Arial", 16, "bold"), bg=self.accent_color, 
                fg="white").pack(pady=12)
        
        # Content
        content_frame = tk.Frame(manager_window, bg=self.bg_color, padx=20, pady=20)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Instructions
        instructions = tk.Label(content_frame, 
                               text="Add reference images to maintain consistent faces, objects, or styles in your generated images.",
                               font=("Arial", 10), bg=self.bg_color, fg=self.fg_color, wraplength=750)
        instructions.pack(anchor=tk.W, pady=(0, 15))
        
        # Reference list frame
        list_frame = tk.Frame(content_frame, bg=self.bg_color)
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create scrollable frame for references
        canvas = tk.Canvas(list_frame, bg=self.entry_bg, highlightthickness=0)
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=self.entry_bg)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Add reference button
        add_btn = tk.Button(content_frame, text="➕ Add Image Reference", 
                           command=self.add_image_reference,
                           bg=self.accent_color, fg="white", 
                           font=("Arial", 10, "bold"), relief=tk.FLAT, 
                           padx=15, pady=8, cursor="hand2")
        add_btn.pack(anchor=tk.W, pady=(15, 0))
        
        # Reference items container
        self.reference_items_frame = scrollable_frame
        self.refresh_reference_items()
        
        # Close button
        close_btn = tk.Button(content_frame, text="Close", 
                             command=manager_window.destroy,
                             bg=self.hover_bg, fg=self.fg_color, 
                             font=("Arial", 10), relief=tk.FLAT, 
                             padx=15, pady=8, cursor="hand2")
        close_btn.pack(anchor=tk.E, pady=(15, 0))
    
    def open_auto_reference_manager(self):
        """Open auto reference manager window"""
        manager_window = tk.Toplevel(self.root)
        manager_window.title("👥 Auto Reference Manager")
        manager_window.geometry("800x600")
        manager_window.configure(bg=self.bg_color)
        manager_window.transient(self.root)
        
        # Header
        header_frame = tk.Frame(manager_window, bg=self.accent_color, height=50)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)
        
        tk.Label(header_frame, text="Auto Reference Manager", 
                font=("Arial", 16, "bold"), bg=self.accent_color, 
                fg="white").pack(pady=12)
        
        # Content
        content_frame = tk.Frame(manager_window, bg=self.bg_color, padx=20, pady=20)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Instructions
        instructions = tk.Label(content_frame, 
                               text="Auto references are automatically captured from generated images to maintain character consistency.",
                               font=("Arial", 10), bg=self.bg_color, fg=self.fg_color, wraplength=750)
        instructions.pack(anchor=tk.W, pady=(0, 15))
        
        # Character list frame
        list_frame = tk.Frame(content_frame, bg=self.bg_color)
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create scrollable frame for characters
        canvas = tk.Canvas(list_frame, bg=self.entry_bg, highlightthickness=0)
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=self.entry_bg)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Character items container
        self.auto_ref_items_frame = scrollable_frame
        self.refresh_auto_ref_items()
        
        # Clear button
        clear_btn = tk.Button(content_frame, text="🗑️ Clear All References", 
                             command=self.clear_auto_references,
                             bg="#B22222", fg="white", 
                             font=("Arial", 10, "bold"), relief=tk.FLAT, 
                             padx=15, pady=8, cursor="hand2")
        clear_btn.pack(anchor=tk.W, pady=(15, 0))
        
        # Close button
        close_btn = tk.Button(content_frame, text="Close", 
                             command=manager_window.destroy,
                             bg=self.hover_bg, fg=self.fg_color, 
                             font=("Arial", 10), relief=tk.FLAT, 
                             padx=15, pady=8, cursor="hand2")
        close_btn.pack(anchor=tk.E, pady=(15, 0))
    
    def refresh_reference_items(self):
        """Refresh the reference items in the manager"""
        # Clear existing items
        for widget in self.reference_items_frame.winfo_children():
            widget.destroy()
        
        # Add reference items
        for i, ref in enumerate(self.image_references):
            self.create_reference_item(ref, i)
    
    def refresh_auto_ref_items(self):
        """Refresh the auto reference items in the manager"""
        # Clear existing items
        for widget in self.auto_ref_items_frame.winfo_children():
            widget.destroy()
        
        # Add character items
        for name, data in self.auto_reference_system.references.items():
            self.create_auto_ref_item(name, data)
    
    def create_reference_item(self, ref, index):
        """Create a reference item widget"""
        item_frame = tk.Frame(self.reference_items_frame, bg=self.button_bg, relief=tk.RAISED, bd=1)
        item_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Reference image
        try:
            # Load and resize image
            image_data = base64.b64decode(ref["image_data"])
            image = Image.open(io.BytesIO(image_data))
            image.thumbnail((80, 80))
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(image)
            
            # Image label
            img_label = tk.Label(item_frame, image=photo, bg=self.button_bg)
            img_label.image = photo  # Keep a reference
            img_label.pack(side=tk.LEFT, padx=10, pady=10)
        except Exception as e:
            print(f"Error loading reference image: {e}")
            # Placeholder if image fails to load
            img_label = tk.Label(item_frame, text="Image", bg=self.button_bg, fg=self.fg_color, width=10, height=5)
            img_label.pack(side=tk.LEFT, padx=10, pady=10)
        
        # Reference info
        info_frame = tk.Frame(item_frame, bg=self.button_bg)
        info_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Name
        name_label = tk.Label(info_frame, text=f"Name: {ref['name']}", 
                             font=("Arial", 10, "bold"), bg=self.button_bg, 
                             fg=self.fg_color, anchor=tk.W)
        name_label.pack(fill=tk.X, pady=(0, 5))
        
        # Size
        size_label = tk.Label(info_frame, text=f"Size: {ref.get('width', 'N/A')}x{ref.get('height', 'N/A')}", 
                             font=("Arial", 9), bg=self.button_bg, 
                             fg=self.fg_color, anchor=tk.W)
        size_label.pack(fill=tk.X, pady=(0, 5))
        
        # Buttons
        btn_frame = tk.Frame(info_frame, bg=self.button_bg)
        btn_frame.pack(fill=tk.X, pady=(5, 0))
        
        # Edit button
        edit_btn = tk.Button(btn_frame, text="✏️ Edit", 
                            command=lambda i=index: self.edit_image_reference(i),
                            bg=self.hover_bg, fg=self.fg_color, 
                            font=("Arial", 9), relief=tk.FLAT, 
                            padx=10, pady=2, cursor="hand2")
        edit_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        # Delete button
        delete_btn = tk.Button(btn_frame, text="🗑️ Delete", 
                              command=lambda i=index: self.delete_image_reference(i),
                              bg="#B22222", fg="white", 
                              font=("Arial", 9), relief=tk.FLAT, 
                              padx=10, pady=2, cursor="hand2")
        delete_btn.pack(side=tk.LEFT)
    
    def create_auto_ref_item(self, name, data):
        """Create an auto reference item widget"""
        item_frame = tk.Frame(self.auto_ref_items_frame, bg=self.button_bg, relief=tk.RAISED, bd=1)
        item_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Character image (show the latest one)
        try:
            if data["images"]:
                # Load and resize the latest image
                image_data = base64.b64decode(data["images"][-1])
                image = Image.open(io.BytesIO(image_data))
                image.thumbnail((80, 80))
                
                # Convert to PhotoImage
                photo = ImageTk.PhotoImage(image)
                
                # Image label
                img_label = tk.Label(item_frame, image=photo, bg=self.button_bg)
                img_label.image = photo  # Keep a reference
                img_label.pack(side=tk.LEFT, padx=10, pady=10)
            else:
                # Placeholder if no images
                img_label = tk.Label(item_frame, text="Character", bg=self.button_bg, fg=self.fg_color, width=10, height=5)
                img_label.pack(side=tk.LEFT, padx=10, pady=10)
        except Exception as e:
            print(f"Error loading character image: {e}")
            # Placeholder if image fails to load
            img_label = tk.Label(item_frame, text="Character", bg=self.button_bg, fg=self.fg_color, width=10, height=5)
            img_label.pack(side=tk.LEFT, padx=10, pady=10)
        
        # Character info
        info_frame = tk.Frame(item_frame, bg=self.button_bg)
        info_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Name
        name_label = tk.Label(info_frame, text=f"Name: {name}", 
                             font=("Arial", 10, "bold"), bg=self.button_bg, 
                             fg=self.fg_color, anchor=tk.W)
        name_label.pack(fill=tk.X, pady=(0, 5))
        
        # Image count
        count_label = tk.Label(info_frame, text=f"Images: {len(data.get('images', []))}", 
                             font=("Arial", 9), bg=self.button_bg, 
                             fg=self.fg_color, anchor=tk.W)
        count_label.pack(fill=tk.X, pady=(0, 5))
        
        # Created at
        created_label = tk.Label(info_frame, text=f"Created: {data.get('created_at', 'N/A')}", 
                               font=("Arial", 9), bg=self.button_bg, 
                               fg=self.fg_color, anchor=tk.W)
        created_label.pack(fill=tk.X, pady=(0, 5))
        
        # Buttons
        btn_frame = tk.Frame(info_frame, bg=self.button_bg)
        btn_frame.pack(fill=tk.X, pady=(5, 0))
        
        # Delete button
        delete_btn = tk.Button(btn_frame, text="🗑️ Delete", 
                              command=lambda n=name: self.delete_auto_reference(n),
                              bg="#B22222", fg="white", 
                              font=("Arial", 9), relief=tk.FLAT, 
                              padx=10, pady=2, cursor="hand2")
        delete_btn.pack(side=tk.LEFT)
    
    def add_image_reference(self):
        """Add a new image reference"""
        # Ask for image file
        file_path = filedialog.askopenfilename(
            title="Select Reference Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")]
        )
        
        if not file_path:
            return
        
        # Ask for reference name
        name_window = tk.Toplevel(self.root)
        name_window.title("Reference Name")
        name_window.geometry("400x150")
        name_window.configure(bg=self.bg_color)
        name_window.transient(self.root)
        name_window.grab_set()
        
        # Header
        header_frame = tk.Frame(name_window, bg=self.accent_color, height=40)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)
        
        tk.Label(header_frame, text="Reference Name", 
                font=("Arial", 14, "bold"), bg=self.accent_color, 
                fg="white").pack(pady=10)
        
        # Content
        content_frame = tk.Frame(name_window, bg=self.bg_color, padx=20, pady=20)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Name field
        tk.Label(content_frame, text="Enter a name for this reference:", 
                font=("Arial", 10), bg=self.bg_color, fg=self.fg_color).pack(anchor=tk.W)
        
        name_var = tk.StringVar()
        name_entry = tk.Entry(content_frame, textvariable=name_var, 
                             font=("Arial", 10), bg=self.entry_bg, 
                             fg=self.fg_color, relief=tk.FLAT)
        name_entry.pack(fill=tk.X, pady=(10, 15))
        
        # Buttons
        button_frame = tk.Frame(content_frame, bg=self.bg_color)
        button_frame.pack(fill=tk.X)
        
        def save_reference():
            name = name_var.get().strip()
            if not name:
                messagebox.showwarning("Warning", "Please enter a name for the reference!")
                return
            
            try:
                # Load image
                image = Image.open(file_path)
                width, height = image.size
                
                # Convert to base64
                img_buffer = io.BytesIO()
                image.save(img_buffer, format="PNG")
                img_data = img_buffer.getvalue()
                img_base64 = base64.b64encode(img_data).decode('utf-8')
                
                # Add to references
                self.image_references.append({
                    "id": ''.join(random.choices(string.ascii_lowercase + string.digits, k=8)),
                    "name": name,
                    "image_data": img_base64,
                    "width": width,
                    "height": height,
                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                
                self.save_image_references()
                self.update_reference_label()
                self.refresh_reference_items()
                messagebox.showinfo("Success", "Image reference added successfully!")
                name_window.destroy()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to add image reference: {str(e)}")
        
        save_btn = tk.Button(button_frame, text="💾 Save", 
                            command=save_reference,
                            bg=self.accent_color, fg="white", 
                            font=("Arial", 10, "bold"), relief=tk.FLAT, 
                            padx=15, pady=8, cursor="hand2")
        save_btn.pack(side=tk.RIGHT)
        
        cancel_btn = tk.Button(button_frame, text="Cancel", 
                              command=name_window.destroy,
                              bg=self.hover_bg, fg=self.fg_color, 
                              font=("Arial", 10), relief=tk.FLAT, 
                              padx=15, pady=8, cursor="hand2")
        cancel_btn.pack(side=tk.RIGHT, padx=(0, 10))
    
    def clear_auto_references(self):
        """Clear all auto references"""
        result = messagebox.askyesno("Confirm Clear", 
                                   "Are you sure you want to clear all auto references?")
        if result:
            self.auto_reference_system.references = {}
            self.auto_reference_system.character_prompts = defaultdict(list)
            self.auto_reference_system.save_references()
            self.update_auto_ref_label()
            self.refresh_auto_ref_items()
            messagebox.showinfo("Success", "All auto references cleared!")
    
    def delete_auto_reference(self, name):
        """Delete an auto reference"""
        if name not in self.auto_reference_system.references:
            return
        
        result = messagebox.askyesno("Confirm Delete", 
                                   f"Are you sure you want to delete the auto reference for '{name}'?")
        if result:
            del self.auto_reference_system.references[name]
            if name in self.auto_reference_system.character_prompts:
                del self.auto_reference_system.character_prompts[name]
            self.auto_reference_system.save_references()
            self.update_auto_ref_label()
            self.refresh_auto_ref_items()
    
    def edit_image_reference(self, index):
        """Edit an image reference"""
        if index >= len(self.image_references):
            return
        
        ref = self.image_references[index]
        
        # Create edit window
        edit_window = tk.Toplevel(self.root)
        edit_window.title("Edit Reference")
        edit_window.geometry("400x150")
        edit_window.configure(bg=self.bg_color)
        edit_window.transient(self.root)
        edit_window.grab_set()
        
        # Header
        header_frame = tk.Frame(edit_window, bg=self.accent_color, height=40)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)
        
        tk.Label(header_frame, text="Edit Reference Name", 
                font=("Arial", 14, "bold"), bg=self.accent_color, 
                fg="white").pack(pady=10)
        
        # Content
        content_frame = tk.Frame(edit_window, bg=self.bg_color, padx=20, pady=20)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Name field
        tk.Label(content_frame, text="Reference name:", 
                font=("Arial", 10), bg=self.bg_color, fg=self.fg_color).pack(anchor=tk.W)
        
        name_var = tk.StringVar(value=ref["name"])
        name_entry = tk.Entry(content_frame, textvariable=name_var, 
                             font=("Arial", 10), bg=self.entry_bg, 
                             fg=self.fg_color, relief=tk.FLAT)
        name_entry.pack(fill=tk.X, pady=(10, 15))
        
        # Buttons
        button_frame = tk.Frame(content_frame, bg=self.bg_color)
        button_frame.pack(fill=tk.X)
        
        def save_changes():
            name = name_var.get().strip()
            if not name:
                messagebox.showwarning("Warning", "Please enter a name for the reference!")
                return
            
            # Update reference
            self.image_references[index]["name"] = name
            self.save_image_references()
            self.update_reference_label()
            self.refresh_reference_items()
            messagebox.showinfo("Success", "Reference updated successfully!")
            edit_window.destroy()
        
        save_btn = tk.Button(button_frame, text="💾 Save", 
                            command=save_changes,
                            bg=self.accent_color, fg="white", 
                            font=("Arial", 10, "bold"), relief=tk.FLAT, 
                            padx=15, pady=8, cursor="hand2")
        save_btn.pack(side=tk.RIGHT)
        
        cancel_btn = tk.Button(button_frame, text="Cancel", 
                              command=edit_window.destroy,
                              bg=self.hover_bg, fg=self.fg_color, 
                              font=("Arial", 10), relief=tk.FLAT, 
                              padx=15, pady=8, cursor="hand2")
        cancel_btn.pack(side=tk.RIGHT, padx=(0, 10))
    
    def delete_image_reference(self, index):
        """Delete an image reference"""
        if index >= len(self.image_references):
            return
        
        result = messagebox.askyesno("Confirm Delete", 
                                   "Are you sure you want to delete this image reference?")
        if result:
            del self.image_references[index]
            self.save_image_references()
            self.update_reference_label()
            self.refresh_reference_items()
    
    def process_placeholders(self, prompt):
        """Replace placeholders in prompt"""
        placeholders = {
            "{style}": "photorealistic",
            "{mood}": "serene",
            "{subject}": "landscape",
            "{background}": "mountain",
            "{lighting}": "dramatic lighting"
        }
        
        for placeholder, value in placeholders.items():
            prompt = prompt.replace(placeholder, value)
        
        return prompt
    
    def enhance_prompt(self, prompt):
        """Auto-enhance prompt with quality descriptors"""
        if not self.auto_enhance_var.get():
            return prompt
        
        template_idx = self.enhancement_var.get()
        if 0 <= template_idx < len(self.quality_enhancers):
            enhancer = self.quality_enhancers[template_idx]
            return f"{prompt}, {enhancer}"
        
        return prompt
    
    def convert_sensitive_language(self, prompt):
        """Convert sensitive language to acceptable terms"""
        if not self.adult_content_var.get():
            return prompt
        
        # Sort phrases by length (longest first) to avoid partial replacements
        sorted_phrases = sorted(self.sensitive_language_dict.keys(), key=len, reverse=True)
        
        converted_prompt = prompt
        for phrase in sorted_phrases:
            # Case-insensitive replacement
            pattern = re.compile(re.escape(phrase), re.IGNORECASE)
            converted_prompt = pattern.sub(self.sensitive_language_dict[phrase], converted_prompt)
        
        return converted_prompt
    
    def generate_images(self):
        """Generate images from prompts"""
        # Get prompts
        prompt_content = self.prompt_text.get(1.0, tk.END).strip()
        if not prompt_content:
            messagebox.showwarning("Warning", "Please enter at least one prompt!")
            return
        
        # Split prompts by empty lines
        prompt_lines = prompt_content.split('\n')
        prompts = []
        current_prompt = []
        
        for line in prompt_lines:
            line = line.strip()
            if line:
                current_prompt.append(line)
            elif current_prompt:  # Empty line and we have content
                prompts.append(' '.join(current_prompt))
                current_prompt = []
        
        # Add the last prompt if there is one
        if current_prompt:
            prompts.append(' '.join(current_prompt))
        
        if not prompts:
            messagebox.showwarning("Warning", "Please enter valid prompts!")
            return
        
        # Start generation in separate thread
        thread = threading.Thread(target=self._generate_images_thread, args=(prompts,))
        thread.daemon = True
        thread.start()
    
    def _generate_images_thread(self, prompts):
        """Generate images in a separate thread"""
        total_images = len(prompts)
        generated_count = 0
        
        for i, prompt in enumerate(prompts):
            try:
                # Update status
                self.root.after(0, lambda i=i, p=prompt: self.status_var.set(f"Processing image {i+1}/{total_images}: {p[:50]}..."))
                
                # Process placeholders and enhance prompt
                processed_prompt = self.process_placeholders(prompt)
                enhanced_prompt = self.enhance_prompt(processed_prompt)
                
                # Convert sensitive language if adult content is enabled
                if self.adult_content_var.get():
                    enhanced_prompt = self.convert_sensitive_language(enhanced_prompt)
                
                # Find auto references for this prompt
                auto_references = {}
                if self.auto_reference_var.get():
                    auto_references = self.auto_reference_system.find_references(enhanced_prompt)
                    
                    # Add reference prompts to the enhanced prompt
                    for name, ref_data in auto_references.items():
                        ref_prompt = self.auto_reference_system.get_reference_prompt(name)
                        if ref_prompt:
                            enhanced_prompt = f"{enhanced_prompt}, {ref_prompt}"
                
                # Add reference names to prompt if available
                if self.image_references:
                    ref_names = [ref["name"] for ref in self.image_references]
                    enhanced_prompt = f"{enhanced_prompt}, reference faces: {', '.join(ref_names)}"
                
                # Try to generate image with retries
                image_data = None
                retry_count = 0
                max_retries = self.config.get("retry_attempts", 5)
                
                while retry_count < max_retries and image_data is None:
                    try:
                        # For adult content, use different approaches in retries
                        if self.adult_content_var.get() and retry_count > 0:
                            # For retries, try with a different seed
                            modified_prompt = f"{enhanced_prompt}, seed={random.randint(1, 10000)}"
                            image_data = self.generate_pollinations_image(modified_prompt)
                        else:
                            image_data = self.generate_pollinations_image(enhanced_prompt)
                    except Exception as e:
                        retry_count += 1
                        if retry_count >= max_retries:
                            raise Exception(f"Failed after {max_retries} attempts: {str(e)}")
                        time.sleep(2)  # Wait before retrying
                
                if image_data:
                    # Check image quality if enabled
                    if self.quality_check_var.get():
                        quality_ok, quality_msg = self.quality_checker.check_image_quality(image_data)
                        if not quality_ok:
                            # Try to enhance the image
                            try:
                                # Convert to PIL Image
                                image = Image.open(io.BytesIO(image_data))
                                
                                # Apply sharpening filter
                                image = image.filter(ImageFilter.SHARPEN)
                                
                                # Convert back to bytes
                                img_buffer = io.BytesIO()
                                image.save(img_buffer, format="PNG")
                                image_data = img_buffer.getvalue()
                                
                                # Check quality again
                                quality_ok, quality_msg = self.quality_checker.check_image_quality(image_data)
                            except Exception as e:
                                print(f"Error enhancing image: {e}")
                    
                    # Save image
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{self.config['output_dir']}/image_{timestamp}_{i+1}.png"
                    
                    with open(filename, 'wb') as f:
                        f.write(image_data)
                    
                    generated_count += 1
                    
                    # Add to auto references if enabled
                    if self.auto_reference_var.get():
                        character_names = self.auto_reference_system.extract_character_names(prompt)
                        if character_names:
                            self.auto_reference_system.add_reference(image_data, prompt, character_names)
                            self.root.after(0, lambda: self.update_auto_ref_label())
                    
                    # Update preview
                    self.root.after(0, lambda data=image_data: self.update_preview(data))
                
                # Update progress
                progress = ((i + 1) / total_images) * 100
                self.root.after(0, lambda p=progress: self.progress_var.set(p))
                
                # Small delay to avoid rate limiting
                time.sleep(1)
                
            except Exception as e:
                error_msg = f"Error generating image {i+1}: {str(e)}"
                self.root.after(0, lambda msg=error_msg: messagebox.showerror("Error", msg))
        
        # Final status update
        self.root.after(0, lambda: self.status_var.set(f"Generated {generated_count}/{total_images} images successfully!"))
        self.root.after(0, lambda: self.progress_var.set(0))
    
    def update_preview(self, image_data):
        """Update the preview canvas with the latest image"""
        try:
            # Convert image data to PIL Image
            image = Image.open(io.BytesIO(image_data))
            
            # Resize for preview
            image.thumbnail((100, 100))
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(image)
            
            # Clear canvas and add new image
            self.preview_canvas.delete("all")
            self.preview_canvas.create_image(10, 10, anchor=tk.NW, image=photo)
            
            # Keep a reference to prevent garbage collection
            self.preview_canvas.image = photo
            
        except Exception as e:
            print(f"Error updating preview: {e}")
    
    def generate_pollinations_image(self, prompt):
        """Generate image using Pollinations.AI with no watermark"""
        try:
            # Parse size
            size_str = self.size_var.get()
            if "x" in size_str:
                width, height = map(int, size_str.split("x"))
            else:
                width, height = 1024, 1024
            
            # Build URL with parameters
            base_url = "https://pollinations.ai/p"
            params = {
                "prompt": prompt,
                "width": width,
                "height": height,
                "model": "flux"
            }
            
            # Add no logo parameter if enabled
            if self.remove_watermark_var.get():
                params["nologo"] = "true"
            
            # Build URL with parameters
            encoded_prompt = urllib.parse.quote(prompt)
            url = f"{base_url}/{encoded_prompt}"
            
            # Add other parameters as query string
            query_params = []
            for key, value in params.items():
                if key != "prompt":
                    query_params.append(f"{key}={value}")
            
            if query_params:
                url += "?" + "&".join(query_params)
            
            # Increase timeout for large images
            response = requests.get(url, timeout=180)  # Increased timeout to 3 minutes
            response.raise_for_status()
            
            return response.content
        except Exception as e:
            raise Exception(f"Pollinations.AI error: {str(e)}")

def main():
    root = tk.Tk()
    app = AdvancedImageGenerator(root)
    root.mainloop()

if __name__ == "__main__":
    main()