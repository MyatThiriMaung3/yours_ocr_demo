import tensorflow as tf
from tensorflow import keras
import numpy as np
import pickle
import json
from pathlib import Path
from PIL import Image
import cv2

class WordRecognizerCTC:
    def __init__(self, model_path, char_config_path):
        """
        Initialize the CTC-based word recognizer
        
        Args:
            model_path: Path to the saved .keras or .h5 model file
            char_config_path: Path to char_config.pkl or char_config.json
        """
        self.model = None
        self.char_to_num = None
        self.num_to_char = None
        self.characters = None
        self.max_len = 21  # From your training
        self.width = 128
        self.height = 32
        self.padding_token = 99
        
        self._load_model(model_path)
        self._load_char_mappings(char_config_path)
    
    def _load_model(self, model_path):
        """Load the trained model"""
        try:
            model_path = Path(model_path)
            
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            print(f"Loading model from: {model_path}")
            
            # Try loading as .keras file first
            if str(model_path).endswith('.keras'):
                self.model = keras.models.load_model(model_path, compile=False)
            elif str(model_path).endswith('.h5'):
                # For .h5 files, try different approaches
                try:
                    self.model = keras.models.load_model(model_path, compile=False)
                except:
                    # If that fails, try loading weights only
                    print("Trying to load as weights file...")
                    # We'll need to rebuild the model architecture
                    raise Exception("Please use HTR_prediction_model.keras file")
            else:
                raise ValueError(f"Unsupported model format: {model_path.suffix}")
            
            print(f"Model loaded successfully!")
            print(f"Model input shape: {self.model.input_shape}")
            print(f"Model output shape: {self.model.output_shape}")
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise Exception(f"Failed to load model from {model_path}: {str(e)}")

    def _load_char_mappings(self, char_config_path):
        """Load character mappings"""
        try:
            char_config_path = Path(char_config_path)
            
            if char_config_path.suffix == '.pkl':
                with open(char_config_path, 'rb') as f:
                    config = pickle.load(f)
            elif char_config_path.suffix == '.json':
                with open(char_config_path, 'r') as f:
                    config = json.load(f)
            else:
                raise ValueError("Config file must be .pkl or .json")
            
            self.characters = config['characters']
            
            # Create TensorFlow StringLookup layers
            self.char_to_num = tf.keras.layers.StringLookup(
                vocabulary=list(self.characters), mask_token=None
            )
            self.num_to_char = tf.keras.layers.StringLookup(
                vocabulary=self.char_to_num.get_vocabulary(), 
                mask_token=None, 
                invert=True
            )
            
            print(f"Character mappings loaded: {len(self.characters)} characters")
            
        except Exception as e:
            raise Exception(f"Failed to load character mappings: {str(e)}")
    
    def distortion_free_resize(self, image, w, h):
        """Resize image without distortion, maintaining aspect ratio"""
        image = tf.image.resize(image, size=(h, w), preserve_aspect_ratio=True)
        
        pad_height = h - tf.shape(image)[0]
        pad_width = w - tf.shape(image)[1]
        
        if pad_height % 2 != 0:
            height = pad_height // 2
            pad_height_top = height + 1
            pad_height_bottom = height
        else:
            pad_height_top = pad_height_bottom = pad_height // 2
        
        if pad_width % 2 != 0:
            width = pad_width // 2
            pad_width_left = width + 1
            pad_width_right = width
        else:
            pad_width_left = pad_width_right = pad_width // 2
        
        image = tf.pad(
            image,
            paddings=[
                [pad_height_top, pad_height_bottom],
                [pad_width_left, pad_width_right],
                [0, 0],
            ],
        )
        
        image = tf.transpose(image, perm=[1, 0, 2])
        image = tf.image.flip_left_right(image)
        return image
    
    def preprocess_image(self, image):
        """
        Preprocess a single image for prediction
        
        Args:
            image: PIL Image or numpy array
            
        Returns:
            Preprocessed tensor ready for model input
        """
        # Convert PIL Image to numpy if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Convert to grayscale if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Add channel dimension if needed
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
        
        # Convert to tensor
        image = tf.convert_to_tensor(image, dtype=tf.float32)
        
        # Resize with distortion-free method
        image = self.distortion_free_resize(image, self.width, self.height)
        
        # Normalize to [0, 1]
        image = tf.cast(image, tf.float32) / 255.0
        
        return image
    
    def decode_predictions(self, pred):
        """
        Decode CTC predictions to text
        
        Args:
            pred: Model predictions (batch_size, time_steps, num_classes)
            
        Returns:
            List of decoded text strings
        """
        input_len = np.ones(pred.shape[0]) * pred.shape[1]
        
        # Use CTC decode
        results = keras.backend.ctc_decode(
            pred, 
            input_length=input_len, 
            greedy=True
        )[0][0][:, :self.max_len]
        
        output_text = []
        for res in results:
            # Remove padding tokens (-1)
            res = tf.gather(res, tf.where(tf.math.not_equal(res, -1)))
            # Convert to characters
            res = tf.strings.reduce_join(self.num_to_char(res)).numpy().decode('utf-8')
            output_text.append(res)
        
        return output_text
    
    def recognize(self, image):
        """
        Recognize text from a single image
        
        Args:
            image: PIL Image or numpy array
            
        Returns:
            Recognized text string
        """
        # Preprocess
        processed_image = self.preprocess_image(image)
        
        # Add batch dimension
        batch_image = tf.expand_dims(processed_image, axis=0)
        
        # Predict
        prediction = self.model.predict(batch_image, verbose=0)
        
        # Decode
        decoded_text = self.decode_predictions(prediction)
        
        return decoded_text[0] if decoded_text else ""
    
    def recognize_batch(self, images):
        """
        Recognize text from multiple images
        
        Args:
            images: List of PIL Images or numpy arrays
            
        Returns:
            List of recognized text strings
        """
        # Preprocess all images
        processed_images = [self.preprocess_image(img) for img in images]
        
        # Stack into batch
        batch_images = tf.stack(processed_images, axis=0)
        
        # Predict
        predictions = self.model.predict(batch_images, verbose=0)
        
        # Decode
        decoded_texts = self.decode_predictions(predictions)
        
        return decoded_texts