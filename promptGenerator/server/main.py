from flask import Flask, request, jsonify
from flask_cors import CORS
from llama_cpp import Llama
import json
import re
import logging
from datetime import datetime
import signal
from functools import wraps
import threading
from queue import Queue
import time
import os

# Konfiguracja timeoutów (w sekundach)
GENERATION_TIMEOUT = 260  # 2 minuty na generowanie tekstu
JSON_PROCESSING_TIMEOUT = 30  # 30 sekund na przetwarzanie JSON
REQUEST_TIMEOUT = 180  # 3 minuty na całe żądanie

app = Flask(__name__)
CORS(app)  # Włącz CORS

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('server.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ModelTimeoutError(Exception):
    pass

class ModelOutputTooLongError(Exception):
    pass

class GPUInitializationError(Exception):
    pass

def timeout_handler(signum, frame):
    raise ModelTimeoutError("Model generation timed out")

# Dekorator do obsługi timeoutu dla funkcji
def timeout_decorator(seconds):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result_queue = Queue()
            
            def target():
                try:
                    result = func(*args, **kwargs)
                    result_queue.put(("success", result))
                except Exception as e:
                    result_queue.put(("error", e))
            
            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            
            thread.join(seconds)
            if thread.is_alive():
                logger.error(f"Function {func.__name__} timed out after {seconds} seconds")
                raise ModelTimeoutError(f"Operation timed out after {seconds} seconds")
            
            status, result = result_queue.get()
            if status == "error":
                raise result
            return result
            
        return wrapper
    return decorator

def initialize_model_with_gpu():
    """
    Inicjalizuje model z obsługą GPU, w przypadku błędu próbuje na CPU
    """
    try:
        model = Llama(
            model_path=r"C:\Users\jakub\Desktop\IN-PROGRES\imageGenerator\promptGenerator\MODELS\Phi-3-mini-4k-instruct-fp16.gguf",
            n_ctx=4096,
            n_threads=4,
            n_batch=512,
            n_gpu_layers=35,    # Liczba warstw na GPU
            main_gpu=0,         # Indeks głównego GPU
            tensor_split=None,   # Można podzielić między kilka GPU
            verbose=True        # Włącz logi dla debugowania GPU
        )
        logger.info("Model initialized successfully with GPU support")
        return model
    except Exception as e:
        logger.error(f"Failed to initialize model with GPU: {e}")
        logger.warning("Attempting CPU initialization...")
        try:
            model = Llama(
            model_path=r"C:\Users\jakub\Desktop\IN-PROGRES\imageGenerator\promptGenerator\MODELS\Phi-3-mini-4k-instruct-fp16.gguf",
                n_ctx=4096,
                n_threads=4,
                n_batch=512,
                verbose=True
            )
            logger.info("Model initialized successfully on CPU")
            return model
        except Exception as e:
            logger.error(f"Failed to initialize model on CPU: {e}")
            raise

# Initialize Phi-3 model
logger.info("Starting model initialization...")
try:
    llm = initialize_model_with_gpu()
except Exception as e:
    logger.error(f"Critical error during model initialization: {e}")
    raise

def create_prompt(categories):
    """
    Tworzy prompt dla modelu Phi-3 na podstawie kategorii
    """
    categories_str = ", ".join(categories)
    prompt = f"""Instruct: Generate 5 creative and diverse prompts for each of the following image generation categories: {categories_str}
Return the response in valid JSON format like this:
{{
    "category_name": [
        "detailed prompt 1",
        "detailed prompt 2",
        "detailed prompt 3",
        "detailed prompt 4",
        "detailed prompt 5"
    ]
}}

Keep the output concise and ensure it's valid JSON. Do not include any additional text or explanations.

Response: """
    
    logger.info(f"Created prompt: {prompt}")
    return prompt

def validate_model_output(text):
    """
    Sprawdza czy output modelu jest prawidłowy
    """
    if len(text) > 10000:
        raise ModelOutputTooLongError("Model output exceeded maximum length")
    
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if not json_match:
        raise ValueError("No valid JSON found in model output")
    
    unwanted_patterns = [
        r'```',
        r'Human:',
        r'Assistant:',
        r'Instruct:',
        r'Response:',
        r'Here is',
        r'I will',
        r'Let me'
    ]
    
    for pattern in unwanted_patterns:
        if re.search(pattern, text):
            text = re.sub(pattern, '', text)
    
    return True

@timeout_decorator(JSON_PROCESSING_TIMEOUT)
def clean_json_output(text, categories):
    """
    Czyści wyjście z modelu i konwertuje go na poprawny format JSON
    """
    logger.info("Starting JSON output cleaning...")
    logger.debug(f"Raw model output: {text}")
    
    try:
        validate_model_output(text)
        
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if not json_match:
            logger.error("No JSON found in the output")
            return {"error": "No JSON found in the output"}
        
        json_str = json_match.group()
        logger.debug(f"Extracted JSON string: {json_str}")
        
        try:
            data = json.loads(json_str)
            logger.info("Successfully parsed initial JSON")
        except json.JSONDecodeError as e:
            logger.warning(f"Initial JSON parsing failed: {e}. Attempting to clean the string...")
            json_str = (json_str
                       .replace('`json\n', '')
                       .replace('`', '')
                       .replace('\n', '')
                       .replace('...', '""]'))
            logger.debug(f"Cleaned JSON string: {json_str}")
            data = json.loads(json_str)
            logger.info("Successfully parsed cleaned JSON")

        result = {}
        
        for category in categories:
            logger.debug(f"Processing category: {category}")
            matching_key = None
            for key in data.keys():
                if key.lower().strip() == category.lower().strip():
                    matching_key = key
                    logger.debug(f"Found matching key: {matching_key} for category: {category}")
                    break
            
            if matching_key:
                prompts = data[matching_key]
                if isinstance(prompts, list):
                    while len(prompts) < 5:
                        prompts.append(f"Generated prompt for {category}")
                    prompts = prompts[:5]
                    logger.info(f"Adjusted prompts for {category}: {len(prompts)} prompts")
                else:
                    logger.warning(f"Prompts for {category} were not in list format. Generating default prompts.")
                    prompts = [f"Generated prompt {i+1} for {category}" for i in range(5)]
                
                result[category] = prompts
            else:
                logger.warning(f"No matching key found for category: {category}. Generating default prompts.")
                result[category] = [f"Generated prompt {i+1} for {category}" for i in range(5)]
        
        logger.info("JSON cleaning completed successfully")
        return result
        
    except Exception as e:
        logger.error(f"Error processing JSON: {e}", exc_info=True)
        return {"error": f"Failed to process the output: {str(e)}"}

@timeout_decorator(GENERATION_TIMEOUT)
def generate_with_phi3(prompt):
    """
    Generuje tekst z modelem Phi-3 z zabezpieczeniami
    """
    try:
        start_time = time.time()
        
        output = llm(
            prompt,
            max_tokens=2000,
            temperature=0.7,
            stop=["</s>", "Human:", "Assistant:", "Instruct:", "Response:"],
            repeat_penalty=1.1,
            top_k=40,
            top_p=0.9,
            stream=False
        )
        
        generation_time = time.time() - start_time
        logger.info(f"Generation completed in {generation_time:.2f} seconds")
        
        return output['choices'][0]['text']
    except Exception as e:
        logger.error(f"Error in Phi-3 generation: {e}")
        raise

@app.route('/generate_prompts', methods=['POST'])
def generate_prompts():
    request_id = datetime.now().strftime('%Y%m%d%H%M%S%f')
    logger.info(f"Received new request - ID: {request_id}")
    
    try:
        if not request.is_json:
            logger.error(f"Request {request_id} - Invalid content type, expected JSON")
            return jsonify({"error": "Request must be JSON"}), 400
            
        data = request.get_json()
        logger.info(f"Request {request_id} - Received data: {data}")
        
        if not isinstance(data.get('categories', None), list):
            logger.error(f"Request {request_id} - Missing or invalid 'categories' field")
            return jsonify({"error": "Request must contain 'categories' list"}), 400
            
        categories = data['categories']
        logger.info(f"Request {request_id} - Categories to process: {categories}")
        
        if not categories:
            logger.error(f"Request {request_id} - Empty categories list")
            return jsonify({"error": "Categories list cannot be empty"}), 400
            
        if len(categories) > 10:
            logger.error(f"Request {request_id} - Too many categories")
            return jsonify({"error": "Maximum 10 categories allowed"}), 400
            
        prompt = create_prompt(categories)
        
        try:
            logger.info(f"Request {request_id} - Starting Phi-3 generation")
            start_time = datetime.now()
            
            model_output = generate_with_phi3(prompt)
            
            generation_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Request {request_id} - Phi-3 generation completed in {generation_time:.2f} seconds")
            
            result = clean_json_output(model_output, categories)
            
            logger.info(f"Request {request_id} - Processing completed successfully")
            return jsonify(result)
            
        except ModelTimeoutError:
            error_msg = "Model generation timed out"
            logger.error(f"Request {request_id} - {error_msg}")
            return jsonify({"error": error_msg}), 503
            
        except ModelOutputTooLongError:
            error_msg = "Model generated too long output"
            logger.error(f"Request {request_id} - {error_msg}")
            return jsonify({"error": error_msg}), 500
            
    except Exception as e:
        logger.error(f"Request {request_id} - Error processing request: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.before_request
def log_request_info():
    logger.info('Headers: %s', request.headers)
    logger.info('Body: %s', request.get_data())

@app.after_request
def log_response_info(response):
    logger.info('Response status: %s', response.status)
    logger.info('Response data: %s', response.get_data())
    return response

if __name__ == '__main__':
    logger.info("Starting Flask server...")
    app.run(debug=True, port=5000)