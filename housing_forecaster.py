import gradio as gr
import pandas as pd
import numpy as np
import json
import csv
import re
import json
import time
from datetime import datetime, timedelta
import requests
from typing import Optional
import logging
from dotenv import load_dotenv
import os
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.font_manager
from huggingface_hub import hf_hub_download
import joblib
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import warnings

# Suppress sklearn version warnings for model loading
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', message='.*Trying to unpickle estimator.*')
warnings.filterwarnings('ignore', message='.*loading a serialized model.*')

class TextboxHandler(logging.Handler):
    """Custom logging handler that stores logs for Gradio textbox"""
    def __init__(self):
        super().__init__()
        self.logs = []
        
    def emit(self, record):
        log_entry = self.format(record)
        self.logs.append(log_entry)
        
    def get_logs(self):
        return "\n".join(self.logs)
        
    def clear(self):
        self.logs = []

class FontDebugFilter(logging.Filter):
    """Filter to exclude matplotlib font debug messages"""
    def filter(self, record):
        return not (record.levelname == 'DEBUG' and 'findfont' in record.msg)

# Load pretrained XGBoost model
model_path = hf_hub_download(
    repo_id="M-Yaqoob/finetune-xgboost",
    filename="fine-tune_xgboost_model.pkl"
)
base_model = joblib.load(model_path)

# Create forecasts directory if it doesn't exist
FORECAST_DIR = Path('./forecasts')
FORECAST_DIR.mkdir(exist_ok=True)

# ============================================================================
# DATA LOADING UTILITIES - Load calibrated parameters from includes/
# ============================================================================

def load_supply_elasticity() -> dict:
    """Load city-specific supply elasticity from includes/"""
    try:
        with open('includes/supply_elasticity_by_city.json', 'r') as f:
            data = json.load(f)
        return {city: vals['long_run'] for city, vals in data.items() if isinstance(vals, dict)}
    except FileNotFoundError:
        logging.warning("Supply elasticity data not found, using defaults")
        return {'Sydney': 0.03, 'Melbourne': 0.05, 'Brisbane': 0.08, 'Perth': 0.07}

def load_rba_parameters() -> dict:
    """Load RBA model coefficients from includes/"""
    try:
        with open('includes/rba_model_parameters.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logging.warning("RBA parameters not found, using defaults")
        return {
            'momentum': {
                'INITIAL_DEVELOPMENT': 0.74,
                'RAPID_GROWTH': 0.60,
                'ESTABLISHED_MATURITY': 0.30,
                'RENEWAL_OR_DECLINE': 0.20
            },
            'error_correction': {
                'INITIAL_DEVELOPMENT': 0.05,
                'RAPID_GROWTH': 0.10,
                'ESTABLISHED_MATURITY': 0.14,
                'RENEWAL_OR_DECLINE': 0.18
            }
        }

def extract_city_from_suburb(suburb: str) -> str:
    """Extract city from 'Suburb STATE' format"""
    suburb_upper = suburb.upper()
    if 'NSW' in suburb_upper or 'SYDNEY' in suburb_upper:
        return 'Sydney'
    elif 'VIC' in suburb_upper or 'MELBOURNE' in suburb_upper:
        return 'Melbourne'
    elif 'QLD' in suburb_upper or 'BRISBANE' in suburb_upper:
        return 'Brisbane'
    elif 'WA' in suburb_upper or 'PERTH' in suburb_upper:
        return 'Perth'
    elif 'SA' in suburb_upper or 'ADELAIDE' in suburb_upper:
        return 'Adelaide'
    elif 'ACT' in suburb_upper or 'CANBERRA' in suburb_upper:
        return 'Canberra'
    else:
        return 'Sydney'  # Default

# Initialize Perplexity API
load_dotenv()
PERPLEXITY_API_KEY = os.getenv('PERPLEXITY_API_KEY')
PERPLEXITY_API_URL = "https://api.perplexity.ai/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
    "Content-Type": "application/json"
}

# Initialize logging handler
textbox_handler = TextboxHandler()
textbox_handler.addFilter(FontDebugFilter())
textbox_handler.setFormatter(
    logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
)

# Configure root logger
logging.getLogger().addHandler(textbox_handler)

# Filter matplotlib's font logger
matplotlib_logger = logging.getLogger('matplotlib.font_manager')
matplotlib_logger.addFilter(FontDebugFilter())

def set_verbose(enabled: bool) -> None:
    """Enable or disable verbose logging"""
    global VERBOSE
    VERBOSE = enabled
    level = logging.DEBUG if enabled else logging.WARNING
    logging.getLogger().setLevel(level)
    textbox_handler.clear() 

def log_api_response(suburb: str, payload: dict, response: requests.Response) -> None:
    """Log API request/response details if verbose mode enabled"""
    if not VERBOSE:
        return
        
    logging.debug(f"\n=== API Call for {suburb} ===")
    logging.debug(f"Request Payload:\n{json.dumps(payload, indent=2)}")
    logging.debug(f"Response Status: {response.status_code}")
    logging.debug(f"Response Content:\n{response.text}\n")

# Debugging Flag
VERBOSE = False

# number cleaner helper
def clean_numeric_value(value):
    """Convert string numbers with commas to float"""
    if isinstance(value, str):
        return float(value.replace(',', ''))
    return float(value)

def fetch_top_growth_suburbs(state=None, limit=5, max_price=None, retries=3, verbose: Optional[bool] = None):
    # Add default return value
    suburbs = []
    
    if verbose is not None:
        set_verbose(verbose)

    price_constraint = f"with median prices under ${max_price:,.0f} " if max_price else ""

    payload = {
        "model": "sonar-pro",
        "messages": [
            {
                "role": "system",
                "content": "You are an expert Australian real estate analyst. Return only a clean JSON array of suburb strings without any additional text or reasoning."
            },
            {
                "role": "user",
                "content": f"Return a JSON array of exactly {limit} top Australian growth suburbs" +
                          "for the last 6 months " +
                          (f"in {state} " if state else "in Australia ") +
                          price_constraint +
                          "based on projected 5-year growth rate. " +
                          "Format each suburb as 'Suburb STATE' where STATE is the 2-3 letter code. " +
                          "Return only the JSON array, no additional text."
            }
        ],
        "temperature": 0.2
    }

    for attempt in range(retries):
        try:
            response = requests.post(PERPLEXITY_API_URL, json=payload, headers=HEADERS, timeout=600)
            response.raise_for_status()

            logging.debug(f"State: {state if state else 'All Australia'}")
            logging.debug(f"Limit: {limit}")
            logging.debug(f"Max Price: {max_price if max_price else 'No limit'}")
            logging.debug(f"Attempt: {attempt + 1}/{retries}")
            log_api_response(state if state else "All Australia", payload, response)

            data = response.json()
            content = data['choices'][0]['message']['content']

            # Handle <think> section if present
            if '</think>' in content:
                content = content.split('</think>')[-1].strip()

            json_start = content.find('[')
            json_end = content.rfind(']') + 1
            json_str = content[json_start:json_end]

            suburbs = json.loads(json_str)
            return suburbs[:limit]

        except Exception as e:
            logging.error(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt == retries - 1:
                return []  # Return empty list on final failure
            time.sleep(2 ** attempt)
    
    return suburbs 

def process_suburb_input(suburb_text, max_price=None):
    """Process suburb input with dynamic API fetching"""
    state_mappings = {
        'NEW SOUTH WALES': 'NSW',
        'NSW': 'NSW',
        'VICTORIA': 'VIC',
        'VIC': 'VIC',
        'QUEENSLAND': 'QLD',
        'QLD': 'QLD',
        'WESTERN AUSTRALIA': 'WA',
        'WA': 'WA',
        'SOUTH AUSTRALIA': 'SA',
        'SA': 'SA',
        'TASMANIA': 'TAS',
        'TAS': 'TAS',
        'NORTHERN TERRITORY': 'NT',
        'NT': 'NT',
        'AUSTRALIAN CAPITAL TERRITORY': 'ACT',
        'ACT': 'ACT'
    }
    
    if not suburb_text.strip():
        # Fetch top 10 nationwide when input is empty
        return fetch_top_growth_suburbs(limit=5, max_price=max_price) or []

    input_text = suburb_text.upper().strip()
    input_lines = [line.strip() for line in input_text.splitlines() if line.strip()]
    
    # Check if input contains ONLY a state reference
    state_only = True
    state_matches = []
    for line in input_lines:
        for full_name, abbrev in state_mappings.items():
            if full_name == line or abbrev == line:
                state_matches.append(abbrev)
                continue
            if full_name in line or abbrev in line:
                state_only = False
                break
    
    if state_only and state_matches:
        # Fetch top suburbs only for explicitly mentioned states
        selected_suburbs = []
        for state in state_matches:
            state_suburbs = fetch_top_growth_suburbs(state=state, limit=5, max_price=max_price) or []
            selected_suburbs.extend(state_suburbs)
        return selected_suburbs or input_lines
    
    # Return exact suburbs provided without fetching additional ones
    return input_lines

def extract_valid_json(response_content: str) -> dict:
    """
    Extracts and returns only the valid JSON part from a response content string.

    This function handles potential extra text or markdown and extracts clean JSON.

    Parameters:
        response_content (str): The content string from the API response.

    Returns:
        dict: The parsed JSON object extracted from the content.

    Raises:
        ValueError: If no valid JSON can be parsed from the content.
    """
    # Try to find JSON directly first (sonar-pro doesn't use <think> tags)
    # But keep support for <think> tags if they appear
    marker = "</think>"
    idx = response_content.rfind(marker)

    if idx == -1:
        # If marker not found, try parsing the entire content.
        # First, remove markdown code fences if present
        content_to_parse = response_content.strip()

        if content_to_parse.startswith("```json"):
            content_to_parse = content_to_parse[7:].strip()
        elif content_to_parse.startswith("```"):
            content_to_parse = content_to_parse[3:].strip()

        if content_to_parse.endswith("```"):
            content_to_parse = content_to_parse[:-3].strip()

        # Remove any trailing explanation text after JSON
        # Look for common separators
        separators = ['\n- ', '\n\n', '\nNote:', '\nAll values']
        for sep in separators:
            if sep in content_to_parse:
                # Find where JSON likely ends
                brace_count = 0
                for i, char in enumerate(content_to_parse):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            # This is likely the end of JSON
                            content_to_parse = content_to_parse[:i+1]
                            break

        try:
            # Try to find JSON boundaries
            json_start = content_to_parse.find('{')
            json_end = content_to_parse.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                json_str = content_to_parse[json_start:json_end]
            else:
                json_str = content_to_parse

            # Fix invalid JSON: Remove underscores from numbers (e.g., 1_299_000 -> 1299000)
            # This is a workaround for Perplexity API sometimes returning numbers with underscores
            json_str = re.sub(r'(\d)_(\d)', r'\1\2', json_str)

            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logging.error(f"No </think> marker found. Response preview: {response_content[:500]}")
            raise ValueError("No </think> marker found and content is not valid JSON") from e

    # Extract the substring after the marker.
    json_str = response_content[idx + len(marker):].strip()

    # Remove markdown code fence markers if present (must be done before JSON parsing)
    # Handle both with and without language specifier
    if json_str.startswith("```json"):
        json_str = json_str[7:].strip()  # Remove ```json
    elif json_str.startswith("```"):
        json_str = json_str[3:].strip()  # Remove ```

    if json_str.endswith("```"):
        json_str = json_str[:-3].strip()  # Remove closing ```

    # Try to extract JSON object or array
    if not json_str:
        logging.error(f"Empty string after </think> marker. Full response: {response_content[:1000]}")
        raise ValueError("Empty content after </think> marker")

    # Find JSON boundaries more robustly - find first complete JSON object
    json_start = json_str.find('{')
    if json_start == -1:
        json_start = json_str.find('[')

    if json_start != -1:
        json_str = json_str[json_start:]

        # Fix invalid JSON: Remove underscores from numbers (e.g., 1_299_000 -> 1299000)
        json_str = re.sub(r'(\d)_(\d)', r'\1\2', json_str)

        # Use a decoder to find the end of the first valid JSON object
        decoder = json.JSONDecoder()
        try:
            parsed_json, end_idx = decoder.raw_decode(json_str)
            return parsed_json
        except json.JSONDecodeError:
            # If that fails, try the old method
            if json_str[0] == '{':
                # Count braces to find matching closing brace
                brace_count = 0
                for i, char in enumerate(json_str):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            json_str = json_str[:i+1]
                            break
            else:
                json_end = json_str.rfind(']') + 1
                if json_end > 0:
                    json_str = json_str[:json_end]

    try:
        parsed_json = json.loads(json_str)
        return parsed_json
    except json.JSONDecodeError as e:
        logging.error(f"JSON decode error. Extracted string: {json_str[:500]}")
        logging.error(f"Full response content: {response_content[:1000]}")
        raise ValueError(f"Failed to parse valid JSON from response content: {str(e)}") from e

def fetch_suburb_data(suburbs, dwelling_type=None, bedrooms=None, bathrooms=None, car_spaces=None, land_size=None, max_price=None, verbose: Optional[bool] = None):
    """
    Fetch suburb data with property type specifications and inflation metrics

    Parameters:
        suburbs (str): Newline-separated list of suburbs
        dwelling_type (str): Type of property (house, apartment, townhouse)
        bedrooms (int, optional): Number of bedrooms
        bathrooms (int, optional): Number of bathrooms
        car_spaces (int, optional): Number of car spaces
        land_size (float, optional): Land size in square meters
        max_price (float, optional): Maximum median price filter
        verbose (bool, optional): Enable detailed logging
    """
    if verbose is not None:
        set_verbose(verbose)

    for attempt in range(3):  # Try up to 3 times
        try:
            all_data = {"suburbs": {}}
            dataframes = []
            
            suburb_list = [s.strip() for s in suburbs.splitlines() if s.strip()]
            
            for suburb in suburb_list:
                try:
                    # First get default averages and inflation metrics
                    default_payload = {
                        "model": "sonar-pro",
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are an expert Australian real estate analyst and economist. Return average property metrics and inflation forecasts."
                            },
                            {
                                "role": "user",
                                "content": f"""Return metrics for {suburb}, Australia as JSON:
                                {{
                                    "bedrooms": {{ "value": (average number), "range": "1-6" }},
                                    "bathrooms": {{ "value": (average number), "range": "1-4" }},
                                    "car_spaces": {{ "value": (average number), "range": "0-3" }},
                                    "land_size": {{ "value": (average size), "unit": "sqm" }},
                                    "inflation": {{
                                        "current": {{ "value": (rate), "unit": "percent" }},
                                        "forecast": {{ "value": (rate), "unit": "percent" }},
                                        "volatility": {{ "value": (rate), "unit": "percent" }}
                                    }}
                                }}"""
                            }
                        ],
                        "temperature": 0.0
                    }

                    default_response = requests.post(PERPLEXITY_API_URL, json=default_payload, headers=HEADERS, timeout=600)
                    default_response.raise_for_status()
                    default_content = default_response.json()['choices'][0]['message']['content']

                    # Parse JSON with <think> section handling
                    default_data = extract_valid_json(default_content)
                    log_api_response(suburb, default_payload, default_response)

                    # Get housing inventory data
                    inventory_payload = {
                        "model": "sonar-pro",
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are an expert Australian real estate analyst. Return only numerical values."
                            },
                            {
                                "role": "user",
                                "content": f"Return the current housing inventory in {suburb} in Australia as a single number only. No additional text. " +
                                        "If the data is unavailable, return market average of 6."
                            }
                        ],
                        "temperature": 0.0
                    }

                    inventory_response = requests.post(PERPLEXITY_API_URL, json=inventory_payload, headers=HEADERS, timeout=600)
                    inventory_response.raise_for_status()
                    inventory_content = inventory_response.json()['choices'][0]['message']['content']
                    # Extract number from response
                    inventory_months = float(re.search(r'\d+\.?\d*', inventory_content).group())

                    supply_ratio = np.where(
                        inventory_months < 1.5,
                        0.25 + (inventory_months/6),  # Hyper-supply constraint premium
                        inventory_months/6  # normalized standard ratio for normal conditions
                    )  
                    
                    log_api_response(suburb, inventory_payload, inventory_response)

                    # Then get specific or default data
                    criteria = []
                    if dwelling_type:
                        criteria.append(f"type: {dwelling_type}")
                    if bedrooms:
                        criteria.append(f"with {bedrooms} bedrooms")
                    if bathrooms:
                        criteria.append(f"{bathrooms} bathrooms")
                    if car_spaces:
                        criteria.append(f"{car_spaces} car spaces")
                    if land_size:
                        criteria.append(f"land size of {land_size} sqm")
                    if max_price:
                        criteria.append(f"median price under ${max_price:,}")
                    
                    criteria_str = " and ".join(criteria)
                    if criteria_str:
                        criteria_str = f" for properties {criteria_str}"

                    suburb_payload = {
                        "model": "sonar-pro",
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are an expert Australian real estate analyst. Return only a JSON object with numerical values and their units/ranges." +
                                    "Only use data published in the last 6 months. "
                            },
                            {
                                "role": "user",
                                "content": f"""Return a JSON object for {suburb}, Australia{criteria_str}, with:
                                {{
                                    "median_price": {{ "value": (price), "unit": "AUD" }},
                                    "dwelling_type": "{dwelling_type if dwelling_type else 'house'}",
                                    "bedrooms": {{ "value": {bedrooms if bedrooms else "(number)"}, "range": "1-6" }},
                                    "bathrooms": {{ "value": {bathrooms if bathrooms else "(number)"}, "range": "1-4" }},
                                    "car_spaces": {{ "value": {car_spaces if car_spaces else "(number)"}, "range": "0-3" }},
                                    "land_size": {{ "value": {land_size if land_size else "(size)"}, "unit": "sqm" }},
                                    "distance_to_cbd": {{ "value": (distance), "unit": "km" }},
                                    "commercial_centers": {{
                                        "major": {{ 
                                            "count": (number), 
                                            "avg_distance": (km),
                                            "size_score": {{ "value": (score), "range": "1-10" }}
                                        }},
                                        "local": {{ 
                                            "count": (number), 
                                            "avg_distance": (km),
                                            "retail_quality": {{ "value": (score), "range": "1-10" }}
                                        }}
                                    }},
                                    "school_quality": {{ "value": (score), "range": "1-10" }},
                                    "infrastructure_score": {{ "value": (score), "range": "1-10" }},
                                    "water_features": {{
                                        "distance": {{ "value": (distance), "unit": "km" }},
                                        "type": "beach/river/harbor"
                                    }},
                                    "parks": {{
                                        "quality_score": {{ "value": (score), "range": "1-10" }},
                                        "distance": {{ "value": (distance), "unit": "km" }}
                                    }},
                                    "zoning": {{
                                        "view_protection": {{ "value": true/false }},
                                        "height_restrictions": {{ "value": (meters) }}
                                    }},
                                    "flood_risk": {{ "value": (risk), "range": "1-10" }},
                                    "population_growth": {{ "value": (rate), "unit": "% per year" }},
                                    "crime_stats": {{
                                        "assault_rate": {{ "value": (per 1000 residents), "source": "ABS" }},
                                        "breakin_rate": {{ "value": (per 1000 households), "source": "ABS" }},
                                        "security_adoption": {{ 
                                            "enhanced_security": {{ "value": (percentage), "source": "ABS" }},
                                            "basic_security": {{ "value": (percentage), "source": "ABS" }}
                                        }}
                                    }},
                                    "climate_risk": {{ "value": (risk), "range": "1-10" }},
                                    "public_transport": {{ "value": (score), "range": "1-10" }},
                                    "demographics": {{
                                        "median_age": {{ "value": (age), "unit": "years" }},
                                        "household_size": {{ "value": (size), "unit": "persons" }},
                                        "household_income": {{ "value": (income), "unit": "AUD_weekly" }},
                                        "family_composition": {{
                                            "couples_with_children": {{ "value": (percentage), "unit": "percent" }},
                                            "couples_no_children": {{ "value": (percentage), "unit": "percent" }},
                                            "single_parent": {{ "value": (percentage), "unit": "percent" }}
                                        }},
                                        "age_distribution": {{
                                            "under_35": {{ "value": (percentage), "unit": "percent" }},
                                            "35_to_65": {{ "value": (percentage), "unit": "percent" }},
                                            "over_65": {{ "value": (percentage), "unit": "percent" }}
                                        }}
                                    }}
                                }}"""
                            }
                        ],
                        "temperature": 0.0
                    }

                    response = requests.post(PERPLEXITY_API_URL, json=suburb_payload, headers=HEADERS, timeout=600)
                    response.raise_for_status()

                    # Get response and validate structure
                    response_json = response.json()
                    if 'choices' not in response_json or not response_json['choices']:
                        print(f"Error for {suburb}: Invalid response structure - no 'choices' field")
                        print(f"Response keys: {response_json.keys()}")
                        continue

                    if 'message' not in response_json['choices'][0]:
                        print(f"Error for {suburb}: Invalid response structure - no 'message' field")
                        print(f"Choice keys: {response_json['choices'][0].keys()}")
                        continue

                    content = response_json['choices'][0]['message']['content']

                    # Parse JSON with <think> section handling
                    try:
                        suburb_data = extract_valid_json(content)
                    except (ValueError, json.JSONDecodeError) as e:
                        error_msg = f"Failed to extract JSON for {suburb}: {str(e)}"
                        print(error_msg)

                        # Save failed response to file for debugging
                        debug_dir = Path('./debug_responses')
                        debug_dir.mkdir(exist_ok=True)
                        debug_file = debug_dir / f"{suburb.replace(' ', '_').replace(',', '')}_response.txt"
                        try:
                            with open(debug_file, 'w', encoding='utf-8') as f:
                                f.write(f"Suburb: {suburb}\n")
                                f.write(f"Error: {str(e)}\n")
                                f.write(f"\n{'='*80}\n")
                                f.write(f"Full Response:\n")
                                f.write(content)
                            print(f"Debug response saved to {debug_file}")
                        except Exception as file_err:
                            print(f"Failed to save debug file: {file_err}")

                        continue  # Skip this suburb and try the next one

                    log_api_response(suburb, suburb_payload, response)
                   
                    # Add supply ratio and inflation data
                    suburb_data["supply_ratio"] = {
                        "value": float(supply_ratio),
                        "benchmark": 6,
                        "unit": "ratio"
                    }
                    
                    suburb_data["inflation"] = default_data["inflation"]
                    
                    # Check if suburb price exceeds maximum price filter
                    if max_price and suburb_data['median_price']['value'] > max_price:
                        continue  
                    
                    # Add default averages to the data structure
                    suburb_data['default_averages'] = {
                        'bedrooms': default_data['bedrooms'],
                        'bathrooms': default_data['bathrooms'],
                        'car_spaces': default_data['car_spaces'],
                        'land_size': default_data['land_size']
                    }
                    
                    all_data["suburbs"][suburb] = suburb_data
                    
                    # # Print debugging information for data types
                    # print(f"\nDebugging {suburb} data structure:")
                    # for key, value in suburb_data.items():
                    #     print(f"{key}: {type(value)}")
                    #     if isinstance(value, dict):
                    #         for sub_key, sub_value in value.items():
                    #             print(f"  {sub_key}: {type(sub_value)}")
                    
                    # Create DataFrame for forecasting with all metrics
                    df = pd.DataFrame([{
                        'suburb': suburb,
                        'median_price': suburb_data['median_price']['value'],
                        'dwelling_type': suburb_data['dwelling_type'],
                        'bedrooms': suburb_data['bedrooms']['value'],
                        'bathrooms': suburb_data['bathrooms']['value'],
                        'car_spaces': suburb_data['car_spaces']['value'],
                        'land_size': suburb_data['land_size']['value'],
                        'distance_to_cbd': suburb_data['distance_to_cbd']['value'],
                        'major_commercial_count': suburb_data['commercial_centers']['major']['count'],
                        'major_commercial_distance': suburb_data['commercial_centers']['major']['avg_distance'],
                        'major_center_size_score': suburb_data['commercial_centers']['major']['size_score']['value'],
                        'local_commercial_count': suburb_data['commercial_centers']['local']['count'],
                        'local_commercial_distance': suburb_data['commercial_centers']['local']['avg_distance'],
                        'retail_quality_score': suburb_data['commercial_centers']['local']['retail_quality']['value'],
                        'school_quality': suburb_data['school_quality']['value'],
                        'infrastructure_score': suburb_data['infrastructure_score']['value'],
                        'flood_risk': suburb_data['flood_risk']['value'],
                        'population_growth': suburb_data['population_growth']['value'],
                        'assault_rate': suburb_data['crime_stats']['assault_rate']['value'],
                        'breakin_rate': suburb_data['crime_stats']['breakin_rate']['value'],
                        'enhanced_security': suburb_data['crime_stats']['security_adoption']['enhanced_security']['value'],
                        'basic_security': suburb_data['crime_stats']['security_adoption']['basic_security']['value'],
                        'security_score': (
                            0.6 * suburb_data['crime_stats']['security_adoption']['enhanced_security']['value'] +
                            0.4 * suburb_data['crime_stats']['security_adoption']['basic_security']['value']
                        ),
                        'climate_risk': suburb_data['climate_risk']['value'],
                        'public_transport': suburb_data['public_transport']['value'],
                        'supply_ratio': suburb_data['supply_ratio']['value'],
                        'water_distance': suburb_data['water_features']['distance']['value'],
                        'water_type': suburb_data['water_features']['type'],
                        'park_distance': suburb_data['parks']['distance']['value'],
                        'park_quality': suburb_data['parks']['quality_score']['value'],
                        'view_protection': suburb_data['zoning']['view_protection']['value'],
                        'height_restriction': suburb_data['zoning']['height_restrictions']['value'],
                        'median_age': suburb_data['demographics']['median_age']['value'],
                        'household_size': suburb_data['demographics']['household_size']['value'],
                        'household_income': suburb_data['demographics']['household_income']['value'],
                        'couples_with_children': suburb_data['demographics']['family_composition']['couples_with_children']['value'],
                        'couples_no_children': suburb_data['demographics']['family_composition']['couples_no_children']['value'],
                        'single_parent': suburb_data['demographics']['family_composition']['single_parent']['value'],
                        'population_under_35': suburb_data['demographics']['age_distribution']['under_35']['value'],
                        'population_35_to_65': suburb_data['demographics']['age_distribution']['35_to_65']['value'],
                        'population_over_65': suburb_data['demographics']['age_distribution']['over_65']['value'],
                        'current_inflation': suburb_data['inflation']['current']['value'],
                        'forecast_inflation': suburb_data['inflation']['forecast']['value'],
                        'inflation_volatility': suburb_data['inflation']['volatility']['value']
                    }])
                    dataframes.append(df)

                except json.JSONDecodeError as e:
                    logging.error(f"JSON parsing error for {suburb}: {str(e)}")
                    logging.error(f"Raw response:\n{response.text}")
                    continue    
                except Exception as e:
                    print(f"Error processing {suburb}: {str(e)}")
                    continue

            if not dataframes:
                raise ValueError("No valid suburb data collected")
            
            # Convert any numpy values to native Python types in all_data
            for suburb, data in all_data['suburbs'].items():
                for key, value in data.items():
                    if isinstance(value, np.ndarray):
                        data[key] = value.tolist()
                    elif isinstance(value, np.generic):
                        data[key] = value.item()
                    elif isinstance(value, dict):
                        for sub_key, sub_value in value.items():
                            if isinstance(sub_value, np.ndarray):
                                value[sub_key] = sub_value.tolist()
                            elif isinstance(sub_value, np.generic):
                                value[sub_key] = sub_value.item()
            
            return pd.concat(dataframes, ignore_index=True), json.dumps(all_data, indent=4)
        
        except Exception as e:
            logging.error(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt == 2:  # Last attempt
                raise gr.Error("Failed to process suburb data after 3 attempts. Please try again.")
            time.sleep(2 ** attempt)  # Exponential backoff

def generate_default_growth_rates(num_years):
    """Generate cyclical growth rates with 15-year market cycles using sine waves"""
    # Long-term base rate of 3.5%
    base_rate = 0.035
    
    # Create array for all years
    rates = np.zeros(num_years)
    
    # Generate sine wave with 15-year period
    # 2π/15 gives us exactly one cycle every 15 years
    cycle_positions = np.arange(num_years)
    sine_wave = np.sin(2 * np.pi * cycle_positions / 15)
    
    # Scale sine wave to desired amplitude range (3-11%)
    # Sine wave ranges from -1 to 1, so we scale and shift to get 3-11%
    min_rate = 0.03  # 3%
    max_rate = 0.11  # 11%
    amplitude = (max_rate - min_rate) / 2
    mean_rate = (max_rate + min_rate) / 2
    
    rates = mean_rate + (amplitude * sine_wave)
    
    # Add controlled random noise (small variations within bounds)
    noise = np.random.normal(0, 0.005, num_years)  # Small noise factor
    rates += noise
    
    # Ensure rates stay within 3-11% bounds
    rates = np.clip(rates, min_rate, max_rate)
    
    # Adjust to maintain long-term average close to base_rate
    average_adjustment = base_rate - np.mean(rates)
    rates += average_adjustment
    
    # Final clip to ensure bounds after adjustment
    rates = np.clip(rates, min_rate, max_rate)
    
    return rates * 100  # Convert to percentage

def fetch_abs_historical_growth(suburb):
    """Fetch historical housing growth data from ABS for the past 50 years"""
    current_year = datetime.now().year
    start_year = current_year - 50
    
    # Not Working ABS doesn't have this data publicly
    # payload = {
    #     "model": "sonar-pro",
    #     "messages": [
    #         {
    #             "role": "system",
    #             "content": "You are an ABS data specialist. Return only numerical year-on-year Australian housing growth rates from ABS housing data."
    #         },
    #         {
    #             "role": "user",
    #             "content": f"""Return a JSON array of year-on-year housing price growth rates for {suburb} 
    #                           from {start_year} to {current_year} using ABS data from the domain abs.gov.au only. 
    #                           Format: [growth_rate1, growth_rate2, ...] 
    #                           Include exactly {current_year - start_year} values."""
    #         }
    #     ],
    #     "temperature": 0.0,
    #     # "search_domain_filter": "abs.gov.au" # Retricted perplexity function. Requires Tier 3 usage.
    # }

    # try:
    #     response = requests.post(PERPLEXITY_API_URL, json=payload, headers=HEADERS)
        
    #     # Log the response for debugging
    #     logging.debug(f"API Response for {suburb}: Status {response.status_code}")
    #     logging.debug(f"Response content: {response.text}")
        
    #     # Handle different response status codes
    #     if response.status_code != 200:
    #         logging.warning(f"API request failed for {suburb} with status {response.status_code}")
    #         return generate_default_growth_rates(current_year - start_year)
            
    #     content = response.json()['choices'][0]['message']['content']
        
    #     # Find JSON array bounds more robustly
    #     json_start = content.find('[')
    #     json_end = content.rfind(']') + 1
        
    #     if json_start == -1 or json_end == 0:
    #         logging.warning(f"No valid JSON array found in response for {suburb}")
    #         return generate_default_growth_rates(current_year - start_year)
            
    #     json_str = content[json_start:json_end]
        
    #     try:
    #         growth_rates = json.loads(json_str)
    #         rates = np.array(growth_rates)
    #         return rates, 'ABS'
    #     except json.JSONDecodeError as e:
    #         logging.warning(f"JSON parsing failed for {suburb}: {str(e)}")
    #         rates = generate_default_growth_rates(current_year - start_year)
    #         return rates, 'DEFAULT'  

    # except Exception as e:
        # Return default rates with source identifier
    rates = generate_default_growth_rates(current_year - start_year)
    return rates, 'DEFAULT'

def calculate_location_premium(distance_km, public_transport_score):
    """
    Calculate location-based price growth premium considering:
    - Non-linear distance decay from CBD
    - Public transport accessibility offset
    - Combined interaction effects
    """
    # Base distance decay using sigmoid function
    # Steeper decline 0-10km, gradual 10km+ 
    base_distance_impact = 0.015 / (1 + np.exp(0.3 * (distance_km - 8)))
    
    # Transport accessibility premium
    # Scale: 0.001 (poor) to 0.008 (excellent)
    transport_premium = (public_transport_score / 10) * 0.008
    
    # Interaction boost for well-connected outer areas
    interaction_boost = 0
    if distance_km > 15 and public_transport_score >= 7:
        interaction_boost = 0.003
        
    return base_distance_impact + transport_premium + interaction_boost

def calculate_commercial_premium(major_count, major_distance, major_size_score,
                                local_count, local_distance, retail_quality):
    """
    Commercial premium calculation: 
    - Major centers (0-1.2% premium)
    - Local centers (0-0.6% premium)
    - Synergy bonus for major + local centers
    - Retail quality impact
    - Distance decay
    """
    # Major center impact (0-1.2% premium)
    major_decay = np.exp(-major_distance/8)  # 8km half-life
    major_premium = min(0.012, 
                      (major_count * 0.004 * major_size_score) * major_decay)
    
    # Local center impact (0-0.6% premium)
    local_decay = np.exp(-local_distance/4)  # 4km half-life
    local_premium = min(0.006, 
                       (local_count * 0.002 * retail_quality) * local_decay)
    
    # Combined premium with synergy bonus
    synergy_bonus = 0.0015 if (major_count >=1 and local_count >=3) else 0
    return min(0.015, major_premium + local_premium + synergy_bonus)  # Max 1.5%

def calculate_water_premium(distance_km, water_type):
    """Calculate premium based on water proximity and type"""
    base_premium = 0.12 * np.exp(-distance_km / 2)
    
    # Additional multipliers by water type
    type_multipliers = {
        'beach': 1.2,  # Beaches command highest premium
        'harbor': 1.1, # Harbor views also valuable
        'river': 0.9   # River proximity slightly lower
    }
    
    return base_premium * type_multipliers.get(water_type.lower(), 1.0)

def calculate_park_premium(distance_km, quality_score):
    """Calculate premium for park proximity and quality"""
    distance_decay = np.exp(-distance_km / 1.5)  # Steeper decay for parks
    base_premium = 0.08 if quality_score >= 7 else 0.03
    return base_premium * distance_decay

def calculate_risk_adjustments(flood_risk_score, climate_risk_score):
    """Calculate risk-based price adjustments"""
    flood_discount = np.where(flood_risk_score > 5, -0.22, 0)  # 22% discount for high flood risk
    climate_discount = np.where(climate_risk_score > 7, -0.15, 0)  # 15% for severe climate risk
    return flood_discount + climate_discount

def calculate_view_premium(has_protection, height_limit):
    """Calculate premium for protected views"""
    base_premium = 0.15 if has_protection else 0
    
    # Additional premium for strict height limits
    height_premium = 0
    if height_limit and height_limit < 15:  # 15m ~ 5 stories
        height_premium = 0.05
        
    return base_premium + height_premium

def calculate_demographic_premium(row):
    """Calculate premium based on demographic factors"""
    # Age-based adjustments with migration consideration
    youth_premium = 0.002 * (row['population_under_35'] / 100)
    aging_impact = -0.001 * (row['population_over_65'] / 100)
    
    # Family composition impact
    family_premium = (
        0.0015 * (row['couples_with_children'] / 100) +
        0.0008 * (row['couples_no_children'] / 100)
    )
    
    # Income effects with dwelling type consideration
    income_multiplier = 1.2 if row['dwelling_type'] == 'house' else 0.9
    income_premium = (
        0.0004 * (row['household_income'] / 1000) * 
        income_multiplier
    )
    
    # Household size impact
    size_premium = 0.0003 * row['household_size'] * (
        1.2 if row['dwelling_type'] == 'house' else 0.8
    )
    
    return (youth_premium + aging_impact + family_premium + 
            income_premium + size_premium)

def calculate_crime_impact(row):
    """
    Calculate crime impact on property values with regional differentiation
    and security mitigation factors
    """
    # Violent crime penalties with regional context
    violent_crime_penalty = np.where(
        row['assault_rate'] > 5,
        -0.0003 * np.log(row['assault_rate']),
        0
    )
    
    # Property crime penalties adjusted for location type
    property_crime_penalty = np.where(
        row['breakin_rate'] > 20,
        -0.00015 * (row['breakin_rate'] - 20),
        0
    )
    
    # Security adoption impact with thresholds
    security_mitigation = 0.0002 * row['security_score']
    
    # Gentrification boost for high-growth areas
    # Using absolute income threshold instead of median comparison
    gentrification_boost = np.where(
        (row['population_growth'] > 2.5) &
        (row['household_income'] > 2000),  # Weekly household income threshold
        0.0005,
        0
    )
    
    return (
        violent_crime_penalty + 
        property_crime_penalty + 
        security_mitigation + 
        gentrification_boost
    )

def fine_tune_model(model, new_data):
    """Incremental training with new suburb data including supply metrics"""
    # Select features including supply ratio
    X = new_data[[
        'population_growth',
        'infrastructure_score', 
        'school_quality',
        'public_transport',
        'flood_risk',
        'climate_risk',
        'supply_ratio',
        'distance_to_cbd',
        'land_size'
    ]]
    y = new_data['median_price']
    
    # Generate synthetic data for tuning with supply-aware variations
    synthetic_data = pd.DataFrame([
        new_data.iloc[0] * np.random.normal(1, 0.1, len(new_data.columns))
        for _ in range(100)
    ])
    
    # Adjust synthetic supply ratios to reflect realistic market conditions
    synthetic_data['supply_ratio'] = np.clip(
        synthetic_data['supply_ratio'],
        0.5,  # Minimum 3-month inventory
        2.0   # Maximum 12-month inventory
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        synthetic_data.drop('median_price', axis=1),
        synthetic_data['median_price'],
        test_size=0.2,
        random_state=42
    )
    
    # Access the XGBoost model from the pipeline
    xgb_model = model.named_steps['regressor']
    
    # Fine-tune with progressive learning and supply-aware parameters
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
        xgb_model=xgb_model  # Warm start from existing model
    )
    
    return model

def estimate_suburb_age(suburb_data: pd.DataFrame) -> np.ndarray:
    """
    Estimate years since maturity using demographic proxies
    Returns array of estimated ages for each suburb

    Lifecycle stages based on AHURI research:
    - Early growth (0-5 years): Large households, young population
    - Peak maturity (5-15 years): Families with children
    - Established (15-30 years): Empty nesters
    - Mature/Renewal (30+ years): Aging demographics or gentrification
    """
    ages = np.zeros(len(suburb_data))

    for i, (idx, row) in enumerate(suburb_data.iterrows()):
        # Early growth: Large households (>3.2), young population
        if row['household_size'] > 3.2:
            ages[i] = 5
        # Peak maturity: Families with children (households 2.8-3.2)
        elif row['household_size'] > 2.8:
            ages[i] = 15
        # Established: Smaller households (2.3-2.8), some empty nesters
        elif row['household_size'] > 2.3:
            ages[i] = 25
        # Mature: Very small households (<2.3), empty nesters/singles
        else:
            ages[i] = 35

    return ages

def calculate_growth_adjustments(suburb_data: pd.DataFrame) -> pd.Series:
    """
    Calculate moderated growth adjustments with research-backed coefficients AND lifecycle-aware maturity decay

    Key Components:
    1. Supply and Demand Metrics
    2. Population and Migration Effects
    3. Infrastructure and Location Impact
    4. Demographic Factors
    5. Risk Adjustments
    6. Affordability Constraints
    7. Crime Statistics
    8. LIFECYCLE MATURITY DECAY (NEW - Phase 1)
    """
    # ====== PHASE 1: MATURITY DECAY FACTOR ======
    # Estimate suburb age using demographic proxies
    years_since_maturity = estimate_suburb_age(suburb_data)

    # Maturity decay factor using S-curve centered at year 15
    # Early suburbs (0-5 years): factor ≈ 0.9-1.0 (minimal decay)
    # Peak suburbs (10-20 years): factor ≈ 0.5 (50% decay)
    # Mature suburbs (30+ years): factor ≈ 0.0-0.1 (near-zero growth from improvements)
    maturity_factor = 1 / (1 + np.exp(0.1 * (years_since_maturity - 15)))

    # Supply impact with continuous scaling (AHURI Report 281)
    supply_impact = np.where(
        suburb_data['supply_ratio'] < 0.5,
        0.001 * (1 - suburb_data['supply_ratio']),  # Max 0.1% boost
        np.where(
            suburb_data['supply_ratio'] > 1.5,
            -0.002 * (suburb_data['supply_ratio'] - 1),  # Scale with oversupply
            0
        )
    )

    # Population growth with migration effects (NOW WITH DECAY)
    base_pop_growth = np.minimum(0.0025, 0.8 * suburb_data['population_growth'])
    migration_boost = np.where(
        suburb_data['population_growth'] > 2.5,
        0.002 * (suburb_data['population_growth'] - 2.5),
        0
    )
    pop_growth_adj = (base_pop_growth + migration_boost) * maturity_factor  # <-- APPLY DECAY

    # Infrastructure impact with quality threshold (NOW WITH DECAY)
    infra_adj = np.where(
        suburb_data['infrastructure_score'] >= 7,
        suburb_data['infrastructure_score'] * 0.0015,
        suburb_data['infrastructure_score'] * 0.0005
    )
    infra_adj = infra_adj * maturity_factor  # <-- APPLY DECAY

    # School quality premium with non-linear scaling (NOW WITH DECAY)
    school_adj = np.power(suburb_data['school_quality']/10, 1.5) * 0.002
    school_adj = school_adj * maturity_factor  # <-- APPLY DECAY

    # Location premium with exponential decay (RBA distance model)
    def location_decay(distance, pt_score):
        base_decay = np.exp(-distance/8)  # 8km half-life
        pt_boost = (pt_score/10) * 0.0005
        return min(0.003, 0.002 * base_decay + pt_boost)

    # Commercial premium with retail quality weighting
    def commercial_impact(row):
        major_decay = np.exp(-row['major_commercial_distance']/6)
        local_decay = np.exp(-row['local_commercial_distance']/3)
        return (
            (row['major_commercial_count'] * 0.0004 * major_decay) +
            (row['local_commercial_count'] * 0.0002 * local_decay * row['retail_quality_score']/10)
        )

    # Water premium with type differentiation
    water_adj = np.where(
        suburb_data['water_type'] == 'ocean',
        0.0012 * np.exp(-suburb_data['water_distance']/2),
        np.where(
            suburb_data['water_type'] == 'river',
            0.0008 * np.exp(-suburb_data['water_distance']/3),
            0
        )
    )

    # Park premium with quality threshold
    park_adj = np.where(
        suburb_data['park_quality'] >= 8,
        0.0015 * np.exp(-suburb_data['park_distance']/2),
        np.where(
            suburb_data['park_quality'] >= 5,
            0.0005 * np.exp(-suburb_data['park_distance']/1.5),
            0
        )
    )

    # Risk adjustments with progressive scaling
    risk_adj = (
        -0.0005 * np.power(suburb_data['flood_risk'], 1.2) +
        -0.0003 * np.power(suburb_data['climate_risk'], 1.1)
    )

    # Housing type differentiation (PropertyUpdate 2025 predictions)
    dwelling_adj = np.where(
        suburb_data['dwelling_type'] == 'apartment',
        0.0002,
        np.where(
            suburb_data['dwelling_type'] == 'townhouse',
            0.0001,  # Moderate premium for townhouses
            -0.0001  # Detached houses becoming less affordable
        )
    )

    # Household size impact
    household_size_adj = 0.0003 * (2.8 - suburb_data['household_size'])

    # Demographic factors
    demographic_adj = suburb_data.apply(calculate_demographic_premium, axis=1)

    # Income-based affordability ceiling
    annual_income = (suburb_data['household_income'] / 52) * 52  # Weekly to annual
    income_ceiling = annual_income * 4.5  # APRA guideline multiplier
    max_sustainable_growth = income_ceiling / suburb_data['median_price']

    # Crime impact calculations
    crime_impact = suburb_data.apply(calculate_crime_impact, axis=1)

    # Combine all adjustments
    base_growth = (
        pop_growth_adj +
        infra_adj +
        school_adj +
        suburb_data.apply(lambda r: location_decay(r['distance_to_cbd'], r['public_transport']), axis=1) +
        suburb_data.apply(commercial_impact, axis=1) +
        water_adj +
        park_adj +
        risk_adj +
        dwelling_adj +
        demographic_adj +
        household_size_adj +
        crime_impact +
        supply_impact
    )

    # Apply affordability ceiling
    final_growth = np.minimum(base_growth, max_sustainable_growth)

    return final_growth

def calculate_growth_rates(projections):
    """Calculate average annual growth rates for different time horizons"""
    years = projections['years']
    nominal = projections['nominal']['median_projection']
    inflation_adj = projections['inflation_adjusted']['median_projection']
    
    # Find indices for time horizons
    yr5_idx = next(i for i, y in enumerate(years) if y-years[0] >= 5)
    yr25_idx = next(i for i, y in enumerate(years) if y-years[0] >= 25)
    yr50_idx = -1  # Last index
    
    # Calculate compound annual growth rates
    def calc_cagr(start_value, end_value, years):
        return (((end_value / start_value) ** (1/years)) - 1) * 100

    growth_rates = {
        "growth_rates": {
            "5_year": {
                "nominal": calc_cagr(nominal[0], nominal[yr5_idx], 5),
                "inflation_adjusted": calc_cagr(inflation_adj[0], inflation_adj[yr5_idx], 5)
            },
            "25_year": {
                "nominal": calc_cagr(nominal[0], nominal[yr25_idx], 25),
                "inflation_adjusted": calc_cagr(inflation_adj[0], inflation_adj[yr25_idx], 25)
            },
            "50_year": {
                "nominal": calc_cagr(nominal[0], nominal[yr50_idx], 50),
                "inflation_adjusted": calc_cagr(inflation_adj[0], inflation_adj[yr50_idx], 50)
            }
        }
    }
    
    return growth_rates

def classify_lifecycle_stage(years_since_maturity: float) -> str:
    """
    Map suburb age to discrete lifecycle stage

    Based on AHURI research patterns:
    - INITIAL_DEVELOPMENT (0-5 years): High growth, infrastructure building
    - RAPID_GROWTH (5-15 years): Peak maturity, infrastructure saturation
    - ESTABLISHED_MATURITY (15-30 years): Stable, moderate growth
    - RENEWAL_OR_DECLINE (30+ years): Either gentrification or decline
    """
    if years_since_maturity < 5:
        return 'INITIAL_DEVELOPMENT'
    elif years_since_maturity < 15:
        return 'RAPID_GROWTH'
    elif years_since_maturity < 30:
        return 'ESTABLISHED_MATURITY'
    else:
        return 'RENEWAL_OR_DECLINE'

def calculate_fundamental_value(suburb_row: pd.Series, state: dict,
                               year_idx: int = 0, rba_params: dict = None) -> float:
    """
    PHASE 4 POLISH: Calculate equilibrium price based on rental yield = user cost
    P* = Annual Rent / (interest rate + depreciation + maintenance - expected appreciation)

    Enhanced with:
    - Dynamic interest rate cycles (10-year RBA cash rate simulation)
    - Current price-based rent (rents adjust with market)
    - Stage-specific expected appreciation
    """
    # Load RBA parameters if not provided
    if rba_params is None:
        rba_params = load_rba_parameters()

    # Estimate annual rent from typical gross yields (distance-based)
    if suburb_row['distance_to_cbd'] < 20:
        gross_yield = 0.03  # Inner city - lower yields, higher growth expectations
    else:
        gross_yield = 0.04  # Regional/outer - higher yields, lower growth

    # ====== PHASE 4: Use current price for rent, not initial ======
    # Rents grow with market conditions
    annual_rent = state['current_price'] * gross_yield

    # ====== PHASE 4: Dynamic interest rate cycles ======
    # Simulates RBA cash rate movements over 10-year cycles
    base_interest_rate = 0.05  # 5% base mortgage rate
    cycle_amplitude = 0.015  # ±1.5% swing
    cycle_phase = (year_idx % 10) / 10 * 2 * np.pi
    interest_rate = base_interest_rate + (cycle_amplitude * np.sin(cycle_phase))

    # User cost components (from RBA model parameters)
    user_cost_params = rba_params.get('user_cost_components', {
        'depreciation': 0.015,
        'maintenance': 0.01,
        'expected_appreciation_long_run': 0.025
    })

    depreciation = user_cost_params.get('depreciation', 0.015)
    maintenance = user_cost_params.get('maintenance', 0.01)

    # ====== PHASE 4: Stage-specific expected appreciation ======
    stage = classify_lifecycle_stage(state['years_since_maturity'] + year_idx)
    stage_appreciation = {
        'INITIAL_DEVELOPMENT': 0.04,  # 4% expected in high-growth areas
        'RAPID_GROWTH': 0.035,  # 3.5% expected
        'ESTABLISHED_MATURITY': 0.025,  # 2.5% long-run average
        'RENEWAL_OR_DECLINE': 0.02  # 2% in mature areas
    }
    expected_appreciation = stage_appreciation.get(stage, 0.025)

    # User cost formula
    user_cost = interest_rate + depreciation + maintenance - expected_appreciation

    # Prevent division by zero
    if user_cost <= 0:
        user_cost = 0.01

    # Fundamental price where rental yield equals user cost
    fundamental_price = annual_rent / user_cost

    return fundamental_price

def calculate_year_growth(state: dict, suburb_row: pd.Series,
                         year_idx: int, rba_params: dict) -> float:
    """
    Calculate growth for THIS specific year based on CURRENT state
    Not Year 0 characteristics!

    Incorporates:
    - Lifecycle stage-specific momentum and mean reversion
    - Current maturity decay
    - Price momentum from prior year
    - Mean reversion to fundamental value
    """
    # Classify current lifecycle stage
    total_age = state['years_since_maturity'] + year_idx
    stage = classify_lifecycle_stage(total_age)

    # Get stage-specific parameters (RBA calibrated)
    momentum_coeff = rba_params['momentum'][stage]
    mean_reversion_speed = rba_params['error_correction'][stage]

    # Base components with current maturity decay
    maturity_factor = 1 / (1 + np.exp(0.1 * (total_age - 15)))

    # Population impact (decays with maturity)
    pop_impact = 0.0008 * suburb_row['population_growth'] * maturity_factor

    # Infrastructure impact (decays with maturity)
    infra_impact = 0.0015 * suburb_row['infrastructure_score'] * maturity_factor

    # School quality impact (decays with maturity)
    school_impact = 0.002 * (suburb_row['school_quality']/10) ** 1.5 * maturity_factor

    # Momentum from prior year (stage-specific coefficient)
    # Early stages: high momentum (0.74), mature stages: low momentum (0.20)
    momentum = momentum_coeff * state['prior_year_growth']

    # Mean reversion to fundamentals (stage-specific speed)
    # Early stages: slow reversion (0.05), mature stages: fast reversion (0.18)
    # PHASE 4: Pass year_idx and rba_params for dynamic interest rates and stage appreciation
    fundamental_price = calculate_fundamental_value(suburb_row, state, year_idx, rba_params)
    price_gap = np.log(fundamental_price / state['current_price'])
    reversion = mean_reversion_speed * price_gap

    # Total growth from all components
    total_growth = (pop_impact + infra_impact + school_impact +
                   momentum + reversion)

    return total_growth

def calculate_supply_adjustment(current_ratio: float, price_change: float,
                               growth_rate: float, years_since_maturity: float,
                               city_elasticity: float) -> float:
    """
    Calculate dynamic supply ratio adjustment based on construction response and demand absorption

    Research basis:
    - RBA: Construction responds with -6.2% approvals per 1pp price decline (detached)
    - US 2008: -78% construction decline during severe oversupply
    - Absorption: 15% baseline, decays with excess inventory severity
    - Time to respond: ~10 years for major demand surges (RBA 2019)

    Returns: Annual change in supply ratio (can be positive or negative)
    """
    supply_ratio_delta = 0.0

    # COMPONENT 1: Construction Response
    # When prices decline and oversupply exists, construction slows
    if current_ratio > 1.2 and growth_rate < 0:
        # RBA: -6.2% detached approvals per 1 percentage point price decline
        # Scale by severity of oversupply and price decline
        construction_slowdown = 0.062 * abs(growth_rate)

        # Elasticity decay with suburb age (mature suburbs less responsive)
        if years_since_maturity < 10:
            elasticity_multiplier = 1.2  # Young suburbs more responsive
        elif years_since_maturity < 30:
            elasticity_multiplier = 1.0  # Standard responsiveness
        else:
            elasticity_multiplier = 0.7  # Mature suburbs less responsive

        construction_slowdown *= elasticity_multiplier

        # Construction slowdown reduces supply ratio (less new inventory added)
        # Scale: 6.2% approval decline → ~0.5% annual supply ratio reduction
        supply_ratio_delta -= construction_slowdown * 0.08

    elif current_ratio < 1.0 and growth_rate > 0:
        # Undersupply: construction accelerates
        # RBA: +30% construction boost for +10% price increase
        construction_boost = min(0.30 * (growth_rate / 0.10), 0.15)  # Cap at 15%
        supply_ratio_delta += construction_boost * 0.05

    # COMPONENT 2: Demand Absorption
    # Excess inventory gets absorbed by demand over time
    if current_ratio > 1.0:
        excess_inventory = current_ratio - 1.0

        # Non-linear absorption: harder to absorb extreme oversupply
        # Base: 15% absorption rate, decays exponentially with severity
        # ratio 1.5 → 11.8%, ratio 5.0 → 3.0%, ratio 30.5 → 0.8%
        base_absorption = 0.15
        absorption_rate = base_absorption * np.exp(-0.15 * excess_inventory)

        # Absorption reduces supply ratio
        supply_reduction = absorption_rate * excess_inventory
        supply_ratio_delta -= supply_reduction

    return supply_ratio_delta

def apply_market_constraints(growth_rate: float, state: dict, year_idx: int) -> float:
    """
    Apply three research-backed constraints to prevent unrealistic growth:
    1. Supply response (Australia's 0.07 elasticity)
    2. Affordability brake (price-to-income threshold)
    3. Infrastructure saturation ceiling

    Based on RBA RDP 2018-03, 2019-01 and AHURI research
    """
    constrained_growth = growth_rate

    # CONSTRAINT 1: Supply dampening
    # When supply ratio >1.2 (>7 months inventory vs 6-month standard), oversupply dampens growth
    if state.get('supply_ratio', 1.0) > 1.2:
        # Research-calibrated dampening based on RBA demand elasticity and US housing crisis data:
        # - RBA: 1% supply increase → 2.5% price decrease (demand elasticity -0.4)
        # - US 2008 crisis: 39% higher inventory → 30% price decline over 3 years (~-11% annually)
        # - Markets stabilize at lower equilibrium, they don't collapse to zero
        #
        # Implementation: Moderate annual penalty approaching -2% ceiling
        # This represents adjustment toward new equilibrium, not perpetual collapse
        # ratio 1.2 → -0.05%, ratio 2.0 → -0.25%, ratio 5.0 → -0.86%, ratio 30.5 → -1.97%
        excess_supply = state['supply_ratio'] - 1.0
        supply_dampening = -0.02 * (1 - np.exp(-0.15 * excess_supply))
        constrained_growth += supply_dampening
        logging.debug(f"Supply dampening: {supply_dampening:.4f} (ratio: {state['supply_ratio']:.2f})")

    # CONSTRAINT 2: Affordability brake
    # Price-to-income ratios >10x trigger demand collapse
    # Historical evidence: ratios >10-12x lead to corrections
    affordability_ratio = state['current_price'] / state['annual_income']
    if affordability_ratio > 10:
        # Non-linear brake: accelerates as ratio increases
        # At 12x: ~-3.6% penalty, at 15x: ~-8.5% penalty
        affordability_brake = -0.02 * ((affordability_ratio - 10) ** 1.2)
        constrained_growth += affordability_brake
        constrained_growth = max(constrained_growth, -0.05)  # Floor at -5% to prevent crashes
        logging.debug(f"Affordability brake: {affordability_brake:.4f} (P/I: {affordability_ratio:.1f}x)")

    # CONSTRAINT 3: Infrastructure saturation ceiling
    # After 10 years, additional infrastructure provides diminishing returns
    # AHURI research: infrastructure saturation at 5-10 years post-development
    years_mature = state.get('years_since_maturity', 20) + year_idx
    if years_mature > 10:
        # Exponential decay: approaches -0.2% ceiling over time
        # Year 10: 0%, Year 20: -0.13%, Year 40: -0.19%
        infra_penalty = -0.002 * (1 - np.exp(-0.1 * (years_mature - 10)))
        constrained_growth += infra_penalty
        logging.debug(f"Infrastructure saturation: {infra_penalty:.4f} (age: {years_mature} years)")

    return constrained_growth

def validate_long_term_projections(suburb_name: str, median_trajectory: np.ndarray,
                                   suburb_data_row: pd.Series, years: np.ndarray) -> dict:
    """
    Sanity checks for 50-year forecasts
    Returns validation results and warnings

    Based on research constraints:
    - No suburb should show >200% real appreciation over 50 years
    - Late-stage growth should be <2% annually
    - Price-to-income ratio shouldn't exceed 20x
    """
    validation_results = {
        'suburb': suburb_name,
        'warnings': [],
        'passed': True
    }

    # Check 1: Total appreciation limit
    total_appreciation = (median_trajectory[-1] / median_trajectory[0]) - 1
    if total_appreciation > 2.0:
        validation_results['warnings'].append(
            f"⚠️ Total appreciation {total_appreciation:.1%} over 50 years - UNREALISTIC (>200%)"
        )
        validation_results['passed'] = False

    # Check 2: Late-stage growth moderation
    late_stage_prices = median_trajectory[-10:]
    late_stage_growth_rates = np.diff(late_stage_prices) / late_stage_prices[:-1]
    avg_late_growth = np.mean(late_stage_growth_rates)

    if avg_late_growth > 0.02:
        validation_results['warnings'].append(
            f"⚠️ Year 40-50 average growth {avg_late_growth:.1%} - should be <2% for mature suburbs"
        )
        validation_results['passed'] = False

    # Check 3: Price-to-income ratio ceiling
    final_price = median_trajectory[-1]
    initial_income = suburb_data_row['household_income'] * 52  # Weekly to annual
    # Assume 2.5% annual income growth over 50 years
    projected_income = initial_income * (1.025 ** 50)
    pti_ratio = final_price / projected_income

    if pti_ratio > 20:
        validation_results['warnings'].append(
            f"⚠️ Final price-to-income ratio {pti_ratio:.1f}x - MARKET WOULD COLLAPSE (>20x)"
        )
        validation_results['passed'] = False

    # PHASE 4: Check 4 - Mean reversion test
    # Prices should not deviate >50% from trend for extended periods
    # Calculate rolling 10-year trend and check for persistent deviations
    if len(median_trajectory) >= 10:
        rolling_deviations = []
        for i in range(10, len(median_trajectory)):
            window = median_trajectory[i-10:i]
            trend = np.polyfit(range(10), window, 1)[0] * 10  # 10-year trend
            current = median_trajectory[i]
            deviation = abs((current - window[-1]) / window[-1])
            rolling_deviations.append(deviation)

        max_deviation = max(rolling_deviations) if rolling_deviations else 0
        if max_deviation > 0.50:
            validation_results['warnings'].append(
                f"⚠️ Maximum 10-year deviation {max_deviation:.1%} - mean reversion may be weak (>50%)"
            )
            validation_results['passed'] = False

    # PHASE 4: Check 5 - Volatility explosion test
    # Standard deviation of year-to-year growth should not exceed 15%
    year_growth_rates = np.diff(median_trajectory) / median_trajectory[:-1]
    growth_volatility = np.std(year_growth_rates)
    if growth_volatility > 0.15:
        validation_results['warnings'].append(
            f"⚠️ Growth volatility {growth_volatility:.1%} - EXCESSIVE (>15%)"
        )
        validation_results['passed'] = False

    # Success metrics
    validation_results['metrics'] = {
        'total_appreciation_pct': total_appreciation * 100,
        'late_stage_avg_growth_pct': avg_late_growth * 100,
        'final_price_to_income_ratio': pti_ratio,
        'projected_50yr_income': projected_income,
        'max_10yr_deviation_pct': max_deviation * 100 if len(median_trajectory) >= 10 else 0,
        'growth_volatility_pct': growth_volatility * 100
    }

    if validation_results['passed']:
        logging.info(f"✅ {suburb_name} validation passed: "
                    f"{total_appreciation:.1%} appreciation, "
                    f"{avg_late_growth:.1%} late growth, "
                    f"{pti_ratio:.1f}x P/I ratio")
    else:
        for warning in validation_results['warnings']:
            logging.warning(f"{suburb_name}: {warning}")

    return validation_results

def forecast_prices(suburbs, dwelling_type=None, bedrooms=None, bathrooms=None, car_spaces=None, land_size=None, max_price=None):
    """Generate forecasts for multiple suburbs with optional property criteria"""
    # Fetch current data and API response with property criteria
    suburb_data, api_response = fetch_suburb_data(
        suburbs,
        dwelling_type=dwelling_type,
        bedrooms=bedrooms,
        bathrooms=bathrooms,
        car_spaces=car_spaces,
        land_size=land_size,
        max_price=max_price
    )
    plot_paths = []
    
    # Parse existing data
    data_dict = json.loads(api_response)
    
    # Calculate base growth rate adjustments from suburb metrics
    growth_adjustments = calculate_growth_adjustments(suburb_data)
    
    # Generate year range for projections
    current_year = datetime.now().year
    years = np.arange(current_year, current_year + 50)

    # For each suburb, add price projections
    for suburb, suburb_info in data_dict['suburbs'].items():
        # Get base price and inflation metrics for this suburb
        base_price = clean_numeric_value(suburb_info['median_price']['value'])
        forecast_inflation = suburb_info['inflation']['forecast']['value']
        inflation_volatility = suburb_info['inflation']['volatility']['value']
        
        # Calculate suburb-specific growth rate
        suburb_idx = suburb_data[suburb_data['suburb'] == suburb].index[0]
        suburb_row = suburb_data.iloc[suburb_idx]
        adjusted_growth = growth_adjustments.iloc[suburb_idx]

        # Calculate projections using Monte Carlo with adjusted rates
        n_simulations = 1000
        all_projections = []

        # Create base growth rate trajectory
        base_growth_rate = np.linspace(
            max(0.015, adjusted_growth - 0.005),  # Floor at 1.5%
            min(0.045, adjusted_growth + 0.01),   # Ceiling at 4.5%
            len(years)
        )

        # ====== PHASE 2: Apply market constraints year-by-year ======
        # Get initial state for constraint calculations
        initial_state = {
            'current_price': base_price,
            'annual_income': suburb_row['household_income'] * 52,  # Weekly to annual
            'supply_ratio': suburb_row['supply_ratio'],
            'years_since_maturity': estimate_suburb_age(pd.DataFrame([suburb_row]))[0]
        }

        # Generate nominal price projections with proper inflation handling
        for _ in range(n_simulations):
            # Initialize state for this simulation
            state = initial_state.copy()
            projections = [state['current_price']]

            # Generate growth and inflation noise arrays
            growth_noise = np.random.normal(0, 0.005, len(years) - 1)
            inflation_noise = np.random.normal(0, inflation_volatility/100, len(years) - 1)

            # Year-by-year projection with constraints
            for year_idx in range(len(years) - 1):
                # Base growth rate for this year
                base_rate = base_growth_rate[year_idx] + growth_noise[year_idx]
                inflation_rate = forecast_inflation/100 + inflation_noise[year_idx]

                # Calculate real growth with partial inflation pass-through
                real_growth = base_rate - (inflation_rate * 0.6)
                combined_growth = real_growth + inflation_rate

                # ====== APPLY MARKET CONSTRAINTS ======
                constrained_growth = apply_market_constraints(combined_growth, state, year_idx)

                # Update price for this year
                new_price = state['current_price'] * (1 + constrained_growth)
                projections.append(new_price)

                # Update state for next year
                state['current_price'] = new_price

                # Update supply ratio (very inelastic response - 0.07 elasticity)
                # Price growth → modest supply increase → dampens future growth
                if year_idx > 0:
                    cumulative_price_growth = (new_price / base_price) - 1
                    # Supply increases by 0.07 * price growth, spread over 5 years
                    supply_increase = (cumulative_price_growth * 0.07) / 5
                    state['supply_ratio'] = initial_state['supply_ratio'] + supply_increase

            all_projections.append(np.array(projections))
        
        # Convert to numpy array for calculations
        nominal_projections = np.array(all_projections)
        
        # Calculate nominal statistics
        nominal_median = np.median(nominal_projections, axis=0)
        nominal_lower_95 = np.percentile(nominal_projections, 2.5, axis=0)
        nominal_upper_95 = np.percentile(nominal_projections, 97.5, axis=0)
        nominal_lower_std = nominal_median - np.std(nominal_projections, axis=0)
        nominal_upper_std = nominal_median + np.std(nominal_projections, axis=0)

        # Calculate inflation factors for price adjustments
        inflation_noise = np.random.normal(0, inflation_volatility/100, (n_simulations, len(years)))
        inflation_factors = np.cumprod(1 + (forecast_inflation/100 + inflation_noise), axis=1)
        
        # Calculate inflation-adjusted projections
        inflation_adjusted_projections = nominal_projections * inflation_factors
        
        # Calculate inflation-adjusted statistics
        inflation_adj_median = np.median(inflation_adjusted_projections, axis=0)
        inflation_adj_lower_95 = np.percentile(inflation_adjusted_projections, 2.5, axis=0)
        inflation_adj_upper_95 = np.percentile(inflation_adjusted_projections, 97.5, axis=0)
        inflation_adj_lower_std = inflation_adj_median - np.std(inflation_adjusted_projections, axis=0)
        inflation_adj_upper_std = inflation_adj_median + np.std(inflation_adjusted_projections, axis=0)

        # ====== VALIDATION: Sanity checks for long-term projections ======
        suburb_row = suburb_data[suburb_data['suburb'] == suburb].iloc[0]
        validation_result = validate_long_term_projections(
            suburb, nominal_median, suburb_row, years
        )

        # Get historical ABS growth rates
        historical_rates, data_source = fetch_abs_historical_growth(suburb)
        historical_prices = base_price * np.cumprod(1 + historical_rates/100)
        
        # Verify dimensions match before plotting
        if len(historical_prices) < len(years):
            # Pad historical prices to match years if needed
            historical_prices = np.pad(historical_prices, 
                                    (0, len(years) - len(historical_prices)),
                                    'edge')
        elif len(historical_prices) > len(years):
            # Trim historical prices if too long
            historical_prices = historical_prices[:len(years)]

        # Generate visualization
        plt.figure(figsize=(12, 7))
        
        # Plot nominal projections (blue)
        plt.fill_between(years, nominal_lower_95, nominal_upper_95, alpha=0.1, color='#3498db', label='Nominal 95% CI')
        plt.fill_between(years, nominal_lower_std, nominal_upper_std, alpha=0.2, color='#3498db', label='Nominal 68% CI')
        plt.plot(years, nominal_median, label='Nominal Median', color='#3498db', linewidth=2)
        
        # Plot inflation-adjusted projections (green)
        plt.fill_between(years, inflation_adj_lower_95, inflation_adj_upper_95, alpha=0.1, color='#2ecc71', label='Inflation-Adjusted 95% CI')
        plt.fill_between(years, inflation_adj_lower_std, inflation_adj_upper_std, alpha=0.2, color='#2ecc71', label='Inflation-Adjusted 68% CI')
        plt.plot(years, inflation_adj_median, label='Inflation-Adjusted Median', color='#2ecc71', linewidth=2)
        
        # Set label based on data source
        history_label = "ABS Historical Growth Pattern" if data_source == 'ABS' else "Default Historical Growth Pattern"
    
        # Add historical line to plot
        plt.plot(years, historical_prices, 
                label=history_label, 
                color='#e74c3c',
                linestyle='--',
                linewidth=1.5,
                alpha=0.8)

        # Add price annotations at 5-year intervals
        interval_years = years[::5]
        nominal_prices = nominal_median[::5]
        inflation_adj_prices = inflation_adj_median[::5]
        
        for year, nom_price, infl_price in zip(interval_years, nominal_prices, inflation_adj_prices):
            # Nominal price annotation - now always below the line
            plt.annotate(
                f'${nom_price:,.0f}',
                xy=(year, nom_price),
                xytext=(0, -20),  
                textcoords='offset points',
                ha='center',
                va='top',  
                bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='gray', alpha=0.7)
            )
            # Inflation-adjusted price annotation remains above
            plt.annotate(
                f'${infl_price:,.0f}',
                xy=(year, infl_price),
                xytext=(0, 10), 
                textcoords='offset points',
                ha='center',
                va='bottom', 
                bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='gray', alpha=0.7)
            )

        # Configure plot styling
        plt.title(f"50-Year Housing Price Forecast for {suburb}\nNominal vs Inflation-Adjusted Values", fontsize=14)
        plt.xlabel("Year", fontsize=12)
        plt.ylabel("Price (AUD)", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # Save plot and store path
        plot_path = FORECAST_DIR / f"{suburb.replace(' ', '_')}_forecast.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        plot_paths.append(plot_path)
        
        # Add projections to suburb data
        suburb_info['price_projections'] = {
            'years': years.tolist(),
            'nominal': {
                'median_projection': nominal_median.tolist(),
                'confidence_intervals': {
                    'lower_95': nominal_lower_95.tolist(),
                    'upper_95': nominal_upper_95.tolist(),
                    'lower_std': nominal_lower_std.tolist(),
                    'upper_std': nominal_upper_std.tolist()
                }
            },
            'inflation_adjusted': {
                'median_projection': inflation_adj_median.tolist(),
                'confidence_intervals': {
                    'lower_95': inflation_adj_lower_95.tolist(),
                    'upper_95': inflation_adj_upper_95.tolist(),
                    'lower_std': inflation_adj_lower_std.tolist(),
                    'upper_std': inflation_adj_upper_std.tolist()
                }
            },
            'growth_assumptions': {
                'base_rate_range': [float(base_growth_rate[0]), float(base_growth_rate[-1])],
                'volatility': 0.0055,
                'growth_adjustment': float(adjusted_growth),
                'inflation': {
                    'forecast_rate': float(forecast_inflation),
                    'volatility': float(inflation_volatility)
                }
            }
        }

        # Calculate YoY growth rates
        growth_rates = calculate_growth_rates(suburb_info['price_projections'])
        suburb_info.update(growth_rates)
    
    return plot_paths, json.dumps(data_dict, indent=4)

def forecast_prices_dynamic(suburbs, dwelling_type=None, bedrooms=None, bathrooms=None,
                           car_spaces=None, land_size=None, max_price=None):
    """
    PHASE 3: Generate forecasts with year-by-year state evolution
    CRITICAL: Each year's growth depends on that year's state, not Year 0

    This is the lifecycle-aware implementation that:
    - Updates state variables every year (price, income, supply, age)
    - Applies stage-specific momentum and mean reversion
    - Uses current maturity decay factors
    - Implements fundamental value anchoring
    """
    # Fetch initial data
    suburb_data, api_response = fetch_suburb_data(
        suburbs, dwelling_type=dwelling_type, bedrooms=bedrooms,
        bathrooms=bathrooms, car_spaces=car_spaces,
        land_size=land_size, max_price=max_price
    )

    # Load calibration data
    supply_elasticity = load_supply_elasticity()
    rba_params = load_rba_parameters()

    current_year = datetime.now().year
    years = np.arange(current_year, current_year + 50)
    n_simulations = 1000

    # Parse existing data
    data_dict = json.loads(api_response)
    plot_paths = []

    # For each suburb, generate dynamic projections
    for _, row in suburb_data.iterrows():
        suburb = row['suburb']

        # Get suburb info from API response
        if suburb not in data_dict.get('suburbs', {}):
            logging.warning(f"Skipping {suburb} - no API data available")
            continue

        suburb_info = data_dict['suburbs'][suburb]
        base_price = clean_numeric_value(suburb_info['median_price']['value'])

        # Initialize state for this suburb
        initial_state = {
            'current_price': base_price,
            'initial_price': base_price,
            'annual_income': row['household_income'] * 52,
            'supply_ratio': row['supply_ratio'],
            'years_since_maturity': estimate_suburb_age(pd.DataFrame([row]))[0],
            'prior_year_growth': 0.03,  # Assume 3% prior
            'city': extract_city_from_suburb(suburb),
            'household_size': row['household_size']  # PHASE 4: Track demographic evolution
        }

        # Run Monte Carlo simulations
        sim_trajectories = []

        for sim in range(n_simulations):
            state = initial_state.copy()
            trajectory = [state['current_price']]

            for year_idx in range(1, len(years)):
                # Calculate THIS YEAR's growth based on CURRENT state
                base_growth = calculate_year_growth(state, row, year_idx, rba_params)

                # Apply market constraints
                constrained_growth = apply_market_constraints(base_growth, state, year_idx)

                # Add stochastic noise
                noise = np.random.normal(0, 0.008)  # 0.8% volatility
                final_growth = constrained_growth + noise

                # Update price
                new_price = state['current_price'] * (1 + final_growth)
                trajectory.append(new_price)

                # ====== UPDATE STATE FOR NEXT ITERATION ======
                state['current_price'] = new_price
                state['prior_year_growth'] = final_growth
                state['years_since_maturity'] += 1

                # DYNAMIC SUPPLY RATIO ADJUSTMENT
                # Combines construction response and demand absorption
                # Research: RBA construction elasticity + US absorption patterns
                price_change = (new_price / state['initial_price']) - 1
                supply_city = supply_elasticity.get(state['city'], 0.07)
                current_total_age = state['years_since_maturity']

                # Calculate dynamic adjustment
                supply_delta = calculate_supply_adjustment(
                    current_ratio=state['supply_ratio'],
                    price_change=price_change,
                    growth_rate=final_growth,
                    years_since_maturity=current_total_age,
                    city_elasticity=supply_city
                )

                # Update supply ratio (can increase or decrease)
                state['supply_ratio'] = max(0.5, state['supply_ratio'] + supply_delta)
                # Floor at 0.5 (3 months supply) prevents unrealistic undersupply

                # Update income (2.5% annual growth assumption)
                state['annual_income'] = initial_state['annual_income'] * ((1.025) ** year_idx)

                # PHASE 4: Demographic evolution - household size decays as suburb ages
                # Reflects empty-nester effect, aging population, smaller new households
                # Decay from ~3.2 (young families) toward ~2.1 (mature/aging) over 30 years
                target_household_size = 2.1
                decay_rate = 0.015  # 1.5% annual decay toward target
                state['household_size'] = (state['household_size'] - target_household_size) * (1 - decay_rate) + target_household_size

            sim_trajectories.append(trajectory)

        # Convert to numpy array
        all_projections = np.array(sim_trajectories)

        # Calculate statistics
        nominal_median = np.median(all_projections, axis=0)
        nominal_lower_95 = np.percentile(all_projections, 2.5, axis=0)
        nominal_upper_95 = np.percentile(all_projections, 97.5, axis=0)
        nominal_lower_std = nominal_median - np.std(all_projections, axis=0)
        nominal_upper_std = nominal_median + np.std(all_projections, axis=0)

        # Calculate inflation-adjusted projections
        forecast_inflation = suburb_info['inflation']['forecast']['value']
        inflation_volatility = suburb_info['inflation']['volatility']['value']
        inflation_noise = np.random.normal(0, inflation_volatility/100, (n_simulations, len(years)))
        inflation_factors = np.cumprod(1 + (forecast_inflation/100 + inflation_noise), axis=1)

        inflation_adjusted_projections = all_projections / inflation_factors

        inflation_adj_median = np.median(inflation_adjusted_projections, axis=0)
        inflation_adj_lower_95 = np.percentile(inflation_adjusted_projections, 2.5, axis=0)
        inflation_adj_upper_95 = np.percentile(inflation_adjusted_projections, 97.5, axis=0)
        inflation_adj_lower_std = inflation_adj_median - np.std(inflation_adjusted_projections, axis=0)
        inflation_adj_upper_std = inflation_adj_median + np.std(inflation_adjusted_projections, axis=0)

        # ====== VALIDATION: Sanity checks ======
        validation_result = validate_long_term_projections(
            suburb, nominal_median, row, years
        )

        # Get historical ABS growth rates
        historical_rates, data_source = fetch_abs_historical_growth(suburb)
        historical_prices = base_price * np.cumprod(1 + historical_rates/100)

        # Verify dimensions match
        if len(historical_prices) < len(years):
            historical_prices = np.pad(historical_prices,
                                      (0, len(years) - len(historical_prices)),
                                      'edge')
        elif len(historical_prices) > len(years):
            historical_prices = historical_prices[:len(years)]

        # Generate visualization
        plt.figure(figsize=(12, 7))

        # Plot nominal projections
        plt.fill_between(years, nominal_lower_95, nominal_upper_95, alpha=0.1, color='#3498db', label='Nominal 95% CI')
        plt.fill_between(years, nominal_lower_std, nominal_upper_std, alpha=0.2, color='#3498db', label='Nominal 68% CI')
        plt.plot(years, nominal_median, label='Nominal Median (Phase 3 Dynamic)', color='#3498db', linewidth=2.5)

        # Plot inflation-adjusted projections
        plt.fill_between(years, inflation_adj_lower_95, inflation_adj_upper_95, alpha=0.1, color='#2ecc71', label='Inflation-Adjusted 95% CI')
        plt.fill_between(years, inflation_adj_lower_std, inflation_adj_upper_std, alpha=0.2, color='#2ecc71', label='Inflation-Adjusted 68% CI')
        plt.plot(years, inflation_adj_median, label='Inflation-Adjusted Median', color='#2ecc71', linewidth=2)

        # Plot historical baseline
        plt.plot(years, historical_prices, label=f'Historical Pattern ({data_source})', color='#95a5a6', linewidth=1.5, linestyle='--')

        plt.xlabel('Year', fontsize=12)
        plt.ylabel('Median Price (AUD)', fontsize=12)
        plt.title(f'{suburb} - 50-Year Price Forecast (Lifecycle-Aware Model)', fontsize=14, fontweight='bold')
        plt.legend(loc='upper left', fontsize=9)
        plt.grid(True, alpha=0.3)

        # Save plot
        plot_filename = f"{suburb.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plot_path = FORECAST_DIR / plot_filename
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        plot_paths.append(str(plot_path))

        # Store projections
        suburb_info['price_projections'] = {
            'years': years.tolist(),
            'nominal': {
                'median_projection': nominal_median.tolist(),
                'lower_95_ci': nominal_lower_95.tolist(),
                'upper_95_ci': nominal_upper_95.tolist(),
                'lower_std': nominal_lower_std.tolist(),
                'upper_std': nominal_upper_std.tolist()
            },
            'inflation_adjusted': {
                'median_projection': inflation_adj_median.tolist(),
                'lower_95_ci': inflation_adj_lower_95.tolist(),
                'upper_95_ci': inflation_adj_upper_95.tolist(),
                'lower_std': inflation_adj_lower_std.tolist(),
                'upper_std': inflation_adj_upper_std.tolist()
            },
            'model_version': 'Phase3_Dynamic_Lifecycle',
            'validation': validation_result
        }

        # Calculate YoY growth rates
        growth_rates = calculate_growth_rates(suburb_info['price_projections'])
        suburb_info.update(growth_rates)

    return plot_paths, json.dumps(data_dict, indent=4)

def generate_forecast_reasoning(df: pd.DataFrame, api_data: str, reasoning_effort: str = "medium", verbose: bool = False) -> tuple[str, list[str]]:
    """
    Generate detailed reasoning for price forecasts using DataFrame data and price projections
    Uses Sonar Deep Research endpoint for comprehensive market analysis

    Args:
        df: DataFrame containing suburb data
        api_data: JSON string with forecast data
        reasoning_effort: Level of research depth - "low", "medium", or "high"
        verbose: Enable debug logging
    """
    all_reasoning = []
    all_citations = []

    data_dict = json.loads(api_data)

    for _, row in df.iterrows():
        suburb = row['suburb']

        # Skip suburbs that don't have data (failed to fetch)
        if suburb not in data_dict.get('suburbs', {}):
            logging.warning(f"Skipping reasoning for {suburb} - no data available")
            all_reasoning.append(f"""
## Market Analysis: {suburb}
Unable to generate analysis - suburb data not available.

---
""")
            continue

        projections = data_dict['suburbs'][suburb]['price_projections']

        years = projections['years']
        year_indices = [
            next(i for i, y in enumerate(years) if y-years[0] >= 5),
            next(i for i, y in enumerate(years) if y-years[0] >= 25),
            -1  # Last index (50 years)
        ]

        nominal_prices = [
            projections['nominal']['median_projection'][i]
            for i in year_indices
        ]
        inflation_adj_prices = [
            projections['inflation_adjusted']['median_projection'][i]
            for i in year_indices
        ]

        reasoning_payload = {
            "model": "sonar-deep-research",
            "reasoning_effort": reasoning_effort,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert Australian real estate analyst. Focus analysis on 25-year timeframe as primary, with 50-year projections as supplementary long-term indicators. Provide detailed growth reasoning with citations. DO NOT include your planning or thinking process in the output - provide only the final comprehensive analysis. Write in a professional, direct style suitable for an investment report."
                },
                {
                    "role": "user",
                    "content": f"""Analyze price projections for {suburb} {row['dwelling_type']} properties with emphasis on 25-year outlook:
                    
                    Primary Growth Outlook (25-Year Forecast):
                    - Nominal: ${nominal_prices[1]:,.2f}
                    - Inflation-Adjusted: ${inflation_adj_prices[1]:,.2f}
                    - Confidence Level: High
                    
                    Supporting Projections:
                    5-Year Forecast (Near-Term):
                    - Nominal: ${nominal_prices[0]:,.2f}
                    - Inflation-Adjusted: ${inflation_adj_prices[0]:,.2f}
                    - Confidence Level: Very High
                    
                    Extended Outlook (50-Year):
                    - Nominal: ${nominal_prices[2]:,.2f}
                    - Inflation-Adjusted: ${inflation_adj_prices[2]:,.2f}
                    - Confidence Level: Moderate
                    
                    Growth Assumptions:
                    - Base Growth Rate: {projections['growth_assumptions']['base_rate_range'][0]:.3f} to {projections['growth_assumptions']['base_rate_range'][1]:.3f}
                    - Growth Volatility: {projections['growth_assumptions']['volatility']:.4f}
                    - Growth Adjustment: {projections['growth_assumptions']['growth_adjustment']:.4f}

                    Property Characteristics:
                    - Current median price: ${row['median_price']:,.2f}
                    - Property type: {row['dwelling_type']}
                    - Configuration: {row['bedrooms']} bed, {row['bathrooms']} bath
                    - Land size: {row['land_size']} sqm
                    
                    Location & Infrastructure:
                    - Distance to CBD: {row['distance_to_cbd']}km
                    - Public transport score: {row['public_transport']}/10
                    - Infrastructure score: {row['infrastructure_score']}/10
                    - School quality: {row['school_quality']}/10
                    
                    Commercial Centers:
                    - Major centers: {row['major_commercial_count']} within {row['major_commercial_distance']}km
                    - Major center quality: {row['major_center_size_score']}/10
                    - Local centers: {row['local_commercial_count']} within {row['local_commercial_distance']}km
                    - Retail quality: {row['retail_quality_score']}/10
                    
                    Natural Features:
                    - Water proximity: {row['water_distance']}km to {row['water_type']}
                    - Park quality: {row['park_quality']}/10 at {row['park_distance']}km
                    - View protection: {row['view_protection']} with {row['height_restriction']}m limit
                    
                    Market Conditions:
                    - Supply ratio: {row['supply_ratio']:.2f}
                    - Population growth: {row['population_growth']}%
                    - Current inflation: {row['current_inflation']}%
                    - Forecast inflation: {row['forecast_inflation']}%
                    - Inflation volatility: {row['inflation_volatility']}%
                    
                    Risk Factors:
                    - Flood risk: {row['flood_risk']}/10
                    - Climate risk: {row['climate_risk']}/10
                    
                    Crime Statistics:
                    - Assault rate: {row['assault_rate']} per 1000 residents
                    - Break-in rate: {row['breakin_rate']} per 1000 households
                    - Enhanced security: {row['enhanced_security']}%
                    - Basic security: {row['basic_security']}%
                    - Security score: {row['security_score']:.1f}
                    
                    Demographics:
                    - Median age: {row['median_age']} years
                    - Household size: {row['household_size']} persons
                    - Weekly household income: ${row['household_income']:,.2f}
                    - Couples with children: {row['couples_with_children']}%
                    - Couples no children: {row['couples_no_children']}%
                    - Single parent: {row['single_parent']}%
                    - Under 35: {row['population_under_35']}%
                    - 35-65: {row['population_35_to_65']}%
                    - Over 65: {row['population_over_65']}%
                    
                    Provide comprehensive analysis covering:
                    1. Primary 25-year growth trajectory and key drivers
                    2. Near-term (5-year) market dynamics and opportunities
                    3. Long-term (50-year) trend indicators and sustainability
                    4. Supply-demand dynamics
                    5. Infrastructure and location impact
                    6. Demographic trends and implications
                    7. Risk assessment and mitigation factors
                    8. Crime impact and security considerations
                    9. Inflation effects on nominal vs adjusted values
                    Include relevant citations to research and market data.
                    """
                }
            ],
            "temperature": 0.2
        }
        
        try:
            # Deep Research can take up to 10 minutes, especially with high reasoning effort
            timeout = 900 if reasoning_effort == "high" else 600
            response = requests.post(PERPLEXITY_API_URL, json=reasoning_payload, headers=HEADERS, timeout=timeout)
            response.raise_for_status()

            if verbose:
                logging.debug(f"\n=== Reasoning API Call for {suburb} ===")
                logging.debug(f"Request Payload:\n{json.dumps(reasoning_payload, indent=2)}")
                logging.debug(f"Response Status: {response.status_code}")
                logging.debug(f"Response Content:\n{response.text}\n")
            
            response_data = response.json()
            content = response_data['choices'][0]['message']['content']
            citations = response_data.get('citations', [])

            # Deep Research provides direct output without thinking tags
            reasoning = content.strip()
            
            suburb_analysis = f"""
## Market Analysis: {suburb} ({row['dwelling_type'].title()} Properties)

{reasoning}

### Research Sources
{chr(10).join(f'- {cite}' for cite in citations)}

---
"""
            
            all_reasoning.append(suburb_analysis)
            all_citations.extend(citations)
            
        except Exception as e:
            error_msg = f"Error generating reasoning for {suburb}: {str(e)}"
            logging.error(error_msg)
            all_reasoning.append(f"""
## Error: {suburb}
{error_msg}

---
""")
            continue
    
    return "\n".join(all_reasoning), list(set(all_citations))

def save_forecast_data(api_data, format='json'):
    """Save forecast data in selected format with complete metrics and calculations"""
    export_dir = Path('exports')
    export_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_name = f"{export_dir}/suburb_forecast_{timestamp}"
    
    # Parse data if needed
    data_dict = api_data if isinstance(api_data, dict) else json.loads(api_data)
    
    if format == 'json':
        output_path = f"{base_name}.json"
        with open(output_path, 'w') as f:
            json.dump(data_dict, f, indent=4)
            
    elif format == 'txt':
        output_path = f"{base_name}.txt"
        with open(output_path, 'w') as f:
            for suburb, data in data_dict['suburbs'].items():
                f.write(f"=== {suburb} ===\n\n")
                
                # Property Details
                f.write("Property Details:\n")
                f.write(f"Median Price: ${data['median_price']['value']:,.2f}\n")
                f.write(f"Dwelling Type: {data['dwelling_type']}\n")
                f.write(f"Bedrooms: {data['bedrooms']['value']}\n")
                f.write(f"Bathrooms: {data['bathrooms']['value']}\n")
                f.write(f"Car Spaces: {data['car_spaces']['value']}\n")
                f.write(f"Land Size: {data['land_size']['value']} sqm\n\n")
                
                # Crime Statistics
                f.write("Crime Statistics:\n")
                f.write(f"Assault Rate: {data['crime_stats']['assault_rate']['value']} per 1000 residents\n")
                f.write(f"Break-in Rate: {data['crime_stats']['breakin_rate']['value']} per 1000 households\n")
                f.write("Security Adoption:\n")
                f.write(f"  Enhanced Security: {data['crime_stats']['security_adoption']['enhanced_security']['value']}%\n")
                f.write(f"  Basic Security: {data['crime_stats']['security_adoption']['basic_security']['value']}%\n\n")
                
                # Demographics
                f.write("Demographics:\n")
                f.write(f"Median Age: {data['demographics']['median_age']['value']} years\n")
                f.write(f"Household Size: {data['demographics']['household_size']['value']} persons\n")
                f.write(f"Weekly Household Income: ${data['demographics']['household_income']['value']:,.2f}\n\n")
                
                f.write("Family Composition:\n")
                f.write(f"  Couples with Children: {data['demographics']['family_composition']['couples_with_children']['value']}%\n")
                f.write(f"  Couples without Children: {data['demographics']['family_composition']['couples_no_children']['value']}%\n")
                f.write(f"  Single Parent Families: {data['demographics']['family_composition']['single_parent']['value']}%\n\n")
                
                f.write("Age Distribution:\n")
                f.write(f"  Under 35: {data['demographics']['age_distribution']['under_35']['value']}%\n")
                f.write(f"  35-65: {data['demographics']['age_distribution']['35_to_65']['value']}%\n")
                f.write(f"  Over 65: {data['demographics']['age_distribution']['over_65']['value']}%\n\n")
                
                # Location Metrics
                f.write("Location Details:\n")
                f.write(f"Distance to CBD: {data['distance_to_cbd']['value']} km\n")
                f.write(f"Public Transport Score: {data['public_transport']['value']}/10\n\n")
                
                # Commercial Centers
                f.write("Commercial Centers:\n")
                f.write("Major Centers:\n")
                f.write(f"  Count: {data['commercial_centers']['major']['count']}\n")
                f.write(f"  Average Distance: {data['commercial_centers']['major']['avg_distance']} km\n")
                f.write(f"  Size Score: {data['commercial_centers']['major']['size_score']['value']}/10\n\n")
                
                f.write("Local Centers:\n")
                f.write(f"  Count: {data['commercial_centers']['local']['count']}\n")
                f.write(f"  Average Distance: {data['commercial_centers']['local']['avg_distance']} km\n")
                f.write(f"  Retail Quality: {data['commercial_centers']['local']['retail_quality']['value']}/10\n\n")
                
                # Area Features
                f.write("Area Features:\n")
                f.write(f"School Quality: {data['school_quality']['value']}/10\n")
                f.write(f"Infrastructure Score: {data['infrastructure_score']['value']}/10\n")
                f.write(f"Population Growth: {data['population_growth']['value']}%\n\n")
                
                # Natural Features
                f.write("Natural Features:\n")
                f.write("Water Features:\n")
                f.write(f"  Distance: {data['water_features']['distance']['value']} km\n")
                f.write(f"  Type: {data['water_features']['type']}\n")
                f.write(f"  Premium: {calculate_water_premium(data['water_features']['distance']['value'], data['water_features']['type']):.3f}\n\n")
                
                f.write("Parks:\n")
                f.write(f"  Distance: {data['parks']['distance']['value']} km\n")
                f.write(f"  Quality: {data['parks']['quality_score']['value']}/10\n")
                f.write(f"  Premium: {calculate_park_premium(data['parks']['distance']['value'], data['parks']['quality_score']['value']):.3f}\n\n")
                
                # Zoning and Risk
                f.write("Zoning and Risk:\n")
                f.write("View Protection:\n")
                f.write(f"  Protected: {data['zoning']['view_protection']['value']}\n")
                f.write(f"  Height Limit: {data['zoning']['height_restrictions']['value']} m\n")
                f.write(f"  Premium: {calculate_view_premium(data['zoning']['view_protection']['value'], data['zoning']['height_restrictions']['value']):.3f}\n\n")
                
                f.write("Risk Factors:\n")
                f.write(f"  Flood Risk: {data['flood_risk']['value']}/10\n")
                f.write(f"  Climate Risk: {data['climate_risk']['value']}/10\n")
                f.write(f"  Risk Adjustment: {calculate_risk_adjustments(data['flood_risk']['value'], data['climate_risk']['value']):.3f}\n\n")
                
                # Market Conditions
                f.write("Market Conditions:\n")
                f.write(f"Supply Ratio: {data['supply_ratio']['value']:.2f} (Benchmark: {data['supply_ratio']['benchmark']} months)\n")
                f.write(f"Current Inflation: {data['inflation']['current']['value']}%\n")
                f.write(f"Forecast Inflation: {data['inflation']['forecast']['value']}%\n")
                f.write(f"Inflation Volatility: {data['inflation']['volatility']['value']}%\n\n")
                f.write("-" * 80 + "\n\n")

                # Growth rates 
                f.write("Growth Rates:\n")
                f.write("5-Year:\n")
                f.write(f"  Nominal: {data['growth_rates']['5_year']['nominal']:.2f}%\n")
                f.write(f"  Inflation-Adjusted: {data['growth_rates']['5_year']['inflation_adjusted']:.2f}%\n")
                f.write("25-Year:\n")
                f.write(f"  Nominal: {data['growth_rates']['25_year']['nominal']:.2f}%\n")
                f.write(f"  Inflation-Adjusted: {data['growth_rates']['25_year']['inflation_adjusted']:.2f}%\n")
                f.write("50-Year:\n")
                f.write(f"  Nominal: {data['growth_rates']['50_year']['nominal']:.2f}%\n")
                f.write(f"  Inflation-Adjusted: {data['growth_rates']['50_year']['inflation_adjusted']:.2f}%\n\n")
    
    elif format == 'xlsx' or format == 'csv':
        df = pd.DataFrame(columns=[
            'Suburb',
            'Dwelling Type',
            'Median Price (AUD)',
            'Bedrooms',
            'Bathrooms',
            'Car Spaces',
            'Land Size (sqm)',
            'Distance to CBD (km)',
            'Major Commercial Count',
            'Major Commercial Distance (km)',
            'Major Center Size Score (1-10)',
            'Local Commercial Count',
            'Local Commercial Distance (km)',
            'Retail Quality Score (1-10)',
            'School Quality (1-10)',
            'Infrastructure Score (1-10)',
            'Population Growth (%)',
            'Public Transport (1-10)',
            'Supply Ratio',
            'Water Distance (km)',
            'Water Type',
            'Water Premium',
            'Park Distance (km)',
            'Park Quality (1-10)',
            'Park Premium',
            'View Protection',
            'Height Restriction (m)',
            'View Premium',
            'Risk Adjustment',
            'Flood Risk (1-10)',
            'Climate Risk (1-10)',
            'Assault Rate (per 1000)',
            'Break-in Rate (per 1000)',
            'Enhanced Security (%)',
            'Basic Security (%)',
            'Security Score',
            'Median Age (years)',
            'Household Size (persons)',
            'Weekly Household Income (AUD)',
            'Couples with Children (%)',
            'Couples No Children (%)',
            'Single Parent (%)',
            'Population Under 35 (%)',
            'Population 35-65 (%)',
            'Population Over 65 (%)',
            'Demographic Premium',
            'Current Inflation (%)',
            'Forecast Inflation (%)',
            'Inflation Volatility (%)',
            '5yr Nominal Growth (%)',
            '5yr Real Growth (%)',
            '25yr Nominal Growth (%)',
            '25yr Real Growth (%)',
            '50yr Nominal Growth (%)',
            '50yr Real Growth (%)'
        ])
        
        for suburb, data in data_dict['suburbs'].items():
            df = pd.concat([df, pd.DataFrame([{
                'Suburb': suburb,
                'Dwelling Type': data['dwelling_type'],
                'Median Price (AUD)': data['median_price']['value'],
                'Bedrooms': data['bedrooms']['value'],
                'Bathrooms': data['bathrooms']['value'],
                'Car Spaces': data['car_spaces']['value'],
                'Land Size (sqm)': data['land_size']['value'],
                'Distance to CBD (km)': data['distance_to_cbd']['value'],
                'Major Commercial Count': data['commercial_centers']['major']['count'],
                'Major Commercial Distance (km)': data['commercial_centers']['major']['avg_distance'],
                'Major Center Size Score (1-10)': data['commercial_centers']['major']['size_score']['value'],
                'Local Commercial Count': data['commercial_centers']['local']['count'],
                'Local Commercial Distance (km)': data['commercial_centers']['local']['avg_distance'],
                'Retail Quality Score (1-10)': data['commercial_centers']['local']['retail_quality']['value'],
                'School Quality (1-10)': data['school_quality']['value'],
                'Infrastructure Score (1-10)': data['infrastructure_score']['value'],
                'Population Growth (%)': data['population_growth']['value'],
                'Public Transport (1-10)': data['public_transport']['value'],
                'Supply Ratio': data['supply_ratio']['value'],
                'Water Distance (km)': data['water_features']['distance']['value'],
                'Water Type': data['water_features']['type'],
                'Water Premium': calculate_water_premium(
                    data['water_features']['distance']['value'],
                    data['water_features']['type']
                ),
                'Park Distance (km)': data['parks']['distance']['value'],
                'Park Quality (1-10)': data['parks']['quality_score']['value'],
                'Park Premium': calculate_park_premium(
                    data['parks']['distance']['value'],
                    data['parks']['quality_score']['value']
                ),
                'View Protection': data['zoning']['view_protection']['value'],
                'Height Restriction (m)': data['zoning']['height_restrictions']['value'],
                'View Premium': calculate_view_premium(
                    data['zoning']['view_protection']['value'],
                    data['zoning']['height_restrictions']['value']
                ),
                'Risk Adjustment': calculate_risk_adjustments(
                    data['flood_risk']['value'],
                    data['climate_risk']['value']
                ),
                'Flood Risk (1-10)': data['flood_risk']['value'],
                'Climate Risk (1-10)': data['climate_risk']['value'],
                'Assault Rate (per 1000)': data['crime_stats']['assault_rate']['value'],
                'Break-in Rate (per 1000)': data['crime_stats']['breakin_rate']['value'],
                'Enhanced Security (%)': data['crime_stats']['security_adoption']['enhanced_security']['value'],
                'Basic Security (%)': data['crime_stats']['security_adoption']['basic_security']['value'],
                'Security Score': (
                    0.6 * data['crime_stats']['security_adoption']['enhanced_security']['value'] +
                    0.4 * data['crime_stats']['security_adoption']['basic_security']['value']
                ),
                'Median Age (years)': data['demographics']['median_age']['value'],
                'Household Size (persons)': data['demographics']['household_size']['value'],
                'Weekly Household Income (AUD)': data['demographics']['household_income']['value'],
                'Couples with Children (%)': data['demographics']['family_composition']['couples_with_children']['value'],
                'Couples No Children (%)': data['demographics']['family_composition']['couples_no_children']['value'],
                'Single Parent (%)': data['demographics']['family_composition']['single_parent']['value'],
                'Population Under 35 (%)': data['demographics']['age_distribution']['under_35']['value'],
                'Population 35-65 (%)': data['demographics']['age_distribution']['35_to_65']['value'],
                'Population Over 65 (%)': data['demographics']['age_distribution']['over_65']['value'],
                'Demographic Premium': calculate_demographic_premium(pd.Series({
                    'population_under_35': data['demographics']['age_distribution']['under_35']['value'],
                    'population_over_65': data['demographics']['age_distribution']['over_65']['value'],
                    'couples_with_children': data['demographics']['family_composition']['couples_with_children']['value'],
                    'couples_no_children': data['demographics']['family_composition']['couples_no_children']['value'],
                    'household_income': data['demographics']['household_income']['value'],
                    'household_size': data['demographics']['household_size']['value'],
                    'dwelling_type': data['dwelling_type']
                })),
                'Current Inflation (%)': data['inflation']['current']['value'],
                'Forecast Inflation (%)': data['inflation']['forecast']['value'],
                'Inflation Volatility (%)': data['inflation']['volatility']['value'],
                '5yr Nominal Growth (%)': data['growth_rates']['5_year']['nominal'],
                '5yr Real Growth (%)': data['growth_rates']['5_year']['inflation_adjusted'],
                '25yr Nominal Growth (%)': data['growth_rates']['25_year']['nominal'],
                '25yr Real Growth (%)': data['growth_rates']['25_year']['inflation_adjusted'],
                '50yr Nominal Growth (%)': data['growth_rates']['50_year']['nominal'],
                '50yr Real Growth (%)': data['growth_rates']['50_year']['inflation_adjusted']
            }])], ignore_index=True)
        
        if format == 'xlsx':
            output_path = f"{base_name}.xlsx"
            df.to_excel(output_path, index=False)
        else:
            output_path = f"{base_name}.csv"
            df.to_csv(output_path, index=False)
    
    return output_path

def export_data(api_data, formats):
    """Export data to multiple formats with timestamped status updates"""
    if not api_data:
        return "Please generate forecasts first"
    
    status_messages = []
    timestamp_format = "%Y-%m-%d %H:%M:%S"
    
    for format in formats:
        timestamp = datetime.now().strftime(timestamp_format)
        try:
            saved_file = save_forecast_data(api_data, format)
            status_messages.append(f"[{timestamp}] Successfully exported to {saved_file}")
        except Exception as e:
            status_messages.append(f"[{timestamp}] Failed to export {format} format: {str(e)}")
    
    # Return all status messages, preserving history
    return "\n".join(status_messages)

# Gradio Interface
with gr.Blocks(theme=gr.themes.Default()) as app:
    gr.Markdown("""
        ## 🏡 Australian Suburb Housing Price Forecaster
        
        This application uses machine learning and real world hedonic real estate market data to forecast 50-year housing prices for Australian suburbs, with confidence intervals and detailed analytics.
        
        ### Input Options:
        - Empty Input → Analyzes top growth suburbs nationally
        - State Names Only (e.g., "NSW", "Victoria") → Top growth suburbs per state
        - Otherwise → Analyzes suburbs matching input
        - Property Criteria (0 = Any): Bedrooms, Bathrooms, Car Spaces, Land Size, Max Median Price
        
        #### NOTE: Include Detailed Reasoning uses Perplexity's Sonar Deep Research endpoint. Higher research depth provides more thorough analysis but is slower and more expensive. Use with care.
                           
        """)
        
    # Main content
    suburb_input = gr.Textbox(
        label="Enter Suburbs (Suburb STATE. One per line. Max 25.)",
        placeholder="e.g.\nSydney CBD NSW\nNoosa QLD\nMelbourne CBD VIC",
        max_lines=25,
        lines=5
    )
    
    with gr.Row():
        bedrooms_input = gr.Number(
            label="Number of Bedrooms (optional)",
            minimum=0, 
            maximum=15,
            step=1,
            value=None
        )
        
        bathrooms_input = gr.Number(
            label="Number of Bathrooms (optional)",
            minimum=0,  
            maximum=20,
            step=1,
            value=None
        )
        
        car_spaces_input = gr.Number(
            label="Number of Car Spaces (optional)",
            minimum=0,
            maximum=10,
            step=1,
            value=None
        )
        
        land_size_input = gr.Number(
            label="Land Size (sqm, optional)",
            minimum=0,
            value=None
        )

        max_price_input = gr.Number(
            label="Max Median Price (AUD, optional)",
            minimum=0,
            value=None
        )

    with gr.Row():
        dwelling_type = gr.Radio(
            choices=["house", "apartment", "townhouse"],
            label="Dwelling Type",
            value="house",
        )

    submit_btn = gr.Button("Generate Forecasts", variant="primary")
    
    forecast_gallery = gr.Gallery(
        label="Price Projections",
        height="h-120", 
        columns=[2],     # Optional: adjust columns for better layout
        allow_preview=True
    )

    reasoning_output = gr.Textbox(
        label="Forecast Reasoning",
        interactive=False,
        lines=5,
        max_lines=1000,
        show_copy_button=True,
        visible=True
    )
    
    with gr.Row():
        include_reasoning = gr.Checkbox(
            label="Include Detailed Reasoning",
            value=False,
        )
        reasoning_effort = gr.Dropdown(
            choices=["low", "medium", "high"],
            value="medium",
            label="Research Depth",
            info="Higher depth = more thorough analysis but slower and more expensive"
        )
        txt_btn = gr.Button("Download Reasoning (TXT)")
        rtf_btn = gr.Button("Download Reasoning (RTF)")
    
    def save_reasoning(reasoning: str, format: str) -> str:
        """Save reasoning to file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if format == 'txt':
            filepath = f'exports/forecast_reasoning_{timestamp}.txt'
            with open(filepath, 'w') as f:
                f.write(reasoning)
        else:
            filepath = f'exports/forecast_reasoning_{timestamp}.rtf'
            # Convert to RTF with basic formatting
            rtf_content = "{\\rtf1\\ansi\n" + reasoning.replace('\n', '\\par\n') + "}"
            with open(filepath, 'w') as f:
                f.write(rtf_content)
        
        return filepath
    
    def update_forecast_display(api_data, verbose, reasoning_effort="medium"):
        """Update forecast display with reasoning"""
        if not api_data:
            return None, gr.update(visible=False)

        reasoning, citations = generate_forecast_reasoning(json.loads(api_data), reasoning_effort, verbose)
        return reasoning, gr.update(visible=True)

    api_output = gr.JSON(label="Suburb Data")
    
    export_format = gr.CheckboxGroup(
        choices=['json', 'csv', 'txt', 'xlsx'],
        value=['json'],
        label="Export Data Formats"
    )

    export_btn = gr.Button("Export Data")
    
    # Unified logging interface
    logging_output = gr.Textbox(
        label="Logging", 
        interactive=False,  # Keep false to prevent user edits
        lines=10,          # Shows 10 lines at a time
        max_lines=1000,     # Allows scrolling through up to 1000 lines
        show_copy_button=True,  # Useful for copying log content
        autoscroll=True,   # Automatically scrolls to newest logs
        value=""
    )

    # Debug controls
    with gr.Row():
        verbose_checkbox = gr.Checkbox(
            label="Enable Debug Logging",
            value=False,
            info="Show verbose output for debugging."
        )
    
    def process_input(value):
        """Convert zero values to None for optional inputs"""
        return None if value == 0 else value
    
    def forecast_with_processing(suburbs, dwelling_type, bedrooms, bathrooms, car_spaces,
                           land_size, max_price, verbose, include_reasoning, reasoning_effort):
        """Enhanced wrapper with verbose logging integration and optional reasoning"""
        set_verbose(verbose)

        processed_suburbs = process_suburb_input(suburbs)
        suburbs_text = '\n'.join(processed_suburbs)

        # First get suburb data
        suburb_df, api_data = fetch_suburb_data(
            suburbs_text,
            dwelling_type,
            process_input(bedrooms),
            process_input(bathrooms),
            process_input(car_spaces),
            process_input(land_size),
            process_input(max_price)
        )

        # Then generate price forecasts
        plots, forecast_data = forecast_prices(
            suburbs_text,
            dwelling_type,
            process_input(bedrooms),
            process_input(bathrooms),
            process_input(car_spaces),
            process_input(land_size),
            process_input(max_price)
        )

        # Generate reasoning only if requested
        reasoning_text = None
        if include_reasoning:
            # Use the DataFrame and forecast data with Deep Research endpoint
            reasoning_text, _ = generate_forecast_reasoning(suburb_df, forecast_data, reasoning_effort, verbose)
        
        # Get current logs
        logs = textbox_handler.get_logs()
        
        return [
            plots, 
            forecast_data,  # Return forecast_data instead of api_data
            reasoning_text if reasoning_text else "Enable 'Include Detailed Reasoning' for analysis", 
            logs
        ]
    
    submit_btn.click(
        fn=forecast_with_processing,
        inputs=[
            suburb_input,
            dwelling_type,
            bedrooms_input,
            bathrooms_input,
            car_spaces_input,
            land_size_input,
            max_price_input,
            verbose_checkbox,
            # logging_output,
            include_reasoning,
            reasoning_effort
        ],
        outputs=[forecast_gallery, api_output, reasoning_output, logging_output]
    )
    
    def update_export_status(api_data, formats, current_logs):
        """Update logging output with export status"""
        new_status = export_data(api_data, formats)
        if current_logs:
            return current_logs + "\n" + new_status
        return new_status
    
    export_btn.click(
        fn=update_export_status,
        inputs=[api_output, export_format, logging_output],
        outputs=logging_output
    )

    txt_btn.click(
        fn=lambda x: save_reasoning(x, 'txt'),
        inputs=[reasoning_output],
        outputs=[logging_output]
    )
    
    rtf_btn.click(
        fn=lambda x: save_reasoning(x, 'rtf'),
        inputs=[reasoning_output],
        outputs=[logging_output]
    )

if __name__ == "__main__":
    app.launch()