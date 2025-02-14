import gradio as gr
import pandas as pd
import numpy as np
import json
import csv
import json
import time
import requests
from dotenv import load_dotenv
import os
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download
import joblib
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

# Load pretrained XGBoost model
model_path = hf_hub_download(
    repo_id="M-Yaqoob/finetune-xgboost",
    filename="fine-tune_xgboost_model.pkl"
)
base_model = joblib.load(model_path)

# Create forecasts directory if it doesn't exist
FORECAST_DIR = Path('./forecasts')
FORECAST_DIR.mkdir(exist_ok=True)

# Initialize Perplexity API
load_dotenv()
PERPLEXITY_API_KEY = os.getenv('PERPLEXITY_API_KEY')
PERPLEXITY_API_URL = "https://api.perplexity.ai/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
    "Content-Type": "application/json"
}

def fetch_top_growth_suburbs(state=None, limit=10, retries=3):
    payload = {
        "model": "sonar-pro",
        "messages": [
            {
                "role": "system",
                "content": "You are an expert Australian real estate analyst. Return only a JSON array of suburb strings."
            },
            {
                "role": "user",
                "content": f"Return a JSON array of exactly {limit} top growth suburbs " +
                          (f"in {state} " if state else "in Australia ") +
                          "based on projected 5-year growth rate. " +
                          "Format each suburb as 'Suburb STATE' where STATE is the 2-3 letter code. " +
                          "Return only the JSON array, no additional text."
            }
        ],
        "temperature": 0.2,
        "search_recency_filter": "month"
    }
    
    for attempt in range(retries):
        try:
            response = requests.post(PERPLEXITY_API_URL, json=payload, headers=HEADERS)
            response.raise_for_status()
            
            data = response.json()
            content = data['choices'][0]['message']['content']
            
            json_start = content.find('[')
            json_end = content.rfind(']') + 1
            json_str = content[json_start:json_end]
            
            suburbs = json.loads(json_str)
            return suburbs[:limit]
            
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt == retries - 1:
                raise gr.Error(f"Unable to fetch rankings for {state if state else 'Australia'}.")
            time.sleep(2 ** attempt)

def process_suburb_input(suburb_text):
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
        return fetch_top_growth_suburbs(limit=10)
    
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
            state_suburbs = fetch_top_growth_suburbs(state=state, limit=5)
            selected_suburbs.extend(state_suburbs)
        return selected_suburbs
    
    # Return exact suburbs provided without fetching additional ones
    return input_lines

def fetch_suburb_data(suburbs, bedrooms=None, bathrooms=None, car_spaces=None, land_size=None, max_price=None):
    for attempt in range(3):  # Try up to 3 times
        try:
            all_data = {"suburbs": {}}
            dataframes = []
            
            suburb_list = [s.strip() for s in suburbs.splitlines() if s.strip()]
            
            for suburb in suburb_list:
                try:
                    # First get default averages
                    default_payload = {
                        "model": "sonar-pro",
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are an expert Australian real estate analyst. Return average property metrics from the last 12 months."
                            },
                            {
                                "role": "user",
                                "content": f"""Return average property metrics for {suburb}, Australia as JSON:
                                {{
                                    "bedrooms": {{ "value": (average number), "range": "1-6" }},
                                    "bathrooms": {{ "value": (average number), "range": "1-4" }},
                                    "car_spaces": {{ "value": (average number), "range": "0-3" }},
                                    "land_size": {{ "value": (average size), "unit": "sqm" }}
                                }}"""
                            }
                        ],
                        "temperature": 0.0,
                        "top_k": 4
                        # "search_recency_filter": "month"
                    }
                    
                    default_response = requests.post(PERPLEXITY_API_URL, json=default_payload, headers=HEADERS)
                    default_response.raise_for_status()
                    default_content = default_response.json()['choices'][0]['message']['content']
                    default_json_start = default_content.find('{')
                    default_json_end = default_content.rfind('}') + 1
                    default_data = json.loads(default_content[default_json_start:default_json_end])

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
                                "content": f"Return the current months of housing inventory in {suburb} as a single number only. No additional text."
                            }
                        ],
                        "temperature": 0.0,
                        "top_k": 2
                        # "search_recency_filter": "month" # Too Granular
                    }

                    inventory_response = requests.post(PERPLEXITY_API_URL, json=inventory_payload, headers=HEADERS)
                    inventory_response.raise_for_status()
                    inventory_content = inventory_response.json()['choices'][0]['message']['content']
                    inventory_months = float(inventory_content)
                    supply_ratio = inventory_months / 6  # Normalize against 6-month benchmark

                    # Then get specific or default data
                    criteria = []
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
                                "content": "You are an expert Australian real estate analyst. Return only a JSON object with numerical values and their units/ranges."
                            },
                            {
                                "role": "user",
                                "content": f"""Return a JSON object for {suburb}, Australia{criteria_str}, in the last 12 months, with:
                                {{
                                    "median_price": {{ "value": (price), "unit": "AUD" }},
                                    "bedrooms": {{ "value": {bedrooms if bedrooms else "(number)"}, "range": "1-6" }},
                                    "bathrooms": {{ "value": {bathrooms if bathrooms else "(number)"}, "range": "1-4" }},
                                    "car_spaces": {{ "value": {car_spaces if car_spaces else "(number)"}, "range": "0-3" }},
                                    "land_size": {{ "value": {land_size if land_size else "(size)"}, "unit": "sqm" }},
                                    "distance_to_cbd": {{ "value": (distance), "unit": "km" }},
                                    "school_quality": {{ "value": (score), "range": "1-10" }},
                                    "infrastructure_score": {{ "value": (score), "range": "1-10" }},
                                    "flood_risk": {{ "value": (risk), "range": "1-10" }},
                                    "population_growth": {{ "value": (rate), "unit": "% per year" }},
                                    "climate_risk": {{ "value": (risk), "range": "1-10" }},
                                    "public_transport": {{ "value": (score), "range": "1-10" }}
                                }}"""
                            }
                        ],
                        "temperature": 0.0,
                        # "top_k": 2
                        # "search_recency_filter": "month"
                    }

                    response = requests.post(PERPLEXITY_API_URL, json=suburb_payload, headers=HEADERS)
                    response.raise_for_status()
                    content = response.json()['choices'][0]['message']['content']
                    json_start = content.find('{')
                    json_end = content.rfind('}') + 1
                    suburb_data = json.loads(content[json_start:json_end])
                    
                    # Add supply ratio to suburb_data
                    suburb_data["supply_ratio"] = {
                        "value": supply_ratio,
                        "benchmark": 6,
                        "unit": "ratio"
                    }
                    
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
                    
                    # Create DataFrame for forecasting
                    df = pd.DataFrame([{
                        'suburb': suburb,
                        'median_price': suburb_data['median_price']['value'],
                        'bedrooms': suburb_data['bedrooms']['value'],
                        'bathrooms': suburb_data['bathrooms']['value'],
                        'car_spaces': suburb_data['car_spaces']['value'],
                        'land_size': suburb_data['land_size']['value'],
                        'distance_to_cbd': suburb_data['distance_to_cbd']['value'],
                        'school_quality': suburb_data['school_quality']['value'],
                        'infrastructure_score': suburb_data['infrastructure_score']['value'],
                        'flood_risk': suburb_data['flood_risk']['value'],
                        'population_growth': suburb_data['population_growth']['value'],
                        'climate_risk': suburb_data['climate_risk']['value'],
                        'public_transport': suburb_data['public_transport']['value'],
                        'supply_ratio': suburb_data['supply_ratio']['value']
                    }])
                    dataframes.append(df)
                    
                except Exception as e:
                    print(f"Error processing {suburb}: {str(e)}")
                    continue

            if not dataframes:
                raise ValueError("No valid suburb data collected")
            
            return pd.concat(dataframes, ignore_index=True), json.dumps(all_data, indent=4)
        
        except Exception as e:
            if attempt == 2:  # Last attempt
                raise gr.Error("Failed to process suburb data after 3 attempts. Please try again.")
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            time.sleep(2 ** attempt)  # Exponential backoff
            continue

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

def forecast_prices(suburbs, bedrooms=None, bathrooms=None, car_spaces=None, land_size=None, max_price=None):
    """Generate forecasts for multiple suburbs with optional property criteria"""
    # Fetch current data and API response with property criteria
    suburb_data, api_response = fetch_suburb_data(
        suburbs,
        bedrooms=bedrooms,
        bathrooms=bathrooms,
        car_spaces=car_spaces,
        land_size=land_size,
        max_price=max_price
    )
    plot_paths = []
    
    # Parse existing data
    data_dict = json.loads(api_response)

    # Calculate supply impact on growth
    supply_impact = np.where(
        suburb_data['supply_ratio'] < 0.6,  # Severe undersupply
        0.004,  # Strong growth premium
        np.where(
            suburb_data['supply_ratio'] < 0.8,  # Moderate undersupply
            0.003,  # Moderate growth premium
            np.where(
                suburb_data['supply_ratio'] > 1.4,  # Severe oversupply
                -0.005,  # Strong growth penalty
                np.where(
                    suburb_data['supply_ratio'] > 1.2,  # Moderate oversupply
                    -0.003,  # Moderate growth penalty
                    0  # Balanced market
                )
            )
        )
    )
    
    # Calculate base growth rate adjustments from suburb metrics
    growth_adjustments = (
        suburb_data['population_growth'] * 0.004 +  # Population growth has strong impact
        suburb_data['infrastructure_score'] * 0.003 +  # Good infrastructure supports growth
        suburb_data['school_quality'] * 0.002 +  # Quality schools attract families
        suburb_data['public_transport'] * 0.0015 -  # Access increases demand
        suburb_data['flood_risk'] * 0.002 -  # Risk factors reduce growth
        suburb_data['climate_risk'] * 0.002 + # Climate impact on long-term value
        supply_impact  # Supply impact on growth refer above
    )
    
    # Generate year range
    years = np.arange(2025, 2076)
    
    # For each suburb, add price projections
    for suburb, suburb_info in data_dict['suburbs'].items():
        # Get base price for this suburb
        base_price = suburb_info['median_price']['value']
        
        # Calculate suburb-specific growth rate
        suburb_idx = suburb_data[suburb_data['suburb'] == suburb].index[0]
        adjusted_growth = growth_adjustments.iloc[suburb_idx]
        
        # Calculate projections using Monte Carlo with adjusted rates
        n_simulations = 1000
        all_projections = []
        base_growth_rate = np.linspace(
            max(0.015, adjusted_growth - 0.005),  # Floor at 1.5%
            min(0.045, adjusted_growth + 0.01),    # Ceiling at 4.5%
            len(years)
        )

        for _ in range(n_simulations):
            growth_noise = np.random.normal(0, 0.005, len(years))
            growth_rates = base_growth_rate + growth_noise
            projections = base_price * np.cumprod(1 + growth_rates)
            all_projections.append(projections)
        
        # Calculate statistics for each year
        projections_array = np.array(all_projections)
        median_projection = np.median(projections_array, axis=0)
        lower_95 = np.percentile(projections_array, 2.5, axis=0)
        upper_95 = np.percentile(projections_array, 97.5, axis=0)
        lower_std = median_projection - np.std(projections_array, axis=0)
        upper_std = median_projection + np.std(projections_array, axis=0)

        # Add projections to suburb data
        suburb_info['price_projections'] = {
            'years': years.tolist(),
            'median_projection': median_projection.tolist(),
            'confidence_intervals': {
                'lower_95': lower_95.tolist(),
                'upper_95': upper_95.tolist(),
                'lower_std': lower_std.tolist(),
                'upper_std': upper_std.tolist()
            },
            'growth_assumptions': {
                'base_rate_range': [float(base_growth_rate[0]), float(base_growth_rate[-1])],
                'volatility': 0.0055, # Average of Regional and Suburban Volatility
                'growth_adjustment': float(adjusted_growth)
            }
        }

        # Generate and save plot
        plt.figure(figsize=(12, 7))
        plt.fill_between(years, lower_95, upper_95, alpha=0.1, color='#2ecc71', label='95% Confidence Interval')
        plt.fill_between(years, lower_std, upper_std, alpha=0.2, color='#2ecc71', label='68% Confidence Interval')
        plt.plot(years, median_projection, label='Median Projection', color='#2ecc71', linewidth=2)
        
        # Add price annotations at 5-year intervals
        interval_years = years[::5]  # Select every 5th year
        interval_prices = median_projection[::5]  # Get corresponding prices

        for year, price in zip(interval_years, interval_prices):
            plt.annotate(
                f'${price:,.0f}',
                xy=(year, price),
                xytext=(0, 10),
                textcoords='offset points',
                ha='center',
                va='bottom',
                bbox=dict(
                    boxstyle='round,pad=0.5',
                    fc='white',
                    ec='gray',
                    alpha=0.7
                )
            )

        plt.title(f"50-Year Housing Price Forecast for {suburb}", fontsize=14)
        plt.xlabel("Year", fontsize=12)
        plt.ylabel("Median Price (AUD)", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        plot_path = FORECAST_DIR / f"{suburb.replace(' ', '_')}_forecast.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        plot_paths.append(plot_path)
    
    return plot_paths, json.dumps(data_dict, indent=4)

def save_forecast_data(api_data, format='json'):
    """Save forecast data in selected format"""
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
                f.write(f"=== {suburb} ===\n")
                f.write(f"Median Price: ${data['median_price']['value']:,.2f}\n")
                f.write(f"Bedrooms: {data['bedrooms']['value']} (Average: {data['default_averages']['bedrooms']['value']})\n")
                f.write(f"Bathrooms: {data['bathrooms']['value']} (Average: {data['default_averages']['bathrooms']['value']})\n")
                f.write(f"Car Spaces: {data['car_spaces']['value']} (Average: {data['default_averages']['car_spaces']['value']})\n")
                f.write(f"Land Size: {data['land_size']['value']} sqm (Average: {data['default_averages']['land_size']['value']})\n")
                f.write(f"Distance to CBD: {data['distance_to_cbd']['value']} km\n")
                f.write(f"School Quality: {data['school_quality']['value']}/10\n")
                f.write(f"Infrastructure Score: {data['infrastructure_score']['value']}/10\n")
                f.write(f"Flood Risk: {data['flood_risk']['value']}/10\n")
                f.write(f"Population Growth: {data['population_growth']['value']}%\n")
                f.write(f"Climate Risk: {data['climate_risk']['value']}/10\n")
                f.write(f"Public Transport: {data['public_transport']['value']}/10\n")
                f.write(f"Supply Ratio: {data['supply_ratio']['value']:.2f} (Benchmark: {data['supply_ratio']['benchmark']} months)\n\n")
    
    elif format == 'xlsx':
        output_path = f"{base_name}.xlsx"
        df = pd.DataFrame(columns=[
            'Suburb',
            'Median Price (AUD)',
            'Bedrooms',
            'Average Bedrooms',
            'Bathrooms',
            'Average Bathrooms',
            'Car Spaces',
            'Average Car Spaces',
            'Land Size (sqm)',
            'Average Land Size (sqm)',
            'Distance to CBD (km)',
            'School Quality (1-10)',
            'Infrastructure Score (1-10)',
            'Flood Risk (1-10)',
            'Population Growth (%)',
            'Climate Risk (1-10)',
            'Public Transport (1-10)',
            'Supply Ratio',
            'Supply Benchmark (months)'
        ])
        
        for suburb, data in data_dict['suburbs'].items():
            df = pd.concat([df, pd.DataFrame([{
                'Suburb': suburb,
                'Median Price (AUD)': data['median_price']['value'],
                'Bedrooms': data['bedrooms']['value'],
                'Average Bedrooms': data['default_averages']['bedrooms']['value'],
                'Bathrooms': data['bathrooms']['value'],
                'Average Bathrooms': data['default_averages']['bathrooms']['value'],
                'Car Spaces': data['car_spaces']['value'],
                'Average Car Spaces': data['default_averages']['car_spaces']['value'],
                'Land Size (sqm)': data['land_size']['value'],
                'Average Land Size (sqm)': data['default_averages']['land_size']['value'],
                'Distance to CBD (km)': data['distance_to_cbd']['value'],
                'School Quality (1-10)': data['school_quality']['value'],
                'Infrastructure Score (1-10)': data['infrastructure_score']['value'],
                'Flood Risk (1-10)': data['flood_risk']['value'],
                'Population Growth (%)': data['population_growth']['value'],
                'Climate Risk (1-10)': data['climate_risk']['value'],
                'Public Transport (1-10)': data['public_transport']['value'],
                'Supply Ratio': data['supply_ratio']['value'],
                'Supply Benchmark (months)': data['supply_ratio']['benchmark']
            }])], ignore_index=True)
        
        df.to_excel(output_path, index=False)

    else:  # csv format
        output_path = f"{base_name}.csv"
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Define headers with units and ranges
            headers = [
                'Suburb',
                'Median Price (AUD)',
                'Bedrooms (1-6)',
                'Bathrooms (1-4)',
                'Car Spaces (0-3)',
                'Land Size (sqm)',
                'Distance To CBD (km)',
                'School Quality (1-10)',
                'Infrastructure Score (1-10)',
                'Flood Risk (1-10)',
                'Population Growth (% per year)',
                'Climate Risk (1-10)',
                'Public Transport (1-10)',
                'Average Bedrooms (1-6)',
                'Average Bathrooms (1-4)',
                'Average Car Spaces (0-3)',
                'Average Land Size (sqm)',
                'Supply Ratio',
                'Supply Benchmark (months)'
            ]
            
            # Write headers
            writer.writerow(headers)
            
            # Write data for each suburb
            for suburb, suburb_data in data_dict['suburbs'].items():
                row_data = [
                    suburb,
                    suburb_data['median_price']['value'],
                    suburb_data['bedrooms']['value'],
                    suburb_data['bathrooms']['value'],
                    suburb_data['car_spaces']['value'],
                    suburb_data['land_size']['value'],
                    suburb_data['distance_to_cbd']['value'],
                    suburb_data['school_quality']['value'],
                    suburb_data['infrastructure_score']['value'],
                    suburb_data['flood_risk']['value'],
                    suburb_data['population_growth']['value'],
                    suburb_data['climate_risk']['value'],
                    suburb_data['public_transport']['value'],
                    suburb_data['default_averages']['bedrooms']['value'],
                    suburb_data['default_averages']['bathrooms']['value'],
                    suburb_data['default_averages']['car_spaces']['value'],
                    suburb_data['default_averages']['land_size']['value'],
                    suburb_data['supply_ratio']['value'],
                    suburb_data['supply_ratio']['benchmark']
                ]
                writer.writerow(row_data)
    
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
        
        This application uses machine learning and real estate market data to forecast 50-year housing prices for Australian suburbs, with confidence intervals and detailed analytics.
        
        ### Input Options:
        - Empty Input → Analyzes top growth suburbs nationally
        - State Names Only (e.g., "NSW", "Victoria") → Top growth suburbs per state
        - Otherwise → Analyzes suburbs matching input
        - Property Criteria (0 = Any): Bedrooms, Bathrooms, Car Spaces, Land Size, Max Median Price
        \n            
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

    def process_input(value):
        """Convert zero values to None for optional inputs"""
        return None if value == 0 else value
    
    def forecast_with_processing(suburbs, bedrooms, bathrooms, car_spaces, land_size, max_price):
        """Enhanced wrapper function with smart suburb processing"""
        processed_suburbs = process_suburb_input(suburbs)
        suburbs_text = '\n'.join(processed_suburbs)
        
        return forecast_prices(
            suburbs_text,
            process_input(bedrooms),
            process_input(bathrooms),
            process_input(car_spaces),
            process_input(land_size),
            process_input(max_price)
        )
    
    submit_btn = gr.Button("Generate Forecasts", variant="primary")
    
    forecast_gallery = gr.Gallery(label="Price Projections")
    
    api_output = gr.JSON(label="Suburb Data")
    
    export_format = gr.CheckboxGroup(
        choices=['json', 'csv', 'txt', 'xlsx'],
        value=['json'],
        label="Export Data Formats"
    )

    export_btn = gr.Button("Export Data")
    
    export_status = gr.Textbox(
        label="Export Status", 
        interactive=False,
        lines=10,
        value=""
    )
    
    submit_btn.click(
        fn=forecast_with_processing,
        inputs=[
            suburb_input,
            bedrooms_input,
            bathrooms_input,
            car_spaces_input,
            land_size_input,
            max_price_input
        ],
        outputs=[forecast_gallery, api_output]
    )
    
    def update_export_status(api_data, formats, current_status):
        """Update export status while preserving history"""
        new_status = export_data(api_data, formats)
        if current_status:
            return current_status + "\n" + new_status
        return new_status
    
    export_btn.click(
        fn=update_export_status,
        inputs=[api_output, export_format, export_status],
        outputs=export_status
    )

if __name__ == "__main__":
    app.launch()