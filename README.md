# 🏡 Australian Suburb Housing Price Forecaster
### Version 1.0
### Author: Robert Li ([dr-robert-li](https://github.com/dr-robert-li/))
    
This Machine Learning powered application uses finetuned XGBoost advanced regression model to forecast housing prices for Australian suburbs over a 50-year period using machine learning and real estate market data.

## Features

### Smart Suburb Analysis
- Empty input: Returns top growth suburbs nationally
- State input: Analyzes top suburbs per specified state
- Custom input: Detailed analysis for specific suburbs

### Property Criteria Customization
- Bedrooms (0-15)
- Bathrooms (0-20)
- Car Spaces (0-10)
- Land Size (sqm)
- Max Initial Median Price (AUD)
- Setting any criteria to 0 analyzes all properties

### Advanced Analytics
- 50-year price projections with confidence intervals
- Monte Carlo simulations for price modeling
- Multiple confidence intervals (1 Std.d 68% and 2 Std.d 95%)
- Key market indicators:
  - School quality scores
  - Infrastructure ratings
  - Flood and climate risk assessments
  - Population growth trends
  - Public transport accessibility
  - Distance to CBD based decay
  - Supply Market Analysis
    - Current months of housing inventory
    - Supply ratio benchmarking against 6-month standard
    - Dynamic growth adjustments based on supply conditions

### Model Fine-tuning
- Progressive learning with synthetic data generation
- Supply-aware parameter adjustments
- Volatility modeling for different market conditions
- Confidence interval calculations using Monte Carlo simulations

### Data Export Options
- JSON: Complete dataset with all metrics
- CSV: Tabular format for spreadsheet analysis
- TXT: Human-readable detailed reports
- XLSX: Excel workbook with formatted data

## Environment Setup

```bash
python -m venv venv
```

Activate the virtual environment using:

```bash
source venv/bin/activate # For Unix/Linux
venv\Scripts\activate ## For Windows
```

## Installation

```bash
pip install -r requirements.txt
```

Put your `PERPLEXITY_API_KEY` key in an .env file. 

## Usage

1. Launch the application:

```bash
python housing_forecaster.py
```

2. Access the web interface at http://localhost:7860

3. Input suburbs using any format:

```bash
Sydney CBD NSW
Noosa QLD
Melbourne CBD VIC
```

4. Set optional property criteria
5. Click "Generate Forecasts"
6. Export data in preferred formats

## Technical Details

- Uses XGBoost model fine-tuned on Australian housing data
- Implements Perplexity API for real-time market analysis
- Generates high-resolution forecast plots with matplotlib
- Supports incremental model training with new data

## Data Storage
- Forecasts saved in ./forecasts directory
- Exported data stored in ./exports directory
- Timestamped files for version tracking

## Dependencies

```bash
gradio: Web interface
pandas: Data processing
numpy: Numerical computations
matplotlib: Visualization
xgboost: Machine learning
huggingface_hub: Model management
```

## License

MIT License

Copyright (c) 2025

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Disclaimer

This software is provided for informational purposes only. The forecasts and analyses generated by this application are based on machine learning models and available data, which may have limitations or inaccuracies. Users should not rely solely on these projections for financial decisions. Always consult with qualified real estate and financial professionals before making investment decisions.

The developers and contributors of this software accept no responsibility for any losses or damages arising from its use.
