# 🏡 Australian Suburb Housing Price Forecaster
### Version 2.0 - Lifecycle-Aware Model
### Author: Robert Li ([dr-robert-li](https://github.com/dr-robert-li/))

This Machine Learning powered application uses a **lifecycle-aware forecasting model** combining finetuned XGBoost regression with RBA-calibrated econometric parameters to forecast housing prices for Australian suburbs over a 50-year period. The model incorporates suburb maturity stages, market equilibrium mechanisms, and empirically-validated constraints to prevent unrealistic exponential growth.

**Note:** This application is very compute, memory and time intensive. It can consume API credits very quickly. Use with care. It is recommended to use a GPU or a machine with at least 16GB of RAM. Each suburb may take several minutes to generate a response.

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
- Dwelling Type Selection:
  - House: Traditional single-family homes
  - Apartment: Multi-unit residential buildings
  - Townhouse: Connected row housing

### Advanced Analytics
- Price projections with confidence intervals:
  - 5-year (Very High Confidence): Near-term market dynamics
  - 25-year (High Confidence): Primary analysis timeframe
  - 50-year (Moderate Confidence): Long-term trend indicators
- Monte Carlo simulations for price modeling
- Multiple confidence intervals (1 Std.d 68% and 2 Std.d 95%)
- Nominal vs Inflation-Adjusted Values:
  - Current inflation impact
  - Forecast inflation rates
  - Inflation volatility modeling
- Key market indicators:
  - Dwelling type specific growth adjustments
  - School quality scores
  - Infrastructure ratings
  - Flood and climate risk assessments
  - Population growth trends
  - Crime rates
  - Public transport accessibility
  - Distance to CBD based decay
  - Quality and distance of commercial centers
  - Quality and distance of parks and bodies of water
  - Household demographics
    - Income
    - Age
    - Size
  - Supply Market Analysis
    - Current months of housing inventory
    - Supply ratio benchmarking against 6-month standard
    - Dynamic growth adjustments based on supply conditions

### Lifecycle-Aware Forecasting Model (Phase 1-4 Implementation)

#### Phase 1: Suburb Maturity Decay
- **Four lifecycle stages** based on years since maturity:
  - INITIAL_DEVELOPMENT (0-5 years): High momentum, rapid infrastructure impacts
  - RAPID_GROWTH (5-15 years): Peak appreciation, strong migration effects
  - ESTABLISHED_MATURITY (15-30 years): Stabilizing growth, moderate dynamics
  - RENEWAL_OR_DECLINE (30+ years): Low momentum, gentrification potential
- **S-curve maturity decay**: Growth bonuses decay via sigmoid function `1/(1+exp(0.1*(age-15)))`
- **Demographic estimation**: Household size proxies estimate suburb age (3.5→2.1 persons)

#### Phase 2: Market Constraints
- **Supply dampening** (Australia's 0.07 elasticity):
  - Very inelastic housing supply response to price growth
  - Supply ratio >1.2 triggers -1.5% per month excess inventory penalty
- **Affordability brake** (Price-to-Income thresholds):
  - P/I ratio >10x triggers exponential demand collapse
  - Historical evidence: ratios >12x lead to market corrections
  - Non-linear penalty: -2% × ((ratio-10)^1.2)
- **Infrastructure saturation ceiling**:
  - After 10 years maturity, diminishing returns from additional infrastructure
  - Exponential decay approaching -0.2% ceiling over time

#### Phase 3: Year-by-Year State Evolution
- **Dynamic state tracking**: Each year's growth depends on CURRENT state, not Year 0
- **State variables updated annually**:
  - Current price (previous year growth applied)
  - Supply ratio (0.07 elasticity response to cumulative price changes)
  - Annual income (2.5% growth assumption)
  - Years since maturity (increments annually)
  - Household size (demographic decay toward 2.1 persons)
- **RBA-calibrated parameters by stage**:
  - Momentum coefficients: 0.74 → 0.60 → 0.30 → 0.20 (decays with maturity)
  - Error correction speed: 0.05 → 0.10 → 0.14 → 0.18 (accelerates with maturity)

#### Phase 4: Fundamental Value Anchoring
- **Mean reversion to user cost equilibrium**:
  - Fundamental Price = Annual Rent / (interest + depreciation + maintenance - expected appreciation)
  - Stage-specific expected appreciation: 4% → 3.5% → 2.5% → 2%
- **Dynamic interest rate cycles**: 10-year sine wave, ±1.5% amplitude around 5% base
- **Current price-based rents**: Rents adjust with market prices, not fixed at Year 0
- **Distance-based yields**: Inner city 3%, regional/outer 4%

#### Enhanced Validation
- **Five-point sanity checks**:
  1. Total appreciation <200% over 50 years (real terms)
  2. Late-stage growth <2% annually (years 40-50)
  3. Price-to-income ratio <20x (market collapse threshold)
  4. Mean reversion: 10-year deviations <50% from trend
  5. Growth volatility <15% (prevents explosive dynamics)
- **Research-backed thresholds** from RBA, AHURI, NSW Productivity Commission sources

### Market Cycle Modeling
- Default to ABS data but will otherwise use a market cycle model:
  - 10-Year Market Cycles
    - Years 1-3: Strong boom phase (+4-6% above baseline)
    - Years 4-6: Moderate growth (+1-2% above baseline)
    - Years 7-8: Market slowdown (-1-3% below baseline)
    - Years 9-10: Sharp correction (-3-5% below baseline)
  - Long-term average growth maintained at 3.5%
  - Volatility modeling with 0.8% random variation
  - Realistic boom-bust patterns based on historical data

### Optional Deep Market Reasoning
- Comprehensive market analysis for each suburb:
  - Primary 25-year growth trajectory analysis
  - Near-term (5-year) market opportunities
  - Long-term (50-year) sustainability indicators
  - Supply-demand dynamics
  - Infrastructure impact assessment
  - Demographic trend implications
  - Risk assessment and mitigation strategies
  - Crime impact and security considerations
  - Inflation effects analysis
- Research citations and market data sources
- Exportable in RTF and TXT formats
- Configurable research depth:
  - **Low**: Faster analysis with basic research
  - **Medium**: Balanced depth and speed (default)
  - **High**: Exhaustive research and detailed analysis (slowest, most expensive)

#### NOTE: This uses Perplexity's Sonar Deep Research endpoint. It is both expensive and slow, especially with "high" research depth. Each suburb can take several minutes to analyze. Use with care for large datasets of multiple suburbs.

### Model Fine-tuning
- Progressive learning with synthetic data generation
- Supply-aware parameter adjustments
- Volatility modeling for different market conditions and dwelling types
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
- Implements Perplexity API for real-time market analysis (Sonar Pro endpoint)
- Generates high-resolution forecast plots with matplotlib
- **Lifecycle-aware modeling** with year-by-year state evolution
- **RBA-calibrated parameters** from Research Discussion Papers 2018-03 and 2019-01
- **Monte Carlo simulations** (1000 iterations) for confidence intervals
- Supports incremental model training with new data

## Data Sources & Research

All external data sources are documented in `includes/DATA_SOURCES.md`:

- **Supply Elasticity**: RBA RDP 2018-03 (Kendall & Tulip) - 0.07 long-run elasticity
- **Price-to-Income Ratios**: Statista, RBA Bulletin, 50-year historical data
- **Infrastructure Effects**: NSW Productivity Commission 2024, Journal of Housing Economics 2021
- **RBA Model Parameters**: RBA RDP 2019-01 (Saunders & Tulip) - 62-equation econometric model
- **Quality ratings**: ✅ Very High to ⚠️ Medium with recency tracking

See `includes/` directory for:
- `supply_elasticity_by_city.json` - City-specific housing supply constraints
- `price_to_income_ratios.json` - Historical affordability thresholds
- `infrastructure_effects.json` - Calibrated school/transport premiums
- `rba_model_parameters.json` - Momentum, error correction, interest rate sensitivity

## Data Storage

- Forecasts saved in `./forecasts/` directory
- Exported data stored in `./exports/` directory
- Research data in `./includes/` directory (excluded from version control)
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
