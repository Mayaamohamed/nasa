from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
from typing import Optional, List, Dict, Any
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import io
import json

app = FastAPI(title="NASA Weather Probability API", 
              description="Historical weather probability dashboard backend",
              version="2.0")

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# NASA API Configuration
NASA_API_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

# Available weather variables with metadata
WEATHER_VARIABLES = {
    "T2M_MAX": {"name": "Maximum Temperature", "units": "°C", "description": "Daily maximum temperature at 2m height"},
    "T2M_MIN": {"name": "Minimum Temperature", "units": "°C", "description": "Daily minimum temperature at 2m height"},
    "PRECTOTCORR": {"name": "Precipitation", "units": "mm/day", "description": "Corrected precipitation"},
    "WS10M": {"name": "Wind Speed", "units": "m/s", "description": "Wind speed at 10m height"},
    "RH2M": {"name": "Relative Humidity", "units": "%", "description": "Relative humidity at 2m height"},
    "ALLSKY_SFC_SW_DWN": {"name": "Solar Radiation", "units": "kWh/m²/day", "description": "Solar radiation"},
    "CLRSKY_DAYS": {"name": "Clear Sky Days", "units": "days", "description": "Number of clear sky days"},
    "T2MDEW": {"name": "Dew Point", "units": "°C", "description": "Dew point temperature"},
}

# Default probability thresholds
DEFAULT_THRESHOLDS = {
    "extreme_heat": 35,      # °C
    "heat_wave": 30,         # °C (for consecutive days)
    "extreme_cold": 0,       # °C
    "cold_wave": 5,          # °C (for consecutive days)
    "heavy_rain": 10,        # mm/day
    "very_heavy_rain": 25,   # mm/day
    "high_wind": 8,          # m/s
    "very_high_wind": 12,    # m/s
    "high_humidity": 80,     # %
    "low_visibility": 70,    # % humidity threshold for fog/comfort
}

# Pydantic Models for Request Validation
class WeatherQuery(BaseModel):
    city: str
    month: int
    day: int
    variables: Optional[List[str]] = None

    @validator('month')
    def validate_month(cls, v):
        if not 1 <= v <= 12:
            raise ValueError('Month must be between 1 and 12')
        return v

    @validator('day')
    def validate_day(cls, v, values):
        if 'month' in values:
            month = values['month']
            if month in [4, 6, 9, 11] and v > 30:
                raise ValueError(f'Day must be <= 30 for month {month}')
            elif month == 2 and v > 29:
                raise ValueError(f'Day must be <= 29 for February')
            elif v > 31:
                raise ValueError('Day must be <= 31')
        return v




# Utility Functions
def get_city_coords(city_name: str):
    """Get coordinates for a city name using Nominatim"""
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": city_name, "format": "json", "limit": 1}
    headers = {"User-Agent": "NASA-Weather-App/2.0 (contact@weatherapp.com)"}
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        data = resp.json()
        if len(data) == 0:
            return None
        return {"lat": float(data[0]["lat"]), "lon": float(data[0]["lon"])}
    except (requests.RequestException, ValueError, KeyError) as e:
        print(f"Geocoding error: {e}")
        return None

def get_nasa_weather_data(lat: float, lon: float,  variables: List[str]):
    start_year = 2000
    end_year = 2020
    """Fetch weather data from NASA POWER API"""
    parameters = ",".join(variables)
    url = f"{NASA_API_URL}?latitude={lat}&longitude={lon}&start={start_year}&end={end_year}&parameters={parameters}&format=JSON&community=RE"
    
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data
    except requests.RequestException as e:
        print(f"NASA API error: {e}")
        return None

def process_weather_data(raw_data: Dict, variables: List[str]) -> pd.DataFrame:
    """Process and clean NASA weather data"""
    if "properties" not in raw_data or "parameter" not in raw_data["properties"]:
        return pd.DataFrame()

    records = []
    dates = list(raw_data["properties"]["parameter"][variables[0]].keys())
    
    for date_str in dates:
        record = {"date": date_str}
        for var in variables:
            value = raw_data["properties"]["parameter"][var].get(date_str, pd.NA)
            record[var] = value
        records.append(record)

    df = pd.DataFrame(records)
    
    # Data cleaning
    df = df.replace([-99, -999, -9999, -999.9], pd.NA)
    df = df.dropna()
    
    # Convert date and create day-of-year
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["date"])
    df["month_day"] = df["date"].dt.strftime("%m-%d")
    df["year"] = df["date"].dt.year
    
    return df

def calculate_extended_probabilities(df: pd.DataFrame, target_doy: str, thresholds: Dict[str, float]) -> Dict[str, Any]:
    """Calculate comprehensive weather probabilities"""
    subset = df[df["month_day"] == target_doy]
    if subset.empty:
        return {}
    
    # Basic statistics
    stats = {}
    for var in df.columns:
        if var not in ['date', 'month_day', 'year']:
            stats[f"{var}_mean"] = round(subset[var].mean(), 2)
            stats[f"{var}_std"] = round(subset[var].std(), 2)
            stats[f"{var}_min"] = round(subset[var].min(), 2)
            stats[f"{var}_max"] = round(subset[var].max(), 2)
            stats[f"{var}_q25"] = round(subset[var].quantile(0.25), 2)
            stats[f"{var}_q75"] = round(subset[var].quantile(0.75), 2)
    
    # Probability calculations
    probabilities = {}
    
    # Temperature probabilities
    if "T2M_MAX" in subset.columns:
        probabilities["extreme_heat"] = round((subset["T2M_MAX"] > thresholds["extreme_heat"]).mean() * 100, 1)
        probabilities["moderate_heat"] = round((subset["T2M_MAX"] > 25).mean() * 100, 1)
    
    if "T2M_MIN" in subset.columns:
        probabilities["extreme_cold"] = round((subset["T2M_MIN"] < thresholds["extreme_cold"]).mean() * 100, 1)
        probabilities["moderate_cold"] = round((subset["T2M_MIN"] < 10).mean() * 100, 1)
    
    # Precipitation probabilities
    if "PRECTOTCORR" in subset.columns:
        probabilities["heavy_rain"] = round((subset["PRECTOTCORR"] > thresholds["heavy_rain"]).mean() * 100, 1)
        probabilities["very_heavy_rain"] = round((subset["PRECTOTCORR"] > thresholds["very_heavy_rain"]).mean() * 100, 1)
        probabilities["any_rain"] = round((subset["PRECTOTCORR"] > 0.1).mean() * 100, 1)
    
    # Wind probabilities
    if "WS10M" in subset.columns:
        probabilities["high_wind"] = round((subset["WS10M"] > thresholds["high_wind"]).mean() * 100, 1)
        probabilities["very_high_wind"] = round((subset["WS10M"] > thresholds["very_high_wind"]).mean() * 100, 1)
    
    # Humidity and comfort probabilities
    if "RH2M" in subset.columns:
        probabilities["high_humidity"] = round((subset["RH2M"] > thresholds["high_humidity"]).mean() * 100, 1)
        
        # Discomfort index (simplified)
        if "T2M_MAX" in subset.columns:
            discomfort = (subset["T2M_MAX"] > 30) & (subset["RH2M"] > 70)
            probabilities["uncomfortable"] = round(discomfort.mean() * 100, 1)
    
    # Clear sky probability
    if "CLRSKY_DAYS" in subset.columns:
        probabilities["clear_sky"] = round((subset["CLRSKY_DAYS"] == 1).mean() * 100, 1)
    
    return {
        "statistics": stats,
        "probabilities": probabilities,
        "data_points": len(subset),
        "years_covered": f"{subset['year'].min()}-{subset['year'].max()}"
    }

def detect_trends(df: pd.DataFrame, target_doy: str, variable: str) -> Dict[str, Any]:
    """Detect trends in weather data over time"""
    subset = df[df["month_day"] == target_doy]
    if len(subset) < 5:
        return {"trend": "insufficient_data", "slope": 0, "p_value": 1}
    
    # Simple linear regression for trend
    years = subset["year"].values
    values = subset[variable].values
    
    if len(years) < 2:
        return {"trend": "insufficient_data", "slope": 0, "p_value": 1}
    
    # Calculate trend
    slope = np.polyfit(years, values, 1)[0]
    
    # Simple trend classification
    if abs(slope) < 0.01:
        trend = "stable"
    elif slope > 0:
        trend = "increasing"
    else:
        trend = "decreasing"
    
    return {
        "trend": trend,
        "slope": round(slope, 4),
        "change_per_decade": round(slope * 10, 2),
        "unit_change_per_decade": f"{round(slope * 10, 2)} per decade"
    }

# API Endpoints
@app.get("/")
async def root():
    return {
        "message": "NASA Weather Probability API",
        "version": "2.0",
        "endpoints": {
            "weather_probabilities": "/api/weather/probabilities", 
            "download_data": "/api/weather/download",
            "variables_info": "/api/variables",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

@app.get("/api/variables")
async def get_variables_info():
    """Get information about available weather variables"""
    return WEATHER_VARIABLES

@app.get("/api/weather/probabilities")
async def get_weather_probabilities(
    city: Optional[str] = Query(None, description="City name"),
    month: int = Query(..., ge=1, le=12, description="Month (1-12)"),
    day: int = Query(..., ge=1, le=31, description="Day (1-31)"),
    variables: str = Query("T2M_MAX,T2M_MIN,PRECTOTCORR,WS10M,RH2M", description="Comma-separated variables")
):
    """Get comprehensive weather probabilities for a location and date"""
    
    # Fixed years for NASA data
    start_year = 2000
    end_year = 2020

    # Get coordinates
    if city:
        coords = get_city_coords(city)
        if not coords:
            raise HTTPException(status_code=404, detail=f"City '{city}' not found")
        lat, lon = coords["lat"], coords["lon"]
        location_name = city
    else:
        raise HTTPException(status_code=400, detail="City must be provided")

    # Parse variables
    variable_list = [v.strip() for v in variables.split(",")]
    invalid_vars = [v for v in variable_list if v not in WEATHER_VARIABLES]
    if invalid_vars:
        raise HTTPException(status_code=400, detail=f"Invalid variables: {invalid_vars}")
    
    # Format target date
    target_doy = f"{month:02d}-{day:02d}"
    
    # Fetch data from NASA
    raw_data = get_nasa_weather_data(lat, lon, variable_list)
    if not raw_data:
        raise HTTPException(status_code=500, detail="Failed to fetch data from NASA API")
    
    # Process data
    df = process_weather_data(raw_data, variable_list)
    if df.empty:
        raise HTTPException(status_code=404, detail="No data available for the specified parameters")
    
    # Calculate probabilities
    results = calculate_extended_probabilities(df, target_doy, DEFAULT_THRESHOLDS)
    if not results:
        raise HTTPException(status_code=404, detail=f"No data available for {target_doy}")
    
    # Add location info
    results.update({
        "location": location_name,
        "coordinates": {"lat": lat, "lon": lon},
        "target_date": target_doy,
        "year_range": f"{start_year}-{end_year}",
        "variables_used": variable_list
    })
    
    return results

@app.get("/api/weather/download")
async def download_weather_data(
    city: str = Query(..., description="City name"),
    month: int = Query(..., ge=1, le=12, description="Month (1-12)"),
    day: int = Query(..., ge=1, le=31, description="Day (1-31)"),
    format: str = Query("csv", regex="^(csv|json)$")
):
    """Download weather data in CSV or JSON format"""

    # fixed years for NASA POWER dataset
    start_year = 1980
    end_year = 2020
    
    coords = get_city_coords(city)
    if not coords:
        raise HTTPException(status_code=404, detail=f"City '{city}' not found")
    
    # Get all available variables
    all_variables = list(WEATHER_VARIABLES.keys())
    raw_data = get_nasa_weather_data(coords["lat"], coords["lon"], all_variables)
    if not raw_data:
        raise HTTPException(status_code=500, detail="Failed to fetch data from NASA API")
    
    df = process_weather_data(raw_data, all_variables)
    if df.empty:
        raise HTTPException(status_code=404, detail="No data available for download")
    
    # Filter for the specific day of year across all years
    target_doy = f"{month:02d}-{day:02d}"
    filtered_df = df[df["month_day"] == target_doy]
    
    if filtered_df.empty:
        raise HTTPException(status_code=404, detail=f"No data available for {target_doy}")
    
    # Add metadata
    metadata = {
        "location": city,
        "coordinates": coords,
        "target_doy": target_doy,
        "year_range": f"{start_year}-{end_year}",
        "download_date": datetime.utcnow().isoformat(),
        "data_points": len(filtered_df),
        "variables": list(WEATHER_VARIABLES.keys())
    }
    
    if format == "csv":
        # Create CSV with metadata as header comments
        output = io.StringIO()
        
        # Write metadata as comments
        for key, value in metadata.items():
            output.write(f"# {key}: {value}\n")
        output.write("\n")
        
        # Write data
        filtered_df.to_csv(output, index=False)
        output.seek(0)
        
        filename = f"weather_data_{city}_{target_doy}_{start_year}_{end_year}.csv"
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    
    else:  # JSON format
        return {
            "metadata": metadata,
            "data": filtered_df.to_dict(orient="records")
        }

@app.get("/api/weather/chart")
async def get_chart_data(
    city: str = Query(..., description="City name"),
    month: int = Query(..., ge=1, le=12),
    day: int = Query(..., ge=1, le=31),
    variable: str = Query("T2M_MAX", description="Variable for chart")
):
    """Get data formatted for charts"""
    
    if variable not in WEATHER_VARIABLES:
        raise HTTPException(status_code=400, detail=f"Invalid variable: {variable}")
    
    coords = get_city_coords(city)
    if not coords:
        raise HTTPException(status_code=404, detail=f"City '{city}' not found")
    
    target_doy = f"{month:02d}-{day:02d}"
    raw_data = get_nasa_weather_data(coords["lat"], coords["lon"], [variable])
    if not raw_data:
        raise HTTPException(status_code=500, detail="Failed to fetch data from NASA API")
    
    df = process_weather_data(raw_data, [variable])
    if df.empty:
        raise HTTPException(status_code=404, detail="No data available")
    
    subset = df[df["month_day"] == target_doy]
    if subset.empty:
        raise HTTPException(status_code=404, detail=f"No data for {target_doy}")
    
    # Format data for charts
    chart_data = {
        "labels": subset["year"].tolist(),
        "datasets": [{
            "label": f"{WEATHER_VARIABLES[variable]['name']} ({WEATHER_VARIABLES[variable]['units']})",
            "data": subset[variable].tolist(),
            "borderColor": "rgb(75, 192, 192)",
            "backgroundColor": "rgba(75, 192, 192, 0.2)"
        }]
    }
    
    return {
        "location": city,
        "target_date": target_doy,
        "variable": variable,
        "variable_info": WEATHER_VARIABLES[variable],
        "chart_data": chart_data,
        "statistics": {
            "mean": round(subset[variable].mean(), 2),
            "std": round(subset[variable].std(), 2),
            "min": round(subset[variable].min(), 2),
            "max": round(subset[variable].max(), 2)
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)