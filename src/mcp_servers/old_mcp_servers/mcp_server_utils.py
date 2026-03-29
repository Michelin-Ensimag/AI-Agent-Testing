"""
mcp_server_utils.py - general-purpose MCP tools: Wikipedia, weather, unit conversion

Complements mcp_server_stock.py. Used together, both servers give the agent
a broader tool set to pick from, which is useful for testing tool selection
and multi-step reasoning (e.g. look up a city on Wikipedia, then get its weather).

Run standalone:
    python mcp_server_utils.py
"""

import requests
import wikipediaapi
from fastmcp import FastMCP

mcp = FastMCP("Utils Agent Server")


@mcp.tool()
def wikipedia_search(query: str) -> dict:
    """
    Fetch the Wikipedia summary for a given topic.

    Args:
        query: e.g. 'Apple Inc', 'Eiffel Tower', 'compound interest'

    Returns:
        Article title, a short summary, and the Wikipedia URL.
    """
    wiki = wikipediaapi.Wikipedia(
        language="en",
        user_agent="ai-agent-testing/1.0 (educational project)"
    )

    page = wiki.page(query)

    if not page.exists():
        return {"error": f"No Wikipedia article found for '{query}'. Try a different search term."}

    summary = page.summary[:600].rsplit(" ", 1)[0] + "..."  # clean word boundary

    return {
        "title": page.title,
        "summary": summary,
        "url": page.fullurl,
    }


_GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
_WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

_WMO_CODES = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Depositing rime fog",
    51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
    61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
    71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
    80: "Slight showers", 81: "Moderate showers", 82: "Violent showers",
    95: "Thunderstorm", 96: "Thunderstorm with hail", 99: "Thunderstorm with heavy hail",
}


@mcp.tool()
def get_weather(city: str) -> dict:
    """
    Get current weather for a city via Open-Meteo (no API key needed).

    Args:
        city: e.g. 'Paris', 'Tokyo', 'New York'

    Returns:
        Temperature (°C), wind speed (km/h), and weather condition.
    """
    # geocode city name → lat/lon
    geo_resp = requests.get(
        _GEOCODE_URL,
        params={"name": city, "count": 1, "language": "en", "format": "json"},
        timeout=10,
    )
    geo_resp.raise_for_status()
    geo_data = geo_resp.json()

    if not geo_data.get("results"):
        return {"error": f"City '{city}' not found. Try a different spelling."}

    result = geo_data["results"][0]
    lat, lon = result["latitude"], result["longitude"]
    full_name = result.get("name", city)
    country = result.get("country", "")

    weather_resp = requests.get(
        _WEATHER_URL,
        params={
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,wind_speed_10m,weathercode",
            "timezone": "auto",
        },
        timeout=10,
    )
    weather_resp.raise_for_status()
    weather_data = weather_resp.json()

    current = weather_data["current"]
    wmo = current.get("weathercode", -1)
    condition = _WMO_CODES.get(wmo, f"Unknown (WMO code {wmo})")

    return {
        "city": f"{full_name}, {country}",
        "temperature_c": current["temperature_2m"],
        "wind_speed_kmh": current["wind_speed_10m"],
        "condition": condition,
        "latitude": lat,
        "longitude": lon,
    }


# All factors convert to a common SI base (metres for length, kg for mass, m/s for speed).
# Temperature is handled separately since it needs offset logic, not just multiplication.
_CONVERSIONS: dict[str, dict[str, float]] = {
    # Length - base: metre
    "m":   {"category": "length", "factor": 1.0},
    "km":  {"category": "length", "factor": 1000.0},
    "cm":  {"category": "length", "factor": 0.01},
    "mm":  {"category": "length", "factor": 0.001},
    "mi":  {"category": "length", "factor": 1609.344},
    "mile": {"category": "length", "factor": 1609.344},
    "miles": {"category": "length", "factor": 1609.344},
    "ft":  {"category": "length", "factor": 0.3048},
    "foot": {"category": "length", "factor": 0.3048},
    "feet": {"category": "length", "factor": 0.3048},
    "in":  {"category": "length", "factor": 0.0254},
    "inch": {"category": "length", "factor": 0.0254},
    "inches": {"category": "length", "factor": 0.0254},
    "yd":  {"category": "length", "factor": 0.9144},
    "yard": {"category": "length", "factor": 0.9144},
    "yards": {"category": "length", "factor": 0.9144},
    # Mass - base: kg
    "kg":  {"category": "mass", "factor": 1.0},
    "g":   {"category": "mass", "factor": 0.001},
    "mg":  {"category": "mass", "factor": 1e-6},
    "lb":  {"category": "mass", "factor": 0.453592},
    "lbs": {"category": "mass", "factor": 0.453592},
    "pound": {"category": "mass", "factor": 0.453592},
    "pounds": {"category": "mass", "factor": 0.453592},
    "oz":  {"category": "mass", "factor": 0.0283495},
    "ounce": {"category": "mass", "factor": 0.0283495},
    "ounces": {"category": "mass", "factor": 0.0283495},
    "t":   {"category": "mass", "factor": 1000.0},
    "tonne": {"category": "mass", "factor": 1000.0},
    # Speed - base: m/s
    "m/s":  {"category": "speed", "factor": 1.0},
    "km/h": {"category": "speed", "factor": 1/3.6},
    "kmh":  {"category": "speed", "factor": 1/3.6},
    "mph":  {"category": "speed", "factor": 0.44704},
    "knot": {"category": "speed", "factor": 0.514444},
    "knots": {"category": "speed", "factor": 0.514444},
}

_TEMPERATURE_UNITS = {"c", "celsius", "f", "fahrenheit", "k", "kelvin"}


def _convert_temperature(value: float, from_unit: str, to_unit: str) -> float:
    fu = from_unit.lower().lstrip("°")
    tu = to_unit.lower().lstrip("°")
    if fu in ("f", "fahrenheit"):
        celsius = (value - 32) * 5 / 9
    elif fu in ("k", "kelvin"):
        celsius = value - 273.15
    else:
        celsius = value

    if tu in ("f", "fahrenheit"):
        return celsius * 9 / 5 + 32
    elif tu in ("k", "kelvin"):
        return celsius + 273.15
    else:
        return celsius


@mcp.tool()
def convert_units(value: float, from_unit: str, to_unit: str) -> dict:
    """
    Convert a numeric value between units.

    Supported: length (m, km, mi, ft, in, yd), mass (kg, g, lb, oz, t),
    speed (m/s, km/h, mph, knots), temperature (C, F, K).

    Args:
        value: numeric value to convert
        from_unit: source unit, e.g. 'km', 'lbs', 'F'
        to_unit: target unit, e.g. 'miles', 'kg', 'C'
    """
    fu = from_unit.lower().lstrip("°")
    tu = to_unit.lower().lstrip("°")

    if fu in _TEMPERATURE_UNITS or tu in _TEMPERATURE_UNITS:
        converted = _convert_temperature(value, fu, tu)
        return {
            "value": value,
            "from_unit": from_unit,
            "to_unit": to_unit,
            "result": round(converted, 6),
        }

    from_info = _CONVERSIONS.get(fu)
    to_info = _CONVERSIONS.get(tu)

    if from_info is None:
        return {"error": f"Unknown unit '{from_unit}'. Check supported units in the tool description."}
    if to_info is None:
        return {"error": f"Unknown unit '{to_unit}'. Check supported units in the tool description."}
    if from_info["category"] != to_info["category"]:
        return {
            "error": (
                f"Incompatible categories: '{from_unit}' is {from_info['category']}, "
                f"'{to_unit}' is {to_info['category']}."
            )
        }

    si_value = value * from_info["factor"]
    result = si_value / to_info["factor"]

    return {
        "value": value,
        "from_unit": from_unit,
        "to_unit": to_unit,
        "result": round(result, 8),
        "category": from_info["category"],
    }


if __name__ == "__main__":
    mcp.run(show_banner=False)
