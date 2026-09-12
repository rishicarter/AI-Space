from dotenv import load_dotenv

from google.adk.agents import Agent, SequentialAgent
# from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.tools.google_search_tool import google_search
from google.adk.models.lite_llm import LiteLlm

from duckduckgo_search import DDGS

load_dotenv()

AGENT_MODEL = "lm_studio/ministral-3-3b"
OPENROUTER_AGENT_MODEL = "openrouter/google/gemma-4-26b-a4b-it:free"
GOOGLE_MODEL = "gemini/gemini-3.5-flash-lite"

def get_weather(city: str) -> dict:
    """Retrieves the current weather report for a specified city.

    Args:
        city (str): The name of the city (e.g., "New York", "London", "Tokyo").

    Returns:
        dict: A dictionary containing the weather information.
              Includes a 'status' key ('success' or 'error').
              If 'success', includes a 'report' key with weather details.
              If 'error', includes an 'error_message' key.
    """
    print(f"--- Tool: get_weather called for city: {city} ---")  # Log tool execution
    city_normalized = city.lower().replace(" ", "")  # Basic normalization

    # Mock weather data
    # api call
    mock_weather_db = {
        "newyork": {
            "status": "success",
            "report": "The weather in New York is sunny with a temperature of 25°C.",
        },
        "london": {
            "status": "success",
            "report": "It's cloudy in London with a temperature of 15°C.",
        },
        "tokyo": {
            "status": "success",
            "report": "Tokyo is experiencing light rain and a temperature of 18°C.",
        },
        "paris": {
            "status": "success",
            "report": "The weather in Paris is sunny with a temperature of 22°C.",
        },
    }

    if city_normalized in mock_weather_db:
        return mock_weather_db[city_normalized]
    else:
        return {
            "status": "error",
            "error_message": f"Sorry, I don't have weather information for '{city}'.",
        }


def web_search(query: str, max_results: int = 5) -> str:
    """Search the web and return formatted results."""

    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=max_results))

    if not results:
        return "No results found."

    output = []

    for i, result in enumerate(results, 1):
        output.append(
            f"""
Result {i}
Title: {result.get('title')}
URL: {result.get('href')}
Snippet: {result.get('body')}
"""
        )

    return "\n".join(output)

destination_research_agent = Agent(
    name="DestinationResearchAgent",
    model=AGENT_MODEL,
    tools=[web_search],
    description="An agent that researches travel destinations and gathers essential information",
    instruction="""
    You are a travel researcher. You will be given a destination and travel preferences, and you will research:
    - Best time to visit and weather patterns
    - Top attractions and must-see locations
    - Local culture, customs, and etiquette tips
    - Transportation options within the destination
    - Safety considerations and travel requirements
    Provide comprehensive destination insights for trip planning.
    """,
    output_key="destination_research",
)


# Itinerary Builder Agent - Creates detailed travel schedule
itinerary_builder_agent = Agent(
    model=LiteLlm(AGENT_MODEL),
    name="ItineraryBuilderAgent",
    description="An agent that creates structured travel itineraries with daily schedules",
    instruction="""
    You are a professional travel planner. Using the research from "destination_research" output, create a detailed itinerary that includes:
    - Day-by-day schedule with recommended activities
    - Suggested accommodation areas or districts
    - Estimated time requirements for each activity
    - Meal recommendations and dining suggestions
    - Budget estimates for major expenses
    Structure it logically for easy following during the trip.
    """,
    output_key="travel_itinerary",
)

# Travel Optimizer Agent - Adds practical tips and optimizations
travel_optimizer_agent = Agent(
    model=LiteLlm(AGENT_MODEL),
    name="TravelOptimizerAgent",
    description="An agent that optimizes travel plans with practical advice and alternatives",
    instruction="""
    You are a seasoned travel consultant. Using the itinerary from "travel_itinerary" output, optimize it by adding:
    - Money-saving tips and budget alternatives
    - Packing recommendations specific to the destination
    - Backup plans for weather or unexpected situations
    - Local apps, websites, or resources to download
    - Cultural do's and don'ts for respectful travel
    
    Format the final output as:
    
    ITINERARY: {travel_itinerary}
    
    OPTIMIZATION TIPS: [your money-saving and practical tips here]
    
    TRAVEL ESSENTIALS: [packing and preparation advice here]
    
    BACKUP PLANS: [alternative options and contingencies here]
    """,
)

# root_agent = Agent(
#     name="MultiAgent",
#     model=LiteLlm(AGENT_MODEL),
#     description="A Multi Agent Setup",
#     instruction="You are a travel planner agent. Help the user plan their trip.",
#     tools=[get_weather]
# )
root_agent = SequentialAgent(
    name="MultiAgent",
    description="A Multi Agent Setup",
    # instruction="You are a travel planner agent. Help the user plan their trip.",
    sub_agents=[
        destination_research_agent,
        itinerary_builder_agent,
        travel_optimizer_agent,
    ],
)