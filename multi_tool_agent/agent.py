from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import google_search
from google.genai import types

# Define constants
APP_NAME = "simple_search_agent"
USER_ID = "user1234"
SESSION_ID = "search_session_1"

# Initialize the agent with just the google_search tool
search_agent = Agent(
    name="simple_search_agent",
    model="gemini-2.0-flash-lite",
    description="A simple agent that performs Google searches.",
    instruction=(
        "I'm a search assistant that can find information on the web. "
        "Ask me to search for anything, and I'll provide the most relevant results."
    ),
    tools=[google_search]  # Only using the built-in google_search tool
)

# Set up session service and create a session
session_service = InMemorySessionService()
session = session_service.create_session(
    app_name=APP_NAME, 
    user_id=USER_ID, 
    session_id=SESSION_ID
)

# Create a runner for the agent
runner = Runner(
    agent=search_agent, 
    app_name=APP_NAME, 
    session_service=session_service
)

# Function to interact with the agent
def call_agent(query):
    """
    Send a query to the agent and print its response.
    
    Args:
        query (str): The search query to send to the agent
    """
    print(f"\nUser Query: {query}")
    
    # Create a content object with the user's query
    content = types.Content(
        role='user', 
        parts=[types.Part(text=query)]
    )
    
    # Run the agent with the query
    events = runner.run(
        user_id=USER_ID, 
        session_id=SESSION_ID, 
        new_message=content
    )
    
    # Process the response events
    for event in events:
        if event.is_final_response():
            final_response = event.content.parts[0].text
            print("\nAgent Response:")
            print(final_response)

# Example usage
if __name__ == "__main__":
    # Test the agent with some example queries
    call_agent("What is machine learning?")
    call_agent("Who won the last World Cup?")
    call_agent("What are the latest advances in renewable energy?")