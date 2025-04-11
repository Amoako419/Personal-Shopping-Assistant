from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.agents.llm_agent import LlmAgent
from google.adk.tools import google_search
from google.genai import types
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner

# --- Constants ---
APP_NAME = "shopping_assistant_app"
USER_ID = "shopper_user_01"
SESSION_ID = "shopping_session_01"
GEMINI_MODEL = "gemini-2.0-flash-exp"

# --- 1. Define Sub-Agents for Each Shopping Task ---

# Product Search Agent
# Takes the user's query about a product and searches for relevant information
product_search_agent = LlmAgent(
    name="ProductSearchAgent",
    model=GEMINI_MODEL,
    instruction="""You are a Product Search AI.
    Based on the user's request for a product, formulate 2-3 specific search queries to find:
    1. Product details and specifications
    2. Price comparisons across different retailers
    3. Customer reviews and ratings
    
    Use clear, specific search terms that will yield relevant results.
    Output these search queries, one per line.
    """,
    description="Creates effective search queries for product information.",
    # Stores its output (the search queries) into the session state
    output_key="search_queries",
    tools=[google_search]
)

# Information Retrieval Agent
# Takes the search queries from the previous agent and performs searches
information_retrieval_agent = LlmAgent(
    name="InformationRetrievalAgent",
    model=GEMINI_MODEL,
    instruction="""You are an Information Retrieval AI.
    Take the search queries provided in the session state under the key 'search_queries'.
    Perform a Google search for each query and collect the most relevant information about the product.
    Focus on finding:
    - Current prices from major retailers
    - Key product specifications
    - Summary of customer reviews and ratings
    - Any special deals or promotions
    
    Organize this information clearly, but retain all the factual details.
    """,
    description="Retrieves and organizes product information from search results.",
    # Stores the raw search results in the session state
    output_key="raw_product_info",
    tools=[google_search]
)

# Product Analysis Agent
# Analyzes the information gathered and prepares recommendations
product_analysis_agent = LlmAgent(
    name="ProductAnalysisAgent",
    model=GEMINI_MODEL,
    instruction="""You are a Product Analysis AI.
    Analyze the product information provided in the session state under the key 'raw_product_info'.
    Create a comprehensive analysis including:
    
    1. PRICE COMPARISON:
       - List the best prices found
       - Note price variations across retailers
       - Identify any special offers or discounts
    
    2. FEATURES ASSESSMENT:
       - Summarize key features and specifications
       - Highlight standout features
       - Note any limitations or drawbacks
    
    3. CUSTOMER SENTIMENT:
       - Summarize what customers like about the product
       - Note common complaints or issues
       - Provide an overall rating if available
    
    Be factual and objective in your analysis.
    """,
    description="Analyzes product information to create a detailed assessment.",
    # Stores the analysis in the session state
    output_key="product_analysis"
)

# Recommendation Agent
# Takes all the information and creates a final recommendation for the user
recommendation_agent = LlmAgent(
    name="RecommendationAgent",
    model=GEMINI_MODEL,
    instruction="""You are a Shopping Recommendation AI.
    Review the product analysis in the session state under the key 'product_analysis'.
    Based on this analysis, create a helpful, personalized shopping recommendation.
    
    Your recommendation should include:
    
    1. A clear verdict on whether the product is worth buying
    2. The best place to purchase (best combination of price and reliability)
    3. Alternative options if applicable
    4. Any timing considerations (e.g., wait for a sale, new model coming soon)
    
    Make your recommendation conversational and helpful, as if you're advising a friend.
    Address the user directly and be concise but thorough.
    """,
    description="Creates final personalized shopping recommendations.",
    # The output will be the final response to the user
    output_key="final_recommendation"
)

# --- 2. Create the SequentialAgent ---
# This agent orchestrates the shopping assistant pipeline
root_agent = SequentialAgent(
    name="ShoppingAssistantAgent",
    sub_agents=[product_search_agent, information_retrieval_agent, product_analysis_agent, recommendation_agent],
    description="A comprehensive shopping assistant that searches for products, analyzes information, and provides recommendations."
)

# Session and Runner
session_service = InMemorySessionService()
session = session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)
runner = Runner(agent=root_agent, app_name=APP_NAME, session_service=session_service)


# Agent Interaction
def call_agent(query):
    """
    Send a query to the shopping assistant and print its response.
    
    Args:
        query (str): The shopping query from the user
    """
    print(f"\nUser Query: {query}")
    content = types.Content(role='user', parts=[types.Part(text=query)])
    events = runner.run(user_id=USER_ID, session_id=SESSION_ID, new_message=content)

    for event in events:
        if event.is_final_response():
            final_response = event.content.parts[0].text
            print("\nShopping Assistant Response:")
            print(final_response)


