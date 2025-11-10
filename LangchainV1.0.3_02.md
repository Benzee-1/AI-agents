# LangChain Mastery: Complete Agent Development Course

## Course Overview
This comprehensive course teaches you to build production-ready AI agents using LangChain v1.0. You'll master everything from basic concepts to advanced multi-agent systems, with extensive hands-on projects and real-world examples.

**Duration**: 60+ hours | **Level**: Beginner to Advanced | **Prerequisites**: Python basics, API familiarity

---

# Module 1: Foundations of LangChain

## Lesson 1.1: Introduction and Philosophy (45 minutes)

### Learning Objectives
- Understand LangChain's core philosophy and design principles
- Differentiate between LangChain and LangGraph
- Grasp the evolution from prototyping to production
- Set up your first working agent

### Content

#### What is LangChain?
LangChain is built on two fundamental beliefs:
1. **LLMs are better when combined with external data sources**
2. **The future of applications is increasingly agentic**

The framework provides:
- **Easy onboarding**: Build agents in under 10 lines of code
- **Production readiness**: Built on LangGraph for durability and streaming
- **Standardized interfaces**: Work with any model provider seamlessly

#### Core Architecture Overview
```python
# The simplest possible LangChain agent
from langchain.agents import create_agent

def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"It's sunny in {city}!"

agent = create_agent(
    model="gpt-4o",
    tools=[get_weather],
    system_prompt="You are a helpful weather assistant"
)

# Run the agent
result = agent.invoke({
    "messages": [{"role": "user", "content": "What's the weather in Paris?"}]
})
print(result["messages"][-1].content)
```

#### LangChain vs LangGraph Relationship
- **LangChain**: High-level abstractions for quick development
- **LangGraph**: Low-level orchestration for custom workflows
- **Integration**: LangChain agents are built on LangGraph under the hood

#### Historical Evolution
- **2022**: Initial release with chains and LLM abstractions
- **2023**: Function calling, LangSmith observability, JavaScript support
- **2024**: LangGraph introduction, 700+ integrations
- **2025**: v1.0 with unified agent abstraction and multimodal support

### Practical Exercise: Your First Agent
Create a simple greeting agent to understand the basic structure:

```python
from langchain.agents import create_agent

def greet_user(name: str, time_of_day: str = "day") -> str:
    """Greet a user by name with appropriate time greeting."""
    greetings = {
        "morning": "Good morning",
        "afternoon": "Good afternoon", 
        "evening": "Good evening",
        "day": "Hello"
    }
    greeting = greetings.get(time_of_day, "Hello")
    return f"{greeting} {name}! How can I help you today?"

def get_current_time() -> str:
    """Get the current time."""
    from datetime import datetime
    now = datetime.now()
    hour = now.hour
    
    if 5 <= hour < 12:
        return "morning"
    elif 12 <= hour < 17:
        return "afternoon"
    elif 17 <= hour < 22:
        return "evening"
    else:
        return "night"

greeting_agent = create_agent(
    model="gpt-4o-mini",
    tools=[greet_user, get_current_time],
    system_prompt="""You are a friendly greeting assistant. 
    Always get the current time first, then greet users appropriately.
    Be warm and welcoming in your responses."""
)

# Test the agent
test_cases = [
    "Hi, my name is Alice",
    "Hello there, I'm Bob", 
    "Good day, Sarah here"
]

for test in test_cases:
    print(f"\nUser: {test}")
    response = greeting_agent.invoke({
        "messages": [{"role": "user", "content": test}]
    })
    print(f"Agent: {response['messages'][-1].content}")
```

### Key Takeaways
1. LangChain prioritizes ease of use while maintaining production capabilities
2. Agents combine models with tools for dynamic problem-solving
3. The framework abstracts complexity while providing flexibility
4. Built-in observability helps debug and improve agent behavior

---

## Lesson 1.2: Installation and Environment Setup (30 minutes)

### Learning Objectives
- Install LangChain and configure development environment
- Set up API keys and LangSmith integration
- Verify installation with a working example
- Understand provider-specific configurations

### Content

#### Core Installation
```bash
# Basic LangChain installation
pip install -U langchain

# Provider-specific packages (install as needed)
pip install -U langchain-openai      # OpenAI models
pip install -U langchain-anthropic   # Claude models  
pip install -U langchain-google-genai # Google Gemini
pip install -U langchain-community   # Community integrations

# Additional utilities
pip install -U langgraph            # For advanced orchestration
pip install -U langsmith            # For observability
```

#### Environment Configuration
Create a `.env` file for your API keys:
```bash
# .env file
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
LANGSMITH_API_KEY=lsv2_...
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=my-langchain-project
```

#### Python Environment Setup
```python
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Verify API keys are loaded
required_keys = ["OPENAI_API_KEY", "LANGSMITH_API_KEY"]
for key in required_keys:
    if not os.getenv(key):
        print(f"⚠️  Warning: {key} not found in environment")
    else:
        print(f"✅ {key} loaded")

# Set up LangSmith tracing
os.environ["LANGSMITH_TRACING"] = "true"
print("🔍 LangSmith tracing enabled")
```

#### Installation Verification
```python
# verify_setup.py - Complete verification script
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
import sys

def test_model_initialization():
    """Test that models can be initialized."""
    try:
        # Test OpenAI model
        model = init_chat_model("gpt-4o-mini")
        response = model.invoke([{"role": "user", "content": "Hello"}])
        print("✅ Model initialization successful")
        return True
    except Exception as e:
        print(f"❌ Model initialization failed: {e}")
        return False

def test_agent_creation():
    """Test agent creation and execution."""
    try:
        def simple_tool(text: str) -> str:
            """Echo the input text."""
            return f"You said: {text}"
        
        agent = create_agent(
            model="gpt-4o-mini",
            tools=[simple_tool],
            system_prompt="You are a test assistant."
        )
        
        result = agent.invoke({
            "messages": [{"role": "user", "content": "Test message"}]
        })
        
        print("✅ Agent creation and execution successful")
        return True
    except Exception as e:
        print(f"❌ Agent creation failed: {e}")
        return False

def test_langsmith_integration():
    """Test LangSmith integration."""
    try:
        import langsmith
        client = langsmith.Client()
        # Simple check - if no exception, likely working
        print("✅ LangSmith integration available")
        return True
    except Exception as e:
        print(f"⚠️  LangSmith integration issue: {e}")
        return False

def main():
    """Run all verification tests."""
    print("🚀 LangChain Installation Verification")
    print("=" * 50)
    
    tests = [
        ("Model Initialization", test_model_initialization),
        ("Agent Creation", test_agent_creation),
        ("LangSmith Integration", test_langsmith_integration)
    ]
    
    passed = 0
    for test_name, test_func in tests:
        print(f"\n🧪 Testing {test_name}...")
        if test_func():
            passed += 1
    
    print(f"\n📊 Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All systems ready! You can start building agents.")
    else:
        print("⚠️  Some tests failed. Check your configuration.")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

#### Provider-Specific Setup

**OpenAI Configuration**
```python
from langchain_openai import ChatOpenAI

# Basic setup
model = ChatOpenAI(
    model="gpt-4o",
    api_key="your-api-key",  # or use environment variable
    temperature=0.7,
    max_tokens=1000
)

# Advanced configuration
model = ChatOpenAI(
    model="gpt-4o",
    temperature=0.7,
    max_tokens=1000,
    timeout=30,
    max_retries=3,
    base_url="https://api.openai.com/v1",  # Custom endpoint if needed
    organization="your-org-id"  # If using organization
)
```

**Anthropic Configuration**
```python
from langchain_anthropic import ChatAnthropic

model = ChatAnthropic(
    model="claude-3-sonnet-20240229",
    api_key="your-anthropic-key",
    temperature=0.7,
    max_tokens=1000
)
```

### Practical Exercise: Environment Health Check
Create a comprehensive health check script:

```python
# health_check.py
import os
import sys
from typing import Dict, List, Tuple
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent

class EnvironmentHealthCheck:
    """Comprehensive environment health check."""
    
    def __init__(self):
        self.results = {}
        self.issues = []
    
    def check_python_version(self) -> bool:
        """Check Python version compatibility."""
        version = sys.version_info
        required = (3, 10)
        
        if version >= required:
            self.results["python_version"] = f"✅ Python {version.major}.{version.minor}.{version.micro}"
            return True
        else:
            issue = f"❌ Python {version.major}.{version.minor} (requires 3.10+)"
            self.results["python_version"] = issue
            self.issues.append("Upgrade Python to 3.10 or higher")
            return False
    
    def check_required_packages(self) -> bool:
        """Check that required packages are installed."""
        required_packages = [
            "langchain", "langsmith", "openai", "anthropic"
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if not missing_packages:
            self.results["packages"] = "✅ All required packages installed"
            return True
        else:
            self.results["packages"] = f"❌ Missing: {', '.join(missing_packages)}"
            self.issues.append(f"Install missing packages: pip install {' '.join(missing_packages)}")
            return False
    
    def check_api_keys(self) -> bool:
        """Check API key configuration."""
        keys_to_check = [
            ("OPENAI_API_KEY", "OpenAI"),
            ("ANTHROPIC_API_KEY", "Anthropic"),
            ("LANGSMITH_API_KEY", "LangSmith")
        ]
        
        configured_keys = []
        missing_keys = []
        
        for env_var, service in keys_to_check:
            if os.getenv(env_var):
                configured_keys.append(service)
            else:
                missing_keys.append(f"{service} ({env_var})")
        
        if configured_keys:
            self.results["api_keys"] = f"✅ Configured: {', '.join(configured_keys)}"
        
        if missing_keys:
            self.results["missing_keys"] = f"⚠️  Missing: {', '.join(missing_keys)}"
            self.issues.append("Configure missing API keys in .env file")
        
        return len(configured_keys) > 0
    
    def check_model_access(self) -> bool:
        """Test actual model access."""
        test_models = []
        
        # Test OpenAI if key available
        if os.getenv("OPENAI_API_KEY"):
            try:
                model = init_chat_model("gpt-4o-mini")
                response = model.invoke([{"role": "user", "content": "Hi"}])
                test_models.append("OpenAI ✅")
            except Exception as e:
                test_models.append(f"OpenAI ❌ ({str(e)[:50]}...)")
                self.issues.append("OpenAI API access issue - check API key and credits")
        
        # Test Anthropic if key available
        if os.getenv("ANTHROPIC_API_KEY"):
            try:
                model = init_chat_model("claude-3-haiku-20240307")
                response = model.invoke([{"role": "user", "content": "Hi"}])
                test_models.append("Anthropic ✅")
            except Exception as e:
                test_models.append(f"Anthropic ❌ ({str(e)[:50]}...)")
                self.issues.append("Anthropic API access issue - check API key and credits")
        
        if test_models:
            self.results["model_access"] = " | ".join(test_models)
            return any("✅" in model for model in test_models)
        else:
            self.results["model_access"] = "❌ No API keys configured for testing"
            return False
    
    def check_langsmith_connection(self) -> bool:
        """Test LangSmith connection."""
        if not os.getenv("LANGSMITH_API_KEY"):
            self.results["langsmith"] = "⚠️  API key not configured"
            return False
        
        try:
            import langsmith
            client = langsmith.Client()
            # Try to access client info (lightweight operation)
            self.results["langsmith"] = "✅ LangSmith connected"
            return True
        except Exception as e:
            self.results["langsmith"] = f"❌ Connection failed: {str(e)[:50]}..."
            self.issues.append("LangSmith connection issue - check API key")
            return False
    
    def run_full_check(self) -> Dict[str, any]:
        """Run all health checks."""
        print("🏥 Running Environment Health Check")
        print("=" * 50)
        
        checks = [
            ("Python Version", self.check_python_version),
            ("Required Packages", self.check_required_packages),
            ("API Keys", self.check_api_keys),
            ("Model Access", self.check_model_access),
            ("LangSmith", self.check_langsmith_connection)
        ]
        
        passed_checks = 0
        for check_name, check_func in checks:
            print(f"\n🔍 {check_name}...")
            if check_func():
                passed_checks += 1
        
        # Print results
        print(f"\n📋 Health Check Results:")
        print("-" * 30)
        for key, value in self.results.items():
            print(f"{key}: {value}")
        
        # Print issues if any
        if self.issues:
            print(f"\n🔧 Issues to Fix:")
            for i, issue in enumerate(self.issues, 1):
                print(f"{i}. {issue}")
        
        print(f"\n📊 Overall: {passed_checks}/{len(checks)} checks passed")
        
        if passed_checks >= 3:  # Minimum viable setup
            print("✅ Environment ready for development!")
        else:
            print("❌ Environment needs configuration before use.")
        
        return {
            "passed_checks": passed_checks,
            "total_checks": len(checks),
            "ready": passed_checks >= 3,
            "issues": self.issues,
            "results": self.results
        }

# Run the health check
if __name__ == "__main__":
    health_checker = EnvironmentHealthCheck()
    results = health_checker.run_full_check()
```

### Troubleshooting Common Issues

**API Key Issues**
- Ensure keys are properly formatted and not expired
- Check that you have sufficient credits/quota
- Verify the key has necessary permissions

**Import Errors**
- Use virtual environments to avoid conflicts
- Ensure you're using compatible Python version (3.10+)
- Install packages in correct order (langchain first, then providers)

**Network Issues**
- Check firewall settings for API access
- Verify proxy configuration if behind corporate firewall
- Test basic internet connectivity

---

## Lesson 1.3: Building Your First Production Agent (60 minutes)

### Learning Objectives
- Build a complete weather forecasting agent with all production features
- Implement tools with runtime context access
- Configure structured output and conversation memory
- Add error handling and user personalization
- Test and debug agent functionality

### Content

#### Production Agent Architecture
A production-ready agent includes:
1. **Robust tool definitions** with proper error handling
2. **Runtime context** for user-specific behavior
3. **Structured output** for consistent responses
4. **Memory persistence** for conversation continuity
5. **Comprehensive testing** and debugging

#### Step 1: Define Runtime Context Schema
```python
from dataclasses import dataclass
from typing import Optional, List
from datetime import datetime

@dataclass
class WeatherContext:
    """Runtime context for weather agent."""
    user_id: str
    location_preference: str = "metric"  # metric or imperial
    default_location: Optional[str] = None
    subscription_tier: str = "free"  # free, premium, enterprise
    timezone: str = "UTC"
    language: str = "english"
    
    # Usage tracking
    requests_today: int = 0
    max_daily_requests: int = 100
    
    def __post_init__(self):
        """Validate context after initialization."""
        if self.location_preference not in ["metric", "imperial"]:
            self.location_preference = "metric"
        
        if self.subscription_tier not in ["free", "premium", "enterprise"]:
            self.subscription_tier = "free"
```

#### Step 2: Create Advanced Tools with Error Handling
```python
from langchain.tools import tool, ToolRuntime
import random
import json
from typing import Dict, Any

class WeatherAPIError(Exception):
    """Custom exception for weather API errors."""
    pass

class RateLimitError(Exception):
    """Exception for rate limiting."""
    pass

@tool
def get_current_weather(
    city: str,
    runtime: ToolRuntime[WeatherContext]
) -> str:
    """Get current weather for a city with comprehensive error handling."""
    
    try:
        # Check rate limits
        context = runtime.context
        if context.requests_today >= context.max_daily_requests:
            raise RateLimitError(f"Daily limit of {context.max_daily_requests} requests exceeded")
        
        # Simulate API call with potential failures
        if random.random() < 0.1:  # 10% chance of API error
            raise WeatherAPIError("Weather service temporarily unavailable")
        
        # Mock weather data with user preferences
        weather_data = {
            "london": {"temp_c": 15, "temp_f": 59, "condition": "cloudy", "humidity": 65},
            "paris": {"temp_c": 18, "temp_f": 64, "condition": "sunny", "humidity": 45}, 
            "tokyo": {"temp_c": 22, "temp_f": 72, "condition": "rainy", "humidity": 80},
            "new york": {"temp_c": 20, "temp_f": 68, "condition": "partly cloudy", "humidity": 55},
            "san francisco": {"temp_c": 16, "temp_f": 61, "condition": "foggy", "humidity": 70}
        }
        
        city_lower = city.lower()
        data = weather_data.get(city_lower, {
            "temp_c": 20, "temp_f": 68, "condition": "partly cloudy", "humidity": 50
        })
        
        # Format response based on user preferences
        if context.location_preference == "imperial":
            temp = f"{data['temp_f']}°F"
        else:
            temp = f"{data['temp_c']}°C"
        
        # Add subscription tier specific features
        basic_info = f"Weather in {city}: {temp}, {data['condition']}"
        
        if context.subscription_tier in ["premium", "enterprise"]:
            basic_info += f", humidity: {data['humidity']}%"
        
        if context.subscription_tier == "enterprise":
            basic_info += f", feels like: {temp}, UV index: moderate"
        
        return basic_info
        
    except RateLimitError as e:
        return f"❌ Rate Limit: {e}"
    except WeatherAPIError as e:
        return f"❌ Weather Service Error: {e}"
    except Exception as e:
        return f"❌ Unexpected Error: {e}"

@tool
def get_user_location(runtime: ToolRuntime[WeatherContext]) -> str:
    """Get user's default location from context."""
    
    context = runtime.context
    
    if context.default_location:
        return f"Your default location is set to: {context.default_location}"
    
    # Mock location based on user_id for demo
    user_locations = {
        "user_1": "San Francisco",
        "user_2": "London", 
        "user_3": "Tokyo",
        "user_4": "New York",
        "user_5": "Paris"
    }
    
    location = user_locations.get(context.user_id, "Unknown")
    
    if location == "Unknown":
        return "No default location set. Please specify a city or set your default location."
    
    return f"Based on your profile, your location is: {location}"

@tool  
def get_weather_forecast(
    city: str,
    days: int = 3,
    runtime: ToolRuntime[WeatherContext] = None
) -> str:
    """Get weather forecast with subscription tier restrictions."""
    
    context = runtime.context
    
    # Check subscription limits
    if context.subscription_tier == "free" and days > 3:
        return "❌ Free tier limited to 3-day forecasts. Upgrade for extended forecasts."
    elif context.subscription_tier == "premium" and days > 7:
        return "❌ Premium tier limited to 7-day forecasts. Enterprise tier offers 14-day forecasts."
    elif days > 14:
        days = 14  # Hard limit
    
    try:
        # Mock forecast data
        forecast_days = []
        conditions = ["sunny", "cloudy", "rainy", "partly cloudy"]
        
        for i in range(min(days, 14)):
            day_temp = random.randint(15, 25) if context.location_preference == "metric" else random.randint(59, 77)
            unit = "°C" if context.location_preference == "metric" else "°F"
            condition = random.choice(conditions)
            
            forecast_days.append(f"Day {i+1}: {day_temp}{unit}, {condition}")
        
        forecast_text = f"{days}-day forecast for {city}:\n" + "\n".join(forecast_days)
        
        return forecast_text
        
    except Exception as e:
        return f"❌ Forecast Error: {e}"

@tool
def set_weather_preferences(
    preference_type: str,
    preference_value: str,
    runtime: ToolRuntime[WeatherContext]
) -> str:
    """Set user weather preferences (simulated - would update database in production)."""
    
    valid_preferences = {
        "units": ["metric", "imperial"],
        "language": ["english", "spanish", "french", "german"],
        "default_location": None  # Any string allowed
    }
    
    if preference_type not in valid_preferences:
        return f"❌ Invalid preference type. Valid options: {list(valid_preferences.keys())}"
    
    if preference_type != "default_location":
        if preference_value not in valid_preferences[preference_type]:
            return f"❌ Invalid value for {preference_type}. Valid options: {valid_preferences[preference_type]}"
    
    # In production, this would update the user's profile in database
    return f"✅ Updated {preference_type} to '{preference_value}' (Note: This is a demo - changes not persisted)"
```

#### Step 3: Define Structured Output Schema
```python
from dataclasses import dataclass
from typing import Optional, List

@dataclass
class WeatherResponse:
    """Structured weather response format."""
    
    location: str
    current_conditions: str
    temperature: str
    recommendation: str
    
    # Optional detailed information
    humidity: Optional[str] = None
    forecast_summary: Optional[str] = None
    alerts: Optional[List[str]] = None
    
    # Metadata
    data_source: str = "weather_api"
    subscription_tier: Optional[str] = None
    
    def __post_init__(self):
        """Validate the response after creation."""
        if not self.location:
            raise ValueError("Location is required")
        if not self.current_conditions:
            raise ValueError("Current conditions are required")
```

#### Step 4: Create the Production Agent
```python
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver

# Configure the model with appropriate parameters
model = init_chat_model(
    "gpt-4o",
    temperature=0.3,  # Slightly creative but consistent
    max_tokens=800,   # Reasonable response length
    timeout=30        # 30-second timeout
)

# Comprehensive system prompt
WEATHER_SYSTEM_PROMPT = """
You are WeatherBot, an expert weather assistant with the following capabilities:

🌤️ CORE FUNCTIONS:
- Provide current weather conditions for any city
- Offer weather forecasts (length depends on user's subscription tier)
- Give personalized recommendations based on weather conditions
- Help users set weather preferences

👤 USER PERSONALIZATION:
- Always consider the user's preferred units (metric/imperial)
- Adapt responses to their subscription tier (free/premium/enterprise)
- Use their default location when they ask about "here" or "my location"
- Provide appropriate detail level based on their tier

🎯 RESPONSE GUIDELINES:
- Be helpful and conversational, not robotic
- Always include practical recommendations (what to wear, activities, etc.)
- Mention subscription benefits when relevant (but don't be pushy)
- Handle errors gracefully with helpful alternatives

⚡ SUBSCRIPTION TIERS:
- Free: Basic weather, 3-day forecast, standard features
- Premium: Detailed weather data, 7-day forecast, humidity/UV info
- Enterprise: All features, 14-day forecast, advanced metrics

Always be accurate, helpful, and personalized to the user's needs.
"""

# Create the production weather agent
weather_agent = create_agent(
    model=model,
    tools=[
        get_current_weather,
        get_user_location,
        get_weather_forecast,
        set_weather_preferences
    ],
    system_prompt=WEATHER_SYSTEM_PROMPT,
    context_schema=WeatherContext,
    response_format=WeatherResponse,
    checkpointer=InMemorySaver()  # For conversation memory
)
```

#### Step 5: Comprehensive Testing Suite
```python
def test_weather_agent_comprehensive():
    """Comprehensive test suite for the weather agent."""
    
    print("🧪 Running Comprehensive Weather Agent Tests")
    print("=" * 60)
    
    # Test configurations for different user types
    test_configs = [
        {
            "name": "Free User - Metric",
            "context": WeatherContext(
                user_id="user_1",
                location_preference="metric",
                default_location="San Francisco",
                subscription_tier="free",
                requests_today=5,
                max_daily_requests=10
            ),
            "thread_id": "test_free_user"
        },
        {
            "name": "Premium User - Imperial", 
            "context": WeatherContext(
                user_id="user_2",
                location_preference="imperial",
                default_location="New York",
                subscription_tier="premium",
                requests_today=15,
                max_daily_requests=100
            ),
            "thread_id": "test_premium_user"
        },
        {
            "name": "Enterprise User - Metric",
            "context": WeatherContext(
                user_id="user_3", 
                location_preference="metric",
                default_location="London",
                subscription_tier="enterprise",
                requests_today=50,
                max_daily_requests=1000
            ),
            "thread_id": "test_enterprise_user"
        }
    ]
    
    # Test scenarios for each configuration
    test_scenarios = [
        {
            "description": "Basic weather query",
            "query": "What's the weather in Tokyo?",
            "expected_elements": ["Tokyo", "temperature", "condition"]
        },
        {
            "description": "User location query",
            "query": "How's the weather where I am?",
            "expected_elements": ["location", "weather"]
        },
        {
            "description": "Forecast request",
            "query": "Can I get a 5-day forecast for Paris?",
            "expected_elements": ["forecast", "Paris"]
        },
        {
            "description": "Preference setting",
            "query": "Set my preferred units to imperial",
            "expected_elements": ["preference", "imperial", "updated"]
        },
        {
            "description": "Conversational follow-up",
            "query": "What should I wear today?",
            "expected_elements": ["recommendation", "wear"]
        }
    ]
    
    # Run tests
    for config in test_configs:
        print(f"\n👤 Testing: {config['name']}")
        print("-" * 40)
        
        config_obj = {"configurable": {"thread_id": config["thread_id"]}}
        
        for scenario in test_scenarios:
            print(f"\n🧪 Scenario: {scenario['description']}")
            print(f"📝 Query: {scenario['query']}")
            
            try:
                response = weather_agent.invoke(
                    {"messages": [{"role": "user", "content": scenario["query"]}]},
                    config=config_obj,
                    context=config["context"]
                )
                
                # Extract response content
                agent_response = response["messages"][-1].content
                structured_response = response.get("structured_response")
                
                print(f"🤖 Response: {agent_response[:150]}...")
                
                # Check for expected elements
                found_elements = []
                for element in scenario["expected_elements"]:
                    if element.lower() in agent_response.lower():
                        found_elements.append(element)
                
                if structured_response:
                    print(f"📊 Structured: Location={structured_response.location}, "
                          f"Temp={structured_response.temperature}")
                
                success_rate = len(found_elements) / len(scenario["expected_elements"])
                status = "✅" if success_rate >= 0.7 else "⚠️" if success_rate >= 0.5 else "❌"
                
                print(f"{status} Test Result: {len(found_elements)}/{len(scenario['expected_elements'])} elements found")
                
            except Exception as e:
                print(f"❌ Test Error: {e}")
        
        print("\n" + "="*40)

def test_error_handling():
    """Test error handling scenarios."""
    
    print("\n🚨 Testing Error Handling")
    print("=" * 40)
    
    # Rate limit test
    rate_limited_context = WeatherContext(
        user_id="rate_test",
        requests_today=100,  # At limit
        max_daily_requests=100
    )
    
    config = {"configurable": {"thread_id": "error_test"}}
    
    try:
        response = weather_agent.invoke(
            {"messages": [{"role": "user", "content": "What's the weather in London?"}]},
            config=config,
            context=rate_limited_context
        )
        
        agent_response = response["messages"][-1].content
        
        if "rate limit" in agent_response.lower() or "limit" in agent_response.lower():
            print("✅ Rate limiting handled correctly")
        else:
            print("⚠️ Rate limiting may not be working as expected")
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")

def interactive_test_session():
    """Interactive testing session."""
    
    print("\n🎮 Interactive Test Session")
    print("=" * 40)
    print("Commands: 'quit' to exit, 'switch' to change user type")
    
    # Default test user
    current_context = WeatherContext(
        user_id="interactive_user",
        location_preference="metric",
        subscription_tier="premium",
        default_location="San Francisco"
    )
    
    config = {"configurable": {"thread_id": "interactive_session"}}
    
    while True:
        user_input = input("\n👤 You: ").strip()
        
        if user_input.lower() == 'quit':
            print("👋 Goodbye!")
            break
        
        if user_input.lower() == 'switch':
            print("Available tiers: free, premium, enterprise")
            tier = input("Select tier: ").strip()
            if tier in ["free", "premium", "enterprise"]:
                current_context.subscription_tier = tier
                print(f"✅ Switched to {tier} tier")
            continue
        
        if not user_input:
            continue
        
        try:
            response = weather_agent.invoke(
                {"messages": [{"role": "user", "content": user_input}]},
                config=config,
                context=current_context
            )
            
            agent_response = response["messages"][-1].content
            print(f"\n🤖 WeatherBot: {agent_response}")
            
            # Show structured response if available
            if "structured_response" in response:
                structured = response["structured_response"]
                print(f"\n📊 Structured Data:")
                print(f"   Location: {structured.location}")
                print(f"   Temperature: {structured.temperature}")
                print(f"   Conditions: {structured.current_conditions}")
                print(f"   Recommendation: {structured.recommendation}")
            
        except Exception as e:
            print(f"❌ Error: {e}")

# Run all tests
if __name__ == "__main__":
    # Comprehensive automated tests
    test_weather_agent_comprehensive()
    
    # Error handling tests
    test_error_handling()
    
    # Interactive session
    print("\nWould you like to start an interactive session? (y/n)")
    if input().lower().startswith('y'):
        interactive_test_session()
```

### Key Production Features Demonstrated

1. **Runtime Context Integration**: User-specific behavior and preferences
2. **Comprehensive Error Handling**: Graceful handling of API failures and rate limits
3. **Subscription Tier Logic**: Different feature sets based on user tier
4. **Structured Output**: Consistent, parseable response format
5. **Memory Persistence**: Conversation continuity across interactions
6. **Comprehensive Testing**: Automated and interactive testing strategies

### Best Practices Illustrated

- **Tool Design**: Clear, focused tools with proper error handling
- **Context Usage**: Leveraging runtime context for personalization
- **User Experience**: Helpful error messages and upgrade suggestions
- **Testing Strategy**: Multiple test types for thorough validation
- **Production Readiness**: Rate limiting, timeouts, and graceful degradation

### Next Steps

This production agent demonstrates all core LangChain concepts working together. You can now:

1. Deploy this agent to a production environment
2. Add additional tools and capabilities
3. Integrate with real weather APIs
4. Implement persistent user preferences
5. Add monitoring and analytics

---

# Module 2: Core Components Deep Dive

## Lesson 2.1: Models and Chat Integration (75 minutes)

### Learning Objectives
- Master model selection and configuration across providers
- Implement dynamic model routing based on task complexity
- Understand multimodal capabilities and token optimization
- Build cost-effective model management systems
- Handle model errors and implement fallback strategies

### Content

#### Model Selection and Configuration Strategies

**Understanding Model Capabilities**
```python
from langchain.chat_models import init_chat_model
from typing import Dict, List, Any
import json

# Model capability matrix for informed selection
MODEL_CAPABILITIES = {
    "gpt-4o": {
        "strengths": ["reasoning", "code", "multimodal", "function_calling"],
        "context_window": 128000,
        "cost_per_1k_tokens": {"input": 0.005, "output": 0.015},
        "speed": "medium",
        "multimodal": True,
        "function_calling": True
    },
    "gpt-4o-mini": {
        "strengths": ["speed", "cost_efficiency", "general_tasks"],
        "context_window": 128000,
        "cost_per_1k_tokens": {"input": 0.00015, "output": 0.0006},
        "speed": "fast",
        "multimodal": True,
        "function_calling": True
    },
    "claude-3-sonnet-20240229": {
        "strengths": ["reasoning", "writing", "analysis", "safety"],
        "context_window": 200000,
        "cost_per_1k_tokens": {"input": 0.003, "output": 0.015},
        "speed": "medium",
        "multimodal": True,
        "function_calling": True
    },
    "claude-3-haiku-20240307": {
        "strengths": ["speed", "cost_efficiency", "conciseness"],
        "context_window": 200000,
        "cost_per_1k_tokens": {"input": 0.00025, "output": 0.00125},
        "speed": "very_fast",
        "multimodal": True,
        "function_calling": True
    }
}

class ModelSelector:
    """Intelligent model selection based on task requirements."""
    
    def __init__(self):
        self.models = {}
        self.load_models()
    
    def load_models(self):
        """Initialize all available models."""
        for model_name in MODEL_CAPABILITIES.keys():
            try:
                self.models[model_name] = init_chat_model(model_name)
                print(f"✅ Loaded {model_name}")
            except Exception as e:
                print(f"⚠️ Failed to load {model_name}: {e}")
    
    def select_model_for_task(
        self,
        task_type: str,
        context_length: int = 0,
        priority: str = "balanced",  # speed, cost, quality, balanced
        multimodal_required: bool = False
    ) -> str:
        """Select the best model for a given task."""
        
        # Filter models by requirements
        suitable_models = []
        
        for model_name, capabilities in MODEL_CAPABILITIES.items():
            # Check context window requirement
            if context_length > capabilities["context_window"]:
                continue
            
            # Check multimodal requirement
            if multimodal_required and not capabilities["multimodal"]:
                continue
            
            # Check if model is loaded
            if model_name not in self.models:
                continue
            
            suitable_models.append(model_name)
        
        if not suitable_models:
            return "gpt-4o-mini"  # Fallback
        
        # Score models based on task and priority
        scored_models = []
        
        for model_name in suitable_models:
            score = self.calculate_model_score(
                model_name, task_type, priority
            )
            scored_models.append((model_name, score))
        
        # Return highest scored model
        best_model = max(scored_models, key=lambda x: x[1])[0]
        return best_model
    
    def calculate_model_score(
        self, 
        model_name: str, 
        task_type: str, 
        priority: str
    ) -> float:
        """Calculate score for model based on task and priority."""
        
        capabilities = MODEL_CAPABILITIES[model_name]
        score = 0.0
        
        # Task-specific scoring
        task_strengths = {
            "reasoning": ["reasoning", "analysis"],
            "coding": ["code", "reasoning"],
            "writing": ["writing", "general_tasks"],
            "multimodal": ["multimodal"],
            "function_calling": ["function_calling"],
            "general": ["general_tasks"]
        }
        
        if task_type in task_strengths:
            for strength in task_strengths[task_type]:
                if strength in capabilities["strengths"]:
                    score += 2.0
        
        # Priority-based adjustments
        if priority == "speed":
            speed_scores = {
                "very_fast": 3.0, "fast": 2.0, "medium": 1.0, "slow": 0.0
            }
            score += speed_scores.get(capabilities["speed"], 0.0)
        
        elif priority == "cost":
            # Lower cost = higher score (inverse relationship)
            avg_cost = (capabilities["cost_per_1k_tokens"]["input"] + 
                       capabilities["cost_per_1k_tokens"]["output"]) / 2
            score += max(0, 3.0 - (avg_cost * 1000))  # Scale to reasonable range
        
        elif priority == "quality":
            # GPT-4 variants and Claude Sonnet get quality bonus
            if "gpt-4o" in model_name or "sonnet" in model_name:
                score += 2.0
        
        else:  # balanced
            # Balanced scoring considers all factors
            score += 1.0  # Base score for suitable models
        
        return score
    
    def get_model(self, model_name: str):
        """Get loaded model instance."""
        return self.models.get(model_name)

# Initialize model selector
model_selector = ModelSelector()

def demonstrate_model_selection():
    """Demonstrate intelligent model selection."""
    
    test_scenarios = [
        {
            "task": "Write a complex analysis of market trends",
            "task_type": "reasoning", 
            "priority": "quality",
            "context_length": 5000
        },
        {
            "task": "Quick customer service response",
            "task_type": "general",
            "priority": "speed", 
            "context_length": 1000
        },
        {
            "task": "Analyze this image and write code",
            "task_type": "multimodal",
            "priority": "balanced",
            "multimodal_required": True
        },
        {
            "task": "Process large document with cost constraints",
            "task_type": "general",
            "priority": "cost",
            "context_length": 50000
        }
    ]
    
    print("🤖 Model Selection Demonstration")
    print("=" * 50)
    
    for scenario in test_scenarios:
        selected_model = model_selector.select_model_for_task(
            task_type=scenario["task_type"],
            context_length=scenario["context_length"],
            priority=scenario["priority"],
            multimodal_required=scenario.get("multimodal_required", False)
        )
        
        print(f"\n📋 Task: {scenario['task']}")
        print(f"🎯 Requirements: {scenario['task_type']}, {scenario['priority']} priority")
        print(f"🤖 Selected Model: {selected_model}")
        
        # Show model capabilities
        caps = MODEL_CAPABILITIES[selected_model]
        print(f"💪 Strengths: {', '.join(caps['strengths'])}")
        print(f"💰 Cost: ${caps['cost_per_1k_tokens']['input']:.4f}/${caps['cost_per_1k_tokens']['output']:.4f} per 1K tokens")

demonstrate_model_selection()
```

#### Dynamic Model Routing Implementation

**Context-Aware Model Router**
```python
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from typing import Callable
import time

class IntelligentModelRouter:
    """Advanced model routing with learning and adaptation."""
    
    def __init__(self, model_selector: ModelSelector):
        self.model_selector = model_selector
        self.performance_history = {}
        self.cost_tracking = {}
        self.error_counts = {}
    
    def route_request(
        self, 
        request: ModelRequest, 
        handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        """Route request to optimal model with performance tracking."""
        
        # Analyze request to determine optimal model
        optimal_model = self.analyze_and_select_model(request)
        
        # Update request with selected model
        original_model = request.model
        request.model = self.model_selector.get_model(optimal_model)
        
        start_time = time.time()
        
        try:
            # Execute with selected model
            response = handler(request)
            
            # Track successful performance
            execution_time = time.time() - start_time
            self.record_performance(optimal_model, execution_time, True, request, response)
            
            print(f"🎯 Routed to {optimal_model} (took {execution_time:.2f}s)")
            
            return response
            
        except Exception as e:
            # Track errors and potentially retry with fallback
            execution_time = time.time() - start_time
            self.record_performance(optimal_model, execution_time, False, request, None)
            
            # Try fallback model
            fallback_model = self.get_fallback_model(optimal_model)
            if fallback_model and fallback_model != optimal_model:
                print(f"⚠️ {optimal_model} failed, trying fallback: {fallback_model}")
                
                request.model = self.model_selector.get_model(fallback_model)
                try:
                    return handler(request)
                except Exception as fallback_error:
                    print(f"❌ Fallback also failed: {fallback_error}")
            
            # If all fails, restore original and re-raise
            request.model = original_model
            raise e
    
    def analyze_and_select_model(self, request: ModelRequest) -> str:
        """Analyze request and select optimal model."""
        
        # Extract request characteristics
        message_count = len(request.messages)
        total_chars = sum(len(str(msg.content)) for msg in request.messages)
        
        # Estimate context length (rough approximation)
        estimated_tokens = total_chars // 4
        
        # Analyze content for task type detection
        last_message = str(request.messages[-1].content).lower() if request.messages else ""
        
        task_type = "general"
        priority = "balanced"
        multimodal_required = False
        
        # Task type detection
        if any(word in last_message for word in ["code", "program", "debug", "implement"]):
            task_type = "coding"
            priority = "quality"
        elif any(word in last_message for word in ["analyze", "complex", "detailed", "research"]):
            task_type = "reasoning"
            priority = "quality"
        elif any(word in last_message for word in ["quick", "brief", "short", "simple"]):
            priority = "speed"
        elif any(word in last_message for word in ["image", "picture", "photo", "visual"]):
            multimodal_required = True
            task_type = "multimodal"
        
        # Check for cost-sensitive scenarios (long conversations)
        if message_count > 20 or estimated_tokens > 10000:
            if priority != "quality":  # Don't override quality requirements
                priority = "cost"
        
        # Select model based on analysis
        selected_model = self.model_selector.select_model_for_task(
            task_type=task_type,
            context_length=estimated_tokens,
            priority=priority,
            multimodal_required=multimodal_required
        )
        
        return selected_model
    
    def record_performance(
        self, 
        model_name: str, 
        execution_time: float, 
        success: bool,
        request: ModelRequest,
        response: ModelResponse
    ):
        """Record model performance for future optimization."""
        
        if model_name not in self.performance_history:
            self.performance_history[model_name] = {
                "total_calls": 0,
                "successful_calls": 0,
                "total_time": 0.0,
                "avg_time": 0.0,
                "success_rate": 0.0
            }
        
        stats = self.performance_history[model_name]
        stats["total_calls"] += 1
        stats["total_time"] += execution_time
        
        if success:
            stats["successful_calls"] += 1
            
            # Track token usage and cost if available
            if response and hasattr(response, 'usage_metadata') and response.usage_metadata:
                self.track_cost(model_name, response.usage_metadata)
        
        # Update derived metrics
        stats["avg_time"] = stats["total_time"] / stats["total_calls"]
        stats["success_rate"] = stats["successful_calls"] / stats["total_calls"]
    
    def track_cost(self, model_name: str, usage_metadata: dict):
        """Track cost per model."""
        if model_name not in self.cost_tracking:
            self.cost_tracking[model_name] = {
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_cost": 0.0
            }
        
        cost_data = self.cost_tracking[model_name]
        model_costs = MODEL_CAPABILITIES[model_name]["cost_per_1k_tokens"]
        
        input_tokens = usage_metadata.get("input_tokens", 0)
        output_tokens = usage_metadata.get("output_tokens", 0)
        
        cost_data["total_input_tokens"] += input_tokens
        cost_data["total_output_tokens"] += output_tokens
        
        # Calculate cost
        input_cost = (input_tokens / 1000) * model_costs["input"]
        output_cost = (output_tokens / 1000) * model_costs["output"]
        cost_data["total_cost"] += input_cost + output_cost
    
    def get_fallback_model(self, failed_model: str) -> str:
        """Get fallback model when primary fails."""
        
        fallback_chain = {
            "gpt-4o": "gpt-4o-mini",
            "claude-3-sonnet-20240229": "claude-3-haiku-20240307", 
            "gpt-4o-mini": "claude-3-haiku-20240307",
            "claude-3-haiku-20240307": "gpt-4o-mini"
        }
        
        return fallback_chain.get(failed_model, "gpt-4o-mini")
    
    def get_performance_report(self) -> str:
        """Generate performance report for all models."""
        
        report = "📊 Model Performance Report\n"
        report += "=" * 50 + "\n\n"
        
        for model_name, stats in self.performance_history.items():
            report += f"🤖 {model_name}:\n"
            report += f"   Total Calls: {stats['total_calls']}\n"
            report += f"   Success Rate: {stats['success_rate']:.1%}\n"
            report += f"   Avg Response Time: {stats['avg_time']:.2f}s\n"
            
            if model_name in self.cost_tracking:
                cost_stats = self.cost_tracking[model_name]
                report += f"   Total Cost: ${cost_stats['total_cost']:.4f}\n"
                report += f"   Total Tokens: {cost_stats['total_input_tokens'] + cost_stats['total_output_tokens']:,}\n"
            
            report += "\n"
        
        return report

# Create router with model selector
router = IntelligentModelRouter(model_selector)

@wrap_model_call
def intelligent_model_routing(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """Middleware for intelligent model routing."""
    return router.route_request(request, handler)
```

#### Multimodal Model Integration

**Advanced Multimodal Processing**
```python
import base64
from pathlib import Path
from typing import Union, List, Dict, Any
import mimetypes

class MultimodalProcessor:
    """Handle multimodal inputs with various data types."""
    
    def __init__(self):
        self.supported_image_types = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
        self.supported_document_types = {".pdf", ".docx", ".txt", ".md"}
        self.supported_audio_types = {".mp3", ".wav", ".m4a", ".ogg"}
        self.supported_video_types = {".mp4", ".avi", ".mov", ".webm"}
    
    def process_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Process a file and return appropriate content block."""
        
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Determine file type
        mime_type, _ = mimetypes.guess_type(str(file_path))
        file_extension = file_path.suffix.lower()
        
        # Read and encode file
        with open(file_path, "rb") as f:
            file_content = f.read()
        
        base64_content = base64.b64encode(file_content).decode('utf-8')
        
        # Create appropriate content block
        if file_extension in self.supported_image_types:
            return {
                "type": "image",
                "base64": base64_content,
                "mime_type": mime_type or f"image/{file_extension[1:]}"
            }
        
        elif file_extension in self.supported_audio_types:
            return {
                "type": "audio",
                "base64": base64_content,
                "mime_type": mime_type or f"audio/{file_extension[1:]}"
            }
        
        elif file_extension in self.supported_video_types:
            return {
                "type": "video", 
                "base64": base64_content,
                "mime_type": mime_type or f"video/{file_extension[1:]}"
            }
        
        elif file_extension in self.supported_document_types:
            return {
                "type": "file",
                "base64": base64_content,
                "mime_type": mime_type or "application/octet-stream",
                "filename": file_path.name
            }
        
        else:
            # Generic file handling
            return {
                "type": "file",
                "base64": base64_content,
                "mime_type": mime_type or "application/octet-stream",
                "filename": file_path.name
            }
    
    def create_multimodal_message(
        self, 
        text: str, 
        files: List[Union[str, Path]] = None,
        urls: List[str] = None
    ) -> Dict[str, Any]:
        """Create a multimodal message with text and various media."""
        
        content_blocks = []
        
        # Add text content
        if text:
            content_blocks.append({
                "type": "text",
                "text": text
            })
        
        # Add file content
        if files:
            for file_path in files:
                try:
                    file_block = self.process_file(file_path)
                    content_blocks.append(file_block)
                except Exception as e:
                    print(f"⚠️ Failed to process file {file_path}: {e}")
        
        # Add URL content (for images)
        if urls:
            for url in urls:
                content_blocks.append({
                    "type": "image",
                    "url": url
                })
        
        return {
            "role": "user",
            "content": content_blocks
        }

# Multimodal tools
from langchain.tools import tool

multimodal_processor = MultimodalProcessor()

@tool
def analyze_image(
    image_path: str,
    analysis_type: str = "general",
    runtime=None
) -> str:
    """Analyze an image with specified analysis type."""
    
    try:
        # Process the image file
        image_block = multimodal_processor.process_file(image_path)
        
        # Create analysis prompt based on type
        analysis_prompts = {
            "general": "Describe what you see in this image in detail.",
            "technical": "Provide a technical analysis of this image including composition, lighting, and visual elements.",
            "business": "Analyze this image from a business perspective. What insights can you provide?",
            "accessibility": "Describe this image for accessibility purposes, focusing on important visual information.",
            "ocr": "Extract and transcribe any text visible in this image."
        }
        
        prompt = analysis_prompts.get(analysis_type, analysis_prompts["general"])
        
        # Create multimodal message
        message = {
            "role": "user", 
            "content": [
                {"type": "text", "text": prompt},
                image_block
            ]
        }
        
        # Use a multimodal-capable model
        model = model_selector.get_model("gpt-4o")  # Ensure multimodal capability
        response = model.invoke([message])
        
        return response.content
        
    except Exception as e:
        return f"❌ Image analysis failed: {e}"

@tool
def compare_images(
    image1_path: str,
    image2_path: str,
    comparison_focus: str = "similarities_and_differences"
) -> str:
    """Compare two images and analyze their relationship."""
    
    try:
        # Process both images
        image1_block = multimodal_processor.process_file(image1_path)
        image2_block = multimodal_processor.process_file(image2_path)
        
        # Create comparison prompt
        comparison_prompts = {
            "similarities_and_differences": "Compare these two images. What are the key similarities and differences?",
            "quality": "Compare the quality and technical aspects of these two images.",
            "style": "Compare the artistic style and visual approach of these two images.",
            "content": "Compare the content and subject matter of these two images."
        }
        
        prompt = comparison_prompts.get(comparison_focus, comparison_prompts["similarities_and_differences"])
        
        # Create multimodal message with both images
        message = {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "text", "text": "Image 1:"},
                image1_block,
                {"type": "text", "text": "Image 2:"},
                image2_block
            ]
        }
        
        model = model_selector.get_model("gpt-4o")
        response = model.invoke([message])
        
        return response.content
        
    except Exception as e:
        return f"❌ Image comparison failed: {e}"

def demonstrate_multimodal_capabilities():
    """Demonstrate multimodal processing capabilities."""
    
    print("🖼️ Multimodal Processing Demonstration")
    print("=" * 50)
    
    # Example multimodal scenarios (would use real files in practice)
    scenarios = [
        {
            "name": "Single Image Analysis",
            "description": "Analyze a product photo for e-commerce",
            # In real usage: "files": ["product_photo.jpg"]
            "simulated": True
        },
        {
            "name": "Document Processing",
            "description": "Extract information from a PDF document", 
            # In real usage: "files": ["contract.pdf"]
            "simulated": True
        },
        {
            "name": "Multi-Image Comparison", 
            "description": "Compare before and after photos",
            # In real usage: "files": ["before.jpg", "after.jpg"]
            "simulated": True
        }
    ]
    
    for scenario in scenarios:
        print(f"\n📋 Scenario: {scenario['name']}")
        print(f"📝 Description: {scenario['description']}")
        
        if scenario["simulated"]:
            print("🎭 (Simulated - would process actual files in practice)")
            
            # Simulate multimodal message structure
            simulated_message = {
                "role": "user",
                "content": [
                    {"type": "text", "text": scenario["description"]},
                    {"type": "image", "base64": "simulated_image_data", "mime_type": "image/jpeg"}
                ]
            }
            
            print(f"📄 Message Structure: {len(simulated_message['content'])} content blocks")
            for i, block in enumerate(simulated_message["content"]):
                print(f"   Block {i+1}: {block['type']}")

demonstrate_multimodal_capabilities()
```

#### Token Management and Cost Optimization

**Advanced Cost Management System**
```python
from langchain_core.callbacks import UsageMetadataCallbackHandler
import json
from datetime import datetime, timedelta
from typing import Optional

class AdvancedCostTracker:
    """Comprehensive cost tracking and optimization system."""
    
    def __init__(self):
        self.callback = UsageMetadataCallbackHandler()
        self.daily_costs = {}
        self.model_costs = {}
        self.cost_alerts = []
        
        # Cost thresholds
        self.daily_budget = 50.0  # $50 daily budget
        self.warning_threshold = 0.8  # 80% of budget
        self.critical_threshold = 0.95  # 95% of budget
        
        # Model pricing (update with current rates)
        self.pricing = MODEL_CAPABILITIES
    
    def track_request(
        self, 
        model_name: str, 
        messages: List[Any],
        response: Any = None
    ) -> Dict[str, Any]:
        """Track a model request with comprehensive cost analysis."""
        
        today = datetime.now().date().isoformat()
        
        if today not in self.daily_costs:
            self.daily_costs[today] = {
                "total_cost": 0.0,
                "requests": 0,
                "tokens_used": 0,
                "models_used": {}
            }
        
        # Calculate cost
        cost_info = self.calculate_request_cost(model_name, messages, response)
        
        # Update daily tracking
        daily_data = self.daily_costs[today]
        daily_data["total_cost"] += cost_info["total_cost"]
        daily_data["requests"] += 1
        daily_data["tokens_used"] += cost_info["total_tokens"]
        
        if model_name not in daily_data["models_used"]:
            daily_data["models_used"][model_name] = {
                "requests": 0, "cost": 0.0, "tokens": 0
            }
        
        model_data = daily_data["models_used"][model_name]
        model_data["requests"] += 1
        model_data["cost"] += cost_info["total_cost"]
        model_data["tokens"] += cost_info["total_tokens"]
        
        # Check budget alerts
        self.check_budget_alerts(today)
        
        return cost_info
    
    def calculate_request_cost(
        self,
        model_name: str,
        messages: List[Any], 
        response: Any = None
    ) -> Dict[str, Any]:
        """Calculate cost for a specific request."""
        
        # Estimate input tokens (rough approximation)
        input_text = ""
        for msg in messages:
            if hasattr(msg, 'content'):
                input_text += str(msg.content)
            elif isinstance(msg, dict):
                if isinstance(msg.get('content'), str):
                    input_text += msg['content']
                elif isinstance(msg.get('content'), list):
                    for block in msg['content']:
                        if isinstance(block, dict) and block.get('type') == 'text':
                            input_text += block.get('text', '')
        
        estimated_input_tokens = len(input_text) // 4  # Rough estimation
        
        # Get output tokens from response if available
        output_tokens = 0
        if response and hasattr(response, 'usage_metadata') and response.usage_metadata:
            output_tokens = response.usage_metadata.get("output_tokens", 0)
            estimated_input_tokens = response.usage_metadata.get("input_tokens", estimated_input_tokens)
        else:
            # Estimate output tokens
            if response and hasattr(response, 'content'):
                output_tokens = len(str(response.content)) // 4
        
        # Calculate costs
        model_pricing = self.pricing.get(model_name, {}).get("cost_per_1k_tokens", {"input": 0.001, "output": 0.001})
        
        input_cost = (estimated_input_tokens / 1000) * model_pricing["input"]
        output_cost = (output_tokens / 1000) * model_pricing["output"]
        total_cost = input_cost + output_cost
        
        return {
            "model_name": model_name,
            "input_tokens": estimated_input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": estimated_input_tokens + output_tokens,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost,
            "timestamp": datetime.now().isoformat()
        }
    
    def check_budget_alerts(self, date: str):
        """Check if budget thresholds have been exceeded."""
        
        daily_cost = self.daily_costs[date]["total_cost"]
        usage_percentage = daily_cost / self.daily_budget
        
        alert_message = None
        
        if usage_percentage >= self.critical_threshold:
            alert_message = f"🚨 CRITICAL: {usage_percentage:.1%} of daily budget used (${daily_cost:.2f}/${self.daily_budget})"
            alert_level = "critical"
        elif usage_percentage >= self.warning_threshold:
            alert_message = f"⚠️ WARNING: {usage_percentage:.1%} of daily budget used (${daily_cost:.2f}/${self.daily_budget})"
            alert_level = "warning"
        
        if alert_message:
            alert = {
                "date": date,
                "message": alert_message,
                "level": alert_level,
                "usage_percentage": usage_percentage,
                "cost": daily_cost,
                "timestamp": datetime.now().isoformat()
            }
            
            self.cost_alerts.append(alert)
            print(alert_message)
    
    def get_cost_optimization_suggestions(self) -> List[str]:
        """Provide cost optimization suggestions based on usage patterns."""
        
        suggestions = []
        
        # Analyze recent usage
        recent_dates = sorted(self.daily_costs.keys())[-7:]  # Last 7 days
        
        if not recent_dates:
            return ["No usage data available for analysis"]
        
        # Calculate average daily cost
        total_cost = sum(self.daily_costs[date]["total_cost"] for date in recent_dates)
        avg_daily_cost = total_cost / len(recent_dates)
        
        if avg_daily_cost > self.daily_budget * 0.8:
            suggestions.append("Consider using more cost-effective models like gpt-4o-mini for simple tasks")
        
        # Analyze model usage patterns
        model_costs = {}
        for date in recent_dates:
            for model, data in self.daily_costs[date]["models_used"].items():
                if model not in model_costs:
                    model_costs[model] = {"cost": 0.0, "requests": 0}
                model_costs[model]["cost"] += data["cost"]
                model_costs[model]["requests"] += data["requests"]
        
        # Find most expensive model
        if model_costs:
            most_expensive = max(model_costs.items(), key=lambda x: x[1]["cost"])
            most_used = max(model_costs.items(), key=lambda x: x[1]["requests"])
            
            if most_expensive[1]["cost"] > total_cost * 0.5:
                suggestions.append(f"Model '{most_expensive[0]}' accounts for 50%+ of costs. Consider alternatives for routine tasks.")
            
            if most_used[0] in ["gpt-4o", "claude-3-sonnet-20240229"]:
                suggestions.append("Consider using faster, cheaper models (gpt-4o-mini, claude-haiku) for simple queries")
        
        # Token usage analysis
        total_tokens = sum(self.daily_costs[date]["tokens_used"] for date in recent_dates)
        avg_tokens_per_request = total_tokens / sum(self.daily_costs[date]["requests"] for date in recent_dates)
        
        if avg_tokens_per_request > 2000:
            suggestions.append("High token usage detected. Consider implementing conversation summarization or context trimming")
        
        if not suggestions:
            suggestions.append("✅ Your usage patterns look efficient!")
        
        return suggestions
    
    def generate_cost_report(self) -> str:
        """Generate comprehensive cost report."""
        
        report = "💰 Cost Analysis Report\n"
        report += "=" * 50 + "\n\n"
        
        # Recent usage summary
        recent_dates = sorted(self.daily_costs.keys())[-7:]
        
        if recent_dates:
            total_cost = sum(self.daily_costs[date]["total_cost"] for date in recent_dates)
            total_requests = sum(self.daily_costs[date]["requests"] for date in recent_dates)
            total_tokens = sum(self.daily_costs[date]["tokens_used"] for date in recent_dates)
            
            report += f"📊 Last 7 Days Summary:\n"
            report += f"   Total Cost: ${total_cost:.4f}\n"
            report += f"   Total Requests: {total_requests:,}\n" 
            report += f"   Total Tokens: {total_tokens:,}\n"
            report += f"   Avg Cost/Request: ${total_cost/total_requests:.4f}\n"
            report += f"   Avg Tokens/Request: {total_tokens/total_requests:.0f}\n\n"
            
            # Daily breakdown
            report += "📅 Daily Breakdown:\n"
            for date in recent_dates[-3:]:  # Last 3 days
                data = self.daily_costs[date]
                report += f"   {date}: ${data['total_cost']:.4f} ({data['requests']} requests)\n"
            
            report += "\n"
        
        # Model usage breakdown
        model_totals = {}
        for date_data in self.daily_costs.values():
            for model, data in date_data["models_used"].items():
                if model not in model_totals:
                    model_totals[model] = {"cost": 0.0, "requests": 0, "tokens": 0}
                model_totals[model]["cost"] += data["cost"]
                model_totals[model]["requests"] += data["requests"]  
                model_totals[model]["tokens"] += data["tokens"]
        
        if model_totals:
            report += "🤖 Model Usage:\n"
            for model, data in sorted(model_totals.items(), key=lambda x: x[1]["cost"], reverse=True):
                report += f"   {model}:\n"
                report += f"      Cost: ${data['cost']:.4f}\n"
                report += f"      Requests: {data['requests']:,}\n"
                report += f"      Tokens: {data['tokens']:,}\n"
            report += "\n"
        
        # Cost optimization suggestions
        suggestions = self.get_cost_optimization_suggestions()
        report += "💡 Optimization Suggestions:\n"
        for i, suggestion in enumerate(suggestions, 1):
            report += f"   {i}. {suggestion}\n"
        
        # Recent alerts
        recent_alerts = [alert for alert in self.cost_alerts if 
                        datetime.fromisoformat(alert["timestamp"]) > datetime.now() - timedelta(days=7)]
        
        if recent_alerts:
            report += "\n🚨 Recent Alerts:\n"
            for alert in recent_alerts[-3:]:  # Last 3 alerts
                report += f"   {alert['date']}: {alert['message']}\n"
        
        return report

# Initialize cost tracker
cost_tracker = AdvancedCostTracker()

# Middleware for cost tracking
@wrap_model_call
def cost_tracking_middleware(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """Middleware to track costs of model calls."""
    
    # Get model name
    model_name = getattr(request.model, 'model_name', 'unknown')
    
    # Execute request
    response = handler(request)
    
    # Track cost
    cost_info = cost_tracker.track_request(model_name, request.messages, response)
    
    # Add cost info to response metadata (optional)
    if hasattr(response, 'response_metadata'):
        response.response_metadata["cost_info"] = cost_info
    
    return response

def demonstrate_cost_optimization():
    """Demonstrate cost tracking and optimization."""
    
    print("💰 Cost Optimization Demonstration")
    print("=" * 50)
    
    # Simulate some model usage
    test_scenarios = [
        ("gpt-4o", "Write a detailed analysis", "This is a complex analytical task..."),
        ("gpt-4o-mini", "Quick summary", "Brief summary response"),
        ("claude-3-sonnet-20240229", "Creative writing", "Long creative writing response..."),
        ("gpt-4o-mini", "Simple question", "Short answer")
    ]
    
    print("🧪 Simulating model usage...")
    for model_name, query, response_text in test_scenarios:
        # Simulate request
        messages = [{"role": "user", "content": query}]
        
        # Create mock response
        class MockResponse:
            def __init__(self, content):
                self.content = content
                self.usage_metadata = {
                    "input_tokens": len(query) // 4,
                    "output_tokens": len(content) // 4,
                    "total_tokens": (len(query) + len(content)) // 4
                }
        
        response = MockResponse(response_text)
        
        # Track cost
        cost_info = cost_tracker.track_request(model_name, messages, response)
        print(f"📊 {model_name}: ${cost_info['total_cost']:.4f} ({cost_info['total_tokens']} tokens)")
    
    # Generate report
    print(f"\n{cost_tracker.generate_cost_report()}")

demonstrate_cost_optimization()
```

### Practical Exercise: Build an Adaptive Model Management System

Create a comprehensive system that intelligently manages multiple models:

```python
# Exercise: Adaptive Model Management Agent

from langchain.agents import create_agent
from langchain.tools import tool
from typing import Dict, Any, List

class AdaptiveModelManager:
    """Complete model management system with learning capabilities."""
    
    def __init__(self):
        self.model_selector = ModelSelector()
        self.router = IntelligentModelRouter(self.model_selector)
        self.cost_tracker = AdvancedCostTracker()
        self.performance_baseline = {}
    
    def create_adaptive_agent(self, tools: List, system_prompt: str):
        """Create agent with adaptive model management."""
        
        @wrap_model_call
        def adaptive_routing(request, handler):
            """Combined routing, cost tracking, and performance monitoring."""
            
            # Route to optimal model
            response = self.router.route_request(request, handler)
            
            # Track costs
            model_name = getattr(request.model, 'model_name', 'unknown')
            self.cost_tracker.track_request(model_name, request.messages, response)
            
            return response
        
        return create_agent(
            model=self.model_selector.get_model("gpt-4o-mini"),  # Default
            tools=tools,
            system_prompt=system_prompt,
            middleware=[adaptive_routing]
        )

# Tools for model management
@tool
def get_model_performance_report(runtime=None) -> str:
    """Get comprehensive model performance report."""
    return manager.router.get_performance_report()

@tool  
def get_cost_analysis(runtime=None) -> str:
    """Get cost analysis and optimization suggestions."""
    return manager.cost_tracker.generate_cost_report()

@tool
def optimize_model_selection(
    task_description: str,
    priority: str = "balanced",
    runtime=None
) -> str:
    """Get model recommendation for a specific task."""
    
    # Analyze task
    task_type = "general"
    if "code" in task_description.lower():
        task_type = "coding"
    elif "analyze" in task_description.lower():
        task_type = "reasoning"
    elif "image" in task_description.lower():
        task_type = "multimodal"
    
    recommended_model = manager.model_selector.select_model_for_task(
        task_type=task_type,
        priority=priority
    )
    
    model_info = MODEL_CAPABILITIES[recommended_model]
    
    return f"""
🎯 Recommended Model: {recommended_model}

💪 Strengths: {', '.join(model_info['strengths'])}
💰 Cost: ${model_info['cost_per_1k_tokens']['input']:.4f}/${model_info['cost_per_1k_tokens']['output']:.4f} per 1K tokens
⚡ Speed: {model_info['speed']}
🧠 Context Window: {model_info['context_window']:,} tokens

This model is optimal for your task type ({task_type}) with {priority} priority.
"""

# Create the adaptive manager
manager = AdaptiveModelManager()

# Create agent with adaptive capabilities
adaptive_agent = manager.create_adaptive_agent(
    tools=[
        get_model_performance_report,
        get_cost_analysis,
        optimize_model_selection,
        analyze_image,  # From multimodal section
        compare_images
    ],
    system_prompt="""
    You are an AI assistant with advanced model management capabilities.
    You can analyze performance, optimize costs, and recommend models for different tasks.
    
    Key capabilities:
    - Intelligent model routing based on task complexity
    - Real-time cost tracking and optimization
    - Performance monitoring and reporting
    - Multimodal processing for images and documents
    
    Always consider cost-effectiveness while maintaining quality.
    Provide detailed explanations of your model selection rationale.
    """
)

def test_adaptive_system():
    """Test the complete adaptive model management system."""
    
    print("🚀 Testing Adaptive Model Management System")
    print("=" * 60)
    
    test_queries = [
        "Analyze the performance of different models I've been using",
        "What's my current cost situation and how can I optimize it?", 
        "I need to write complex code for a machine learning algorithm. What model should I use?",
        "Recommend a model for analyzing customer feedback data with cost efficiency in mind",
        "Show me the performance report for all models"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🧪 Test {i}: {query}")
        print("-" * 40)
        
        try:
            response = adaptive_agent.invoke({
                "messages": [{"role": "user", "content": query}]
            })
            
            agent_response = response["messages"][-1].content
            print(f"🤖 Response: {agent_response[:200]}...")
            
            # Show routing information if available
            print("📊 System performed automatic model selection and cost tracking")
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
    
    print(f"\n📈 Final Performance Summary:")
    print(manager.router.get_performance_report())

if __name__ == "__main__":
    test_adaptive_system()
```

### Key Concepts Mastered

1. **Intelligent Model Selection**: Automated model choice based on task requirements
2. **Dynamic Routing**: Real-time model switching with fallback strategies  
3. **Cost Optimization**: Comprehensive cost tracking and budget management
4. **Multimodal Processing**: Advanced handling of images, documents, and media
5. **Performance Monitoring**: Detailed analytics and optimization recommendations
6. **Production Readiness**: Error handling, fallbacks, and monitoring

This module provides the foundation for building sophisticated, cost-effective, and reliable model management systems in production environments.

---

## Lesson 2.2: Messages and Communication Patterns (60 minutes)

### Learning Objectives
- Master different message types and their strategic use cases  
- Implement advanced multimodal message handling
- Build sophisticated conversation management systems
- Create custom message types for domain-specific applications
- Optimize message flow for performance and user experience

### Content

#### Advanced Message Architecture

**Message Type Strategy and Use Cases**
```python
from langchain.messages import (
    SystemMessage, HumanMessage, AIMessage, ToolMessage, BaseMessage
)
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import json
import uuid

class MessageArchitect:
    """Strategic message construction and management."""
    
    def __init__(self):
        self.message_patterns = {
            "instruction": self.create_instruction_pattern,
            "conversation": self.create_conversation_pattern,
            "analysis": self.create_analysis_pattern,
            "workflow": self.create_workflow_pattern,
            "multimodal": self.create_multimodal_pattern
        }
    
    def create_instruction_pattern(
        self, 
        task: str,
        context: Dict[str, Any] = None,
        constraints: List[str] = None,
        examples: List[Dict[str, str]] = None
    ) -> List[BaseMessage]:
        """Create messages for instruction-following tasks."""
        
        messages = []
        
        # System message with clear instructions
        system_content = f"Task: {task}\n\n"
        
        if context:
            system_content += "Context:\n"
            for key, value in context.items():
                system_content += f"- {key}: {value}\n"
            system_content += "\n"
        
        if constraints:
            system_content += "Constraints:\n"
            for constraint in constraints:
                system_content += f"- {constraint}\n"
            system_content += "\n"
        
        if examples:
            system_content += "Examples:\n"
            for i, example in enumerate(examples, 1):
                system_content += f"Example {i}:\n"
                system_content += f"Input: {example.get('input', '')}\n"
                system_content += f"Output: {example.get('output', '')}\n\n"
        
        messages.append(SystemMessage(content=system_content))
        
        return messages
    
    def create_conversation_pattern(
        self,
        persona: str,
        conversation_style: str = "helpful",
        memory_context: Dict[str, Any] = None
    ) -> List[BaseMessage]:
        """Create messages for conversational interactions."""
        
        style_prompts = {
            "helpful": "Be helpful, friendly, and informative.",
            "professional": "Maintain a professional, business-appropriate tone.",
            "casual": "Use a casual, relaxed conversational style.",
            "educational": "Focus on teaching and explaining concepts clearly.",
            "creative": "Be imaginative and creative in your responses."
        }
        
        system_content = f"You are {persona}. {style_prompts.get(conversation_style, style_prompts['helpful'])}\n\n"
        
        if memory_context:
            system_content += "Conversation Context:\n"
            for key, value in memory_context.items():
                system_content += f"- {key}: {value}\n"
            system_content += "\n"
        
        system_content += "Maintain consistency with previous interactions and build rapport with the user."
        
        return [SystemMessage(content=system_content)]
    
    def create_analysis_pattern(
        self,
        analysis_type: str,
        data_sources: List[str] = None,
        output_format: str = "detailed",
        focus_areas: List[str] = None
    ) -> List[BaseMessage]:
        """Create messages for analytical tasks."""
        
        system_content = f"Perform {analysis_type} analysis with the following specifications:\n\n"
        
        if data_sources:
            system_content += "Data Sources:\n"
            for source in data_sources:
                system_content += f"- {source}\n"
            system_content += "\n"
        
        if focus_areas:
            system_content += "Focus Areas:\n"
            for area in focus_areas:
                system_content += f"- {area}\n"
            system_content += "\n"
        
        output_formats = {
            "detailed": "Provide comprehensive analysis with supporting evidence.",
            "summary": "Focus on key insights and actionable recommendations.", 
            "structured": "Use clear headings, bullet points, and organized sections.",
            "technical": "Include technical details, methodologies, and data analysis."
        }
        
        system_content += f"Output Format: {output_formats.get(output_format, output_formats['detailed'])}\n"
        system_content += "\nEnsure your analysis is objective, evidence-based, and actionable."
        
        return [SystemMessage(content=system_content)]
    
    def create_workflow_pattern(
        self,
        steps: List[str],
        decision_points: List[Dict[str, Any]] = None,
        error_handling: str = "graceful"
    ) -> List[BaseMessage]:
        """Create messages for workflow execution."""
        
        system_content = "Execute the following workflow systematically:\n\n"
        
        system_content += "Workflow Steps:\n"
        for i, step in enumerate(steps, 1):
            system_content += f"{i}. {step}\n"
        system_content += "\n"
        
        if decision_points:
            system_content += "Decision Points:\n"
            for point in decision_points:
                system_content += f"- At step {point.get('step', 'N/A')}: {point.get('description', '')}\n"
                if 'criteria' in point:
                    system_content += f"  Criteria: {point['criteria']}\n"
            system_content += "\n"
        
        error_strategies = {
            "graceful": "Handle errors gracefully and continue with remaining steps.",
            "strict": "Stop execution if any step fails and report the issue.",
            "adaptive": "Adapt the workflow based on encountered issues."
        }
        
        system_content += f"Error Handling: {error_strategies.get(error_handling, error_strategies['graceful'])}\n"
        system_content += "\nReport progress after each step and provide a final summary."
        
        return [SystemMessage(content=system_content)]
    
    def create_multimodal_pattern(
        self,
        media_types: List[str],
        processing_instructions: Dict[str, str] = None
    ) -> List[BaseMessage]:
        """Create messages for multimodal processing."""
        
        system_content = f"Process multimodal content including: {', '.join(media_types)}\n\n"
        
        default_instructions = {
            "image": "Analyze visual content, describe key elements, and extract relevant information.",
            "audio": "Transcribe speech, identify speakers, and note audio characteristics.",
            "video": "Describe visual scenes, transcribe audio, and note temporal progression.",
            "document": "Extract text content, identify structure, and summarize key information."
        }
        
        system_content += "Processing Instructions:\n"
        for media_type in media_types:
            instruction = processing_instructions.get(media_type) if processing_instructions else default_instructions.get(media_type, "Process and describe content.")
            system_content += f"- {media_type.title()}: {instruction}\n"
        
        system_content += "\nProvide comprehensive analysis while being mindful of accessibility and content appropriateness."
        
        return [SystemMessage(content=system_content)]

# Advanced message content handling
class AdvancedMessageProcessor:
    """Process and manipulate message content with advanced features."""
    
    def __init__(self):
        self.content_analyzers = {
            "sentiment": self.analyze_sentiment,
            "complexity": self.analyze_complexity,
            "topics": self.extract_topics,
            "intent": self.detect_intent,
            "entities": self.extract_entities
        }
    
    def process_message_content(
        self,
        message: BaseMessage,
        analysis_types: List[str] = None
    ) -> Dict[str, Any]:
        """Comprehensive message content analysis."""
        
        content = self.extract_text_content(message)
        
        analysis_results = {
            "original_type": message.type,
            "content_length": len(content),
            "word_count": len(content.split()),
            "has_multimodal": self.has_multimodal_content(message)
        }
        
        # Run requested analyses
        if analysis_types:
            for analysis_type in analysis_types:
                if analysis_type in self.content_analyzers:
                    analysis_results[analysis_type] = self.content_analyzers[analysis_type](content)
        
        return analysis_results
    
    def extract_text_content(self, message: BaseMessage) -> str:
        """Extract all text content from a message."""
        
        if isinstance(message.content, str):
            return message.content
        
        elif isinstance(message.content, list):
            text_parts = []
            for block in message.content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                    elif "text" in block:
                        text_parts.append(str(block["text"]))
            return " ".join(text_parts)
        
        else:
            return str(message.content)
    
    def has_multimodal_content(self, message: BaseMessage) -> bool:
        """Check if message contains multimodal content."""
        
        if isinstance(message.content, list):
            for block in message.content:
                if isinstance(block, dict):
                    block_type = block.get("type", "")
                    if block_type in ["image", "audio", "video", "file"]:
                        return True
        
        return False
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of text content."""
        
        # Simplified sentiment analysis (in production, use proper NLP)
        positive_words = ["good", "great", "excellent", "amazing", "wonderful", "love", "like", "happy", "pleased"]
        negative_words = ["bad", "terrible", "awful", "hate", "dislike", "angry", "frustrated", "disappointed"]
        
        words = text.lower().split()
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        
        if positive_count > negative_count:
            sentiment = "positive"
            confidence = min(0.9, 0.5 + (positive_count - negative_count) * 0.1)
        elif negative_count > positive_count:
            sentiment = "negative"
            confidence = min(0.9, 0.5 + (negative_count - positive_count) * 0.1)
        else:
            sentiment = "neutral"
            confidence = 0.6
        
        return {
            "sentiment": sentiment,
            "confidence": confidence,
            "positive_indicators": positive_count,
            "negative_indicators": negative_count
        }
    
    def analyze_complexity(self, text: str) -> Dict[str, Any]:
        """Analyze complexity of text content."""
        
        words = text.split()
        sentences = text.split('.')
        
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
        avg_sentence_length = sum(len(sentence.split()) for sentence in sentences) / len(sentences) if sentences else 0
        
        # Technical term detection (simplified)
        technical_indicators = ["algorithm", "implementation", "architecture", "framework", "methodology"]
        technical_count = sum(1 for word in words if word.lower() in technical_indicators)
        
        # Complexity scoring
        complexity_score = 0
        if avg_word_length > 6:
            complexity_score += 1
        if avg_sentence_length > 20:
            complexity_score += 1
        if technical_count > 0:
            complexity_score += 1
        if len(words) > 100:
            complexity_score += 1
        
        complexity_levels = {0: "simple", 1: "moderate", 2: "complex", 3: "very_complex", 4: "highly_complex"}
        
        return {
            "complexity_level": complexity_levels.get(complexity_score, "simple"),
            "avg_word_length": avg_word_length,
            "avg_sentence_length": avg_sentence_length,
            "technical_terms": technical_count,
            "total_words": len(words)
        }
    
    def extract_topics(self, text: str) -> List[str]:
        """Extract topics from text (simplified implementation)."""
        
        topic_keywords = {
            "technology": ["technology", "software", "programming", "code", "computer", "digital", "AI", "machine learning"],
            "business": ["business", "marketing", "sales", "revenue", "profit", "strategy", "management", "finance"],
            "science": ["science", "research", "experiment", "data", "analysis", "study", "hypothesis", "theory"],
            "health": ["health", "medical", "medicine", "treatment", "patient", "doctor", "hospital", "wellness"],
            "education": ["education", "learning", "teaching", "school", "university", "student", "course", "training"]
        }
        
        text_lower = text.lower()
        detected_topics = []
        
        for topic, keywords in topic_keywords.items():
            matches = sum(1 for keyword in keywords if keyword in text_lower)
            if matches >= 2:  # Require at least 2 keyword matches
                detected_topics.append(topic)
        
        return detected_topics
    
    def detect_intent(self, text: str) -> Dict[str, Any]:
        """Detect user intent from text."""
        
        intent_patterns = {
            "question": ["what", "how", "why", "when", "where", "who", "?"],
            "request": ["please", "can you", "could you", "would you", "help me"],
            "command": ["do", "make", "create", "generate", "build", "write"],
            "complaint": ["problem", "issue", "wrong", "error", "bug", "broken"],
            "compliment": ["thank", "thanks", "great", "excellent", "good job"]
        }
        
        text_lower = text.lower()
        intent_scores = {}
        
        for intent, patterns in intent_patterns.items():
            score = sum(1 for pattern in patterns if pattern in text_lower)
            if score > 0:
                intent_scores[intent] = score
        
        if intent_scores:
            primary_intent = max(intent_scores, key=intent_scores.get)
            confidence = min(0.9, 0.3 + intent_scores[primary_intent] * 0.1)
        else:
            primary_intent = "unknown"
            confidence = 0.1
        
        return {
            "primary_intent": primary_intent,
            "confidence": confidence,
            "all_intents": intent_scores
        }
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities from text (simplified)."""
        
        import re
        
        entities = {
            "emails": [],
            "urls": [],
            "phone_numbers": [],
            "dates": [],
            "numbers": []
        }
        
        # Email pattern
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        entities["emails"] = re.findall(email_pattern, text)
        
        # URL pattern  
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        entities["urls"] = re.findall(url_pattern, text)
        
        # Phone number pattern (simplified)
        phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
        entities["phone_numbers"] = re.findall(phone_pattern, text)
        
        # Date pattern (simplified)
        date_pattern = r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'
        entities["dates"] = re.findall(date_pattern, text)
        
        # Number pattern
        number_pattern = r'\b\d+(?:\.\d+)?\b'
        entities["numbers"] = re.findall(number_pattern, text)
        
        # Filter empty lists
        entities = {k: v for k, v in entities.items() if v}
        
        return entities

# Custom message types for specific domains
class CustomerServiceMessage(BaseMessage):
    """Enhanced message type for customer service scenarios."""
    
    type: str = "customer_service"
    
    def __init__(
        self,
        content: Union[str, List[Dict[str, Any]]],
        customer_id: Optional[str] = None,
        ticket_id: Optional[str] = None,
        priority: str = "normal",
        category: Optional[str] = None,
        sentiment: Optional[str] = None,
        **kwargs
    ):
        super().__init__(content=content, **kwargs)
        self.customer_id = customer_id
        self.ticket_id = ticket_id or str(uuid.uuid4())
        self.priority = priority
        self.category = category
        self.sentiment = sentiment
        self.timestamp = datetime.now().isoformat()
    
    @classmethod
    def from_user_input(
        cls,
        content: str,
        customer_id: str,
        auto_analyze: bool = True
    ) -> "CustomerServiceMessage":
        """Create customer service message with automatic analysis."""
        
        processor = AdvancedMessageProcessor()
        
        # Auto-detect priority and category
        priority = "normal"
        category = None
        sentiment = None
        
        if auto_analyze:
            analysis = processor.process_message_content(
                HumanMessage(content=content),
                analysis_types=["sentiment", "intent", "topics"]
            )
            
            # Determine priority from sentiment and intent
            if analysis.get("sentiment", {}).get("sentiment") == "negative":
                if analysis["sentiment"]["confidence"] > 0.7:
                    priority = "high"
            
            # Determine category from topics and intent
            topics = analysis.get("topics", [])
            if "technology" in topics:
                category = "technical"
            elif any(word in content.lower() for word in ["bill", "payment", "charge"]):
                category = "billing"
            elif any(word in content.lower() for word in ["account", "login", "password"]):
                category = "account"
            else:
                category = "general"
            
            sentiment = analysis.get("sentiment", {}).get("sentiment")
        
        return cls(
            content=content,
            customer_id=customer_id,
            priority=priority,
            category=category,
            sentiment=sentiment
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/transmission."""
        return {
            "type": self.type,
            "content": self.content,
            "customer_id": self.customer_id,
            "ticket_id": self.ticket_id,
            "priority": self.priority,
            "category": self.category,
            "sentiment": self.sentiment,
            "timestamp": self.timestamp,
            "metadata": getattr(self, 'metadata', {})
        }

class AnalyticsMessage(BaseMessage):
    """Message type for analytics and reporting scenarios."""
    
    type: str = "analytics"
    
    def __init__(
        self,
        content: Union[str, List[Dict[str, Any]]],
        data_source: Optional[str] = None,
        metrics: Optional[Dict[str, float]] = None,
        time_range: Optional[Dict[str, str]] = None,
        visualization_type: Optional[str] = None,
        **kwargs
    ):
        super().__init__(content=content, **kwargs)
        self.data_source = data_source
        self.metrics = metrics or {}
        self.time_range = time_range or {}
        self.visualization_type = visualization_type
        self.generated_at = datetime.now().isoformat()
    
    def add_metric(self, name: str, value: float, unit: str = None):
        """Add a metric to this analytics message."""
        self.metrics[name] = {
            "value": value,
            "unit": unit,
            "timestamp": datetime.now().isoformat()
        }
    
    def format_for_display(self) -> str:
        """Format analytics data for human-readable display."""
        
        display_content = f"📊 Analytics Report\n"
        display_content += f"Generated: {self.generated_at}\n"
        
        if self.data_source:
            display_content += f"Data Source: {self.data_source}\n"
        
        if self.time_range:
            display_content += f"Time Range: {self.time_range.get('start', 'N/A')} to {self.time_range.get('end', 'N/A')}\n"
        
        display_content += "\n📈 Metrics:\n"
        for name, data in self.metrics.items():
            if isinstance(data, dict):
                value = data.get('value', data)
                unit = data.get('unit', '')
            else:
                value = data
                unit = ''
            
            display_content += f"  • {name}: {value} {unit}\n"
        
        if isinstance(self.content, str):
            display_content += f"\n📝 Analysis:\n{self.content}"
        
        return display_content
```

#### Conversation Flow Management

**Advanced Conversation Orchestration**
```python
from enum import Enum
from typing import Callable, Optional
import asyncio

class ConversationState(Enum):
    """States in conversation flow."""
    GREETING = "greeting"
    INFORMATION_GATHERING = "information_gathering"  
    PROCESSING = "processing"
    CLARIFICATION = "clarification"
    RESOLUTION = "resolution"
    FOLLOWUP = "followup"
    CLOSURE = "closure"

class ConversationFlow:
    """Manage complex conversation flows with state transitions."""
    
    def __init__(self):
        self.current_state = ConversationState.GREETING
        self.state_history = [ConversationState.GREETING]
        self.context = {}
        self.transitions = self._define_transitions()
        self.state_handlers = self._define_state_handlers()
    
    def _define_transitions(self) -> Dict[ConversationState, List[ConversationState]]:
        """Define valid state transitions."""
        return {
            ConversationState.GREETING: [
                ConversationState.INFORMATION_GATHERING,
                ConversationState.CLOSURE
            ],
            ConversationState.INFORMATION_GATHERING: [
                ConversationState.PROCESSING,
                ConversationState.CLARIFICATION,
                ConversationState.CLOSURE
            ],
            ConversationState.PROCESSING: [
                ConversationState.RESOLUTION,
                ConversationState.CLARIFICATION,
                ConversationState.INFORMATION_GATHERING
            ],
            ConversationState.CLARIFICATION: [
                ConversationState.INFORMATION_GATHERING,
                ConversationState.PROCESSING,
                ConversationState.RESOLUTION
            ],
            ConversationState.RESOLUTION: [
                ConversationState.FOLLOWUP,
                ConversationState.CLOSURE,
                ConversationState.INFORMATION_GATHERING  # For additional requests
            ],
            ConversationState.FOLLOWUP: [
                ConversationState.CLOSURE,
                ConversationState.INFORMATION_GATHERING
            ],
            ConversationState.CLOSURE: []  # Terminal state
        }
    
    def _define_state_handlers(self) -> Dict[ConversationState, Callable]:
        """Define handlers for each conversation state."""
        return {
            ConversationState.GREETING: self.handle_greeting,
            ConversationState.INFORMATION_GATHERING: self.handle_information_gathering,
            ConversationState.PROCESSING: self.handle_processing,
            ConversationState.CLARIFICATION: self.handle_clarification,
            ConversationState.RESOLUTION: self.handle_resolution,
            ConversationState.FOLLOWUP: self.handle_followup,
            ConversationState.CLOSURE: self.handle_closure
        }
    
    def transition_to(self, new_state: ConversationState) -> bool:
        """Attempt to transition to a new state."""
        
        if new_state in self.transitions.get(self.current_state, []):
            self.state_history.append(new_state)
            self.current_state = new_state
            print(f"🔄 Transitioned to: {new_state.value}")
            return True
        else:
            print(f"❌ Invalid transition from {self.current_state.value} to {new_state.value}")
            return False
    
    def process_message(
        self,
        message: BaseMessage,
        processor: AdvancedMessageProcessor
    ) -> Dict[str, Any]:
        """Process message according to current conversation state."""
        
        # Analyze message content
        analysis = processor.process_message_content(
            message,
            analysis_types=["sentiment", "intent", "complexity", "entities"]
        )
        
        # Update context with analysis
        self.context.update({
            "last_message_analysis": analysis,
            "last_message_timestamp": datetime.now().isoformat()
        })
        
        # Handle based on current state
        handler = self.state_handlers.get(self.current_state)
        if handler:
            result = handler(message, analysis)
        else:
            result = {"response": "I'm not sure how to handle that right now."}
        
        return result
    
    def handle_greeting(self, message: BaseMessage, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Handle greeting state."""
        
        greetings = ["hello", "hi", "hey", "good morning", "good afternoon"]
        message_content = message.content.lower() if isinstance(message.content, str) else ""
        
        if any(greeting in message_content for greeting in greetings):
            # Stay in greeting or move to information gathering
            if analysis.get("intent", {}).get("primary_intent") == "question":
                self.transition_to(ConversationState.INFORMATION_GATHERING)
                return {
                    "response": "Hello! I see you have a question. I'm here to help. What would you like to know?",
                    "next_actions": ["gather_information"]
                }
            else:
                return {
                    "response": "Hello! How can I assist you today?",
                    "next_actions": ["wait_for_request"]
                }
        else:
            # Direct to information gathering
            self.transition_to(ConversationState.INFORMATION_GATHERING)
            return self.handle_information_gathering(message, analysis)
    
    def handle_information_gathering(self, message: BaseMessage, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Handle information gathering state."""
        
        # Collect information from the message
        entities = analysis.get("entities", {})
        topics = analysis.get("topics", [])
        intent = analysis.get("intent", {})
        
        # Update context with gathered information
        self.context.update({
            "topics": topics,
            "entities": entities,
            "primary_intent": intent.get("primary_intent")
        })
        
        # Determine if we have enough information
        required_info = self.determine_required_info()
        missing_info = [info for info in required_info if info not in self.context]
        
        if missing_info:
            return {
                "response": f"I need a bit more information. Could you tell me about: {', '.join(missing_info)}?",
                "missing_information": missing_info,
                "next_actions": ["continue_gathering"]
            }
        else:
            # Enough information gathered, move to processing
            self.transition_to(ConversationState.PROCESSING)
            return {
                "response": "Thank you for the information. Let me process your request.",
                "gathered_context": self.context,
                "next_actions": ["process_request"]
            }
    
    def handle_processing(self, message: BaseMessage, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Handle processing state."""
        
        # Simulate processing logic
        complexity = analysis.get("complexity", {}).get("complexity_level", "simple")
        
        if complexity in ["very_complex", "highly_complex"]:
            # May need clarification
            self.transition_to(ConversationState.CLARIFICATION)
            return {
                "response": "This is quite complex. Let me clarify a few points to ensure I understand correctly.",
                "complexity_level": complexity,
                "next_actions": ["seek_clarification"]
            }
        else:
            # Can proceed to resolution
            self.transition_to(ConversationState.RESOLUTION)
            return {
                "response": "I've processed your request. Here's what I found...",
                "processing_complete": True,
                "next_actions": ["provide_resolution"]
            }
    
    def handle_clarification(self, message: BaseMessage, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Handle clarification state."""
        
        # Process clarification response
        clarification_received = True  # Simplified logic
        
        if clarification_received:
            self.transition_to(ConversationState.PROCESSING)
            return {
                "response": "Thank you for the clarification. Processing your request now.",
                "clarification_received": True,
                "next_actions": ["reprocess_with_clarification"]
            }
        else:
            return {
                "response": "I still need clarification on a few points...",
                "clarification_needed": True,
                "next_actions": ["continue_clarification"]
            }
    
    def handle_resolution(self, message: BaseMessage, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Handle resolution state."""
        
        # Provide resolution based on context
        resolution = self.generate_resolution()
        
        # Check if user is satisfied
        sentiment = analysis.get("sentiment", {})
        if sentiment.get("sentiment") == "positive":
            self.transition_to(ConversationState.FOLLOWUP)
            return {
                "response": f"Great! {resolution} Is there anything else I can help you with?",
                "resolution_provided": resolution,
                "user_satisfied": True,
                "next_actions": ["offer_additional_help"]
            }
        elif sentiment.get("sentiment") == "negative":
            self.transition_to(ConversationState.INFORMATION_GATHERING)
            return {
                "response": f"I see you're not satisfied. Let me try a different approach. {resolution}",
                "resolution_provided": resolution,
                "user_satisfied": False,
                "next_actions": ["gather_more_info", "try_different_approach"]
            }
        else:
            return {
                "response": resolution,
                "resolution_provided": resolution,
                "next_actions": ["await_feedback"]
            }
    
    def handle_followup(self, message: BaseMessage, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Handle followup state."""
        
        intent = analysis.get("intent", {}).get("primary_intent")
        
        if intent in ["question", "request"]:
            # New request
            self.transition_to(ConversationState.INFORMATION_GATHERING)
            return {
                "response": "Of course! I'd be happy to help with that as well.",
                "new_request": True,
                "next_actions": ["handle_new_request"]
            }
        else:
            # Ready to close
            self.transition_to(ConversationState.CLOSURE)
            return {
                "response": "You're welcome! Feel free to reach out if you need any more help.",
                "ready_to_close": True,
                "next_actions": ["close_conversation"]
            }
    
    def handle_closure(self, message: BaseMessage, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Handle closure state."""
        
        return {
            "response": "Thank you for using our service. Have a great day!",
            "conversation_closed": True,
            "next_actions": ["end_session"]
        }
    
    def determine_required_info(self) -> List[str]:
        """Determine what information is required based on context."""
        
        # Simplified logic - in practice, this would be more sophisticated
        intent = self.context.get("primary_intent")
        topics = self.context.get("topics", [])
        
        required = ["user_goal"]
        
        if intent == "question":
            required.append("question_topic")
        elif intent == "request":
            required.extend(["request_details", "preferred_outcome"])
        elif intent == "complaint":
            required.extend(["issue_description", "impact_level"])
        
        if "technology" in topics:
            required.append("technical_context")
        
        return required
    
    def generate_resolution(self) -> str:
        """Generate resolution based on gathered context."""
        
        # Simplified resolution generation
        topics = self.context.get("topics", [])
        intent = self.context.get("primary_intent", "unknown")
        
        if "technology" in topics:
            return "I've analyzed your technical issue and here's the recommended solution..."
        elif intent == "complaint":
            return "I understand your concern and here's how we can resolve this..."
        else:
            return "Based on your request, here's what I recommend..."
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get summary of conversation flow."""
        
        return {
            "current_state": self.current_state.value,
            "state_history": [state.value for state in self.state_history],
            "context": self.context,
            "total_transitions": len(self.state_history) - 1
        }
```

### Practical Exercise: Advanced Message Management System

Build a comprehensive message management system for a customer support application:

```python
# Exercise: Advanced Customer Support Message System

from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from typing import Dict, Any, List

class CustomerSupportMessageManager:
    """Comprehensive message management for customer support."""
    
    def __init__(self):
        self.message_architect = MessageArchitect()
        self.message_processor = AdvancedMessageProcessor()
        self.conversation_flows = {}  # Store flows per customer
        
    def create_support_conversation(self, customer_id: str) -> ConversationFlow:
        """Create new conversation flow for customer."""
        
        flow = ConversationFlow()
        self.conversation_flows[customer_id] = flow
        return flow
    
    def process_customer_message(
        self,
        customer_id: str,
        message_content: str,
        message_type: str = "standard"
    ) -> Dict[str, Any]:
        """Process incoming customer message."""
        
        # Get or create conversation flow
        if customer_id not in self.conversation_flows:
            self.create_support_conversation(customer_id)
        
        flow = self.conversation_flows[customer_id]
        
        # Create appropriate message type
        if message_type == "customer_service":
            message = CustomerServiceMessage.from_user_input(
                content=message_content,
                customer_id=customer_id,
                auto_analyze=True
            )
        else:
            message = HumanMessage(content=message_content)
        
        # Process through conversation flow
        result = flow.process_message(message, self.message_processor)
        
        # Add conversation context
        result.update({
            "customer_id": customer_id,
            "conversation_state": flow.current_state.value,
            "message_analysis": flow.context.get("last_message_analysis", {}),
            "conversation_summary": flow.get_conversation_summary()
        })
        
        return result

# Tools for message management
support_manager = CustomerSupportMessageManager()

@tool
def process_customer_inquiry(
    customer_id: str,
    message: str,
    priority: str = "normal",
    runtime: ToolRuntime = None
) -> str:
    """Process customer inquiry with advanced message handling."""
    
    result = support_manager.process_customer_message(
        customer_id=customer_id,
        message_content=message,
        message_type="customer_service"
    )
    
    # Format response for agent
    response = f"Customer Inquiry Processed:\n"
    response += f"State: {result['conversation_state']}\n"
    response += f"Response: {result['response']}\n"
    
    # Add analysis insights
    analysis = result.get("message_analysis", {})
    if analysis:
        response += f"\nMessage Analysis:\n"
        response += f"- Sentiment: {analysis.get('sentiment', 'unknown')}\n"
        response += f"- Intent: {analysis.get('intent', 'unknown')}\n"
        response += f"- Complexity: {analysis.get('complexity', 'unknown')}\n"
    
    return response

@tool
def analyze_conversation_pattern(
    customer_id: str,
    runtime: ToolRuntime = None
) -> str:
    """Analyze conversation patterns for insights."""
    
    if customer_id not in support_manager.conversation_flows:
        return f"No conversation history found for customer {customer_id}"
    
    flow = support_manager.conversation_flows[customer_id]
    summary = flow.get_conversation_summary()
    
    analysis = f"Conversation Pattern Analysis for {customer_id}:\n\n"
    analysis += f"Current State: {summary['current_state']}\n"
    analysis += f"State Transitions: {' → '.join(summary['state_history'])}\n"
    analysis += f"Total Transitions: {summary['total_transitions']}\n"
    
    # Analyze conversation efficiency
    if summary['total_transitions'] <= 3:
        analysis += "✅ Efficient conversation flow\n"
    elif summary['total_transitions'] <= 6:
        analysis += "⚠️ Moderate conversation complexity\n" 
    else:
        analysis += "🔄 Complex conversation - may need escalation\n"
    
    # Context insights
    context = summary.get('context', {})
    if context:
        analysis += f"\nContext Insights:\n"
        if 'topics' in context:
            analysis += f"- Topics: {', '.join(context['topics'])}\n"
        if 'primary_intent' in context:
            analysis += f"- Primary Intent: {context['primary_intent']}\n"
    
    return analysis

@tool
def generate_personalized_response(
    customer_id: str,
    response_type: str,
    context_details: str = "",
    runtime: ToolRuntime = None
) -> str:
    """Generate personalized response based on conversation context."""
    
    if customer_id not in support_manager.conversation_flows:
        return "No conversation context available for personalization."
    
    flow = support_manager.conversation_flows[customer_id]
    context = flow.context
    
    # Create appropriate message pattern
    if response_type == "resolution":
        messages = support_manager.message_architect.create_analysis_pattern(
            analysis_type="customer_issue_resolution",
            focus_areas=context.get("topics", ["general_inquiry"]),
            output_format="structured"
        )
    elif response_type == "followup":
        messages = support_manager.message_architect.create_conversation_pattern(
            persona="customer support specialist",
            conversation_style="professional",
            memory_context=context
        )
    else:
        messages = support_manager.message_architect.create_instruction_pattern(
            task=f"Generate {response_type} response",
            context=context,
            constraints=["Be empathetic", "Be solution-focused", "Be clear"]
        )
    
    # Format the response template
    response_template = f"Personalized {response_type.title()} Response:\n\n"
    
    if messages:
        system_message = messages[0]
        response_template += f"Guidelines: {system_message.content[:200]}...\n\n"
    
    response_template += f"Context-Aware Elements:\n"
    response_template += f"- Customer State: {flow.current_state.value}\n"
    response_template += f"- Conversation History: {len(flow.state_history)} interactions\n"
    
    if context_details:
        response_template += f"- Additional Context: {context_details}\n"
    
    return response_template

# Create comprehensive support agent
support_agent = create_agent(
    model="gpt-4o",
    tools=[
        process_customer_inquiry,
        analyze_conversation_pattern,
        generate_personalized_response
    ],
    system_prompt="""
    You are an advanced customer support assistant with sophisticated message processing capabilities.
    
    Your abilities include:
    - Processing customer inquiries with automatic sentiment and intent analysis
    - Managing complex conversation flows with state transitions  
    - Generating personalized responses based on conversation context
    - Analyzing conversation patterns for efficiency insights
    
    Key principles:
    1. Always consider conversation state and context
    2. Adapt your communication style to customer sentiment
    3. Provide structured, actionable responses
    4. Escalate complex issues when conversation patterns indicate difficulty
    5. Maintain empathy while being solution-focused
    
    Use the available tools to provide comprehensive customer support.
    """,
    checkpointer=InMemorySaver()
)

def test_advanced_message_system():
    """Test the advanced message management system."""
    
    print("🎯 Testing Advanced Message Management System")
    print("=" * 60)
    
    # Test scenarios with different customer interactions
    test_scenarios = [
        {
            "customer_id": "cust_001",
            "scenarios": [
                "Hi, I'm having trouble logging into my account",
                "I tried resetting my password but it's still not working",
                "This is really frustrating, I need access urgently",
                "Okay, that solution worked. Thank you!"
            ]
        },
        {
            "customer_id": "cust_002", 
            "scenarios": [
                "Hello, I have a question about my billing",
                "I was charged twice for the same service last month",
                "Can you help me understand why this happened?",
                "I'd like a refund for the duplicate charge"
            ]
        }
    ]
    
    config = {"configurable": {"thread_id": "support_session"}}
    
    for customer_scenario in test_scenarios:
        customer_id = customer_scenario
