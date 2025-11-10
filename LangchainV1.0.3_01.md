# LangChain Mastery: Complete Agent Development Course

## Course Overview
Master LangChain v1.0 to build production-ready AI agents and applications. This comprehensive course takes you from basics to advanced multi-agent systems with hands-on projects and real-world examples.

**Duration**: 50+ hours | **Level**: Beginner to Advanced | **Prerequisites**: Python basics, familiarity with APIs

---

# Module 1: Foundations of LangChain

## Lesson 1.1: Introduction and Philosophy (30 minutes)

### Learning Objectives
- Understand LangChain's core philosophy and design principles
- Differentiate between LangChain and LangGraph
- Grasp the evolution from prototyping to production

### Content

#### What is LangChain?
LangChain is a framework designed to be the easiest place to start building with LLMs while remaining flexible and production-ready. It's built on two core beliefs:

1. **LLMs are better when combined with external data sources**
2. **Applications will become increasingly agentic**

#### Core Architecture
```python
# LangChain's simple agent creation
from langchain.agents import create_agent

def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"It's sunny in {city}!"

agent = create_agent(
    model="gpt-4o",
    tools=[get_weather],
    system_prompt="You are a helpful weather assistant"
)
```

#### LangChain vs LangGraph
- **LangChain**: High-level abstractions, quick setup, opinionated patterns
- **LangGraph**: Low-level control, custom workflows, maximum flexibility
- **Relationship**: LangChain agents are built on LangGraph for durability and streaming

#### Evolution Timeline
- **2022**: Initial release with chains and basic LLM abstractions
- **2023**: Function calling, LangSmith observability, JavaScript support
- **2024**: LangGraph for orchestration, 700+ integrations
- **2025**: v1.0 with unified agent abstraction and multimodal support

### Practical Exercise
Set up your development environment and create your first "Hello World" agent.

```python
# Exercise: Create a simple greeting agent
from langchain.agents import create_agent

def greet_user(name: str) -> str:
    """Greet a user by name."""
    return f"Hello {name}! Welcome to LangChain."

agent = create_agent(
    model="gpt-4o-mini",
    tools=[greet_user],
    system_prompt="You are a friendly greeting assistant."
)

# Test your agent
result = agent.invoke({
    "messages": [{"role": "user", "content": "My name is Alice"}]
})
print(result["messages"][-1].content)
```

---

## Lesson 1.2: Installation and Environment Setup (20 minutes)

### Learning Objectives
- Install LangChain and configure development environment
- Set up API keys and LangSmith integration
- Understand provider-specific installations

### Content

#### Basic Installation
```bash
# Core installation
pip install -U langchain

# Provider-specific packages
pip install -U langchain-openai      # For OpenAI models
pip install -U langchain-anthropic   # For Claude models
pip install -U langchain-community   # Community integrations
```

#### Environment Configuration
```python
# .env file
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
LANGSMITH_API_KEY=lsv2_...
LANGSMITH_TRACING=true
```

#### LangSmith Setup
```python
import os
import getpass

# Set up LangSmith for observability
if not os.environ.get("LANGSMITH_API_KEY"):
    os.environ["LANGSMITH_API_KEY"] = getpass.getpass("LangSmith API Key: ")
    
os.environ["LANGSMITH_TRACING"] = "true"
```

#### Verification Script
```python
# verify_setup.py
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent

def test_setup():
    """Verify LangChain setup is working."""
    try:
        # Test model initialization
        model = init_chat_model("gpt-4o-mini")
        
        # Test simple agent
        def hello_world() -> str:
            """Return a greeting."""
            return "Hello from LangChain!"
        
        agent = create_agent(model, tools=[hello_world])
        
        result = agent.invoke({
            "messages": [{"role": "user", "content": "Say hello"}]
        })
        
        print("✅ Setup successful!")
        print(f"Response: {result['messages'][-1].content}")
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")

if __name__ == "__main__":
    test_setup()
```

### Practical Exercise
Complete the environment setup and run the verification script to ensure everything works correctly.

---

## Lesson 1.3: Your First Production Agent (45 minutes)

### Learning Objectives
- Build a complete weather forecasting agent
- Implement tools with runtime context
- Configure structured output and memory
- Test agent functionality

### Content

#### Building a Weather Agent Step-by-Step

**Step 1: Define Tools with Context**
```python
from dataclasses import dataclass
from langchain.tools import tool, ToolRuntime

@dataclass
class Context:
    """Runtime context for our agent."""
    user_id: str
    location_preference: str = "metric"

@tool
def get_weather(city: str, runtime: ToolRuntime[Context]) -> str:
    """Get current weather for a city."""
    # Access user preferences from context
    units = runtime.context.location_preference
    
    # Simulate API call
    temp = "22°C" if units == "metric" else "72°F"
    return f"Current weather in {city}: {temp}, sunny skies"

@tool
def get_user_location(runtime: ToolRuntime[Context]) -> str:
    """Get user's default location."""
    # In real app, this would query a database
    user_locations = {
        "user_1": "San Francisco",
        "user_2": "London"
    }
    return user_locations.get(runtime.context.user_id, "Unknown")
```

**Step 2: Configure the Model**
```python
from langchain.chat_models import init_chat_model

model = init_chat_model(
    "gpt-4o",
    temperature=0.3,  # Slightly creative but consistent
    max_tokens=500,   # Reasonable response length
    timeout=30        # 30-second timeout
)
```

**Step 3: Define Structured Output**
```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class WeatherResponse:
    """Structured weather response."""
    location: str
    temperature: str
    conditions: str
    recommendation: str
    forecast_confidence: Optional[str] = None
```

**Step 4: Add Memory and Create Agent**
```python
from langgraph.checkpoint.memory import InMemorySaver

# System prompt
WEATHER_SYSTEM_PROMPT = """
You are a helpful weather assistant. Use the available tools to:

1. Get the user's location if they ask about "here" or "my location"
2. Fetch weather information for specific cities
3. Provide helpful recommendations based on conditions

Always be friendly and include practical advice like what to wear or activities to do.
"""

# Create agent with all components
agent = create_agent(
    model=model,
    tools=[get_weather, get_user_location],
    system_prompt=WEATHER_SYSTEM_PROMPT,
    context_schema=Context,
    response_format=WeatherResponse,
    checkpointer=InMemorySaver()  # For conversation memory
)
```

**Step 5: Test the Agent**
```python
# Test configuration
config = {"configurable": {"thread_id": "weather_chat_1"}}
context = Context(user_id="user_1", location_preference="metric")

# Test 1: Specific city
response1 = agent.invoke(
    {"messages": [{"role": "user", "content": "What's the weather in Tokyo?"}]},
    config=config,
    context=context
)

print("Structured Response:", response1["structured_response"])
print("Last Message:", response1["messages"][-1].content)

# Test 2: User's location
response2 = agent.invoke(
    {"messages": [{"role": "user", "content": "How's the weather here?"}]},
    config=config,
    context=context
)

print("Structured Response:", response2["structured_response"])
```

#### Complete Working Example
```python
"""
Complete Weather Agent Example
Demonstrates all core LangChain concepts in one application.
"""

from dataclasses import dataclass
from typing import Optional
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool, ToolRuntime
from langgraph.checkpoint.memory import InMemorySaver

# 1. Define Context Schema
@dataclass
class Context:
    user_id: str
    location_preference: str = "metric"

# 2. Define Structured Output
@dataclass
class WeatherResponse:
    location: str
    temperature: str
    conditions: str
    recommendation: str
    forecast_confidence: Optional[str] = None

# 3. Create Tools
@tool
def get_weather(city: str, runtime: ToolRuntime[Context]) -> str:
    """Get current weather for a city."""
    units = runtime.context.location_preference
    
    # Simulate weather API
    weather_data = {
        "tokyo": {"temp_c": 18, "temp_f": 64, "condition": "cloudy"},
        "london": {"temp_c": 12, "temp_f": 54, "condition": "rainy"},
        "san francisco": {"temp_c": 22, "temp_f": 72, "condition": "sunny"}
    }
    
    data = weather_data.get(city.lower(), {"temp_c": 20, "temp_f": 68, "condition": "partly cloudy"})
    temp = f"{data['temp_c']}°C" if units == "metric" else f"{data['temp_f']}°F"
    
    return f"Weather in {city}: {temp}, {data['condition']}"

@tool
def get_user_location(runtime: ToolRuntime[Context]) -> str:
    """Get user's preferred location."""
    locations = {
        "user_1": "San Francisco",
        "user_2": "London",
        "user_3": "Tokyo"
    }
    return locations.get(runtime.context.user_id, "San Francisco")

# 4. Configure Model
model = init_chat_model(
    "gpt-4o-mini",
    temperature=0.3,
    max_tokens=500
)

# 5. Create Agent
agent = create_agent(
    model=model,
    tools=[get_weather, get_user_location],
    system_prompt="""
    You are an expert weather assistant. Provide accurate weather information 
    and practical recommendations. Use tools to get current conditions and 
    always include helpful advice about what to wear or activities to do.
    """,
    context_schema=Context,
    response_format=WeatherResponse,
    checkpointer=InMemorySaver()
)

# 6. Interactive Testing Function
def test_weather_agent():
    """Interactive test function."""
    config = {"configurable": {"thread_id": "test_session"}}
    context = Context(user_id="user_1", location_preference="metric")
    
    test_queries = [
        "What's the weather in Tokyo?",
        "How about in London?",
        "What's the weather like where I am?",
        "Should I bring an umbrella today?"
    ]
    
    for query in test_queries:
        print(f"\n🌤️  Query: {query}")
        print("-" * 50)
        
        response = agent.invoke(
            {"messages": [{"role": "user", "content": query}]},
            config=config,
            context=context
        )
        
        # Show structured output
        if "structured_response" in response:
            weather_resp = response["structured_response"]
            print(f"📍 Location: {weather_resp.location}")
            print(f"🌡️  Temperature: {weather_resp.temperature}")
            print(f"☁️  Conditions: {weather_resp.conditions}")
            print(f"💡 Recommendation: {weather_resp.recommendation}")
        
        # Show conversational response
        print(f"💬 Agent: {response['messages'][-1].content}")

if __name__ == "__main__":
    test_weather_agent()
```

### Practical Exercise
1. **Setup**: Create the complete weather agent following the example
2. **Customize**: Modify the weather data to include your local cities
3. **Extend**: Add a new tool for weather alerts or forecasts
4. **Test**: Run the interactive test and try various queries

### Expected Output
```
🌤️  Query: What's the weather in Tokyo?
--------------------------------------------------
📍 Location: Tokyo
🌡️  Temperature: 18°C
☁️  Conditions: cloudy
💡 Recommendation: It's a bit cool and cloudy, so bring a light jacket!
💬 Agent: The weather in Tokyo is currently 18°C and cloudy. I'd recommend wearing layers...
```

---

# Module 2: Core Components Deep Dive

## Lesson 2.1: Models and Chat Integration (60 minutes)

### Learning Objectives
- Master model selection and configuration across providers
- Implement dynamic model routing
- Understand multimodal capabilities and token management
- Optimize for cost and performance

### Content

#### Model Initialization Patterns

**Basic Model Setup**
```python
from langchain.chat_models import init_chat_model

# String-based initialization (recommended)
model = init_chat_model("gpt-4o")

# With parameters
model = init_chat_model(
    "claude-3-sonnet-20240229",
    temperature=0.7,
    max_tokens=1000,
    timeout=30
)

# Provider-specific class (for advanced config)
from langchain_openai import ChatOpenAI
model = ChatOpenAI(
    model="gpt-4o",
    temperature=0.7,
    openai_api_base="https://custom-endpoint.com/v1"
)
```

#### Dynamic Model Selection
```python
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from langchain.chat_models import init_chat_model
from typing import Callable

# Initialize different models for different use cases
MODELS = {
    "fast": init_chat_model("gpt-4o-mini"),
    "balanced": init_chat_model("gpt-4o"),
    "powerful": init_chat_model("claude-3-opus-20240229")
}

@wrap_model_call
def smart_model_router(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """Route to appropriate model based on request complexity."""
    
    # Analyze request complexity
    message_count = len(request.messages)
    last_message = request.messages[-1].content if request.messages else ""
    
    # Simple heuristics for model selection
    if len(last_message) > 2000 or message_count > 10:
        model_key = "powerful"
    elif any(keyword in last_message.lower() for keyword in ["code", "analyze", "complex"]):
        model_key = "balanced"  
    else:
        model_key = "fast"
    
    # Update request with selected model
    request.model = MODELS[model_key]
    print(f"🤖 Using {model_key} model for this request")
    
    return handler(request)

# Use in agent
agent = create_agent(
    model=MODELS["balanced"],  # Default model
    tools=[],
    middleware=[smart_model_router]
)
```

#### Multimodal Model Usage
```python
def analyze_image(image_path: str) -> str:
    """Analyze an image using multimodal model."""
    
    # Read and encode image
    import base64
    with open(image_path, "rb") as image_file:
        image_data = base64.b64encode(image_file.read()).decode()
    
    # Create multimodal message
    message = {
        "role": "user",
        "content": [
            {"type": "text", "text": "What do you see in this image?"},
            {
                "type": "image",
                "base64": image_data,
                "mime_type": "image/jpeg"
            }
        ]
    }
    
    model = init_chat_model("gpt-4o")  # Supports vision
    response = model.invoke([message])
    return response.content

# Example usage
# result = analyze_image("product_photo.jpg")
```

#### Token Management and Cost Optimization
```python
from langchain_core.callbacks import UsageMetadataCallbackHandler

class TokenTracker:
    """Track token usage across model calls."""
    
    def __init__(self):
        self.callback = UsageMetadataCallbackHandler()
        self.total_cost = 0.0
        
        # Pricing per 1K tokens (example rates)
        self.pricing = {
            "gpt-4o": {"input": 0.005, "output": 0.015},
            "gpt-4o-mini": {"input": 0.0015, "output": 0.006},
            "claude-3-sonnet": {"input": 0.003, "output": 0.015}
        }
    
    def calculate_cost(self, model_name: str, usage: dict) -> float:
        """Calculate cost based on token usage."""
        if model_name not in self.pricing:
            return 0.0
            
        rates = self.pricing[model_name]
        input_cost = (usage.get("input_tokens", 0) / 1000) * rates["input"]
        output_cost = (usage.get("output_tokens", 0) / 1000) * rates["output"]
        
        return input_cost + output_cost
    
    def track_call(self, model_name: str, messages: list) -> dict:
        """Make a tracked model call."""
        model = init_chat_model(model_name)
        
        response = model.invoke(
            messages,
            config={"callbacks": [self.callback]}
        )
        
        # Calculate cost
        usage = response.usage_metadata
        if usage:
            cost = self.calculate_cost(model_name, usage)
            self.total_cost += cost
            
            print(f"💰 Call cost: ${cost:.4f}")
            print(f"📊 Tokens: {usage.get('input_tokens', 0)} in, {usage.get('output_tokens', 0)} out")
            print(f"💳 Total cost: ${self.total_cost:.4f}")
        
        return response

# Usage example
tracker = TokenTracker()

response = tracker.track_call(
    "gpt-4o-mini",
    [{"role": "user", "content": "Explain quantum computing in simple terms"}]
)
```

#### Model Configuration Patterns
```python
class ModelManager:
    """Centralized model configuration management."""
    
    def __init__(self):
        self.models = {}
        self.load_models()
    
    def load_models(self):
        """Load and configure all models."""
        
        # Fast model for simple tasks
        self.models["fast"] = init_chat_model(
            "gpt-4o-mini",
            temperature=0.3,
            max_tokens=500,
            timeout=15
        )
        
        # Balanced model for general use
        self.models["balanced"] = init_chat_model(
            "gpt-4o",
            temperature=0.5,
            max_tokens=1500,
            timeout=30
        )
        
        # Creative model for content generation
        self.models["creative"] = init_chat_model(
            "claude-3-sonnet-20240229",
            temperature=0.8,
            max_tokens=2000,
            timeout=45
        )
        
        # Reasoning model for complex analysis
        self.models["reasoning"] = init_chat_model(
            "gpt-4o",
            temperature=0.1,
            max_tokens=3000,
            timeout=60
        )
    
    def get_model(self, task_type: str):
        """Get appropriate model for task type."""
        return self.models.get(task_type, self.models["balanced"])
    
    def create_specialized_agent(self, task_type: str, tools: list, prompt: str):
        """Create agent with appropriate model for task."""
        model = self.get_model(task_type)
        
        return create_agent(
            model=model,
            tools=tools,
            system_prompt=prompt
        )

# Usage
manager = ModelManager()

# Create different specialized agents
research_agent = manager.create_specialized_agent(
    "reasoning",
    [web_search, summarize],
    "You are a research analyst. Provide thorough, well-reasoned analysis."
)

content_agent = manager.create_specialized_agent(
    "creative", 
    [generate_image, write_text],
    "You are a creative content generator. Be imaginative and engaging."
)
```

### Practical Exercise: Multi-Model Agent System

Create an agent that automatically selects the best model for different types of requests:

```python
# Exercise: Build a Smart Model Router Agent
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call
from langchain.tools import tool
from langchain.chat_models import init_chat_model

# Define task classification tool
@tool
def classify_task(query: str) -> str:
    """Classify the type of task for model selection."""
    query_lower = query.lower()
    
    if any(word in query_lower for word in ["code", "program", "debug", "algorithm"]):
        return "coding"
    elif any(word in query_lower for word in ["creative", "story", "poem", "blog"]):
        return "creative"
    elif any(word in query_lower for word in ["analyze", "research", "complex", "detailed"]):
        return "analysis"
    else:
        return "general"

# Initialize models
models = {
    "coding": init_chat_model("gpt-4o", temperature=0.1),
    "creative": init_chat_model("claude-3-sonnet-20240229", temperature=0.8),
    "analysis": init_chat_model("gpt-4o", temperature=0.3),
    "general": init_chat_model("gpt-4o-mini", temperature=0.5)
}

@wrap_model_call
def intelligent_model_router(request, handler):
    """Route to best model based on task type."""
    last_message = request.messages[-1].content
    
    # Classify the task
    task_result = classify_task.invoke(last_message)
    task_type = task_result if isinstance(task_result, str) else "general"
    
    # Select appropriate model
    selected_model = models.get(task_type, models["general"])
    request.model = selected_model
    
    print(f"🎯 Task: {task_type} → Model: {selected_model.model_name}")
    
    return handler(request)

# Create the smart agent
smart_agent = create_agent(
    model=models["general"],
    tools=[classify_task],
    middleware=[intelligent_model_router],
    system_prompt="You are an intelligent assistant that adapts to different types of tasks."
)

# Test with different query types
test_queries = [
    "Write a Python function to sort a list",
    "Create a short story about a time traveler", 
    "Analyze the economic impact of renewable energy",
    "What's the capital of France?"
]

for query in test_queries:
    print(f"\n📝 Query: {query}")
    response = smart_agent.invoke({
        "messages": [{"role": "user", "content": query}]
    })
    print(f"💬 Response: {response['messages'][-1].content[:100]}...")
```

---

## Lesson 2.2: Messages and Communication Patterns (45 minutes)

### Learning Objectives
- Master different message types and their use cases
- Implement multimodal message handling
- Understand message metadata and token tracking
- Build conversation management systems

### Content

#### Message Types and Structure

**Basic Message Types**
```python
from langchain.messages import (
    SystemMessage, 
    HumanMessage, 
    AIMessage, 
    ToolMessage
)

# System message - Sets behavior and context
system_msg = SystemMessage(
    content="You are a helpful customer service representative.",
    name="system"
)

# Human message - User input
human_msg = HumanMessage(
    content="I need help with my order",
    name="customer_alice",
    id="msg_001"
)

# AI message - Model response
ai_msg = AIMessage(
    content="I'd be happy to help! Can you provide your order number?",
    response_metadata={
        "model": "gpt-4o",
        "usage": {"input_tokens": 50, "output_tokens": 15}
    }
)

# Tool message - Tool execution result
tool_msg = ToolMessage(
    content="Order #12345: Status - Shipped, Expected delivery: Tomorrow",
    tool_call_id="call_abc123",
    name="order_lookup"
)
```

**Multimodal Messages**
```python
# Image analysis message
multimodal_msg = HumanMessage(
    content=[
        {"type": "text", "text": "What's in this image?"},
        {
            "type": "image",
            "base64": "iVBORw0KGgoAAAANSUhEUgAA...",
            "mime_type": "image/jpeg"
        }
    ]
)

# Document analysis
document_msg = HumanMessage(
    content=[
        {"type": "text", "text": "Summarize this PDF"},
        {
            "type": "file",
            "base64": "JVBERi0xLjQKMSAwIG9iai...",
            "mime_type": "application/pdf",
            "filename": "report.pdf"
        }
    ]
)

# Audio transcription
audio_msg = HumanMessage(
    content=[
        {"type": "text", "text": "Transcribe this audio"},
        {
            "type": "audio", 
            "base64": "UklGRnoGAABXQVZFZm10...",
            "mime_type": "audio/wav"
        }
    ]
)
```

#### Advanced Message Management

**Conversation History Manager**
```python
from typing import List, Dict, Any
from langchain.messages import BaseMessage
import json

class ConversationManager:
    """Manage conversation history with advanced features."""
    
    def __init__(self, max_tokens: int = 4000):
        self.messages: List[BaseMessage] = []
        self.max_tokens = max_tokens
        self.metadata = {}
    
    def add_message(self, message: BaseMessage, metadata: Dict[str, Any] = None):
        """Add message with optional metadata."""
        self.messages.append(message)
        
        if metadata:
            msg_id = getattr(message, 'id', len(self.messages))
            self.metadata[msg_id] = metadata
    
    def get_context_window(self, preserve_system: bool = True) -> List[BaseMessage]:
        """Get messages that fit in context window."""
        if not self.messages:
            return []
        
        # Always preserve system message
        result = []
        if preserve_system and self.messages[0].type == "system":
            result.append(self.messages[0])
            remaining = self.messages[1:]
        else:
            remaining = self.messages
        
        # Estimate tokens (rough approximation)
        total_tokens = sum(len(msg.content) // 4 for msg in result)
        
        # Add messages from most recent, staying under limit
        for msg in reversed(remaining):
            msg_tokens = len(str(msg.content)) // 4
            if total_tokens + msg_tokens <= self.max_tokens:
                result.insert(-1 if preserve_system else 0, msg)
                total_tokens += msg_tokens
            else:
                break
        
        return result
    
    def summarize_old_messages(self, summary_model) -> None:
        """Replace old messages with summary."""
        if len(self.messages) < 10:
            return
        
        # Keep system message and last 5 messages
        system_msg = self.messages[0] if self.messages[0].type == "system" else None
        old_messages = self.messages[1:-5] if system_msg else self.messages[:-5]
        recent_messages = self.messages[-5:]
        
        if not old_messages:
            return
        
        # Create summary
        summary_prompt = "Summarize this conversation history concisely:\n\n"
        for msg in old_messages:
            summary_prompt += f"{msg.type}: {msg.content}\n"
        
        summary_response = summary_model.invoke([
            {"role": "user", "content": summary_prompt}
        ])
        
        # Replace old messages with summary
        summary_msg = SystemMessage(
            content=f"Previous conversation summary: {summary_response.content}"
        )
        
        new_messages = []
        if system_msg:
            new_messages.append(system_msg)
        new_messages.append(summary_msg)
        new_messages.extend(recent_messages)
        
        self.messages = new_messages
    
    def export_conversation(self, format: str = "json") -> str:
        """Export conversation in various formats."""
        if format == "json":
            return json.dumps([
                {
                    "type": msg.type,
                    "content": msg.content,
                    "metadata": self.metadata.get(getattr(msg, 'id', None), {})
                }
                for msg in self.messages
            ], indent=2)
        
        elif format == "markdown":
            md_content = "# Conversation History\n\n"
            for msg in self.messages:
                role = msg.type.title()
                content = msg.content
                md_content += f"## {role}\n{content}\n\n"
            return md_content
        
        else:
            raise ValueError(f"Unsupported format: {format}")

# Usage example
conversation = ConversationManager(max_tokens=3000)

# Add messages
conversation.add_message(
    SystemMessage("You are a helpful assistant"),
    metadata={"timestamp": "2024-01-01T10:00:00Z"}
)

conversation.add_message(
    HumanMessage("Hello, I need help with Python"),
    metadata={"user_id": "user_123", "session": "session_456"}
)

# Get context-appropriate messages
context_messages = conversation.get_context_window()
print(f"Context window has {len(context_messages)} messages")
```

#### Message Content Blocks and Processing

**Content Block Analysis**
```python
def analyze_message_content(message: BaseMessage) -> Dict[str, Any]:
    """Analyze different types of content in a message."""
    
    analysis = {
        "text_blocks": [],
        "image_blocks": [],
        "file_blocks": [],
        "tool_calls": [],
        "reasoning": []
    }
    
    # Handle different content formats
    if isinstance(message.content, str):
        analysis["text_blocks"].append({
            "type": "text",
            "content": message.content,
            "length": len(message.content)
        })
    
    elif isinstance(message.content, list):
        for block in message.content:
            if isinstance(block, dict):
                block_type = block.get("type", "unknown")
                
                if block_type == "text":
                    analysis["text_blocks"].append({
                        "type": "text",
                        "content": block.get("text", ""),
                        "length": len(block.get("text", ""))
                    })
                
                elif block_type == "image":
                    analysis["image_blocks"].append({
                        "type": "image",
                        "source": "base64" if "base64" in block else "url",
                        "mime_type": block.get("mime_type", "unknown")
                    })
                
                elif block_type == "file":
                    analysis["file_blocks"].append({
                        "type": "file",
                        "filename": block.get("filename", "unknown"),
                        "mime_type": block.get("mime_type", "unknown")
                    })
    
    # Analyze tool calls if present
    if hasattr(message, 'tool_calls') and message.tool_calls:
        for tool_call in message.tool_calls:
            analysis["tool_calls"].append({
                "name": tool_call.get("name"),
                "args": tool_call.get("args", {}),
                "id": tool_call.get("id")
            })
    
    # Check for reasoning content
    if hasattr(message, 'content_blocks'):
        for block in message.content_blocks:
            if block.get("type") == "reasoning":
                analysis["reasoning"].append({
                    "content": block.get("reasoning", ""),
                    "length": len(block.get("reasoning", ""))
                })
    
    return analysis

# Example usage
message = AIMessage(
    content=[
        {"type": "text", "text": "I'll analyze this image for you."},
        {"type": "reasoning", "reasoning": "The user wants image analysis..."}
    ],
    tool_calls=[
        {"name": "analyze_image", "args": {"image_url": "..."}, "id": "call_123"}
    ]
)

content_analysis = analyze_message_content(message)
print(json.dumps(content_analysis, indent=2))
```

#### Custom Message Types for Domain-Specific Applications

```python
from langchain.messages import BaseMessage
from typing import Optional, Dict, Any

class CustomerServiceMessage(BaseMessage):
    """Custom message type for customer service scenarios."""
    
    type: str = "customer_service"
    
    def __init__(
        self,
        content: str,
        customer_id: Optional[str] = None,
        ticket_id: Optional[str] = None,
        priority: str = "normal",
        category: Optional[str] = None,
        **kwargs
    ):
        super().__init__(content=content, **kwargs)
        self.customer_id = customer_id
        self.ticket_id = ticket_id
        self.priority = priority
        self.category = category
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/transmission."""
        return {
            "type": self.type,
            "content": self.content,
            "customer_id": self.customer_id,
            "ticket_id": self.ticket_id,
            "priority": self.priority,
            "category": self.category,
            "metadata": getattr(self, 'metadata', {})
        }

class SystemAlertMessage(BaseMessage):
    """Message type for system alerts and notifications."""
    
    type: str = "system_alert"
    
    def __init__(
        self,
        content: str,
        alert_level: str = "info",  # info, warning, error, critical
        source_system: Optional[str] = None,
        alert_code: Optional[str] = None,
        **kwargs
    ):
        super().__init__(content=content, **kwargs)
        self.alert_level = alert_level
        self.source_system = source_system
        self.alert_code = alert_code
    
    @property
    def is_critical(self) -> bool:
        return self.alert_level == "critical"

# Usage in customer service agent
def create_customer_service_conversation():
    """Create a customer service conversation with custom messages."""
    
    messages = [
        SystemMessage("You are a customer service representative."),
        
        CustomerServiceMessage(
            content="My order hasn't arrived and it's been 2 weeks!",
            customer_id="cust_12345",
            ticket_id="tick_67890",
            priority="high",
            category="shipping"
        ),
        
        AIMessage("I understand your frustration. Let me check your order status right away."),
        
        SystemAlertMessage(
            content="Customer has premium status - prioritize resolution",
            alert_level="info",
            source_system="CRM",
            alert_code="PREMIUM_CUSTOMER"
        )
    ]
    
    return messages

# Example agent that handles custom message types
@tool  
def escalate_ticket(ticket_id: str, reason: str) -> str:
    """Escalate a customer service ticket."""
    return f"Ticket {ticket_id} escalated. Reason: {reason}"

def handle_customer_message(message: CustomerServiceMessage) -> str:
    """Handle customer service message with priority routing."""
    
    if message.priority == "critical":
        return escalate_ticket.invoke({
            "ticket_id": message.ticket_id,
            "reason": "Critical priority customer issue"
        })
    
    return f"Processing {message.category} issue for customer {message.customer_id}"
```

### Practical Exercise: Advanced Message Handler

Build a sophisticated message processing system:

```python
# Exercise: Build a Multi-Modal Conversation Processor

class AdvancedMessageProcessor:
    """Process different types of messages with context awareness."""
    
    def __init__(self):
        self.conversation_history = []
        self.file_cache = {}
        self.analysis_results = {}
    
    def process_message(self, message: BaseMessage) -> Dict[str, Any]:
        """Process a message and return structured analysis."""
        
        result = {
            "message_type": message.type,
            "content_analysis": self.analyze_content(message),
            "context_relevance": self.assess_context_relevance(message),
            "required_capabilities": self.determine_capabilities(message),
            "processing_strategy": self.suggest_processing_strategy(message)
        }
        
        # Store in history
        self.conversation_history.append(message)
        
        return result
    
    def analyze_content(self, message: BaseMessage) -> Dict[str, Any]:
        """Analyze message content for different data types."""
        # Implementation from previous example
        return analyze_message_content(message)
    
    def assess_context_relevance(self, message: BaseMessage) -> float:
        """Assess how relevant this message is to conversation context."""
        if len(self.conversation_history) < 2:
            return 1.0
        
        # Simple keyword overlap analysis
        current_words = set(str(message.content).lower().split())
        recent_words = set()
        
        for prev_msg in self.conversation_history[-3:]:
            recent_words.update(str(prev_msg.content).lower().split())
        
        if not recent_words:
            return 0.5
        
        overlap = len(current_words.intersection(recent_words))
        return min(overlap / len(current_words) if current_words else 0, 1.0)
    
    def determine_capabilities(self, message: BaseMessage) -> List[str]:
        """Determine what capabilities are needed to handle this message."""
        capabilities = []
        
        content_analysis = analyze_message_content(message)
        
        if content_analysis["image_blocks"]:
            capabilities.append("vision")
        if content_analysis["file_blocks"]:
            capabilities.append("document_processing")  
        if content_analysis["tool_calls"]:
            capabilities.append("tool_execution")
        if any("code" in str(block) for block in content_analysis["text_blocks"]):
            capabilities.append("code_analysis")
        
        return capabilities
    
    def suggest_processing_strategy(self, message: BaseMessage) -> str:
        """Suggest the best processing strategy for this message."""
        capabilities = self.determine_capabilities(message)
        
        if "vision" in capabilities and "document_processing" in capabilities:
            return "multimodal_analysis"
        elif "tool_execution" in capabilities:
            return "agentic_processing"
        elif "code_analysis" in capabilities:
            return "code_focused"
        else:
            return "conversational"

# Test the processor
processor = AdvancedMessageProcessor()

# Test different message types
test_messages = [
    HumanMessage("Can you help me debug this Python code?"),
    HumanMessage(content=[
        {"type": "text", "text": "What's in this image?"},
        {"type": "image", "base64": "fake_image_data", "mime_type": "image/jpeg"}
    ]),
    AIMessage("", tool_calls=[{"name": "search", "args": {"query": "python debugging"}}])
]

for msg in test_messages:
    analysis = processor.process_message(msg)
    print(f"\n📝 Message Type: {analysis['message_type']}")
    print(f"🎯 Required Capabilities: {analysis['required_capabilities']}")
    print(f"⚡ Processing Strategy: {analysis['processing_strategy']}")
    print(f"🔗 Context Relevance: {analysis['context_relevance']:.2f}")
```

---

## Lesson 2.3: Tools and External Integration (75 minutes)

### Learning Objectives
- Create sophisticated tools with runtime context access
- Implement error handling and retry mechanisms
- Build tools that interact with databases, APIs, and file systems
- Master tool composition and chaining patterns

### Content

#### Advanced Tool Creation Patterns

**Tools with Runtime Context Access**
```python
from langchain.tools import tool, ToolRuntime
from dataclasses import dataclass
from typing import Optional, Dict, Any
import json

@dataclass
class DatabaseContext:
    """Database connection context."""
    connection_string: str
    user_id: str
    permissions: list[str]

@dataclass  
class APIContext:
    """API integration context."""
    api_key: str
    base_url: str
    rate_limit: int

@tool
def query_user_data(
    query: str,
    table: str,
    runtime: ToolRuntime[DatabaseContext]
) -> str:
    """Query user-specific data with permission checking."""
    
    # Access database context
    db_context = runtime.context
    
    # Check permissions
    if "read_user_data" not in db_context.permissions:
        return "Error: Insufficient permissions to read user data"
    
    # Simulate database query with context
    print(f"🔍 Querying {table} for user {db_context.user_id}")
    print(f"📊 Query: {query}")
    
    # In real implementation, use db_context.connection_string
    # to establish database connection
    
    # Mock result based on context
    mock_results = {
        "users": f"User data for {db_context.user_id}: Active account, Premium tier",
        "orders": f"Recent orders for {db_context.user_id}: 3 orders in last month",
        "preferences": f"Preferences for {db_context.user_id}: Email notifications enabled"
    }
    
    return mock_results.get(table, f"No data found in {table}")

@tool
def call_external_api(
    endpoint: str,
    method: str = "GET",
    data: Optional[Dict[str, Any]] = None,
    runtime: ToolRuntime[APIContext] = None
) -> str:
    """Make external API calls with proper authentication."""
    
    if not runtime:
        return "Error: API context not available"
    
    api_context = runtime.context
    
    # Construct full URL
    full_url = f"{api_context.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
    
    # Simulate API call with rate limiting
    print(f"🌐 Calling {method} {full_url}")
    print(f"🔑 Using API key: {api_context.api_key[:8]}...")
    print(f"⚡ Rate limit: {api_context.rate_limit} requests/minute")
    
    # Mock API response
    if endpoint == "weather":
        return json.dumps({"temperature": 22, "condition": "sunny"})
    elif endpoint == "news":
        return json.dumps({"headlines": ["Tech news 1", "Tech news 2"]})
    else:
        return json.dumps({"message": f"Data from {endpoint}"})
```

**Tools with State Interaction**
```python
from langgraph.types import Command

@tool
def save_user_preference(
    preference_key: str,
    preference_value: str,
    runtime: ToolRuntime
) -> Command:
    """Save user preference to agent state."""
    
    current_preferences = runtime.state.get("user_preferences", {})
    current_preferences[preference_key] = preference_value
    
    # Return Command to update state
    return Command(
        update={"user_preferences": current_preferences},
        # Also return a tool message
        messages=[
            ToolMessage(
                content=f"Saved preference: {preference_key} = {preference_value}",
                tool_call_id=runtime.tool_call_id
            )
        ]
    )

@tool
def get_user_preference(
    preference_key: str,
    runtime: ToolRuntime
) -> str:
    """Retrieve user preference from agent state."""
    
    preferences = runtime.state.get("user_preferences", {})
    value = preferences.get(preference_key)
    
    if value:
        return f"User preference for {preference_key}: {value}"
    else:
        return f"No preference set for {preference_key}"

@tool
def analyze_conversation_sentiment(runtime: ToolRuntime) -> str:
    """Analyze sentiment of current conversation."""
    
    messages = runtime.state.get("messages", [])
    
    # Simple sentiment analysis (in real app, use proper NLP)
    positive_words = ["good", "great", "excellent", "happy", "satisfied"]
    negative_words = ["bad", "terrible", "awful", "unhappy", "frustrated"]
    
    positive_count = 0
    negative_count = 0
    
    for message in messages[-5:]:  # Analyze last 5 messages
        content = str(message.content).lower()
        positive_count += sum(word in content for word in positive_words)
        negative_count += sum(word in content for word in negative_words)
    
    if positive_count > negative_count:
        sentiment = "positive"
    elif negative_count > positive_count:
        sentiment = "negative"
    else:
        sentiment = "neutral"
    
    return f"Conversation sentiment: {sentiment} (positive: {positive_count}, negative: {negative_count})"
```

#### Error Handling and Retry Mechanisms

**Robust Tool Implementation**
```python
import time
import random
from functools import wraps
from typing import Callable, Type, Union

def retry_on_failure(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: Union[Type[Exception], tuple] = Exception
):
    """Decorator to add retry logic to tools."""
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                
                except exceptions as e:
                    last_exception = e
                    
                    if attempt < max_retries:
                        wait_time = delay * (backoff_factor ** attempt)
                        print(f"⚠️  Attempt {attempt + 1} failed: {e}")
                        print(f"⏱️  Retrying in {wait_time:.1f}s...")
                        time.sleep(wait_time)
                    else:
                        print(f"❌ All {max_retries + 1} attempts failed")
            
            # If we get here, all retries failed
            raise last_exception
        
        return wrapper
    return decorator

class APIError(Exception):
    """Custom exception for API errors."""
    pass

class RateLimitError(Exception):
    """Custom exception for rate limiting."""
    pass

@tool
@retry_on_failure(max_retries=3, exceptions=(APIError, RateLimitError))
def reliable_api_call(endpoint: str, runtime: ToolRuntime) -> str:
    """Make API call with built-in retry logic."""
    
    # Simulate occasional failures
    if random.random() < 0.3:  # 30% chance of failure
        failure_type = random.choice([APIError, RateLimitError])
        raise failure_type(f"Simulated {failure_type.__name__} for {endpoint}")
    
    # Simulate successful response
    return f"✅ Successfully called {endpoint}: Data retrieved"

@tool
def safe_file_operation(
    operation: str,
    filename: str,
    content: str = None,
    runtime: ToolRuntime = None
) -> str:
    """Perform file operations with comprehensive error handling."""
    
    try:
        if operation == "read":
            # Simulate file reading
            if "nonexistent" in filename:
                raise FileNotFoundError(f"File {filename} not found")
            return f"File content from {filename}: [simulated content]"
        
        elif operation == "write":
            if not content:
                raise ValueError("Content required for write operation")
            
            # Simulate permission error occasionally
            if "protected" in filename:
                raise PermissionError(f"Permission denied writing to {filename}")
            
            return f"Successfully wrote {len(content)} characters to {filename}"
        
        elif operation == "delete":
            if "important" in filename:
                raise PermissionError(f"Cannot delete protected file {filename}")
            
            return f"Successfully deleted {filename}"
        
        else:
            raise ValueError(f"Unsupported operation: {operation}")
    
    except FileNotFoundError as e:
        return f"❌ File Error: {e}"
    
    except PermissionError as e:
        return f"❌ Permission Error: {e}"
        
    except ValueError as e:
        return f"❌ Input Error: {e}"
    
    except Exception as e:
        return f"❌ Unexpected Error: {e}"
```

#### Database Integration Tools

**Advanced Database Operations**
```python
import sqlite3
from typing import List, Dict, Any
from contextlib import contextmanager

class DatabaseManager:
    """Manage database connections and operations."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database with sample tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create users table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    role TEXT DEFAULT 'user',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create orders table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS orders (
                    id INTEGER PRIMARY KEY,
                    user_id INTEGER,
                    product TEXT NOT NULL,
                    quantity INTEGER DEFAULT 1,
                    total_amount DECIMAL(10,2),
                    status TEXT DEFAULT 'pending',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            ''')
            
            # Insert sample data
            cursor.execute("SELECT COUNT(*) FROM users")
            if cursor.fetchone()[0] == 0:
                sample_users = [
                    ("Alice Johnson", "alice@example.com", "admin"),
                    ("Bob Smith", "bob@example.com", "user"),
                    ("Carol Davis", "carol@example.com", "user")
                ]
                cursor.executemany(
                    "INSERT INTO users (name, email, role) VALUES (?, ?, ?)",
                    sample_users
                )
            
            conn.commit()
    
    @contextmanager
    def get_connection(self):
        """Get database connection with proper cleanup."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        try:
            yield conn
        finally:
            conn.close()
    
    def execute_query(self, query: str, params: tuple = ()) -> List[Dict[str, Any]]:
        """Execute SELECT query and return results."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            
            # Convert rows to dictionaries
            columns = [description[0] for description in cursor.description]
            results = []
            for row in cursor.fetchall():
                results.append(dict(zip(columns, row)))
                
            return results
    
    def execute_update(self, query: str, params: tuple = ()) -> int:
        """Execute INSERT/UPDATE/DELETE query."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            conn.commit()
            return cursor.rowcount

# Initialize database manager
db_manager = DatabaseManager("example.db")

@tool
def search_users(
    search_term: str = None,
    role: str = None,
    runtime: ToolRuntime = None
) -> str:
    """Search users by name, email, or role."""
    
    try:
        query = "SELECT id, name, email, role, created_at FROM users WHERE 1=1"
        params = []
        
        if search_term:
            query += " AND (name LIKE ? OR email LIKE ?)"
            params.extend([f"%{search_term}%", f"%{search_term}%"])
        
        if role:
            query += " AND role = ?"
            params.append(role)
        
        query += " ORDER BY created_at DESC LIMIT 10"
        
        results = db_manager.execute_query(query, tuple(params))
        
        if not results:
            return "No users found matching the criteria."
        
        # Format results
        user_list = []
        for user in results:
            user_list.append(
                f"ID: {user['id']}, Name: {user['name']}, "
                f"Email: {user['email']}, Role: {user['role']}"
            )
        
        return f"Found {len(results)} users:\n" + "\n".join(user_list)
    
    except Exception as e:
        return f"Database error: {e}"

@tool
def get_user_orders(user_id: int, runtime: ToolRuntime = None) -> str:
    """Get orders for a specific user."""
    
    try:
        query = '''
            SELECT o.id, o.product, o.quantity, o.total_amount, o.status, o.created_at,
                   u.name as user_name
            FROM orders o
            JOIN users u ON o.user_id = u.id
            WHERE o.user_id = ?
            ORDER BY o.created_at DESC
        '''
        
        results = db_manager.execute_query(query, (user_id,))
        
        if not results:
            return f"No orders found for user ID {user_id}."
        
        # Format results
        order_list = []
        user_name = results[0]['user_name']
        
        for order in results:
            order_list.append(
                f"Order #{order['id']}: {order['product']} "
                f"(Qty: {order['quantity']}, Amount: ${order['total_amount']}, "
                f"Status: {order['status']})"
            )
        
        return f"Orders for {user_name}:\n" + "\n".join(order_list)
    
    except Exception as e:
        return f"Database error: {e}"

@tool
def create_order(
    user_id: int,
    product: str,
    quantity: int = 1,
    total_amount: float = 0.0,
    runtime: ToolRuntime = None
) -> str:
    """Create a new order for a user."""
    
    try:
        # First verify user exists
        user_query = "SELECT name FROM users WHERE id = ?"
        user_results = db_manager.execute_query(user_query, (user_id,))
        
        if not user_results:
            return f"Error: User with ID {user_id} not found."
        
        user_name = user_results[0]['name']
        
        # Create the order
        insert_query = '''
            INSERT INTO orders (user_id, product, quantity, total_amount, status)
            VALUES (?, ?, ?, ?, 'pending')
        '''
        
        rows_affected = db_manager.execute_update(
            insert_query, 
            (user_id, product, quantity, total_amount)
        )
        
        if rows_affected > 0:
            return f"✅ Order created successfully for {user_name}: {product} (Qty: {quantity}, Amount: ${total_amount})"
        else:
            return "❌ Failed to create order."
    
    except Exception as e:
        return f"Database error: {e}"
```

#### API Integration Tools

**RESTful API Integration**
```python
import requests
import json
from typing import Optional, Dict, Any
from urllib.parse import urljoin

class APIClient:
    """Generic API client with authentication and error handling."""
    
    def __init__(self, base_url: str, api_key: str = None, timeout: int = 30):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        self.session = requests.Session()
        
        # Set default headers
        if api_key:
            self.session.headers.update({
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json'
            })
    
    def make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Make HTTP request with error handling."""
        
        url = urljoin(self.base_url + '/', endpoint.lstrip('/'))
        
        try:
            response = self.session.request(
                method=method.upper(),
                url=url,
                json=data,
                params=params,
                timeout=self.timeout
            )
            
            response.raise_for_status()
            
            # Try to parse JSON response
            try:
                return response.json()
            except ValueError:
                return {"text_response": response.text}
        
        except requests.exceptions.Timeout:
            raise APIError(f"Request timeout after {self.timeout}s")
        
        except requests.exceptions.ConnectionError:
            raise APIError(f"Connection error to {url}")
        
        except requests.exceptions.HTTPError as e:
            if response.status_code == 429:
                raise RateLimitError("Rate limit exceeded")
            else:
                raise APIError(f"HTTP {response.status_code}: {response.text}")

# Initialize API clients
weather_api = APIClient("https://api.openweathermap.org/data/2.5")
news_api = APIClient("https://newsapi.org/v2")

@tool
@retry_on_failure(max_retries=2, exceptions=(APIError, RateLimitError))
def get_weather_data(
    city: str,
    units: str = "metric",
    runtime: ToolRuntime = None
) -> str:
    """Get weather data for a city using external API."""
    
    try:
        # In real implementation, you'd use actual API key
        # params = {"q": city, "units": units, "appid": weather_api.api_key}
        
        # Mock weather data for demonstration
        mock_weather = {
            "london": {"temp": 15, "description": "cloudy", "humidity": 65},
            "paris": {"temp": 18, "description": "sunny", "humidity": 45},
            "tokyo": {"temp": 22, "description": "rainy", "humidity": 80},
            "new york": {"temp": 20, "description": "partly cloudy", "humidity": 55}
        }
        
        city_lower = city.lower()
        if city_lower in mock_weather:
            data = mock_weather[city_lower]
            temp_unit = "°C" if units == "metric" else "°F"
            
            return (f"Weather in {city}: {data['temp']}{temp_unit}, "
                   f"{data['description']}, humidity: {data['humidity']}%")
        else:
            return f"Weather data not available for {city}"
    
    except Exception as e:
        return f"Weather API error: {e}"

@tool
def search_news(
    query: str,
    category: str = "general",
    limit: int = 5,
    runtime: ToolRuntime = None
) -> str:
    """Search for news articles."""
    
    try:
        # Mock news data
        mock_articles = [
            {
                "title": f"Breaking: {query} Update",
                "source": "Tech News Daily",
                "published": "2024-01-15T10:00:00Z",
                "url": "https://example.com/article1"
            },
            {
                "title": f"Analysis: Impact of {query}",
                "source": "Business Times",
                "published": "2024-01-15T09:30:00Z", 
                "url": "https://example.com/article2"
            },
            {
                "title": f"Opinion: The Future of {query}",
                "source": "Opinion Weekly",
                "published": "2024-01-15T08:00:00Z",
                "url": "https://example.com/article3"
            }
        ]
        
        # Format results
        articles = mock_articles[:limit]
        result = f"Found {len(articles)} articles about '{query}':\n\n"
        
        for i, article in enumerate(articles, 1):
            result += (f"{i}. {article['title']}\n"
                      f"   Source: {article['source']}\n"
                      f"   Published: {article['published']}\n"
                      f"   URL: {article['url']}\n\n")
        
        return result
    
    except Exception as e:
        return f"News API error: {e}"
```

#### File System Operations

**Advanced File Handling Tools**
```python
import os
import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional
import mimetypes

@tool
def analyze_file(
    file_path: str,
    runtime: ToolRuntime = None
) -> str:
    """Analyze file and return metadata and content preview."""
    
    try:
        path = Path(file_path)
        
        if not path.exists():
            return f"❌ File not found: {file_path}"
        
        # Get file metadata
        stat = path.stat()
        file_size = stat.st_size
        modified_time = stat.st_mtime
        
        # Determine file type
        mime_type, _ = mimetypes.guess_type(file_path)
        file_extension = path.suffix.lower()
        
        # Basic file info
        info = {
            "name": path.name,
            "size": f"{file_size:,} bytes",
            "type": mime_type or "unknown",
            "extension": file_extension,
            "modified": modified_time
        }
        
        # Content preview based on file type
        preview = ""
        
        if file_extension in ['.txt', '.md', '.py', '.js', '.html', '.css']:
            # Text files
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read(500)  # First 500 chars
                    preview = f"Preview:\n{content}"
                    if len(content) == 500:
                        preview += "\n... (truncated)"
            except UnicodeDecodeError:
                preview = "Binary file - cannot preview text content"
        
        elif file_extension == '.json':
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    preview = f"JSON structure:\n{json.dumps(data, indent=2)[:500]}"
            except json.JSONDecodeError as e:
                preview = f"Invalid JSON: {e}"
        
        elif file_extension == '.csv':
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    rows = list(reader)[:5]  # First 5 rows
                    preview = f"CSV preview ({len(rows)} rows shown):\n"
                    for row in rows:
                        preview += f"{', '.join(row)}\n"
            except Exception as e:
                preview = f"Error reading CSV: {e}"
        
        else:
            preview = f"Binary file type: {mime_type or 'unknown'}"
        
        # Combine info and preview
        result = f"📄 File Analysis: {path.name}\n"
        result += f"Size: {info['size']}\n"
        result += f"Type: {info['type']}\n"
        result += f"Extension: {info['extension']}\n\n"
        result += preview
        
        return result
    
    except Exception as e:
        return f"❌ Error analyzing file: {e}"

@tool
def process_csv_file(
    file_path: str,
    operation: str = "summary",
    column: Optional[str] = None,
    runtime: ToolRuntime = None
) -> str:
    """Process CSV file with various operations."""
    
    try:
        path = Path(file_path)
        
        if not path.exists():
            return f"❌ CSV file not found: {file_path}"
        
        # Read CSV
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        if not rows:
            return "❌ CSV file is empty"
        
        headers = list(rows[0].keys())
        
        if operation == "summary":
            return (f"📊 CSV Summary for {path.name}:\n"
                   f"Rows: {len(rows)}\n"
                   f"Columns: {len(headers)}\n"
                   f"Headers: {', '.join(headers)}")
        
        elif operation == "headers":
            return f"CSV Headers: {', '.join(headers)}"
        
        elif operation == "sample":
            sample_size = min(3, len(rows))
            result = f"📋 Sample data from {path.name} ({sample_size} rows):\n\n"
            
            for i, row in enumerate(rows[:sample_size]):
                result += f"Row {i+1}:\n"
                for header in headers:
                    result += f"  {header}: {row[header]}\n"
                result += "\n"
            
            return result
        
        elif operation == "count" and column:
            if column not in headers:
                return f"❌ Column '{column}' not found. Available: {', '.join(headers)}"
            
            # Count unique values in column
            values = [row[column] for row in rows]
            unique_counts = {}
            for value in values:
                unique_counts[value] = unique_counts.get(value, 0) + 1
            
            result = f"📈 Value counts for column '{column}':\n"
            for value, count in sorted(unique_counts.items(), key=lambda x: x[1], reverse=True):
                result += f"  {value}: {count}\n"
            
            return result
        
        else:
            return f"❌ Unsupported operation: {operation}. Available: summary, headers, sample, count"
    
    except Exception as e:
        return f"❌ Error processing CSV: {e}"

@tool
def create_report_file(
    filename: str,
    data: Dict[str, Any],
    format: str = "json",
    runtime: ToolRuntime = None
) -> str:
    """Create a report file with given data."""
    
    try:
        path = Path(filename)
        
        # Ensure directory exists
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == "json":
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)
            
            return f"✅ JSON report created: {filename}"
        
        elif format.lower() == "csv":
            # Assume data is a list of dictionaries
            if not isinstance(data, list) or not data:
                return "❌ For CSV format, data must be a non-empty list of dictionaries"
            
            headers = list(data[0].keys())
            
            with open(path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                writer.writerows(data)
            
            return f"✅ CSV report created: {filename} ({len(data)} rows)"
        
        elif format.lower() == "txt":
            with open(path, 'w', encoding='utf-8') as f:
                if isinstance(data, dict):
                    for key, value in data.items():
                        f.write(f"{key}: {value}\n")
                else:
                    f.write(str(data))
            
            return f"✅ Text report created: {filename}"
        
        else:
            return f"❌ Unsupported format: {format}. Available: json, csv, txt"
    
    except Exception as e:
        return f"❌ Error creating report: {e}"
```

### Practical Exercise: Comprehensive Tool Integration

Build a complete data analysis agent with database, API, and file system integration:

```python
# Exercise: Build a Business Intelligence Agent

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from dataclasses import dataclass

@dataclass
class BusinessContext:
    company_id: str
    user_role: str
    permissions: List[str]

# Combine all the tools we've created
business_tools = [
    # Database tools
    search_users,
    get_user_orders, 
    create_order,
    
    # API tools  
    get_weather_data,
    search_news,
    
    # File tools
    analyze_file,
    process_csv_file,
    create_report_file,
    
    # State management tools
    save_user_preference,
    get_user_preference,
    analyze_conversation_sentiment
]

# Create the business intelligence agent
bi_agent = create_agent(
    model=init_chat_model("gpt-4o"),
    tools=business_tools,
    context_schema=BusinessContext,
    system_prompt="""
    You are a Business Intelligence Assistant with access to:
    - Customer database for user and order analysis
    - External APIs for market data and news
    - File system for report generation and data analysis
    - User preference management
    
    Help users analyze business data, generate reports, and make data-driven decisions.
    Always consider user permissions and provide appropriate access control.
    """,
    checkpointer=InMemorySaver()
)

# Test the comprehensive agent
def test_business_agent():
    """Test the business intelligence agent with various scenarios."""
    
    context = BusinessContext(
        company_id="company_123",
        user_role="analyst", 
        permissions=["read_user_data", "generate_reports", "access_apis"]
    )
    
    config = {"configurable": {"thread_id": "bi_session"}}
    
    test_scenarios = [
        "Show me information about our users and their recent orders",
        "Create a summary report of user activity in JSON format",
        "What's the current market sentiment about our industry? Search for relevant news",
        "Analyze the conversation sentiment and save my preference for detailed reports",
        "Generate a comprehensive business intelligence report combining all available data"
    ]
    
    for scenario in test_scenarios:
        print(f"\n🔍 Scenario: {scenario}")
        print("=" * 80)
        
        response = bi_agent.invoke(
            {"messages": [{"role": "user", "content": scenario}]},
            context=context,
            config=config
        )
        
        print(f"🤖 Response: {response['messages'][-1].content}")
        print("\n" + "-" * 80)

if __name__ == "__main__":
    test_business_agent()
```

This comprehensive exercise demonstrates:
- **Multi-domain tool integration** (database, API, file system)
- **Context-aware tool behavior** using runtime context
- **Error handling and resilience** with retry mechanisms
- **State management** for user preferences and conversation analysis
- **Permission-based access control** through context validation
- **Complex workflow orchestration** combining multiple tools for business intelligence

The agent can handle sophisticated requests that require coordination across multiple systems, making it suitable for real-world business applications.

---

# Module 3: Memory and State Management

## Lesson 3.1: Short-term Memory (Conversation State) (60 minutes)

### Learning Objectives
- Implement conversation persistence with checkpointers
- Master message history management strategies
- Build custom state schemas for complex applications
- Optimize memory usage for long conversations

### Content

#### Understanding LangChain Memory Architecture

**Memory Layers in LangChain Agents**
```python
"""
LangChain Memory Architecture:

1. Runtime Context - Static configuration (user_id, API keys, permissions)
2. Agent State - Conversation-scoped data (messages, custom fields)  
3. Store - Cross-conversation persistent data (user preferences, history)

Each serves different purposes and has different lifecycles.
"""

from langchain.agents import create_agent, AgentState
from langgraph.checkpoint.memory import InMemorySaver
from typing import TypedDict, List, Optional, Any
import json
from datetime import datetime

# Basic conversation memory setup
def basic_memory_example():
    """Demonstrate basic conversation memory."""
    
    agent = create_agent(
        model="gpt-4o-mini",
        tools=[],
        system_prompt="You are a helpful assistant with memory.",
        checkpointer=InMemorySaver()  # Enables conversation persistence
    )
    
    # Conversation thread configuration
    config = {"configurable": {"thread_id": "user_conversation_1"}}
    
    # First interaction
    response1 = agent.invoke(
        {"messages": [{"role": "user", "content": "My name is Alice and I like Python programming."}]},
        config=config
    )
    print("First response:", response1["messages"][-1].content)
    
    # Second interaction - agent should remember Alice and Python
    response2 = agent.invoke(
        {"messages": [{"role": "user", "content": "What programming language do I like?"}]},
        config=config  # Same thread_id
    )
    print("Second response:", response2["messages"][-1].content)
    
    # Different thread - agent won't remember
    config_new = {"configurable": {"thread_id": "user_conversation_2"}}
    response3 = agent.invoke(
        {"messages": [{"role": "user", "content": "What's my name?"}]},
        config=config_new  # Different thread_id
    )
    print("New thread response:", response3["messages"][-1].content)

# Run the example
basic_memory_example()
```

#### Custom State Schemas for Complex Applications

**Advanced State Management**
```python
from typing_extensions import NotRequired
from datetime import datetime, timedelta

class CustomerServiceState(AgentState):
    """Custom state for customer service scenarios."""
    
    # Core conversation data (inherited from AgentState)
    messages: List[BaseMessage]
    
    # Customer-specific information
    customer_id: NotRequired[str]
    customer_tier: NotRequired[str]  # bronze, silver, gold, platinum
    account_status: NotRequired[str]  # active, suspended, closed
    
    # Conversation metadata
    conversation_start: NotRequired[datetime]
    escalation_level: NotRequired[int]  # 0-3 (0=normal, 3=critical)
    tags: NotRequired[List[str]]  # ["billing", "technical", "complaint"]
    
    # Interaction tracking
    satisfaction_score: NotRequired[float]  # 1-5 scale
    resolution_attempts: NotRequired[int]
    last_activity: NotRequired[datetime]
    
    # Context from previous interactions
    previous_tickets: NotRequired[List[Dict[str, Any]]]
    known_issues: NotRequired[List[str]]
    preferences: NotRequired[Dict[str, str]]

class ProjectManagementState(AgentState):
    """State for project management assistant."""
    
    messages: List[BaseMessage]
    
    # Project context
    current_project: NotRequired[str]
    project_phase: NotRequired[str]  # planning, execution, review, closed
    team_members: NotRequired[List[str]]
    deadlines: NotRequired[Dict[str, datetime]]
    
    # Task tracking
    active_tasks: NotRequired[List[Dict[str, Any]]]
    completed_tasks: NotRequired[List[Dict[str, Any]]]
    blocked_tasks: NotRequired[List[Dict[str, Any]]]
    
    # Meeting and collaboration
    next_meeting: NotRequired[datetime]
    action_items: NotRequired[List[Dict[str, Any]]]
    decisions_made: NotRequired[List[str]]
    
    # Progress tracking
    completion_percentage: NotRequired[float]
    budget_used: NotRequired[float]
    risk_level: NotRequired[str]  # low, medium, high, critical

# Example: Customer Service Agent with Rich State
@tool
def lookup_customer(customer_id: str, runtime: ToolRuntime[CustomerServiceState]) -> str:
    """Look up customer information and update state."""
    
    # Mock customer data
    customer_data = {
        "cust_001": {
            "name": "Alice Johnson",
            "tier": "gold",
            "status": "active",
            "previous_tickets": [
                {"id": "tick_123", "issue": "billing", "resolved": True},
                {"id": "tick_124", "issue": "technical", "resolved": False}
            ],
            "preferences": {"contact_method": "email", "language": "english"}
        },
        "cust_002": {
            "name": "Bob Smith", 
            "tier": "silver",
            "status": "active",
            "previous_tickets": [],
            "preferences": {"contact_method": "phone", "language": "english"}
        }
    }
    
    data = customer_data.get(customer_id, {})
    if not data:
        return f"Customer {customer_id} not found."
    
    # Update agent state with customer information
    from langgraph.types import Command
    return Command(
        update={
            "customer_id": customer_id,
            "customer_tier": data["tier"],
            "account_status": data["status"],
            "previous_tickets": data["previous_tickets"],
            "preferences": data["preferences"],
            "conversation_start": datetime.now()
        }
    )

@tool
def escalate_issue(reason: str, runtime: ToolRuntime[CustomerServiceState]) -> str:
    """Escalate customer issue and update state."""
    
    current_level = runtime.state.get("escalation_level", 0)
    new_level = min(current_level + 1, 3)
    
    escalation_names = {0: "Standard", 1: "Supervisor", 2: "Manager", 3: "Director"}
    
    from langgraph.types import Command
    return Command(
        update={
            "escalation_level": new_level,
            "tags": runtime.state.get("tags", []) + ["escalated"],
            "last_activity": datetime.now()
        }
    )

# Create customer service agent with rich state
customer_service_agent = create_agent(
    model="gpt-4o",
    tools=[lookup_customer, escalate_issue],
    state_schema=CustomerServiceState,
    system_prompt="""
    You are a customer service representative. Use customer information from the state
    to provide personalized service. Consider their tier level, previous issues,
    and preferences when responding.
    
    Current customer tier affects service level:
    - Platinum: Highest priority, dedicated support
    - Gold: Priority support, extended hours
    - Silver: Standard support with faster response
    - Bronze: Standard support
    """,
    checkpointer=InMemorySaver()
)

def test_customer_service():
    """Test customer service agent with rich state."""
    
    config = {"configurable": {"thread_id": "cs_conversation_1"}}
    
    # Start conversation with customer lookup
    response1 = customer_service_agent.invoke(
        {"messages": [{"role": "user", "content": "I'm having trouble with my account. My customer ID is cust_001."}]},
        config=config
    )
    
    # Check if state was updated
    print("Agent found customer information and can now provide personalized service")
    
    # Continue conversation - agent should use customer context
    response2 = customer_service_agent.invoke(
        {"messages": [{"role": "user", "content": "This is really frustrating. I've had issues before!"}]},
        config=config
    )
    
    print("Agent response considers customer history and tier level")

test_customer_service()
```

#### Memory Management Strategies

**Long Conversation Handling**
```python
from langchain.agents.middleware import AgentMiddleware, before_model, after_model
from langchain.messages import RemoveMessage, SystemMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
import tiktoken

class ConversationMemoryManager(AgentMiddleware):
    """Advanced memory management for long conversations."""
    
    def __init__(
        self,
        max_tokens: int = 8000,
        preserve_recent_messages: int = 10,
        summarization_model: str = "gpt-4o-mini"
    ):
        super().__init__()
        self.max_tokens = max_tokens
        self.preserve_recent_messages = preserve_recent_messages
        self.summarization_model = init_chat_model(summarization_model)
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, messages: List[BaseMessage]) -> int:
        """Count tokens in message list."""
        total = 0
        for message in messages:
            content = str(message.content)
            total += len(self.tokenizer.encode(content))
        return total
    
    def create_summary(self, messages: List[BaseMessage]) -> str:
        """Create summary of conversation messages."""
        
        conversation_text = ""
        for msg in messages:
            role = msg.type
            content = str(msg.content)
            conversation_text += f"{role}: {content}\n"
        
        summary_prompt = f"""
        Please create a concise summary of this conversation that preserves:
        1. Key facts and decisions made
        2. Important context for future interactions  
        3. User preferences or requirements mentioned
        4. Any unresolved issues or next steps
        
        Conversation:
        {conversation_text}
        
        Summary:
        """
        
        response = self.summarization_model.invoke([
            {"role": "user", "content": summary_prompt}
        ])
        
        return response.content
    
    @before_model
    def manage_conversation_length(self, state, runtime):
        """Manage conversation length before model calls."""
        
        messages = state.get("messages", [])
        
        if len(messages) < 5:  # Too short to need management
            return None
        
        # Count current tokens
        current_tokens = self.count_tokens(messages)
        
        if current_tokens <= self.max_tokens:
            return None  # Within limits
        
        print(f"🧠 Managing conversation memory: {current_tokens} tokens -> reducing")
        
        # Preserve system message and recent messages
        system_msg = messages[0] if messages and messages[0].type == "system" else None
        recent_messages = messages[-self.preserve_recent_messages:]
        
        # Messages to summarize (everything except system and recent)
        start_idx = 1 if system_msg else 0
        end_idx = len(messages) - self.preserve_recent_messages
        
        if end_idx <= start_idx:
            return None  # Not enough messages to summarize
        
        messages_to_summarize = messages[start_idx:end_idx]
        
        # Create summary
        summary_text = self.create_summary(messages_to_summarize)
        summary_msg = SystemMessage(
            content=f"[Previous conversation summary: {summary_text}]"
        )
        
        # Build new message list
        new_messages = []
        if system_msg:
            new_messages.append(system_msg)
        new_messages.append(summary_msg)
        new_messages.extend(recent_messages)
        
        # Calculate token savings
        new_tokens = self.count_tokens(new_messages)
        saved_tokens = current_tokens - new_tokens
        
        print(f"💾 Memory managed: {saved_tokens} tokens saved, {new_tokens} tokens remaining")
        
        return {
            "messages": [
                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                *new_messages
            ]
        }

# Smart memory management with different strategies
class SmartMemoryManager:
    """Intelligent memory management with multiple strategies."""
    
    def __init__(self):
        self.strategies = {
            "summarize": self.summarize_strategy,
            "trim_oldest": self.trim_oldest_strategy,
            "priority_keep": self.priority_keep_strategy,
            "sliding_window": self.sliding_window_strategy
        }
    
    def summarize_strategy(self, messages: List[BaseMessage], limit: int) -> List[BaseMessage]:
        """Summarize old messages, keep recent ones."""
        if len(messages) <= 10:
            return messages
        
        # Keep system message, summarize middle, keep recent
        system_msg = messages[0] if messages[0].type == "system" else None
        recent_msgs = messages[-5:]
        middle_msgs = messages[1:-5] if system_msg else messages[:-5]
        
        # Create summary (simplified)
        summary = f"Previous conversation covered: {len(middle_msgs)} exchanges about various topics."
        summary_msg = SystemMessage(content=f"[Summary: {summary}]")
        
        result = []
        if system_msg:
            result.append(system_msg)
        result.append(summary_msg)
        result.extend(recent_msgs)
        
        return result
    
    def trim_oldest_strategy(self, messages: List[BaseMessage], limit: int) -> List[BaseMessage]:
        """Remove oldest messages, keep system and recent."""
        if len(messages) <= limit:
            return messages
        
        system_msg = messages[0] if messages[0].type == "system" else None
        
        if system_msg:
            # Keep system message + most recent messages
            keep_count = limit - 1
            return [system_msg] + messages[-keep_count:]
        else:
            return messages[-limit:]
    
    def priority_keep_strategy(self, messages: List[BaseMessage], limit: int) -> List[BaseMessage]:
        """Keep high-priority messages (system, tool results, recent)."""
        if len(messages) <= limit:
            return messages
        
        priority_messages = []
        regular_messages = []
        
        for msg in messages:
            if (msg.type in ["system", "tool"] or 
                messages.index(msg) >= len(messages) - 3):  # Last 3 messages
                priority_messages.append(msg)
            else:
                regular_messages.append(msg)
        
        # Fill remaining space with regular messages (most recent first)
        remaining_space = limit - len(priority_messages)
        if remaining_space > 0:
            priority_messages.extend(regular_messages[-remaining_space:])
        
        # Sort by original order
        return sorted(priority_messages, key=lambda x: messages.index(x))
    
    def sliding_window_strategy(self, messages: List[BaseMessage], limit: int) -> List[BaseMessage]:
        """Keep a sliding window of recent messages."""
        return messages[-limit:] if len(messages) > limit else messages

@before_model
def smart_memory_management(state, runtime):
    """Apply intelligent memory management."""
    
    messages = state.get("messages", [])
    
    if len(messages) <= 20:
        return None
    
    manager = SmartMemoryManager()
    
    # Choose strategy based on conversation characteristics
    has_tools = any(msg.type == "tool" for msg in messages[-10:])
    has_system = messages[0].type == "system" if messages else False
    
    if has_tools and has_system:
        strategy = "priority_keep"
        limit = 15
    elif len(messages) > 50:
        strategy = "summarize" 
        limit = 20
    else:
        strategy = "sliding_window"
        limit = 12
    
    new_messages = manager.strategies[strategy](messages, limit)
    
    if len(new_messages) < len(messages):
        print(f"🧠 Applied {strategy} strategy: {len(messages)} -> {len(new_messages)} messages")
        
        return {
            "messages": [
                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                *new_messages
            ]
        }
    
    return None

# Create agent with smart memory management
memory_optimized_agent = create_agent(
    model="gpt-4o",
    tools=[],
    middleware=[smart_memory_management],
    system_prompt="You are a helpful assistant with optimized memory management.",
    checkpointer=InMemorySaver()
)
```

#### Production Memory Patterns

**Database-Backed Persistence**
```python
from langgraph.checkpoint.postgres import PostgresSaver
from sqlalchemy import create_engine
import asyncio

class ProductionMemorySetup:
    """Production-ready memory configuration."""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.setup_database()
    
    def setup_database(self):
        """Set up database for persistent memory."""
        
        # For PostgreSQL persistence (production recommended)
        self.checkpointer = PostgresSaver.from_conn_string(self.database_url)
        
        # Initialize database tables
        self.checkpointer.setup()
    
    def create_production_agent(
        self,
        model_name: str,
        tools: List,
        system_prompt: str,
        state_schema=None
    ):
        """Create agent with production memory configuration."""
        
        return create_agent(
            model=model_name,
            tools=tools,
            system_prompt=system_prompt,
            state_schema=state_schema,
            checkpointer=self.checkpointer,
            middleware=[
                ConversationMemoryManager(
                    max_tokens=6000,
                    preserve_recent_messages=8
                )
            ]
        )
    
    async def cleanup_old_conversations(self, days_old: int = 30):
        """Clean up old conversation threads."""
        
        cutoff_date = datetime.now() - timedelta(days=days_old)
        
        # Implementation would depend on your database schema
        # This is a conceptual example
        async with self.checkpointer.get_connection() as conn:
            await conn.execute(
                "DELETE FROM checkpoints WHERE created_at < %s",
                (cutoff_date,)
            )
    
    def get_conversation_stats(self, thread_id: str) -> Dict[str, Any]:
        """Get statistics about a conversation thread."""
        
        # Get conversation history
        config = {"configurable": {"thread_id": thread_id}}
        
        # This would require implementing a method to retrieve
        # full conversation history from the checkpointer
        stats = {
            "thread_id": thread_id,
            "message_count": 0,
            "conversation_duration": timedelta(0),
            "total_tokens_used": 0,
            "tools_called": [],
            "last_activity": None
        }
        
        return stats

# Usage example
def setup_production_memory():
    """Example of production memory setup."""
    
    # Database configuration
    DATABASE_URL = "postgresql://user:password@localhost:5432/langchain_db"
    
    memory_setup = ProductionMemorySetup(DATABASE_URL)
    
    # Create production agent
    agent = memory_setup.create_production_agent(
        model_name="gpt-4o",
        tools=[],
        system_prompt="You are a production assistant with persistent memory.",
        state_schema=CustomerServiceState
    )
    
    return agent, memory_setup

# Memory monitoring and maintenance
class MemoryMonitor:
    """Monitor and maintain memory usage."""
    
    def __init__(self, checkpointer):
        self.checkpointer = checkpointer
        self.stats = {
            "total_threads": 0,
            "active_threads": 0,
            "total_messages": 0,
            "storage_used_mb": 0
        }
    
    def collect_stats(self):
        """Collect memory usage statistics."""
        
        # Implementation would query the checkpointer's storage
        # This is a conceptual example
        
        print(f"📊 Memory Statistics:")
        print(f"   Total conversation threads: {self.stats['total_threads']}")
        print(f"   Currently active threads: {self.stats['active_threads']}")
        print(f"   Total messages stored: {self.stats['total_messages']}")
        print(f"   Storage used: {self.stats['storage_used_mb']} MB")
    
    def optimize_storage(self):
        """Optimize memory storage."""
        
        print("🔧 Optimizing memory storage...")
        
        # Strategies could include:
        # 1. Compress old conversations
        # 2. Archive inactive threads
        # 3. Remove redundant data
        # 4. Optimize database indexes
        
        print("✅ Memory optimization complete")

# Example usage
if __name__ == "__main__":
    # Set up production memory
    agent, memory_setup = setup_production_memory()
    
    # Create monitor
    monitor = MemoryMonitor(memory_setup.checkpointer)
    
    # Test conversation with persistent memory
    config = {"configurable": {"thread_id": "prod_conversation_1"}}
    
    response = agent.invoke(
        {"messages": [{"role": "user", "content": "Hello, I'm starting a new project."}]},
        config=config
    )
    
    # Continue conversation later (memory persists)
    response2 = agent.invoke(
        {"messages": [{"role": "user", "content": "What did I mention about my project?"}]},
        config=config
    )
    
    # Monitor memory usage
    monitor.collect_stats()
```

### Practical Exercise: Advanced Memory Management System

Build a comprehensive memory management system for a customer support application:

```python
# Exercise: Customer Support Memory System

from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from langchain.agents import create_agent, AgentState
from langchain.tools import tool, ToolRuntime
from langchain.agents.middleware import AgentMiddleware, before_model
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

class SupportTicketState(AgentState):
    """Comprehensive state for support tickets."""
    
    messages: List[BaseMessage]
    
    # Ticket information
    ticket_id: NotRequired[str]
    priority: NotRequired[str]  # low, medium, high, critical
    category: NotRequired[str]  # technical, billing, account, general
    status: NotRequired[str]   # open, in_progress, resolved, closed
    
    # Customer information
    customer_id: NotRequired[str]
    customer_name: NotRequired[str]
    customer_tier: NotRequired[str]
    
    # Conversation tracking
    agent_name: NotRequired[str]
    conversation_start: NotRequired[datetime]
    last_response: NotRequired[datetime]
    response_time_sla: NotRequired[timedelta]
    
    # Resolution tracking
    escalation_count: NotRequired[int]
    resolution_attempts: NotRequired[List[str]]
    customer_satisfaction: NotRequired[int]  # 1-5 scale
    
    # Context and history
    related_tickets: NotRequired[List[str]]
    knowledge_base_articles: NotRequired[List[str]]
    previous_interactions: NotRequired[List[Dict[str, Any]]]

class SupportMemoryManager(AgentMiddleware[SupportTicketState]):
    """Advanced memory management for customer support."""
    
    state_schema = SupportTicketState
    
    def __init__(self):
        super().__init__()
        self.max_conversation_length = 30
        self.summary_threshold = 20
    
    @before_model
    def manage_support_memory(self, state: SupportTicketState, runtime):
        """Manage memory specifically for support conversations."""
        
        messages = state.get("messages", [])
        
        if len(messages) < self.summary_threshold:
            return None
        
        # Extract important information before summarizing
        important_info = self.extract_important_info(messages, state)
        
        # Create contextual summary
        summary = self.create_support_summary(messages, state, important_info)
        
        # Keep system message, summary, and recent messages
        system_msg = messages[0] if messages and messages[0].type == "system" else None
        recent_messages = messages[-10:]  # Keep last 10 messages
        
        new_messages = []
        if system_msg:
            new_messages.append(system_msg)
        
        # Add context-rich summary
        summary_msg = SystemMessage(
            content=f"[Support Context: {summary}]"
        )
        new_messages.append(summary_msg)
        new_messages.extend(recent_messages)
        
        print(f"🎫 Support memory managed: {len(messages)} -> {len(new_messages)} messages")
        
        return {
            "messages": [
                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                *new_messages
            ]
        }
    
    def extract_important_info(self, messages: List[BaseMessage], state: SupportTicketState) -> Dict[str, Any]:
        """Extract important information from conversation."""
        
        info = {
            "customer_issues": [],
            "solutions_tried": [],
            "error_messages": [],
            "product_versions": [],
            "promises_made": []
        }
        
        for message in messages:
            content = str(message.content).lower()
            
            # Look for common patterns
            if "error" in content or "problem" in content:
                info["customer_issues"].append(content[:100])
            
            if "tried" in content or "attempted" in content:
                info["solutions_tried"].append(content[:100])
            
            if "will" in content and message.type == "ai":
                info["promises_made"].append(content[:100])
        
        return info
    
    def create_support_summary(
        self, 
        messages: List[BaseMessage], 
        state: SupportTicketState, 
        important_info: Dict[str, Any]
    ) -> str:
        """Create a context-rich summary for support conversations."""
        
        summary_parts = []
        
        # Basic ticket info
        if state.get("ticket_id"):
            summary_parts.append(f"Ticket #{state['ticket_id']}")
        
        if state.get("category"):
            summary_parts.append(f"Category: {state['category']}")
        
        if state.get("priority"):
            summary_parts.append(f"Priority: {state['priority']}")
        
        # Customer info
        if state.get("customer_name"):
            summary_parts.append(f"Customer: {state['customer_name']}")
        
        if state.get("customer_tier"):
            summary_parts.append(f"Tier: {state['customer_tier']}")
        
        # Issue summary
        if important_info["customer_issues"]:
            issues = ", ".join(important_info["customer_issues"][:2])
            summary_parts.append(f"Issues: {issues}")
        
        # Solutions attempted
        if important_info["solutions_tried"]:
            solutions = ", ".join(important_info["solutions_tried"][:2])
            summary_parts.append(f"Solutions tried: {solutions}")
        
        return " | ".join(summary_parts)

# Support tools with memory integration
@tool
def create_ticket(
    issue_description: str,
    priority: str = "medium",
    category: str = "general",
    runtime: ToolRuntime[SupportTicketState] = None
) -> Command:
    """Create a new support ticket."""
    
    ticket_id = f"TICK-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    return Command(
        update={
            "ticket_id": ticket_id,
            "priority": priority,
            "category": category,
            "status": "open",
            "conversation_start": datetime.now(),
            "escalation_count": 0,
            "resolution_attempts": []
        }
    )

@tool
def update_ticket_status(
    new_status: str,
    resolution_note: str = None,
    runtime: ToolRuntime[SupportTicketState] = None
) -> Command:
    """Update ticket status and add resolution notes."""
    
    current_attempts = runtime.state.get("resolution_attempts", [])
    
    if resolution_note:
        current_attempts.append({
            "timestamp": datetime.now().isoformat(),
            "note": resolution_note,
            "status": new_status
        })
    
    return Command(
        update={
            "status": new_status,
            "resolution_attempts": current_attempts,
            "last_response": datetime.now()
        }
    )

@tool
def escalate_ticket(
    escalation_reason: str,
    runtime: ToolRuntime[SupportTicketState] = None
) -> Command:
    """Escalate ticket to higher support tier."""
    
    current_escalations = runtime.state.get("escalation_count", 0)
    
    return Command(
        update={
            "escalation_count": current_escalations + 1,
            "status": "escalated",
            "priority": "high" if current_escalations == 0 else "critical"
        }
    )

@tool
def log_customer_satisfaction(
    satisfaction_score: int,
    feedback: str = None,
    runtime: ToolRuntime[SupportTicketState] = None
) -> Command:
    """Log customer satisfaction rating."""
    
    if not 1 <= satisfaction_score <= 5:
        return "Error: Satisfaction score must be between 1 and 5"
    
    return Command(
        update={
            "customer_satisfaction": satisfaction_score,
            "status": "resolved" if satisfaction_score >= 3 else "needs_followup"
        }
    )

# Create comprehensive support agent
support_agent = create_agent(
    model="gpt-4o",
    tools=[create_ticket, update_ticket_status, escalate_ticket, log_customer_satisfaction],
    state_schema=SupportTicketState,
    middleware=[SupportMemoryManager()],
    system_prompt="""
    You are a customer support representative with access to ticketing tools.
    
    For each customer interaction:
    1. Create or reference a ticket ID
    2. Categorize the issue appropriately  
    3. Set proper priority based on urgency and customer tier
    4. Document all resolution attempts
    5. Escalate when necessary
    6. Always aim for customer satisfaction
    
    Use ticket information and conversation history to provide consistent,
    informed support throughout the interaction.
    """,
    checkpointer=InMemorySaver()
)

def test_support_memory_system():
    """Test the comprehensive support memory system."""
    
    config = {"configurable": {"thread_id": "support_session_1"}}
    
    # Simulate a complex support interaction
    interactions = [
        "Hi, I'm having trouble logging into my account. I keep getting an error message.",
        "I tried resetting my password but that didn't work. The error says 'Authentication failed'.",
        "Yes, I'm using Chrome browser. I also tried clearing my cache.",
        "My email is alice@example.com and I'm a premium customer.",
        "I've been trying for 2 hours now. This is really frustrating!",
        "Okay, that solution worked! Thank you so much for your help.",
        "I'd rate my experience as 4 out of 5. The agent was very helpful."
    ]
    
    print("🎫 Starting Support Session with Advanced Memory Management")
    print("=" * 80)
    
    for i, interaction in enumerate(interactions, 1):
        print(f"\n👤 Customer ({i}/{len(interactions)}): {interaction}")
        
        response = support_agent.invoke(
            {"messages": [{"role": "user", "content": interaction}]},
            config=config
        )
        
        print(f"🎧 Agent: {response['messages'][-1].content}")
        
        # Show some state information
        state_info = []
        if "ticket_id" in response:
            state_info.append(f"Ticket: {response['ticket_id']}")
        if "priority" in response:
            state_info.append(f"Priority: {response['priority']}")
        if "status" in response:
            state_info.append(f"Status: {response['status']}")
        
        if state_info:
            print(f"📋 State: {' | '.join(state_info)}")
        
        print("-" * 40)
    
    print("\n✅ Support session completed with full memory retention")

if __name__ == "__main__":
    test_support_memory_system()
```

This comprehensive exercise demonstrates:

1. **Custom State Schema**: Rich state management for support scenarios
2. **Intelligent Memory Management**: Context-aware conversation summarization
3. **Tool Integration**: Tools that read and update agent state
4. **Production Patterns**: Scalable memory management strategies
5. **Business Logic**: Support-specific workflows and escalation paths

The system maintains conversation context while optimizing memory usage, ensuring that important support information is preserved even in long conversations.

---

## Lesson 3.2: Long-term Memory (Persistent Storage) (45 minutes)

### Learning Objectives
- Implement cross-conversation persistent memory using LangGraph store
- Build user preference and knowledge management systems
- Create memory-driven personalization features
- Design scalable memory architectures for production

### Content

#### Understanding LangGraph Store Architecture

**Store vs State vs Context**
```python
"""
Memory Architecture in LangChain:

1. Runtime Context - Static config for a single invocation (API keys, user_id)
2. Agent State - Conversation-scoped data (messages, temporary variables)  
3. Store - Cross-conversation persistent data (user preferences, knowledge)

Store is the foundation for long-term memory that persists across sessions.
"""

from langgraph.store.memory import InMemoryStore
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json

# Basic store setup and operations
def demonstrate_store_basics():
    """Demonstrate basic store operations."""
    
    # Initialize store (in production, use database-backed store)
    store = InMemoryStore()
    
    # Store operations use (namespace, key) tuple for organization
    user_namespace = ("users", "profiles")
    preferences_namespace = ("users", "preferences") 
    knowledge_namespace = ("knowledge", "facts")
    
    # Store user profile
    store.put(
        user_namespace,
        "user_123",
        {
            "name": "Alice Johnson",
            "email": "alice@example.com",
            "account_created": "2024-01-15",
            "tier": "premium",
            "timezone": "UTC-8"
        }
    )
    
    # Store user preferences
    store.put(
        preferences_namespace,
        "user_123",
        {
            "communication_style": "detailed",
            "language": "english",
            "notification_frequency": "daily",
            "preferred_topics": ["technology", "business", "science"]
        }
    )
    
    # Store knowledge facts
    store.put(
        knowledge_namespace,
        "alice_facts",
        {
            "works_at": "TechCorp Inc",
            "role": "Senior Developer",
            "programming_languages": ["Python", "JavaScript", "Go"],
            "current_projects": ["AI Assistant", "Web Platform"],
            "last_updated": datetime.now().isoformat()
        }
    )
    
    # Retrieve data
    profile = store.get(user_namespace, "user_123")
    print(f"User Profile: {profile.value}")
    
    preferences = store.get(preferences_namespace, "user_123") 
    print(f"User Preferences: {preferences.value}")
    
    # Search within namespace
    search_results = store.search(
        knowledge_namespace,
        query="programming languages",
        limit=5
    )
    
    for result in search_results:
        print(f"Knowledge: {result.key} -> {result.value}")

demonstrate_store_basics()
```

#### User Preference Management System

**Advanced Preference Engine**
```python
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional, Union
import json

@dataclass
class UserPreferences:
    """Structured user preferences."""
    
    # Communication preferences
    communication_style: str = "balanced"  # concise, detailed, balanced
    response_length: str = "medium"  # short, medium, long
    formality_level: str = "professional"  # casual, professional, formal
    
    # Content preferences  
    preferred_topics: List[str] = None
    avoided_topics: List[str] = None
    expertise_level: str = "intermediate"  # beginner, intermediate, expert
    
    # Interaction preferences
    wants_explanations: bool = True
    wants_examples: bool = True
    wants_step_by_step: bool = False
    
    # Technical preferences
    preferred_code_style: str = "pythonic"
    preferred_frameworks: List[str] = None
    operating_system: str = "unknown"
    
    # Scheduling and timing
    timezone: str = "UTC"
    work_hours_start: str = "09:00"
    work_hours_end: str = "17:00"
    
    # Metadata
    last_updated: str = None
    preference_source: str = "implicit"  # explicit, implicit, inferred
    
    def __post_init__(self):
        if self.preferred_topics is None:
            self.preferred_topics = []
        if self.avoided_topics is None:
            self.avoided_topics = []
        if self.preferred_frameworks is None:
            self.preferred_frameworks = []
        if self.last_updated is None:
            self.last_updated = datetime.now().isoformat()

class PreferenceManager:
    """Manage user preferences with learning capabilities."""
    
    def __init__(self, store):
        self.store = store
        self.preferences_namespace = ("users", "preferences")
        self.interactions_namespace = ("users", "interactions")
    
    def get_user_preferences(self, user_id: str) -> UserPreferences:
        """Get user preferences, creating defaults if none exist."""
        
        stored_prefs = self.store.get(self.preferences_namespace, user_id)
        
        if stored_prefs:
            # Convert stored dict back to UserPreferences
            return UserPreferences(**stored_prefs.value)
        else:
            # Create default preferences
            return UserPreferences()
    
    def update_preferences(
        self, 
        user_id: str, 
        updates: Dict[str, Any],
        source: str = "explicit"
    ) -> UserPreferences:
        """Update user preferences with new values."""
        
        current_prefs = self.get_user_preferences(user_id)
        
        # Update fields
        for key, value in updates.items():
            if hasattr(current_prefs, key):
                setattr(current_prefs, key, value)
        
        # Update metadata
        current_prefs.last_updated = datetime.now().isoformat()
        current_prefs.preference_source = source
        
        # Save to store
        self.store.put(
            self.preferences_namespace,
            user_id,
            asdict(current_prefs)
        )
        
        return current_prefs
    
    def learn_from_interaction(
        self,
        user_id: str,
        interaction_data: Dict[str, Any]
    ) -> UserPreferences:
        """Learn and update preferences from user interactions."""
        
        current_prefs = self.get_user_preferences(user_id)
        updates = {}
        
        # Analyze interaction for preference signals
        user_message = interaction_data.get("user_message", "").lower()
        agent_response = interaction_data.get("agent_response", "")
        user_feedback = interaction_data.get("feedback", {})
        
        # Infer communication style preferences
        if "be brief" in user_message or "short" in user_message:
            updates["response_length"] = "short"
            updates["communication_style"] = "concise"
        
        elif "more detail" in user_message or "explain more" in user_message:
            updates["response_length"] = "long"
            updates["communication_style"] = "detailed"
        
        # Infer technical level
        technical_terms = ["api", "algorithm", "framework", "architecture", "implementation"]
        if any(term in user_message for term in technical_terms):
            if current_prefs.expertise_level == "beginner":
                updates["expertise_level"] = "intermediate"
        
        # Learn topic preferences from positive feedback
        if user_feedback.get("rating", 0) >= 4:
            topics_mentioned = self.extract_topics(user_message)
            if topics_mentioned:
                current_topics = set(current_prefs.preferred_topics)
                current_topics.update(topics_mentioned)
                updates["preferred_topics"] = list(current_topics)
        
        # Learn from negative feedback
        if user_feedback.get("rating", 0) <= 2:
            topics_mentioned = self.extract_topics(user_message)
            if topics_mentioned:
                current_avoided = set(current_prefs.avoided_topics)
                current_avoided.update(topics_mentioned)
                updates["avoided_topics"] = list(current_avoided)
        
        # Apply updates if any were inferred
        if updates:
            return self.update_preferences(user_id, updates, source="inferred")
        
        return current_prefs
    
    def extract_topics(self, text: str) -> List[str]:
        """Extract topics from text (simplified implementation)."""
        
        topic_keywords = {
            "technology": ["tech", "software", "programming", "code", "development"],
            "business": ["business", "marketing", "sales", "strategy", "revenue"],
            "science": ["science", "research", "experiment", "data", "analysis"],
            "health": ["health", "medical", "fitness", "wellness", "nutrition"],
            "education": ["education", "learning", "teaching", "school", "training"]
        }
        
        text_lower = text.lower()
        found_topics = []
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                found_topics.append(topic)
        
        return found_topics
    
    def get_personalization_context(self, user_id: str) -> str:
        """Generate context string for personalizing responses."""
        
        prefs = self.get_user_preferences(user_id)
        
        context_parts = []
        
        # Communication style guidance
        context_parts.append(f"Communication style: {prefs.communication_style}")
        context_parts.append(f"Response length: {prefs.response_length}")
        context_parts.append(f"Formality: {prefs.formality_level}")
        
        # Technical level
        context_parts.append(f"Technical level: {prefs.expertise_level}")
        
        # Preferences
        if prefs.preferred_topics:
            context_parts.append(f"Interested in: {', '.join(prefs.preferred_topics)}")
        
        if prefs.avoided_topics:
            context_parts.append(f"Avoid topics: {', '.join(prefs.avoided_topics)}")
        
        # Special requests
        if prefs.wants_examples:
            context_parts.append("Include examples when helpful")
        
        if prefs.wants_step_by_step:
            context_parts.append("Provide step-by-step instructions")
        
        return " | ".join(context_parts)

# Initialize preference system
store = InMemoryStore()
preference_manager = PreferenceManager(store)

# Tools for preference management
@tool
def update_user_preferences(
    preference_updates: Dict[str, Any],
    runtime: ToolRuntime
) -> str:
    """Update user preferences based on explicit user requests."""
    
    # Get user ID from context (would be set when agent is invoked)
    user_id = getattr(runtime.context, 'user_id', 'default_user')
    
    updated_prefs = preference_manager.update_preferences(
        user_id, 
        preference_updates,
        source="explicit"
    )
    
    return f"Updated preferences: {preference_updates}"

@tool
def get_user_preferences(runtime: ToolRuntime) -> str:
    """Get current user preferences."""
    
    user_id = getattr(runtime.context, 'user_id', 'default_user')
    prefs = preference_manager.get_user_preferences(user_id)
    
    # Return formatted preferences
    return f"""
    Communication Style: {prefs.communication_style}
    Response Length: {prefs.response_length}
    Expertise Level: {prefs.expertise_level}
    Preferred Topics: {', '.join(prefs.preferred_topics) if prefs.preferred_topics else 'None'}
    Wants Examples: {prefs.wants_examples}
    Wants Step-by-step: {prefs.wants_step_by_step}
    """

@tool
def learn_from_feedback(
    feedback_rating: int,
    feedback_text: str = "",
    runtime: ToolRuntime = None  
) -> str:
    """Learn from user feedback to improve preferences."""
    
    user_id = getattr(runtime.context, 'user_id', 'default_user')
    
    # Get recent interaction context
    recent_messages = runtime.state.get("messages", [])[-4:]  # Last 4 messages
    
    interaction_data = {
        "user_message": "",
        "agent_response": "",
        "feedback": {"rating": feedback_rating, "text": feedback_text}
    }
    
    # Extract user and agent messages
    for msg in recent_messages:
        if msg.type == "human":
            interaction_data["user_message"] = str(msg.content)
        elif msg.type == "ai":
            interaction_data["agent_response"] = str(msg.content)
    
    # Learn from the interaction
    updated_prefs = preference_manager.learn_from_interaction(user_id, interaction_data)
    
    return f"Thank you for the feedback! I've updated my understanding of your preferences."
```

#### Knowledge Management System

**Personal Knowledge Base**
```python
class PersonalKnowledgeBase:
    """Manage personal knowledge and facts about users."""
    
    def __init__(self, store):
        self.store = store
        self.facts_namespace = ("knowledge", "facts")
        self.relationships_namespace = ("knowledge", "relationships") 
        self.memories_namespace = ("knowledge", "memories")
    
    def store_fact(
        self,
        user_id: str,
        fact_type: str,
        fact_data: Dict[str, Any],
        confidence: float = 1.0
    ) -> str:
        """Store a fact about the user."""
        
        fact_key = f"{user_id}_{fact_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        fact_record = {
            "user_id": user_id,
            "type": fact_type,
            "data": fact_data,
            "confidence": confidence,
            "created_at": datetime.now().isoformat(),
            "last_confirmed": datetime.now().isoformat(),
            "confirmation_count": 1
        }
        
        self.store.put(self.facts_namespace, fact_key, fact_record)
        return fact_key
    
    def get_user_facts(
        self,
        user_id: str,
        fact_type: Optional[str] = None,
        min_confidence: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Retrieve facts about a user."""
        
        # Search for facts about this user
        search_results = self.store.search(
            self.facts_namespace,
            query=user_id,
            limit=50
        )
        
        facts = []
        for result in search_results:
            fact = result.value
            
            # Filter by user_id (in case search returns false positives)
            if fact.get("user_id") != user_id:
                continue
            
            # Filter by fact type if specified
            if fact_type and fact.get("type") != fact_type:
                continue
            
            # Filter by confidence
            if fact.get("confidence", 0) < min_confidence:
                continue
            
            facts.append(fact)
        
        # Sort by confidence and recency
        facts.sort(key=lambda x: (x.get("confidence", 0), x.get("created_at", "")), reverse=True)
        
        return facts
    
    def update_fact_confidence(
        self,
        fact_key: str,
        confidence_change: float,
        confirmed: bool = True
    ) -> None:
        """Update confidence in a fact based on new evidence."""
        
        fact_record = self.store.get(self.facts_namespace, fact_key)
        
        if fact_record:
            fact_data = fact_record.value
            
            # Update confidence
            current_confidence = fact_data.get("confidence", 0.5)
            new_confidence = max(0.0, min(1.0, current_confidence + confidence_change))
            fact_data["confidence"] = new_confidence
            
            # Update confirmation tracking
            if confirmed:
                fact_data["last_confirmed"] = datetime.now().isoformat()
                fact_data["confirmation_count"] = fact_data.get("confirmation_count", 0) + 1
            
            # Save updated fact
            self.store.put(self.facts_namespace, fact_key, fact_data)
    
    def store_memory(
        self,
        user_id: str,
        memory_type: str,
        memory_data: Dict[str, Any],
        importance: float = 0.5
    ) -> str:
        """Store a memory/experience about the user."""
        
        memory_key = f"{user_id}_memory_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        memory_record = {
            "user_id": user_id,
            "type": memory_type,
            "data": memory_data,
            "importance": importance,
            "created_at": datetime.now().isoformat(),
            "access_count": 0,
            "last_accessed": None
        }
        
        self.store.put(self.memories_namespace, memory_key, memory_record)
        return memory_key
    
    def get_relevant_memories(
        self,
        user_id: str,
        context: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Get memories relevant to current context."""
        
        # Search for memories about this user
        search_results = self.store.search(
            self.memories_namespace,
            query=f"{user_id} {context}",
            limit=limit * 2  # Get more than needed for filtering
        )
        
        memories = []
        for result in search_results:
            memory = result.value
            
            if memory.get("user_id") == user_id:
                # Update access tracking
                memory["access_count"] = memory.get("access_count", 0) + 1
                memory["last_accessed"] = datetime.now().isoformat()
                
                # Re-store with updated access info
                
