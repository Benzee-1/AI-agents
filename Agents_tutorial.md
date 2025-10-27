Great! Based on your working Azure agent, here are the examples you requested:

## 1. Agent with Custom Tools

```python
# azure_custom_tools.py
from langchain.agents import create_agent
from langchain_openai import AzureChatOpenAI
from langchain.tools import tool
import os
from dotenv import load_dotenv
import json
import requests
from datetime import datetime

load_dotenv()

@tool
def weather_info(city: str) -> str:
    """Get weather information for a city (mock function)"""
    # Mock weather data - in real app, you'd call a weather API
    weather_data = {
        "New York": "Sunny, 22°C",
        "London": "Cloudy, 15°C", 
        "Tokyo": "Rainy, 18°C",
        "Paris": "Partly cloudy, 20°C"
    }
    return weather_data.get(city, f"Weather data not available for {city}")

@tool
def currency_converter(amount: float, from_currency: str, to_currency: str) -> str:
    """Convert currency amounts (mock conversion rates)"""
    # Mock exchange rates - in real app, you'd call a currency API
    rates = {
        ("USD", "EUR"): 0.85,
        ("EUR", "USD"): 1.18,
        ("USD", "GBP"): 0.75,
        ("GBP", "USD"): 1.33,
        ("USD", "JPY"): 110.0,
        ("JPY", "USD"): 0.009
    }
    
    rate = rates.get((from_currency.upper(), to_currency.upper()))
    if rate:
        converted = amount * rate
        return f"{amount} {from_currency} = {converted:.2f} {to_currency}"
    else:
        return f"Conversion rate not available for {from_currency} to {to_currency}"

@tool
def file_manager(action: str, filename: str, content: str = "") -> str:
    """Manage files: create, read, or list files"""
    try:
        if action == "create":
            with open(filename, 'w') as f:
                f.write(content)
            return f"File '{filename}' created successfully"
        
        elif action == "read":
            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    content = f.read()
                return f"Content of '{filename}':\n{content}"
            else:
                return f"File '{filename}' not found"
        
        elif action == "list":
            files = [f for f in os.listdir('.') if os.path.isfile(f)]
            return f"Files in current directory: {', '.join(files)}"
        
        else:
            return "Invalid action. Use: create, read, or list"
            
    except Exception as e:
        return f"File operation error: {e}"

@tool
def system_info() -> str:
    """Get system information"""
    import platform
    import psutil
    
    info = {
        "OS": platform.system(),
        "OS Version": platform.version(),
        "Architecture": platform.architecture()[0],
        "CPU Count": psutil.cpu_count(),
        "Memory": f"{psutil.virtual_memory().total // (1024**3)} GB"
    }
    
    return json.dumps(info, indent=2)

# Create Azure OpenAI model
llm = AzureChatOpenAI(
    azure_deployment="gpt-4.1",
    api_version="2024-02-01",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    temperature=0.3,
    max_tokens=800
)

# Create agent with custom tools
agent = create_agent(
    model=llm,
    tools=[weather_info, currency_converter, file_manager, system_info],
    system_prompt="""You are a versatile assistant with access to multiple tools:
    - Weather information for major cities
    - Currency conversion capabilities
    - File management operations
    - System information retrieval
    
    Use these tools appropriately to help users with their requests."""
)

def chat_with_agent(user_message: str):
    inputs = {"messages": [{"role": "user", "content": user_message}]}
    
    try:
        result = agent.invoke(inputs)
        if 'messages' in result:
            for msg in reversed(result['messages']):
                if hasattr(msg, 'content') and hasattr(msg, 'type'):
                    if msg.type == 'ai' or 'ai' in str(msg.type).lower():
                        return msg.content
        return "No response received"
    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    test_queries = [
        "What's the weather like in New York?",
        "Convert 100 USD to EUR",
        "Create a file called 'test.txt' with content 'Hello World'",
        "Read the test.txt file",
        "Show me system information"
    ]
    
    for query in test_queries:
        print(f"\nUser: {query}")
        response = chat_with_agent(query)
        print(f"Agent: {response}")
```

## 2. ReAct Agent (Alternative Approach)

```python
# azure_react_agent.py
from langchain.agents import create_agent
from langchain_openai import AzureChatOpenAI
from langchain.tools import tool
import os
from dotenv import load_dotenv
import re

load_dotenv()

@tool
def search_knowledge(query: str) -> str:
    """Search for information in a knowledge base (mock function)"""
    # Mock knowledge base
    knowledge = {
        "python": "Python is a high-level programming language known for its simplicity and readability.",
        "machine learning": "Machine learning is a subset of AI that enables computers to learn without explicit programming.",
        "azure": "Microsoft Azure is a cloud computing platform offering various services.",
        "openai": "OpenAI is an AI research laboratory that created GPT models.",
        "langchain": "LangChain is a framework for developing applications powered by language models."
    }
    
    # Simple search
    query_lower = query.lower()
    for key, value in knowledge.items():
        if key in query_lower:
            return f"Found information about '{key}': {value}"
    
    return f"No specific information found for '{query}'. This might require external search."

@tool
def calculate_statistics(numbers: str) -> str:
    """Calculate statistics for a comma-separated list of numbers"""
    try:
        nums = [float(x.strip()) for x in numbers.split(',')]
        
        stats = {
            "count": len(nums),
            "sum": sum(nums),
            "mean": sum(nums) / len(nums),
            "min": min(nums),
            "max": max(nums),
            "range": max(nums) - min(nums)
        }
        
        return f"""Statistics for {numbers}:
        Count: {stats['count']}
        Sum: {stats['sum']}
        Mean: {stats['mean']:.2f}
        Min: {stats['min']}
        Max: {stats['max']}
        Range: {stats['range']}"""
        
    except Exception as e:
        return f"Error calculating statistics: {e}"

@tool
def text_processor(text: str, operation: str) -> str:
    """Process text with various operations: uppercase, lowercase, reverse, word_count, char_count"""
    operations = {
        "uppercase": text.upper(),
        "lowercase": text.lower(),
        "reverse": text[::-1],
        "word_count": f"Word count: {len(text.split())}",
        "char_count": f"Character count: {len(text)}",
        "title_case": text.title()
    }
    
    result = operations.get(operation.lower())
    if result:
        return f"Operation '{operation}' on '{text}': {result}"
    else:
        return f"Invalid operation. Available: {', '.join(operations.keys())}"

# Create Azure OpenAI model with ReAct-style prompting
llm = AzureChatOpenAI(
    azure_deployment="gpt-4.1",
    api_version="2024-02-01",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    temperature=0.1,
    max_tokens=1000
)

# Create ReAct agent with reasoning-focused system prompt
agent = create_agent(
    model=llm,
    tools=[search_knowledge, calculate_statistics, text_processor],
    system_prompt="""You are a ReAct (Reasoning + Acting) agent. For each user query, you should:

    1. **Think** about what the user is asking and what tools might be needed
    2. **Act** by using appropriate tools to gather information
    3. **Observe** the results from the tools
    4. **Reason** about the results and determine if more actions are needed
    5. **Respond** with a comprehensive answer

    Always explain your reasoning process and show your step-by-step thinking.
    
    Available tools:
    - search_knowledge: Search for information about topics
    - calculate_statistics: Calculate stats for number lists
    - text_processor: Perform text operations
    
    Be thorough in your analysis and clearly explain each step."""
)

def react_chat(user_message: str):
    inputs = {"messages": [{"role": "user", "content": user_message}]}
    
    try:
        result = agent.invoke(inputs)
        if 'messages' in result:
            for msg in reversed(result['messages']):
                if hasattr(msg, 'content') and hasattr(msg, 'type'):
                    if msg.type == 'ai' or 'ai' in str(msg.type).lower():
                        return msg.content
        return "No response received"
    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    print("ReAct Agent - Reasoning and Acting")
    print("="*50)
    
    test_queries = [
        "What is machine learning and calculate statistics for these test scores: 85, 92, 78, 96, 88, 91",
        "Tell me about Python and convert 'Hello World' to uppercase",
        "Search for information about Azure and analyze the text 'Cloud Computing' by counting words and characters"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. User: {query}")
        print("-" * 60)
        response = react_chat(query)
        print(f"ReAct Agent: {response}")
        print("=" * 60)
```

## 3. Minimal Agent with Memory

```python
# azure_memory_agent.py
from langchain.agents import create_agent
from langchain_openai import AzureChatOpenAI
from langchain.tools import tool
import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

@tool
def note_taker(action: str, note: str = "", note_id: str = "") -> str:
    """Take notes: 'save' a note, 'recall' by id, or 'list' all notes"""
    # Simple in-memory storage (in real app, use persistent storage)
    if not hasattr(note_taker, 'notes'):
        note_taker.notes = {}
    
    if action == "save":
        note_id = note_id or f"note_{len(note_taker.notes) + 1}"
        note_taker.notes[note_id] = {
            "content": note,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        return f"Note saved with ID: {note_id}"
    
    elif action == "recall":
        if note_id in note_taker.notes:
            note_data = note_taker.notes[note_id]
            return f"Note {note_id}: {note_data['content']} (saved: {note_data['timestamp']})"
        else:
            return f"Note {note_id} not found"
    
    elif action == "list":
        if note_taker.notes:
            notes_list = []
            for nid, data in note_taker.notes.items():
                notes_list.append(f"{nid}: {data['content'][:50]}... ({data['timestamp']})")
            return "All notes:\n" + "\n".join(notes_list)
        else:
            return "No notes saved"
    
    else:
        return "Invalid action. Use: save, recall, or list"

@tool
def calculator_with_history(expression: str) -> str:
    """Calculate expressions and maintain a history"""
    if not hasattr(calculator_with_history, 'history'):
        calculator_with_history.history = []
    
    try:
        result = eval(expression)
        calculation = f"{expression} = {result}"
        calculator_with_history.history.append(calculation)
        
        # Keep only last 10 calculations
        if len(calculator_with_history.history) > 10:
            calculator_with_history.history = calculator_with_history.history[-10:]
        
        return f"Result: {result}\nCalculation added to history."
    except Exception as e:
        return f"Error: {e}"

@tool
def get_calculation_history() -> str:
    """Get the history of calculations"""
    if hasattr(calculator_with_history, 'history') and calculator_with_history.history:
        return "Calculation History:\n" + "\n".join(calculator_with_history.history)
    else:
        return "No calculation history available"

# Create Azure OpenAI model
llm = AzureChatOpenAI(
    azure_deployment="gpt-4.1",
    api_version="2024-02-01",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    temperature=0.3,
    max_tokens=600
)

class MemoryAgent:
    def __init__(self, agent):
        self.agent = agent
        self.conversation_memory = []
    
    def chat(self, user_message: str):
        # Add user message to memory
        self.conversation_memory.append({"role": "user", "content": user_message})
        
        # Create context with conversation history
        context_messages = self.conversation_memory.copy()
        inputs = {"messages": context_messages}
        
        try:
            result = self.agent.invoke(inputs)
            
            if 'messages' in result:
                # Find the assistant's response
                for msg in reversed(result['messages']):
                    if hasattr(msg, 'content') and hasattr(msg, 'type'):
                        if msg.type == 'ai' or 'ai' in str(msg.type).lower():
                            # Add assistant response to memory
                            self.conversation_memory.append({
                                "role": "assistant", 
                                "content": msg.content
                            })
                            
                            # Keep memory manageable (last 20 messages)
                            if len(self.conversation_memory) > 20:
                                self.conversation_memory = self.conversation_memory[-20:]
                            
                            return msg.content
            
            return "No response received"
            
        except Exception as e:
            return f"Error: {e}"
    
    def get_conversation_summary(self):
        """Get a summary of the conversation"""
        if not self.conversation_memory:
            return "No conversation history"
        
        summary = f"Conversation with {len(self.conversation_memory)} messages:\n"
        for i, msg in enumerate(self.conversation_memory[-6:], 1):  # Last 6 messages
            role = msg['role'].title()
            content = msg['content'][:100] + "..." if len(msg['content']) > 100 else msg['content']
            summary += f"{i}. {role}: {content}\n"
        
        return summary

# Create agent with memory-aware system prompt
agent = create_agent(
    model=llm,
    tools=[note_taker, calculator_with_history, get_calculation_history],
    system_prompt="""You are a helpful assistant with memory capabilities. You can:
    1. Save and recall notes for the user
    2. Perform calculations and maintain a history
    3. Remember our conversation context
    
    Always reference previous parts of our conversation when relevant and help maintain continuity."""
)

if __name__ == "__main__":
    memory_agent = MemoryAgent(agent)
    
    print("Memory Agent - Maintains conversation history and notes")
    print("="*60)
    
    # Simulate a conversation with memory
    conversation_flow = [
        "Save a note: 'Meeting with client tomorrow at 2 PM'",
        "Calculate 250 * 1.2 for the project budget",
        "Save another note with ID 'budget': '300 is the total project cost'",
        "What calculations have I done so far?",
        "What notes do I have saved?",
        "Recall the budget note",
        "Calculate 300 - 50 for remaining budget after expenses"
    ]
    
    for i, message in enumerate(conversation_flow, 1):
        print(f"\n{i}. User: {message}")
        response = memory_agent.chat(message)
        print(f"   Agent: {response}")
    
    print(f"\n{'='*60}")
    print("CONVERSATION SUMMARY:")
    print(memory_agent.get_conversation_summary())
```

## 4. Simple Streaming Agent

```python
# azure_streaming_agent.py
from langchain.agents import create_agent
from langchain_openai import AzureChatOpenAI
from langchain.tools import tool
import os
from dotenv import load_dotenv
import time
import sys

load_dotenv()

@tool
def data_processor(data: str, operation: str) -> str:
    """Process data with streaming-like operations"""
    operations = {
        "analyze": f"Analyzing data: {data}\nFound {len(data.split())} words, {len(data)} characters",
        "summarize": f"Summary of '{data}': {data[:50]}..." if len(data) > 50 else f"Summary: {data}",
        "validate": f"Validation result for '{data}': {'Valid' if len(data) > 0 else 'Invalid'}",
        "transform": f"Transformed data: {data.upper().replace(' ', '_')}"
    }
    
    return operations.get(operation, f"Unknown operation: {operation}")

@tool
def progress_simulator(task: str, steps: int = 5) -> str:
    """Simulate a long-running task with progress updates"""
    result = f"Starting task: {task}\n"
    
    for i in range(1, steps + 1):
        result += f"Step {i}/{steps}: Processing...\n"
        time.sleep(0.5)  # Simulate work
    
    result += f"Task '{task}' completed successfully!"
    return result

@tool
def real_time_data() -> str:
    """Get real-time data (simulated)"""
    import random
    
    data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cpu_usage": f"{random.randint(10, 90)}%",
        "memory_usage": f"{random.randint(30, 80)}%",
        "active_connections": random.randint(50, 200),
        "status": random.choice(["healthy", "warning", "normal"])
    }
    
    return f"""Real-time System Data:
    Timestamp: {data['timestamp']}
    CPU Usage: {data['cpu_usage']}
    Memory Usage: {data['memory_usage']}
    Active Connections: {data['active_connections']}
    Status: {data['status']}"""

# Create Azure OpenAI model configured for streaming
llm = AzureChatOpenAI(
    azure_deployment="gpt-4.1",
    api_version="2024-02-01",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    temperature=0.4,
    max_tokens=800,
    streaming=True  # Enable streaming
)

# Create streaming agent
agent = create_agent(
    model=llm,
    tools=[data_processor, progress_simulator, real_time_data],
    system_prompt="""You are a streaming assistant that provides real-time responses. 
    When processing requests:
    1. Acknowledge the request immediately
    2. Use tools to gather/process data
    3. Provide step-by-step updates
    4. Give comprehensive final results
    
    Be responsive and informative throughout the process."""
)

def streaming_chat(user_message: str):
    """Chat with streaming response simulation"""
    inputs = {"messages": [{"role": "user", "content": user_message}]}
    
    print("🔄 Processing your request...", end="", flush=True)
    
    # Simulate streaming with dots
    for _ in range(3):
        time.sleep(0.5)
        print(".", end="", flush=True)
    print(" ✓")
    
    try:
        # Try streaming if available
        try:
            print("\n📡 Streaming response:")
            print("-" * 40)
            
            response_chunks = []
            for chunk in agent.stream(inputs, stream_mode="updates"):
                if chunk:
                    print("📦 Received chunk:", end=" ")
                    chunk_str = str(chunk)
                    if len(chunk_str) > 100:
                        chunk_str = chunk_str[:100] + "..."
                    print(chunk_str)
                    response_chunks.append(str(chunk))
                    time.sleep(0.1)  # Simulate streaming delay
            
            return "\n".join(response_chunks) if response_chunks else "No streaming content"
            
        except Exception as stream_error:
            print(f"Streaming failed: {stream_error}")
            print("Falling back to regular invoke...")
            
            # Fallback to regular invoke
            result = agent.invoke(inputs)
            if 'messages' in result:
                for msg in reversed(result['messages']):
                    if hasattr(msg, 'content') and hasattr(msg, 'type'):
                        if msg.type == 'ai' or 'ai' in str(msg.type).lower():
                            return msg.content
            return "No response received"
            
    except Exception as e:
        return f"Error: {e}"

def simulate_real_time_monitoring():
    """Simulate real-time monitoring with streaming updates"""
    print("\n🔴 LIVE: Real-time System Monitoring")
    print("=" * 50)
    
    for i in range(5):
        print(f"\n📊 Update {i+1}/5:")
        response = streaming_chat("Get real-time system data")
        
        # Extract just the data part for cleaner display
        if "Real-time System Data:" in response:
            data_part = response.split("Real-time System Data:")[1] if "Real-time System Data:" in response else response
            print(data_part.strip())
        else:
            print(response)
        
        if i < 4:  # Don't wait after last update
            print("⏳ Waiting for next update...")
            time.sleep(2)
    
    print("\n✅ Monitoring session completed")

if __name__ == "__main__":
    print("Streaming Agent - Real-time Processing")
    print("=" * 50)
    
    # Test streaming with different types of requests
    test_queries = [
        "Analyze this data: 'The quick brown fox jumps over the lazy dog'",
        "Simulate a data backup task with 4 steps",
        "Get current system status"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. User: {query}")
        print("=" * 60)
        response = streaming_chat(query)
        
        # Clean up response display
        if isinstance(response, str) and len(response) > 500:
            print("📝 Full Response:")
            print(response)
        else:
            print("📝 Response:", response)
        print("=" * 60)
    
    # Demonstrate real-time monitoring
    print("\n" + "="*60)
    print("BONUS: Real-time Monitoring Demo")
    simulate_real_time_monitoring()
```

## 5. Agent with VectorDB and Embedding

```python
# azure_vector_agent.py
from langchain.agents import create_agent
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.tools import tool
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import os
from dotenv import load_dotenv
import tempfile
import shutil

load_dotenv()

# Initialize Azure OpenAI Embeddings
embeddings = AzureOpenAIEmbeddings(
    azure_deployment="text-embedding-ada-002",  # Your embedding deployment name
    api_version="2024-02-01",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY")
)

# Sample knowledge base documents
sample_documents = [
    "Python is a high-level programming language created by Guido van Rossum. It emphasizes code readability and simplicity.",
    "Machine Learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.",
    "Azure OpenAI Service provides REST API access to OpenAI's powerful language models including GPT-4, GPT-3.5-turbo, and embeddings models.",
    "LangChain is a framework for developing applications powered by language models. It provides tools for prompt management, chains, and agents.",
    "Vector databases store high-dimensional vectors and enable similarity search. They are essential for RAG (Retrieval Augmented Generation) applications.",
    "Natural Language Processing (NLP) is a branch of AI that helps computers understand, interpret and manipulate human language.",
    "Docker is a containerization platform that allows developers to package applications and their dependencies into lightweight containers.",
    "Kubernetes is an open-source container orchestration platform that automates deployment, scaling, and management of containerized applications.",
    "REST APIs (Representational State Transfer) are architectural style for designing networked applications using HTTP protocols.",
    "Git is a distributed version control system that tracks changes in source code during software development."
]

# Initialize vector store
def initialize_vector_store():
    """Initialize vector store with sample documents"""
    # Create temporary directory for Chroma
    temp_dir = tempfile.mkdtemp()
    
    # Create documents
    docs = [Document(page_content=doc, metadata={"source": f"doc_{i}"}) 
            for i, doc in enumerate(sample_documents)]
    
    # Split documents (though these are already small)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20
    )
    splits = text_splitter.split_documents(docs)
    
    # Create vector store
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=temp_dir
    )
    
    return vectorstore, temp_dir

# Initialize vector store globally
try:
    vector_store, temp_directory = initialize_vector_store()
    print("✅ Vector store initialized successfully")
except Exception as e:
    print(f"❌ Vector store initialization failed: {e}")
    vector_store, temp_directory = None, None

@tool
def semantic_search(query: str, k: int = 3) -> str:
    """Search the knowledge base using semantic similarity"""
    if not vector_store:
        return "Vector store not available"
    
    try:
        # Perform similarity search
        results = vector_store.similarity_search(query, k=k)
        
        if not results:
            return f"No relevant information found for: {query}"
        
        # Format results
        search_results = f"Found {len(results)} relevant documents for '{query}':\n\n"
        for i, doc in enumerate(results, 1):
            search_results += f"{i}. {doc.page_content}\n"
            if doc.metadata:
                search_results += f"   Source: {doc.metadata.get('source', 'unknown')}\n"
            search_results += "\n"
        
        return search_results.strip()
        
    except Exception as e:
        return f"Search error: {e}"

@tool
def add_to_knowledge_base(content: str, source: str = "user_added") -> str:
    """Add new content to the knowledge base"""
    if not vector_store:
        return "Vector store not available"
    
    try:
        # Create new document
        doc = Document(page_content=content, metadata={"source": source})
        
        # Add to vector store
        vector_store.add_documents([doc])
        
        return f"Successfully added content to knowledge base: '{content[:100]}...'"
        
    except Exception as e:
        return f"Error adding to knowledge base: {e}"

@tool
def similarity_comparison(text1: str, text2: str) -> str:
    """Compare similarity between two texts using embeddings"""
    if not embeddings:
        return "Embeddings not available"
    
    try:
        # Get embeddings for both texts
        embedding1 = embeddings.embed_query(text1)
        embedding2 = embeddings.embed_query(text2)
        
        # Calculate cosine similarity
        from numpy import dot
        from numpy.linalg import norm
        
        similarity = dot(embedding1, embedding2) / (norm(embedding1) * norm(embedding2))
        
        result = f"""Similarity Analysis:
        Text 1: "{text1}"
        Text 2: "{text2}"
        Similarity Score: {similarity:.4f}
        
        Interpretation:
        - 1.0 = Identical meaning
        - 0.8+ = Very similar
        - 0.6+ = Moderately similar  
        - 0.4+ = Somewhat similar
        - <0.4 = Not very similar"""
        
        return result
        
    except Exception as e:
        return f"Similarity comparison error: {e}"

@tool
def knowledge_base_stats() -> str:
    """Get statistics about the knowledge base"""
    if not vector_store:
        return "Vector store not available"
    
    try:
        # Get collection info
        collection = vector_store._collection
        count = collection.count()
        
        # Sample some documents to show topics
        sample_docs = vector_store.similarity_search("", k=5)  # Get any 5 docs
        
        stats = f"""Knowledge Base Statistics:
        Total Documents: {count}
        
        Sample Topics:"""
        
        for i, doc in enumerate(sample_docs[:3], 1):
            preview = doc.page_content[:80] + "..." if len(doc.page_content) > 80 else doc.page_content
            stats += f"\n{i}. {preview}"
        
        return stats
        
    except Exception as e:
        return f"Stats error: {e}"

# Create Azure OpenAI model
llm = AzureChatOpenAI(
    azure_deployment="gpt-4.1",
    api_version="2024-02-01",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    temperature=0.2,
    max_tokens=1000
)

# Create vector-powered agent
agent = create_agent(
    model=llm,
    tools=[semantic_search, add_to_knowledge_base, similarity_comparison, knowledge_base_stats],
    system_prompt="""You are a knowledge assistant powered by vector search and embeddings. You can:

    1. Search the knowledge base using semantic similarity
    2. Add new information to the knowledge base
    3. Compare similarity between texts
    4. Provide statistics about the knowledge base
    
    When users ask questions:
    - First search the knowledge base for relevant information
    - Combine search results with your knowledge to provide comprehensive answers
    - Suggest adding new information if it's not in the knowledge base
    
    Always be helpful and explain how you're using the vector search capabilities."""
)

def vector_chat(user_message: str):
    inputs = {"messages": [{"role": "user", "content": user_message}]}
    
    try:
        result = agent.invoke(inputs)
        if 'messages' in result:
            for msg in reversed(result['messages']):
                if hasattr(msg, 'content') and hasattr(msg, 'type'):
                    if msg.type == 'ai' or 'ai' in str(msg.type).lower():
                        return msg.content
        return "No response received"
    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    print("Vector-Powered Agent with Embeddings")
    print("=" * 60)
    
    if not vector_store:
        print("❌ Vector store not available. Please check your embedding configuration.")
        exit(1)
    
    try:
        test_queries = [
            "What do you know about Python programming?",
            "Tell me about machine learning and find similar information",
            "Compare the similarity between 'artificial intelligence' and 'machine learning'",
            "Add this information: 'FastAPI is a modern web framework for building APIs with Python'",
            "Search for information about web frameworks",
            "What are the statistics of your knowledge base?"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n{i}. User: {query}")
            print("-" * 50)
            response = vector_chat(query)
            print(f"Vector Agent: {response}")
            print("=" * 60)
    
    finally:
        # Cleanup temporary directory
        if temp_directory and os.path.exists(temp_directory):
            shutil.rmtree(temp_directory)
            print(f"\n🧹 Cleaned up temporary directory: {temp_directory}")
```

## Installation Requirements

For all these examples, you'll need:

```bash
pip install langchain==1.0.1 langchain-openai langchain-community python-dotenv chromadb numpy psutil
```

## Environment Variables

Make sure your `.env` file includes:

```env
AZURE_OPENAI_API_KEY=sk-E62CUIjZGalQ0ReGkB6J4Q
AZURE_OPENAI_ENDPOINT=https://genai-gateway.azure-api.net/
AZURE_OPENAI_DEPLOYMENT=gpt-4.1
AZURE_OPENAI_API_VERSION=2024-02-01
```

Each example builds on your working base code and demonstrates different advanced agent capabilities! Start with the Custom Tools example and work your way through them.
