# Apache Airflow 2.10 Logs Analysis with AI-Powered Detection

I'll provide you with a comprehensive solution to analyze Apache Airflow logs for errors, warnings, anomalies, and configuration issues using AI agents.

## Prerequisites

### 1. Environment Setup and Libraries Installation

```bash
# Run this in your terminal or Jupyter cell with !
pip install pandas numpy matplotlib seaborn plotly
pip install openai python-dotenv
pip install scikit-learn textstat
pip install asyncio aiofiles watchdog
pip install regex nltk
pip install azure-ai-textanalytics (optional)
```

### 2. Initial Setup and Imports

```python
# Cell 1: Import necessary libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import re
import json
import asyncio
import logging
from typing import List, Dict, Tuple, Optional, Any
import warnings
from pathlib import Path
import glob
from collections import defaultdict, Counter
import time

# OpenAI and Azure
import openai
from openai import AzureOpenAI
from dotenv import load_dotenv

# Machine Learning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

# Text processing
import textstat
import string

warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

print("✅ All libraries imported successfully")
```

### 3. Azure OpenAI Configuration

```python
# Cell 2: Azure OpenAI Setup
load_dotenv()

# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT', 'https://genai-gateway.azure-api.net/')
AZURE_OPENAI_API_KEY = os.getenv('AZURE_OPENAI_API_KEY')
AZURE_OPENAI_API_VERSION = os.getenv('AZURE_OPENAI_API_VERSION', '2024-02-15-preview')
AZURE_OPENAI_DEPLOYMENT = os.getenv('AZURE_OPENAI_DEPLOYMENT', 'gpt-35-turbo')

# Initialize Azure OpenAI client
client = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION
)

def test_azure_openai_connection():
    """Test Azure OpenAI connection"""
    try:
        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=[{"role": "user", "content": "Test connection"}],
            max_tokens=10
        )
        print("✅ Azure OpenAI connection successful")
        return True
    except Exception as e:
        print(f"❌ Azure OpenAI connection failed: {e}")
        return False

test_azure_openai_connection()
```

## Step 1: Airflow Log Parser and Reader

```python
# Cell 3: Airflow Log Parser
class AirflowLogParser:
    def __init__(self, airflow_home="/opt/GOLD/airflow"):
        self.airflow_home = airflow_home
        self.log_patterns = {
            'scheduler_log': re.compile(r'^\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}\+\d{4})\]\s+\{[^}]*\}\s+(\w+)\s+-\s+(.*)$'),
            'task_log': re.compile(r'^\[(\d{4}-\d{2}-\d{2}),\s+(\d{2}:\d{2}:\d{2})\s+\w+\]\s+\{[^}]*\}\s+(\w+)\s+-\s+(.*)$'),
            'webserver_log': re.compile(r'^\[(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2}:\d{2}),\d+\]\s+(\w+)\s+in\s+\w+:\s+(.*)$'),
            'error_pattern': re.compile(r'(ERROR|CRITICAL|FATAL)', re.IGNORECASE),
            'warning_pattern': re.compile(r'WARNING', re.IGNORECASE),
            'exception_pattern': re.compile(r'(Exception|Error|Traceback)', re.IGNORECASE),
            'config_pattern': re.compile(r'(configuration|config|setting)', re.IGNORECASE)
        }
        
    def get_log_files(self, start_date: str, end_date: str = None) -> List[str]:
        """Get list of log files for given date range"""
        log_files = []
        
        if end_date is None:
            end_date = start_date
            
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        current_dt = start_dt
        while current_dt <= end_dt:
            date_str = current_dt.strftime('%Y-%m-%d')
            
            # Scheduler logs
            scheduler_path = f"{self.airflow_home}/logs/scheduler/{date_str}"
            if os.path.exists(scheduler_path):
                scheduler_files = glob.glob(f"{scheduler_path}/*.log")
                log_files.extend(scheduler_files)
            
            current_dt += timedelta(days=1)
        
        return log_files
    
    def parse_log_line(self, line: str, log_type: str = "scheduler") -> Dict:
        """Parse a single log line"""
        line = line.strip()
        if not line:
            return None
            
        parsed_log = {
            'raw_line': line,
            'timestamp': None,
            'level': 'UNKNOWN',
            'message': line,
            'log_type': log_type,
            'has_error': bool(self.log_patterns['error_pattern'].search(line)),
            'has_warning': bool(self.log_patterns['warning_pattern'].search(line)),
            'has_exception': bool(self.log_patterns['exception_pattern'].search(line)),
            'has_config_issue': bool(self.log_patterns['config_pattern'].search(line))
        }
        
        # Try to parse with scheduler pattern
        if log_type == "scheduler":
            match = self.log_patterns['scheduler_log'].match(line)
            if match:
                parsed_log['timestamp'] = match.group(1)
                parsed_log['level'] = match.group(2)
                parsed_log['message'] = match.group(3)
        
        # Extract additional information
        parsed_log['dag_id'] = self._extract_dag_id(line)
        parsed_log['task_id'] = self._extract_task_id(line)
        parsed_log['message_length'] = len(parsed_log['message'])
        parsed_log['word_count'] = len(parsed_log['message'].split())
        
        return parsed_log
    
    def _extract_dag_id(self, line: str) -> Optional[str]:
        """Extract DAG ID from log line"""
        dag_pattern = re.compile(r'dag_id[\'\":\s=]+([a-zA-Z0-9_\-\.]+)')
        match = dag_pattern.search(line)
        return match.group(1) if match else None
    
    def _extract_task_id(self, line: str) -> Optional[str]:
        """Extract Task ID from log line"""
        task_pattern = re.compile(r'task_id[\'\":\s=]+([a-zA-Z0-9_\-\.]+)')
        match = task_pattern.search(line)
        return match.group(1) if match else None
    
    def read_log_files(self, log_files: List[str]) -> pd.DataFrame:
        """Read and parse multiple log files"""
        all_logs = []
        
        for log_file in log_files:
            try:
                print(f"📖 Reading log file: {log_file}")
                with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                    for line_num, line in enumerate(f, 1):
                        parsed_log = self.parse_log_line(line, "scheduler")
                        if parsed_log:
                            parsed_log['file_path'] = log_file
                            parsed_log['line_number'] = line_num
                            all_logs.append(parsed_log)
                            
            except Exception as e:
                logger.error(f"Error reading file {log_file}: {e}")
        
        df = pd.DataFrame(all_logs)
        if not df.empty:
            # Convert timestamp if possible
            try:
                df['parsed_timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            except:
                df['parsed_timestamp'] = None
                
        print(f"✅ Parsed {len(df)} log entries from {len(log_files)} files")
        return df

# Initialize the parser
airflow_parser = AirflowLogParser()

# Test with current date
current_date = datetime.now().strftime('%Y-%m-%d')
print(f"🔍 Looking for logs on date: {current_date}")

# Get log files for today (adjust date as needed)
log_files = airflow_parser.get_log_files(current_date)
print(f"Found {len(log_files)} log files")

# If no files found for today, try yesterday
if not log_files:
    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    print(f"🔍 Trying yesterday's logs: {yesterday}")
    log_files = airflow_parser.get_log_files(yesterday)
    print(f"Found {len(log_files)} log files")
```

## Step 2: Log Analysis and Feature Extraction

```python
# Cell 4: Advanced Log Analysis
class AirflowLogAnalyzer:
    def __init__(self, df_logs: pd.DataFrame):
        self.df = df_logs.copy()
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.scaler = StandardScaler()
        
    def extract_features(self) -> pd.DataFrame:
        """Extract comprehensive features from logs"""
        if self.df.empty:
            print("⚠️ No log data available for analysis")
            return pd.DataFrame()
            
        print("🔧 Extracting features from logs...")
        
        # Time-based features
        if 'parsed_timestamp' in self.df.columns and self.df['parsed_timestamp'].notna().any():
            self.df['hour'] = self.df['parsed_timestamp'].dt.hour
            self.df['day_of_week'] = self.df['parsed_timestamp'].dt.dayofweek
            self.df['is_weekend'] = self.df['day_of_week'].isin([5, 6])
            self.df['is_night'] = self.df['hour'].isin(range(0, 6))
        else:
            # Default values if timestamp parsing failed
            self.df['hour'] = 12
            self.df['day_of_week'] = 1
            self.df['is_weekend'] = False
            self.df['is_night'] = False
        
        # Text-based features
        self.df['message_length'] = self.df['message'].str.len()
        self.df['word_count'] = self.df['message'].str.split().str.len()
        self.df['uppercase_ratio'] = self.df['message'].apply(
            lambda x: sum(1 for c in x if c.isupper()) / len(x) if len(x) > 0 else 0
        )
        self.df['digit_ratio'] = self.df['message'].apply(
            lambda x: sum(1 for c in x if c.isdigit()) / len(x) if len(x) > 0 else 0
        )
        
        # Log level encoding
        level_encoding = {'DEBUG': 1, 'INFO': 2, 'WARNING': 3, 'ERROR': 4, 'CRITICAL': 5, 'UNKNOWN': 0}
        self.df['level_numeric'] = self.df['level'].map(level_encoding).fillna(0)
        
        # Keyword-based features
        self.df['contains_sql'] = self.df['message'].str.contains(r'(SELECT|INSERT|UPDATE|DELETE|CREATE|DROP)', case=False, na=False)
        self.df['contains_connection'] = self.df['message'].str.contains(r'(connection|connect|disconnect)', case=False, na=False)
        self.df['contains_timeout'] = self.df['message'].str.contains(r'(timeout|timed out)', case=False, na=False)
        self.df['contains_memory'] = self.df['message'].str.contains(r'(memory|out of memory|oom)', case=False, na=False)
        self.df['contains_permission'] = self.df['message'].str.contains(r'(permission|denied|unauthorized)', case=False, na=False)
        
        # Calculate anomaly score based on multiple factors
        self.df['anomaly_score'] = (
            self.df['has_error'].astype(int) * 0.4 +
            self.df['has_exception'].astype(int) * 0.3 +
            self.df['has_warning'].astype(int) * 0.2 +
            (self.df['level_numeric'] >= 3).astype(int) * 0.1
        )
        
        print(f"✅ Feature extraction completed. Shape: {self.df.shape}")
        return self.df
    
    def detect_anomalies(self) -> pd.DataFrame:
        """Detect anomalies using multiple methods"""
        if self.df.empty:
            return self.df
            
        print("🔍 Detecting anomalies...")
        
        # Prepare features for ML
        feature_columns = [
            'message_length', 'word_count', 'uppercase_ratio', 'digit_ratio',
            'level_numeric', 'hour', 'day_of_week'
        ]
        
        # Handle missing values
        X = self.df[feature_columns].fillna(0)
        
        # Isolation Forest
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        self.df['is_anomaly_iso'] = (iso_forest.fit_predict(X) == -1)
        
        # Rule-based anomaly detection
        self.df['is_anomaly_rules'] = (
            self.df['has_error'] |
            self.df['has_exception'] |
            (self.df['level_numeric'] >= 4) |
            (self.df['message_length'] > self.df['message_length'].quantile(0.95)) |
            self.df['contains_timeout'] |
            self.df['contains_memory'] |
            self.df['contains_permission']
        )
        
        # Combined anomaly detection
        self.df['is_anomaly'] = self.df['is_anomaly_iso'] | self.df['is_anomaly_rules']
        
        anomaly_count = self.df['is_anomaly'].sum()
        print(f"✅ Detected {anomaly_count} anomalies out of {len(self.df)} logs")
        
        return self.df
    
    def categorize_issues(self) -> Dict[str, pd.DataFrame]:
        """Categorize different types of issues"""
        categories = {}
        
        # Errors
        categories['errors'] = self.df[self.df['has_error']].copy()
        
        # Warnings
        categories['warnings'] = self.df[self.df['has_warning']].copy()
        
        # Configuration issues
        categories['config_issues'] = self.df[
            self.df['has_config_issue'] |
            self.df['message'].str.contains(r'(config|configuration|setting|parameter)', case=False, na=False)
        ].copy()
        
        # Performance issues
        categories['performance_issues'] = self.df[
            self.df['contains_timeout'] |
            self.df['contains_memory'] |
            self.df['message'].str.contains(r'(slow|performance|latency)', case=False, na=False)
        ].copy()
        
        # Connection issues
        categories['connection_issues'] = self.df[
            self.df['contains_connection'] |
            self.df['message'].str.contains(r'(refused|unreachable|network)', case=False, na=False)
        ].copy()
        
        # Task failures
        categories['task_failures'] = self.df[
            self.df['message'].str.contains(r'(failed|failure|unsuccessful)', case=False, na=False)
        ].copy()
        
        print("📊 Issue categorization completed:")
        for category, df_cat in categories.items():
            print(f"  - {category}: {len(df_cat)} entries")
        
        return categories

# Create sample log data if no real logs are found
def create_sample_airflow_logs() -> pd.DataFrame:
    """Create sample Airflow logs for demonstration"""
    sample_logs = [
        {
            'timestamp': '2024-01-15T10:30:00.123+0000',
            'level': 'INFO',
            'message': 'DAG my_dag_1 is now running task extract_data',
            'dag_id': 'my_dag_1',
            'task_id': 'extract_data'
        },
        {
            'timestamp': '2024-01-15T10:31:00.456+0000',
            'level': 'ERROR',
            'message': 'Task failed with exception: Connection timeout to database server',
            'dag_id': 'my_dag_1',
            'task_id': 'extract_data'
        },
        {
            'timestamp': '2024-01-15T10:32:00.789+0000',
            'level': 'WARNING',
            'message': 'Configuration parameter max_connections not set, using default value',
            'dag_id': None,
            'task_id': None
        },
        {
            'timestamp': '2024-01-15T10:33:00.012+0000',
            'level': 'CRITICAL',
            'message': 'Out of memory error in task transform_data. Available: 512MB, Required: 2GB',
            'dag_id': 'my_dag_2',
            'task_id': 'transform_data'
        },
        {
            'timestamp': '2024-01-15T10:34:00.345+0000',
            'level': 'INFO',
            'message': 'Successfully completed task load_data in 45 seconds',
            'dag_id': 'my_dag_1',
            'task_id': 'load_data'
        }
    ]
    
    # Expand sample logs
    expanded_logs = []
    for i in range(100):  # Create 100 sample logs
        base_log = sample_logs[i % len(sample_logs)].copy()
        base_log['raw_line'] = f"[{base_log['timestamp']}] {base_log['level']} - {base_log['message']}"
        base_log['has_error'] = 'ERROR' in base_log['level'] or 'error' in base_log['message'].lower()
        base_log['has_warning'] = 'WARNING' in base_log['level']
        base_log['has_exception'] = 'exception' in base_log['message'].lower() or 'failed' in base_log['message'].lower()
        base_log['has_config_issue'] = 'configuration' in base_log['message'].lower() or 'config' in base_log['message'].lower()
        base_log['file_path'] = f'/opt/GOLD/airflow/logs/scheduler/2024-01-15/scheduler_{i//20}.log'
        base_log['line_number'] = (i % 20) + 1
        base_log['log_type'] = 'scheduler'
        expanded_logs.append(base_log)
    
    df = pd.DataFrame(expanded_logs)
    df['parsed_timestamp'] = pd.to_datetime(df['timestamp'])
    return df

# Try to read real logs, fallback to sample data
try:
    if log_files:
        df_logs = airflow_parser.read_log_files(log_files)
    else:
        print("📝 No log files found, using sample data for demonstration")
        df_logs = create_sample_airflow_logs()
        
    # Analyze logs
    analyzer = AirflowLogAnalyzer(df_logs)
    df_analyzed = analyzer.extract_features()
    df_analyzed = analyzer.detect_anomalies()
    issue_categories = analyzer.categorize_issues()
    
    print(f"\n📈 Analysis Summary:")
    print(f"Total logs analyzed: {len(df_analyzed)}")
    print(f"Anomalies detected: {df_analyzed['is_anomaly'].sum()}")
    print(f"Errors: {df_analyzed['has_error'].sum()}")
    print(f"Warnings: {df_analyzed['has_warning'].sum()}")
    
except Exception as e:
    print(f"❌ Error during log analysis: {e}")
    df_analyzed = pd.DataFrame()
    issue_categories = {}
```

## Step 3: AI-Powered Log Analysis Agent

```python
# Cell 5: AI Agent for Intelligent Log Analysis
class AirflowAILogAgent:
    def __init__(self, azure_client, deployment_name):
        self.client = azure_client
        self.deployment_name = deployment_name
        self.analysis_cache = {}
        
    async def analyze_log_entry(self, log_entry: Dict, context: str = "") -> Dict:
        """Analyze a single log entry using AI"""
        log_text = log_entry.get('message', '')
        log_level = log_entry.get('level', 'UNKNOWN')
        
        # Create cache key
        cache_key = f"{log_level}_{hash(log_text)}"
        if cache_key in self.analysis_cache:
            return self.analysis_cache[cache_key]
        
        prompt = f"""
        You are an expert Apache Airflow administrator analyzing log entries for issues.
        
        Log Details:
        - Level: {log_level}
        - Message: {log_text}
        - DAG ID: {log_entry.get('dag_id', 'N/A')}
        - Task ID: {log_entry.get('task_id', 'N/A')}
        - Timestamp: {log_entry.get('timestamp', 'N/A')}
        
        Context: {context}
        
        Please analyze this Airflow log entry and provide:
        1. Severity level (LOW, MEDIUM, HIGH, CRITICAL)
        2. Issue category (ERROR, WARNING, CONFIG, PERFORMANCE, CONNECTION, TASK_FAILURE, INFO)
        3. Root cause analysis
        4. Recommended actions
        5. Whether this indicates a systemic issue
        
        Focus on Airflow-specific issues like:
        - DAG parsing errors
        - Task execution failures
        - Scheduler issues
        - Database connectivity problems
        - Resource constraints
        - Configuration problems
        
        Respond in JSON format:
        {{
            "severity": "LEVEL",
            "category": "CATEGORY",
            "root_cause": "ANALYSIS",
            "recommendations": ["ACTION1", "ACTION2"],
            "systemic_issue": true/false,
            "urgency_score": 1-10
        }}
        """
        
        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.deployment_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=600,
                temperature=0.1
            )
            
            content = response.choices[0].message.content.strip()
            
            # Clean JSON response
            if content.startswith('```json'):
                content = content[7:-3]
            elif content.startswith('```'):
                content = content[3:-3]
            
            analysis = json.loads(content)
            
            # Cache the result
            self.analysis_cache[cache_key] = analysis
            
            return analysis
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            return self._create_fallback_analysis(log_entry)
        except Exception as e:
            logger.error(f"Error in AI analysis: {e}")
            return self._create_fallback_analysis(log_entry)
    
    def _create_fallback_analysis(self, log_entry: Dict) -> Dict:
        """Create fallback analysis when AI fails"""
        level = log_entry.get('level', 'UNKNOWN')
        message = log_entry.get('message', '').lower()
        
        # Rule-based fallback
        if level in ['ERROR', 'CRITICAL']:
            severity = 'HIGH' if level == 'ERROR' else 'CRITICAL'
            urgency = 8 if level == 'ERROR' else 10
        elif level == 'WARNING':
            severity = 'MEDIUM'
            urgency = 5
        else:
            severity = 'LOW'
            urgency = 2
        
        # Determine category
        category = 'INFO'
        if 'error' in message or 'failed' in message:
            category = 'ERROR'
        elif 'warning' in message:
            category = 'WARNING'
        elif 'config' in message:
            category = 'CONFIG'
        elif 'timeout' in message or 'slow' in message:
            category = 'PERFORMANCE'
        elif 'connection' in message:
            category = 'CONNECTION'
        
        return {
            "severity": severity,
            "category": category,
            "root_cause": "Automated analysis based on log level and keywords",
            "recommendations": ["Review log details", "Check system status"],
            "systemic_issue": False,
            "urgency_score": urgency
        }
    
    async def analyze_pattern_trends(self, df: pd.DataFrame) -> Dict:
        """Analyze patterns and trends in logs"""
        if df.empty:
            return {"error": "No data to analyze"}
        
        # Aggregate statistics
        stats = {
            "total_logs": len(df),
            "error_rate": df['has_error'].mean() if 'has_error' in df.columns else 0,
            "warning_rate": df['has_warning'].mean() if 'has_warning' in df.columns else 0,
            "most_common_errors": [],
            "peak_activity_hours": [],
            "problematic_dags": []
        }
        
        # Most common error messages
        if 'has_error' in df.columns:
            error_logs = df[df['has_error']]['message']
            if not error_logs.empty:
                error_counts = error_logs.value_counts().head(5)
                stats["most_common_errors"] = [
                    {"message": msg, "count": int(count)} 
                    for msg, count in error_counts.items()
                ]
        
        # Peak activity hours
        if 'hour' in df.columns:
            hourly_counts = df.groupby('hour').size()
            peak_hours = hourly_counts.nlargest(3)
            stats["peak_activity_hours"] = [
                {"hour": int(hour), "log_count": int(count)} 
                for hour, count in peak_hours.items()
            ]
        
        # Problematic DAGs
        if 'dag_id' in df.columns:
            dag_errors = df[df['has_error']].groupby('dag_id').size().sort_values(ascending=False).head(5)
            stats["problematic_dags"] = [
                {"dag_id": dag, "error_count": int(count)} 
                for dag, count in dag_errors.items() if pd.notna(dag)
            ]
        
        return stats
    
    async def generate_recommendations(self, analysis_results: List[Dict]) -> Dict:
        """Generate overall recommendations based on analysis results"""
        if not analysis_results:
            return {"recommendations": ["No analysis results to process"]}
        
        # Aggregate findings
        severities = [r.get('severity', 'LOW') for r in analysis_results]
        categories = [r.get('category', 'INFO') for r in analysis_results]
        systemic_issues = [r.get('systemic_issue', False) for r in analysis_results]
        
        severity_counts = Counter(severities)
        category_counts = Counter(categories)
        
        prompt = f"""
        Based on analysis of {len(analysis_results)} Airflow log entries, provide system-wide recommendations.
        
        Summary:
        - Severity distribution: {dict(severity_counts)}
        - Issue category distribution: {dict(category_counts)}
        - Systemic issues detected: {sum(systemic_issues)}
        
        Provide comprehensive recommendations for:
        1. Immediate actions needed
        2. Long-term improvements
        3. Monitoring enhancements
        4. Configuration optimizations
        5. Resource scaling considerations
        
        Format as JSON:
        {{
            "immediate_actions": ["ACTION1", "ACTION2"],
            "long_term_improvements": ["IMPROVEMENT1", "IMPROVEMENT2"],
            "monitoring_enhancements": ["MONITOR1", "MONITOR2"],
            "config_optimizations": ["CONFIG1", "CONFIG2"],
            "resource_scaling": ["SCALE1", "SCALE2"],
            "priority_score": 1-10
        }}
        """
        
        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.deployment_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800,
                temperature=0.2
            )
            
            content = response.choices[0].message.content.strip()
            if content.startswith('```json'):
                content = content[7:-3]
            elif content.startswith('```'):
                content = content[3:-3]
                
            return json.loads(content)
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return {
                "immediate_actions": ["Review critical errors", "Check system resources"],
                "long_term_improvements": ["Implement better monitoring", "Optimize DAG designs"],
                "monitoring_enhancements": ["Set up alerting", "Create dashboards"],
                "config_optimizations": ["Review Airflow configuration", "Optimize database settings"],
                "resource_scaling": ["Monitor CPU/Memory usage", "Consider cluster scaling"],
                "priority_score": 5
            }

# Initialize AI agent
ai_agent = AirflowAILogAgent(client, AZURE_OPENAI_DEPLOYMENT)
```

## Step 4: Real-time Log Monitoring

```python
# Cell 6: Real-time Log Monitoring System
import asyncio
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import threading
import queue

class AirflowLogMonitor(FileSystemEventHandler):
    def __init__(self, ai_agent, log_parser):
        self.ai_agent = ai_agent
        self.log_parser = log_parser
        self.alert_queue = queue.Queue()
        self.processing_queue = queue.Queue()
        self.is_monitoring = False
        
    def on_modified(self, event):
        """Handle file modification events"""
        if event.is_directory:
            return
            
        if event.src_path.endswith('.log'):
            print(f"📝 Log file modified: {event.src_path}")
            self.processing_queue.put(event.src_path)
    
    async def process_new_logs(self, log_file_path: str):
        """Process newly added log entries"""
        try:
            # Read only new lines (simplified - in production, track file position)
            with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                
            # Process last few lines (new entries)
            recent_lines = lines[-10:] if len(lines) > 10 else lines
            
            for line in recent_lines:
                parsed_log = self.log_parser.parse_log_line(line)
                if parsed_log and (parsed_log['has_error'] or parsed_log['has_warning'] or parsed_log['has_exception']):
                    # Analyze with AI
                    analysis = await self.ai_agent.analyze_log_entry(parsed_log)
                    
                    # Create alert if severity is high
                    if analysis.get('severity') in ['HIGH', 'CRITICAL']:
                        alert = {
                            'timestamp': datetime.now().isoformat(),
                            'log_entry': parsed_log,
                            'ai_analysis': analysis,
                            'source_file': log_file_path
                        }
                        self.alert_queue.put(alert)
                        
        except Exception as e:
            logger.error(f"Error processing new logs from {log_file_path}: {e}")
    
    def start_monitoring(self, log_directory: str):
        """Start monitoring log directory"""
        print(f"🔍 Starting real-time monitoring of: {log_directory}")
        
        if not os.path.exists(log_directory):
            print(f"❌ Directory does not exist: {log_directory}")
            return
        
        observer = Observer()
        observer.schedule(self, log_directory, recursive=True)
        observer.start()
        
        self.is_monitoring = True
        print("✅ Real-time monitoring started")
        
        try:
            while self.is_monitoring:
                # Process queued files
                if not self.processing_queue.empty():
                    log_file = self.processing_queue.get()
                    asyncio.run(self.process_new_logs(log_file))
                
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped by user")
        finally:
            observer.stop()
            observer.join()
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.is_monitoring = False
    
    def get_alerts(self) -> List[Dict]:
        """Get all alerts from queue"""
        alerts = []
        while not self.alert_queue.empty():
            alerts.append(self.alert_queue.get())
        return alerts

# Initialize monitor
log_monitor = AirflowLogMonitor(ai_agent, airflow_parser)
```

## Step 5: Batch Analysis and Reporting

```python
# Cell 7: Batch Analysis and Comprehensive Reporting
class AirflowLogBatchAnalyzer:
    def __init__(self, ai_agent, log_parser):
        self.ai_agent = ai_agent
        self.log_parser = log_parser
        self.analysis_results = []
        
    async def analyze_log_batch(self, df_logs: pd.DataFrame, sample_size: int = 50) -> Dict:
        """Perform comprehensive batch analysis"""
        if df_logs.empty:
            return {"error": "No logs to analyze"}
            
        print(f"🔍 Starting batch analysis of {len(df_logs)} logs...")
        
        # Sample logs for detailed AI analysis (to manage API costs)
        if len(df_logs) > sample_size:
            # Prioritize errors, warnings, and anomalies
            priority_logs = df_logs[
                df_logs['has_error'] | 
                df_logs['has_warning'] | 
                df_logs.get('is_anomaly', False)
            ]
            
            if len(priority_logs) > sample_size:
                sample_logs = priority_logs.sample(n=sample_size)
            else:
                remaining_size = sample_size - len(priority_logs)
                other_logs = df_logs[~df_logs.index.isin(priority_logs.index)]
                if len(other_logs) > remaining_size:
                    additional_logs = other_logs.sample(n=remaining_size)
                    sample_logs = pd.concat([priority_logs, additional_logs])
                else:
                    sample_logs = df_logs
        else:
            sample_logs = df_logs
        
        print(f"📊 Analyzing {len(sample_logs)} selected logs with AI...")
        
        # Analyze each log entry
        analysis_results = []
        for idx, log_entry in sample_logs.iterrows():
            try:
                analysis = await self.ai_agent.analyze_log_entry(log_entry.to_dict())
                analysis_results.append({
                    'log_index': idx,
                    'analysis': analysis,
                    'log_entry': log_entry.to_dict()
                })
            except Exception as e:
                logger.error(f"Error analyzing log {idx}: {e}")
                continue
        
        # Generate pattern analysis
        pattern_stats = await self.ai_agent.analyze_pattern_trends(df_logs)
        
        # Generate recommendations
        ai_analyses = [r['analysis'] for r in analysis_results]
        recommendations = await self.ai_agent.generate_recommendations(ai_analyses)
        
        # Compile comprehensive report
        report = {
            'analysis_timestamp': datetime.now().isoformat(),
            'total_logs_analyzed': len(df_logs),
            'ai_analyzed_count': len(analysis_results),
            'pattern_statistics': pattern_stats,
            'detailed_analyses': analysis_results,
            'recommendations': recommendations,
            'summary_statistics': self._generate_summary_stats(df_logs, analysis_results)
        }
        
        self.analysis_results = analysis_results
        print("✅ Batch analysis completed")
        
        return report
    
    def _generate_summary_stats(self, df_logs: pd.DataFrame, analysis_results: List[Dict]) -> Dict:
        """Generate summary statistics"""
        stats = {
            'log_level_distribution': df_logs['level'].value_counts().to_dict(),
            'error_rate': float(df_logs['has_error'].mean()) if 'has_error' in df_logs.columns else 0,
            'warning_rate': float(df_logs['has_warning'].mean()) if 'has_warning' in df_logs.columns else 0,
            'anomaly_rate': float(df_logs['is_anomaly'].mean()) if 'is_anomaly' in df_logs.columns else 0,
        }
        
        if analysis_results:
            ai_severities = [r['analysis'].get('severity', 'LOW') for r in analysis_results]
            ai_categories = [r['analysis'].get('category', 'INFO') for r in analysis_results]
            
            stats['ai_severity_distribution'] = dict(Counter(ai_severities))
            stats['ai_category_distribution'] = dict(Counter(ai_categories))
        
        return stats
    
    def generate_html_report(self, report: Dict, output_file: str = "airflow_log_analysis.html"):
        """Generate HTML report"""
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Airflow Log Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .error {{ color: red; }}
                .warning {{ color: orange; }}
                .info {{ color: blue; }}
                .critical {{ color: darkred; font-weight: bold; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🔍 Airflow Log Analysis Report</h1>
                <p><strong>Analysis Date:</strong> {report['analysis_timestamp']}</p>
                <p><strong>Total Logs Analyzed:</strong> {report['total_logs_analyzed']}</p>
                <p><strong>AI Detailed Analysis:</strong> {report['ai_analyzed_count']} entries</p>
            </div>
            
            <div class="section">
                <h2>📊 Summary Statistics</h2>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Error Rate</td><td>{report['summary_statistics']['error_rate']:.2%}</td></tr>
                    <tr><td>Warning Rate</td><td>{report['summary_statistics']['warning_rate']:.2%}</td></tr>
                    <tr><td>Anomaly Rate</td><td>{report['summary_statistics']['anomaly_rate']:.2%}</td></tr>
                </table>
            </div>
            
            <div class="section">
                <h2>🚨 Critical Issues</h2>
                {self._generate_critical_issues_html(report)}
            </div>
            
            <div class="section">
                <h2>💡 Recommendations</h2>
                {self._generate_recommendations_html(report['recommendations'])}
            </div>
            
            <div class="section">
                <h2>📈 Pattern Analysis</h2>
                {self._generate_patterns_html(report['pattern_statistics'])}
            </div>
        </body>
        </html>
        """
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_template)
        
        print(f"📄 HTML report generated: {output_file}")
    
    def _generate_critical_issues_html(self, report: Dict) -> str:
        """Generate HTML for critical issues"""
        critical_issues = [
            r for r in report['detailed_analyses'] 
            if r['analysis'].get('severity') in ['HIGH', 'CRITICAL']
        ]
        
        if not critical_issues:
            return "<p>✅ No critical issues detected.</p>"
        
        html = "<table><tr><th>Severity</th><th>Category</th><th>Message</th><th>Recommendations</th></tr>"
        for issue in critical_issues[:10]:  # Show top 10
            analysis = issue['analysis']
            message = issue['log_entry']['message'][:100] + "..." if len(issue['log_entry']['message']) > 100 else issue['log_entry']['message']
            recommendations = "; ".join(analysis.get('recommendations', []))
            
            severity_class = 'critical' if analysis['severity'] == 'CRITICAL' else 'error'
            html += f"""
            <tr class="{severity_class}">
                <td>{analysis['severity']}</td>
                <td>{analysis['category']}</td>
                <td>{message}</td>
                <td>{recommendations}</td>
            </tr>
            """
        html += "</table>"
        return html
    
    def _generate_recommendations_html(self, recommendations: Dict) -> str:
        """Generate HTML for recommendations"""
        html = ""
        for category, items in recommendations.items():
            if isinstance(items, list) and items:
                html += f"<h3>{category.replace('_', ' ').title()}</h3><ul>"
                for item in items:
                    html += f"<li>{item}</li>"
                html += "</ul>"
        return html
    
    def _generate_patterns_html(self, patterns: Dict) -> str:
        """Generate HTML for pattern analysis"""
        html = "<ul>"
        for key, value in patterns.items():
            if isinstance(value, (int, float)):
                html += f"<li><strong>{key.replace('_', ' ').title()}:</strong> {value}</li>"
            elif isinstance(value, list):
                html += f"<li><strong>{key.replace('_', ' ').title()}:</strong> {len(value)} items</li>"
        html += "</ul>"
        return html

# Initialize batch analyzer
batch_analyzer = AirflowLogBatchAnalyzer(ai_agent, airflow_parser)
```

## Step 6: Execute Comprehensive Analysis

```python
# Cell 8: Execute Complete Analysis
async def run_comprehensive_analysis():
    """Run complete Airflow log analysis"""
    
    print("🚀 Starting Comprehensive Airflow Log Analysis")
    print("=" * 60)
    
    try:
        # Check if we have analyzed logs
        if df_analyzed.empty:
            print("❌ No log data available for analysis")
            return
        
        # Run batch analysis
        print("\n1️⃣ Running AI-powered batch analysis...")
        analysis_report = await batch_analyzer.analyze_log_batch(df_analyzed, sample_size=20)
        
        # Display key findings
        print("\n📊 Key Findings:")
        print(f"Total logs analyzed: {analysis_report['total_logs_analyzed']}")
        print(f"AI detailed analysis: {analysis_report['ai_analyzed_count']} entries")
        
        # Show summary statistics
        summary_stats = analysis_report['summary_statistics']
        print(f"\n📈 Summary Statistics:")
        print(f"Error Rate: {summary_stats['error_rate']:.2%}")
        print(f"Warning Rate: {summary_stats['warning_rate']:.2%}")
        print(f"Anomaly Rate: {summary_stats['anomaly_rate']:.2%}")
        
        # Show AI analysis distribution
        if 'ai_severity_distribution' in summary_stats:
            print(f"\n🤖 AI Severity Distribution:")
            for severity, count in summary_stats['ai_severity_distribution'].items():
                print(f"  {severity}: {count}")
        
        # Show critical issues
        critical_analyses = [
            r for r in analysis_report['detailed_analyses']
            if r['analysis'].get('severity') in ['HIGH', 'CRITICAL']
        ]
        
        if critical_analyses:
            print(f"\n🚨 Critical Issues Found: {len(critical_analyses)}")
            for i, critical in enumerate(critical_analyses[:3], 1):  # Show first 3
                analysis = critical['analysis']
                log_entry = critical['log_entry']
                print(f"\n  Issue {i}:")
                print(f"    Severity: {analysis['severity']}")
                print(f"    Category: {analysis['category']}")
                print(f"    Message: {log_entry['message'][:100]}...")
                print(f"    Root Cause: {analysis['root_cause']}")
                print(f"    Recommendations: {', '.join(analysis['recommendations'][:2])}")
        
        # Show recommendations
        recommendations = analysis_report['recommendations']
        print(f"\n💡 Top Recommendations:")
        if 'immediate_actions' in recommendations:
            print("  Immediate Actions:")
            for action in recommendations['immediate_actions'][:3]:
                print(f"    - {action}")
        
        if 'long_term_improvements' in recommendations:
            print("  Long-term Improvements:")
            for improvement in recommendations['long_term_improvements'][:3]:
                print(f"    - {improvement}")
        
        # Generate HTML report
        print("\n2️⃣ Generating detailed HTML report...")
        batch_analyzer.generate_html_report(analysis_report)
        
        return analysis_report
        
    except Exception as e:
        logger.error(f"Error in comprehensive analysis: {e}")
        return None

# Run the analysis
analysis_results = await run_comprehensive_analysis()
```

## Step 7: Visualization Dashboard

```python
# Cell 9: Create Visualization Dashboard
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class AirflowLogDashboard:
    def __init__(self, df_logs: pd.DataFrame, analysis_results: List[Dict] = None):
        self.df = df_logs
        self.analysis_results = analysis_results or []
        
    def create_summary_dashboard(self):
        """Create comprehensive summary dashboard"""
        if self.df.empty:
            print("No data available for visualization")
            return
            
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Airflow Log Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Log Level Distribution
        if 'level' in self.df.columns:
            level_counts = self.df['level'].value_counts()
            axes[0, 0].pie(level_counts.values, labels=level_counts.index, autopct='%1.1f%%')
            axes[0, 0].set_title('Log Level Distribution')
        
        # 2. Issues Over Time
        if 'parsed_timestamp' in self.df.columns and self.df['parsed_timestamp'].notna().any():
            hourly_issues = self.df[self.df['has_error'] | self.df['has_warning']].groupby(
                self.df['parsed_timestamp'].dt.hour
            ).size()
            axes[0, 1].bar(hourly_issues.index, hourly_issues.values, color='red', alpha=0.7)
            axes[0, 1].set_title('Issues by Hour')
            axes[0, 1].set_xlabel('Hour')
            axes[0, 1].set_ylabel('Count')
        
        # 3. DAG Error Analysis
        if 'dag_id' in self.df.columns:
            dag_errors = self.df[self.df['has_error']].groupby('dag_id').size().sort_values(ascending=False).head(10)
            if not dag_errors.empty:
                axes[0, 2].barh(range(len(dag_errors)), dag_errors.values)
                axes[0, 2].set_yticks(range(len(dag_errors)))
                axes[0, 2].set_yticklabels([str(dag)[:20] for dag in dag_errors.index])
                axes[0, 2].set_title('Top 10 DAGs with Errors')
                axes[0, 2].set_xlabel('Error Count')
        
        # 4. Message Length Distribution
        if 'message_length' in self.df.columns:
            axes[1, 0].hist(self.df['message_length'], bins=30, alpha=0.7, color='blue')
            axes[1, 0].set_title('Message Length Distribution')
            axes[1, 0].set_xlabel('Message Length')
            axes[1, 0].set_ylabel('Frequency')
        
        # 5. Anomaly Detection Results
        if 'is_anomaly' in self.df.columns:
            anomaly_counts = self.df['is_anomaly'].value_counts()
            axes[1, 1].pie(anomaly_counts.values, 
                          labels=['Normal', 'Anomaly'], 
                          colors=['green', 'red'],
                          autopct='%1.1f%%')
            axes[1, 1].set_title('Anomaly Detection Results')
        
        # 6. AI Analysis Results
        if self.analysis_results:
            severities = [r['analysis'].get('severity', 'UNKNOWN') for r in self.analysis_results]
            severity_counts = Counter(severities)
            axes[1, 2].bar(severity_counts.keys(), severity_counts.values(), 
                          color=['green', 'yellow', 'orange', 'red'])
            axes[1, 2].set_title('AI Analysis - Severity Distribution')
            axes[1, 2].set_xlabel('Severity')
            axes[1, 2].set_ylabel('Count')
        
        plt.tight_layout()
        plt.show()
    
    def create_interactive_dashboard(self):
        """Create interactive Plotly dashboard"""
        if self.df.empty:
            print("No data available for interactive visualization")
            return
            
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Log Levels Over Time', 'Error Categories', 
                          'DAG Performance', 'Hourly Activity'),
            specs=[[{"secondary_y": True}, {"type": "pie"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        # 1. Log levels over time
        if 'parsed_timestamp' in self.df.columns and self.df['parsed_timestamp'].notna().any():
            df_time = self.df.set_index('parsed_timestamp').resample('H').agg({
                'has_error': 'sum',
                'has_warning': 'sum',
                'level': 'count'
            })
            
            fig.add_trace(
                go.Scatter(x=df_time.index, y=df_time['has_error'], 
                          name='Errors', line=dict(color='red')),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=df_time.index, y=df_time['has_warning'], 
                          name='Warnings', line=dict(color='orange')),
                row=1, col=1
            )
        
        # 2. Error categories pie chart
        if self.analysis_results:
            categories = [r['analysis'].get('category', 'UNKNOWN') for r in self.analysis_results]
            category_counts = Counter(categories)
            
            fig.add_trace(
                go.Pie(labels=list(category_counts.keys()), 
                      values=list(category_counts.values()),
                      name="Error Categories"),
                row=1, col=2
            )
        
        # 3. DAG performance
        if 'dag_id' in self.df.columns:
            dag_stats = self.df.groupby('dag_id').agg({
                'has_error': 'sum',
                'has_warning': 'sum'
            }).reset_index()
            
            fig.add_trace(
                go.Bar(x=dag_stats['dag_id'], y=dag_stats['has_error'], 
                      name='DAG Errors', marker_color='red'),
                row=2, col=1
            )
        
        # 4. Hourly activity scatter
        if 'hour' in self.df.columns:
            hourly_stats = self.df.groupby('hour').agg({
                'has_error': 'sum',
                'level': 'count'
            }).reset_index()
            
            fig.add_trace(
                go.Scatter(x=hourly_stats['hour'], y=hourly_stats['level'],
                          mode='markers+lines', name='Total Logs',
                          marker=dict(size=hourly_stats['has_error']*5, color='blue')),
                row=2, col=2
            )
        
        fig.update_layout(height=800, showlegend=True, 
                         title_text="Interactive Airflow Log Analysis Dashboard")
        fig.show()
    
    def create_error_timeline(self):
        """Create detailed error timeline"""
        if self.df.empty or 'parsed_timestamp' not in self.df.columns:
            print("No timestamp data available for timeline")
            return
            
        error_logs = self.df[self.df['has_error']].copy()
        if error_logs.empty:
            print("No errors found for timeline")
            return
        
        fig = px.timeline(
            error_logs.head(20),  # Show first 20 errors
            x_start="parsed_timestamp",
            x_end="parsed_timestamp",
            y="dag_id",
            color="level",
            title="Error Timeline by DAG",
            hover_data=["message"]
        )
        
        fig.update_yaxes(categoryorder="total ascending")
        fig.show()

# Create dashboard
if not df_analyzed.empty:
    dashboard = AirflowLogDashboard(df_analyzed, batch_analyzer.analysis_results)
    
    print("📊 Creating Summary Dashboard...")
    dashboard.create_summary_dashboard()
    
    print("🎯 Creating Interactive Dashboard...")
    dashboard.create_interactive_dashboard()
    
    print("📅 Creating Error Timeline...")
    dashboard.create_error_timeline()
else:
    print("❌ No analyzed data available for dashboard")
```

## Step 8: Continuous Monitoring Setup

```python
# Cell 10: Setup Continuous Monitoring
async def setup_continuous_monitoring():
    """Setup continuous monitoring system"""
    
    print("🔄 Setting up Continuous Airflow Log Monitoring")
    print("=" * 50)
    
    # Configuration
    airflow_logs_dir = "/opt/GOLD/airflow/logs/scheduler"
    current_date = datetime.now().strftime('%Y-%m-%d')
    monitor_directory = f"{airflow_logs_dir}/{current_date}"
    
    print(f"📁 Monitoring directory: {monitor_directory}")
    
    # Check if directory exists
    if not os.path.exists(monitor_directory):
        print(f"⚠️ Directory does not exist: {monitor_directory}")
        print("Creating sample directory structure for demonstration...")
        
        # Create sample directory and files for demo
        os.makedirs(monitor_directory, exist_ok=True)
        
        # Create a sample log file
        sample_log_path = f"{monitor_directory}/scheduler_sample.log"
        sample_logs = [
            f"[{datetime.now().isoformat()}] INFO - DAG sample_dag is running",
            f"[{datetime.now().isoformat()}] ERROR - Task failed: Connection refused",
            f"[{datetime.now().isoformat()}] WARNING - Configuration issue detected"
        ]
        
        with open(sample_log_path, 'w') as f:
            f.write('\n'.join(sample_logs))
        
        print(f"✅ Created sample log file: {sample_log_path}")
    
    # Start monitoring (this would run in background in production)
    print("\n🔍 Continuous monitoring would start here...")
    print("In production, this would:")
    print("1. Monitor log directory for new files and changes")
    print("2. Process new log entries in real-time")
    print("3. Generate alerts for critical issues")
    print("4. Send notifications via email/Slack/etc.")
    
    # Simulate processing some alerts
    alerts = log_monitor.get_alerts()
    if alerts:
        print(f"\n🚨 {len(alerts)} alerts in queue:")
        for alert in alerts:
            ai_analysis = alert['ai_analysis']
            print(f"  - {ai_analysis['severity']}: {ai_analysis['category']}")
    else:
        print("\n✅ No alerts in queue")

# Setup monitoring
await setup_continuous_monitoring()
```

## Step 9: Configuration and Deployment

```python
# Cell 11: Configuration Management
class AirflowLogAnalysisConfig:
    def __init__(self):
        self.config = {
            "airflow": {
                "home_directory": "/opt/GOLD/airflow",
                "log_directories": [
                    "logs/scheduler",
                    "logs/dag_processor_manager", 
                    "logs/webserver"
                ]
            },
            "monitoring": {
                "batch_size": 100,
                "ai_analysis_sample_size": 50,
                "alert_threshold_severity": ["HIGH", "CRITICAL"],
                "monitoring_interval_seconds": 30,
                "max_alerts_per_hour": 100
            },
            "ai_analysis": {
                "azure_endpoint": "https://genai-gateway.azure-api.net/",
                "deployment_name": "gpt-35-turbo",
                "api_version": "2024-02-15-preview",
                "max_tokens": 600,
                "temperature": 0.1,
                "enable_caching": True
            },
            "detection": {
                "anomaly_detection_contamination": 0.1,
                "error_keywords": ["error", "exception", "failed", "timeout", "refused"],
                "warning_keywords": ["warning", "deprecated", "retry"],
                "config_keywords": ["configuration", "config", "setting", "parameter"],
                "performance_keywords": ["slow", "timeout", "memory", "cpu"]
            },
            "reporting": {
                "generate_html_reports": True,
                "report_output_directory": "./reports",
                "retention_days": 30,
                "email_notifications": False,
                "slack_notifications": False
            }
        }
    
    def save_config(self, filepath: str = "airflow_log_config.json"):
        """Save configuration to file"""
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.config, f, indent=2)
        print(f"✅ Configuration saved to {filepath}")
    
    def load_config(self, filepath: str):
        """Load configuration from file"""
        with open(filepath, 'r') as f:
            self.config = json.load(f)
        print(f"✅ Configuration loaded from {filepath}")
    
    def get_config(self):
        return self.config

# Save configuration
config_manager = AirflowLogAnalysisConfig()
config_manager.save_config()

print("📋 Configuration saved. Contents:")
print(json.dumps(config_manager.get_config(), indent=2))
```

## Step 10: Production Deployment Script

```python
# Cell 12: Production Deployment Setup
def create_deployment_script():
    """Create production deployment script"""
    
    deployment_script = '''#!/bin/bash
# Airflow Log Analysis Deployment Script

echo "🚀 Deploying Airflow Log Analysis System"

# Create directories
mkdir -p /opt/airflow-log-analysis/{logs,reports,config}

# Install Python dependencies
pip install -r requirements.txt

# Set environment variables
export AIRFLOW_HOME=/opt/GOLD/airflow
export AZURE_OPENAI_ENDPOINT=https://genai-gateway.azure-api.net/
export AZURE_OPENAI_API_KEY=your_api_key_here

# Create systemd service
cat > /etc/systemd/system/airflow-log-monitor.service << EOF
[Unit]
Description=Airflow Log Analysis Monitor
After=network.target

[Service]
Type=simple
User=airflow
WorkingDirectory=/opt/airflow-log-analysis
ExecStart=/usr/bin/python3 monitor.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
systemctl enable airflow-log-monitor
systemctl start airflow-log-monitor

echo "✅ Deployment completed"
'''
    
    requirements_txt = '''pandas>=1.5.0
numpy>=1.21.0
matplotlib>=3.5.0
seaborn>=0.11.0
plotly>=5.0.0
openai>=1.0.0
python-dotenv>=0.19.0
scikit-learn>=1.1.0
textstat>=0.7.0
watchdog>=2.1.0
azure-ai-textanalytics>=5.2.0
'''
    
    monitor_py = '''#!/usr/bin/env python3
"""
Production Airflow Log Monitor
"""
import asyncio
import sys
import os
sys.path.append(os.path.dirname(__file__))

from airflow_log_analyzer import *

async def main():
    """Main monitoring loop"""
    # Initialize components
    parser = AirflowLogParser()
    ai_agent = AirflowAILogAgent(client, AZURE_OPENAI_DEPLOYMENT)
    monitor = AirflowLogMonitor(ai_agent, parser)
    
    # Start monitoring
    log_directory = "/opt/GOLD/airflow/logs/scheduler"
    monitor.start_monitoring(log_directory)

if __name__ == "__main__":
    asyncio.run(main())
'''
    
    # Write files
    with open("deploy.sh", "w") as f:
        f.write(deployment_script)
    
    with open("requirements.txt", "w") as f:
        f.write(requirements_txt)
    
    with open("monitor.py", "w") as f:
        f.write(monitor_py)
    
    # Make deployment script executable
    os.chmod("deploy.sh", 0o755)
    
    print("✅ Deployment files created:")
    print("  - deploy.sh (deployment script)")
    print("  - requirements.txt (Python dependencies)")
    print("  - monitor.py (production monitor)")
    print("  - airflow_log_config.json (configuration)")

create_deployment_script()
```

## Usage Summary

### Complete End-to-End Example

```python
# Cell 13: Complete Usage Example
async def complete_airflow_log_analysis():
    """Complete end-to-end example"""
    
    print("🎯 Complete Airflow Log Analysis Example")
    print("=" * 50)
    
    try:
        # 1. Parse logs
        print("1️⃣ Parsing Airflow logs...")
        current_date = datetime.now().strftime('%Y-%m-%d')
        log_files = airflow_parser.get_log_files(current_date)
        
        if not log_files:
            print("Using sample data for demonstration")
            df_logs = create_sample_airflow_logs()
        else:
            df_logs = airflow_parser.read_log_files(log_files)
        
        # 2. Analyze logs
        print("\n2️⃣ Analyzing logs...")
        analyzer = AirflowLogAnalyzer(df_logs)
        df_analyzed = analyzer.extract_features()
        df_analyzed = analyzer.detect_anomalies()
        
        # 3. AI-powered analysis
        print("\n3️⃣ Running AI analysis...")
        batch_analyzer = AirflowLogBatchAnalyzer(ai_agent, airflow_parser)
        analysis_report = await batch_analyzer.analyze_log_batch(df_analyzed, sample_size=10)
        
        # 4. Generate reports
        print("\n4️⃣ Generating reports...")
        batch_analyzer.generate_html_report(analysis_report)
        
        # 5. Create visualizations
        print("\n5️⃣ Creating visualizations...")
        dashboard = AirflowLogDashboard(df_analyzed, batch_analyzer.analysis_results)
        dashboard.create_summary_dashboard()
        
        print("\n✅ Complete analysis finished!")
        print(f"📊 Results: {analysis_report['total_logs_analyzed']} logs analyzed")
        print(f"🚨 Critical issues: {len([r for r in analysis_report['detailed_analyses'] if r['analysis'].get('severity') in ['HIGH', 'CRITICAL']])}")
        
        return analysis_report
        
    except Exception as e:
        logger.error(f"Error in complete analysis: {e}")
        return None

# Run complete example
final_results = await complete_airflow_log_analysis()
```

## Key Features Summary

### ✅ What This System Provides:

1. **Real-time Log Parsing**: Monitors Airflow scheduler logs in `/opt/GOLD/airflow/logs/scheduler/`
2. **AI-Powered Analysis**: Uses Azure OpenAI to analyze log entries intelligently  
3. **Multi-level Detection**: 
   - Rule-based anomaly detection
   - Machine learning anomaly detection
   - AI-powered severity classification
4. **Comprehensive Categorization**:
   - Errors and exceptions
   - Configuration issues
   - Performance problems
   - Connection issues
   - Task failures
5. **Interactive Dashboards**: Matplotlib and Plotly visualizations
6. **Automated Reporting**: HTML reports with recommendations
7. **Continuous Monitoring**: File system monitoring for real-time alerts
8. **Production Ready**: Configuration management and deployment scripts

### 🔧 Configuration:
- Airflow home: `/opt/GOLD/airflow/`
- Log path: `$AIRFLOW_HOME/logs/scheduler/${DATE}`
- Azure OpenAI endpoint: `https://genai-gateway.azure-api.net/`
- Configurable thresholds and parameters

### 🚀 Production Deployment:
Run the generated `deploy.sh` script to deploy in production with systemd service management.

This system provides comprehensive Airflow log analysis with AI-powered insights for proactive issue detection and resolution!
