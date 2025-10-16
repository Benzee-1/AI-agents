# Intrusion Detection via Log Analysis using AI Agents

I'll provide you with a comprehensive guide to build an AI-powered intrusion detection system that analyzes logs for unusual access patterns and privilege escalations.

## Prerequisites

### 1. Required Libraries Installation

```bash
# Run this in your terminal or Jupyter cell with !
pip install pandas numpy scikit-learn matplotlib seaborn
pip install openai python-dotenv
pip install asyncio aiofiles
pip install plotly dash (optional for interactive dashboards)
```

### 2. Environment Setup

```python
# Cell 1: Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import re
import json
import asyncio
import logging
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# OpenAI and environment
import openai
from openai import OpenAI
import os
from dotenv import load_dotenv

# Machine Learning
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
```

### 3. OpenAI Configuration

```python
# Cell 2: OpenAI Setup
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Test connection
def test_openai_connection():
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Test connection"}],
            max_tokens=10
        )
        print("✅ OpenAI connection successful")
        return True
    except Exception as e:
        print(f"❌ OpenAI connection failed: {e}")
        return False

test_openai_connection()
```

## Step 1: Log Data Generation and Preprocessing

```python
# Cell 3: Generate Synthetic Log Data
import random
from datetime import datetime, timedelta

class LogGenerator:
    def __init__(self):
        self.users = ['admin', 'user1', 'user2', 'service_account', 'guest', 'root', 'database_user']
        self.actions = ['login', 'logout', 'file_access', 'privilege_escalation', 'system_command', 'data_export']
        self.resources = ['server1', 'database', 'file_server', 'web_app', 'admin_panel', 'backup_system']
        self.ip_addresses = ['192.168.1.10', '192.168.1.20', '10.0.0.5', '172.16.0.100', '203.0.113.5']
        
    def generate_normal_logs(self, num_logs: int = 1000) -> List[Dict]:
        logs = []
        start_time = datetime.now() - timedelta(days=7)
        
        for i in range(num_logs):
            timestamp = start_time + timedelta(
                hours=random.randint(0, 168), 
                minutes=random.randint(0, 59), 
                seconds=random.randint(0, 59)
            )
            
            log_entry = {
                'timestamp': timestamp.isoformat(),
                'user': random.choice(self.users),
                'action': random.choice(self.actions),
                'resource': random.choice(self.resources),
                'source_ip': random.choice(self.ip_addresses),
                'success': random.choice([True, True, True, False]),  # Mostly successful
                'session_duration': random.randint(5, 3600),  # 5 seconds to 1 hour
                'data_size': random.randint(100, 10000),  # bytes
                'severity': random.choice(['INFO', 'WARNING', 'ERROR'])
            }
            logs.append(log_entry)
        
        return logs
    
    def generate_anomalous_logs(self, num_logs: int = 100) -> List[Dict]:
        anomalous_logs = []
        start_time = datetime.now() - timedelta(days=7)
        
        for i in range(num_logs):
            timestamp = start_time + timedelta(
                hours=random.randint(0, 168), 
                minutes=random.randint(0, 59)
            )
            
            # Create different types of anomalies
            anomaly_type = random.choice(['privilege_escalation', 'unusual_access', 'brute_force', 'data_exfiltration'])
            
            if anomaly_type == 'privilege_escalation':
                log_entry = {
                    'timestamp': timestamp.isoformat(),
                    'user': random.choice(['user1', 'user2', 'guest']),
                    'action': 'privilege_escalation',
                    'resource': 'admin_panel',
                    'source_ip': random.choice(self.ip_addresses),
                    'success': True,
                    'session_duration': random.randint(1, 30),  # Very short
                    'data_size': random.randint(50000, 100000),  # Large data access
                    'severity': 'CRITICAL'
                }
            elif anomaly_type == 'unusual_access':
                log_entry = {
                    'timestamp': (timestamp.replace(hour=random.randint(2, 4))).isoformat(),  # Unusual hours
                    'user': random.choice(self.users),
                    'action': 'file_access',
                    'resource': random.choice(self.resources),
                    'source_ip': '198.51.100.42',  # External IP
                    'success': True,
                    'session_duration': random.randint(7200, 14400),  # Very long session
                    'data_size': random.randint(50000, 200000),
                    'severity': 'WARNING'
                }
            elif anomaly_type == 'brute_force':
                log_entry = {
                    'timestamp': timestamp.isoformat(),
                    'user': random.choice(['unknown', 'admin']),
                    'action': 'login',
                    'resource': 'web_app',
                    'source_ip': '203.0.113.100',  # Suspicious IP
                    'success': False,
                    'session_duration': random.randint(1, 5),
                    'data_size': random.randint(100, 500),
                    'severity': 'ERROR'
                }
            else:  # data_exfiltration
                log_entry = {
                    'timestamp': timestamp.isoformat(),
                    'user': random.choice(self.users),
                    'action': 'data_export',
                    'resource': 'database',
                    'source_ip': random.choice(self.ip_addresses),
                    'success': True,
                    'session_duration': random.randint(3600, 7200),
                    'data_size': random.randint(100000, 500000),  # Very large
                    'severity': 'WARNING'
                }
            
            anomalous_logs.append(log_entry)
        
        return anomalous_logs

# Generate sample data
log_generator = LogGenerator()
normal_logs = log_generator.generate_normal_logs(1000)
anomalous_logs = log_generator.generate_anomalous_logs(100)

# Combine and create DataFrame
all_logs = normal_logs + anomalous_logs
df_logs = pd.DataFrame(all_logs)

# Add labels for training (in real scenario, these would be discovered)
df_logs['is_anomaly'] = [False] * len(normal_logs) + [True] * len(anomalous_logs)

print(f"Generated {len(df_logs)} log entries")
print(f"Normal logs: {len(normal_logs)}")
print(f"Anomalous logs: {len(anomalous_logs)}")
print("\nSample logs:")
print(df_logs.head())
```

## Step 2: Data Preprocessing and Feature Engineering

```python
# Cell 4: Data Preprocessing
class LogPreprocessor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def preprocess_logs(self, df: pd.DataFrame) -> pd.DataFrame:
        df_processed = df.copy()
        
        # Convert timestamp to datetime
        df_processed['timestamp'] = pd.to_datetime(df_processed['timestamp'])
        
        # Extract time-based features
        df_processed['hour'] = df_processed['timestamp'].dt.hour
        df_processed['day_of_week'] = df_processed['timestamp'].dt.dayofweek
        df_processed['is_weekend'] = df_processed['day_of_week'].isin([5, 6])
        df_processed['is_night'] = df_processed['hour'].isin(range(0, 6))
        
        # Encode categorical variables
        categorical_columns = ['user', 'action', 'resource', 'source_ip', 'severity']
        for col in categorical_columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
            df_processed[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df_processed[col])
        
        # Convert boolean to int
        df_processed['success'] = df_processed['success'].astype(int)
        df_processed['is_weekend'] = df_processed['is_weekend'].astype(int)
        df_processed['is_night'] = df_processed['is_night'].astype(int)
        
        # Calculate additional features
        df_processed['log_data_size'] = np.log1p(df_processed['data_size'])
        df_processed['log_session_duration'] = np.log1p(df_processed['session_duration'])
        
        return df_processed
    
    def get_feature_columns(self) -> List[str]:
        return [
            'user_encoded', 'action_encoded', 'resource_encoded', 
            'source_ip_encoded', 'severity_encoded', 'success',
            'hour', 'day_of_week', 'is_weekend', 'is_night',
            'log_data_size', 'log_session_duration'
        ]

# Preprocess the data
preprocessor = LogPreprocessor()
df_processed = preprocessor.preprocess_logs(df_logs)
feature_columns = preprocessor.get_feature_columns()

print("Processed DataFrame shape:", df_processed.shape)
print("\nFeature columns:", feature_columns)
print("\nProcessed data sample:")
print(df_processed[feature_columns].head())
```

## Step 3: Anomaly Detection Models

```python
# Cell 5: Anomaly Detection Models
class AnomalyDetector:
    def __init__(self):
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.dbscan = DBSCAN(eps=0.5, min_samples=5)
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def fit(self, X: np.ndarray):
        """Train the anomaly detection models"""
        # Scale the features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit Isolation Forest
        self.isolation_forest.fit(X_scaled)
        
        # Fit DBSCAN
        self.dbscan.fit(X_scaled)
        
        self.is_fitted = True
        print("✅ Anomaly detection models trained successfully")
        
    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Predict anomalies using multiple methods"""
        if not self.is_fitted:
            raise ValueError("Models must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        
        # Isolation Forest predictions (-1 for anomaly, 1 for normal)
        if_predictions = self.isolation_forest.predict(X_scaled)
        if_anomalies = (if_predictions == -1).astype(int)
        
        # DBSCAN predictions (-1 for noise/anomaly)
        dbscan_predictions = self.dbscan.fit_predict(X_scaled)
        dbscan_anomalies = (dbscan_predictions == -1).astype(int)
        
        # Anomaly scores from Isolation Forest
        anomaly_scores = self.isolation_forest.decision_function(X_scaled)
        
        return {
            'isolation_forest': if_anomalies,
            'dbscan': dbscan_anomalies,
            'anomaly_scores': anomaly_scores
        }

# Train anomaly detection models
X = df_processed[feature_columns].values
y = df_processed['is_anomaly'].values

detector = AnomalyDetector()
detector.fit(X)

# Predict anomalies
predictions = detector.predict(X)

# Add predictions to dataframe
df_processed['pred_isolation_forest'] = predictions['isolation_forest']
df_processed['pred_dbscan'] = predictions['dbscan']
df_processed['anomaly_score'] = predictions['anomaly_scores']

print("Anomaly Detection Results:")
print(f"Isolation Forest detected {predictions['isolation_forest'].sum()} anomalies")
print(f"DBSCAN detected {predictions['dbscan'].sum()} anomalies")
```

## Step 4: AI Agent for Log Analysis

```python
# Cell 6: AI Agent for Intelligent Log Analysis
class IntelligentLogAnalyzer:
    def __init__(self, openai_client):
        self.client = openai_client
        
    async def analyze_log_entry(self, log_entry: Dict, context: str = "") -> Dict:
        """Analyze a single log entry using OpenAI"""
        prompt = f"""
        Analyze this security log entry for potential threats or anomalies:
        
        Log Entry:
        - Timestamp: {log_entry.get('timestamp')}
        - User: {log_entry.get('user')}
        - Action: {log_entry.get('action')}
        - Resource: {log_entry.get('resource')}
        - Source IP: {log_entry.get('source_ip')}
        - Success: {log_entry.get('success')}
        - Session Duration: {log_entry.get('session_duration')} seconds
        - Data Size: {log_entry.get('data_size')} bytes
        - Severity: {log_entry.get('severity')}
        
        Additional Context: {context}
        
        Please provide:
        1. Risk level (LOW, MEDIUM, HIGH, CRITICAL)
        2. Potential threat type
        3. Explanation of concerns
        4. Recommended actions
        
        Respond in JSON format:
        {{
            "risk_level": "LEVEL",
            "threat_type": "TYPE",
            "explanation": "EXPLANATION",
            "recommendations": ["ACTION1", "ACTION2"]
        }}
        """
        
        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.1
            )
            
            # Parse JSON response
            content = response.choices[0].message.content.strip()
            # Remove markdown formatting if present
            if content.startswith('```json'):
                content = content[7:-3]
            elif content.startswith('```'):
                content = content[3:-3]
                
            return json.loads(content)
            
        except Exception as e:
            logger.error(f"Error analyzing log entry: {e}")
            return {
                "risk_level": "UNKNOWN",
                "threat_type": "Analysis Error",
                "explanation": f"Failed to analyze: {str(e)}",
                "recommendations": ["Manual review required"]
            }
    
    def analyze_patterns(self, df: pd.DataFrame, user: str = None) -> Dict:
        """Analyze patterns for a specific user or overall"""
        if user:
            user_logs = df[df['user'] == user]
            if len(user_logs) == 0:
                return {"error": f"No logs found for user {user}"}
        else:
            user_logs = df
            
        analysis = {
            "total_logs": len(user_logs),
            "unique_actions": user_logs['action'].nunique(),
            "unique_resources": user_logs['resource'].nunique(),
            "unique_ips": user_logs['source_ip'].nunique(),
            "success_rate": user_logs['success'].mean(),
            "avg_session_duration": user_logs['session_duration'].mean(),
            "avg_data_size": user_logs['data_size'].mean(),
            "night_activity": user_logs['is_night'].sum(),
            "weekend_activity": user_logs['is_weekend'].sum()
        }
        
        return analysis

# Initialize AI analyzer
ai_analyzer = IntelligentLogAnalyzer(client)
```

## Step 5: Real-time Monitoring System

```python
# Cell 7: Real-time Log Monitoring System
class LogMonitoringSystem:
    def __init__(self, anomaly_detector, ai_analyzer, alert_threshold=0.7):
        self.detector = anomaly_detector
        self.ai_analyzer = ai_analyzer
        self.alert_threshold = alert_threshold
        self.alerts = []
        
    async def process_log_batch(self, logs: List[Dict]) -> List[Dict]:
        """Process a batch of logs and return alerts"""
        alerts = []
        
        for log in logs:
            # Convert to DataFrame for processing
            df_single = pd.DataFrame([log])
            df_processed = preprocessor.preprocess_logs(df_single)
            
            # Extract features
            X_single = df_processed[feature_columns].values
            
            # Predict anomaly
            predictions = self.detector.predict(X_single)
            anomaly_score = predictions['anomaly_scores'][0]
            is_anomaly_if = predictions['isolation_forest'][0]
            is_anomaly_dbscan = predictions['dbscan'][0]
            
            # Check if alert threshold is exceeded
            if is_anomaly_if or is_anomaly_dbscan or anomaly_score < -0.5:
                # Get AI analysis
                ai_analysis = await self.ai_analyzer.analyze_log_entry(log)
                
                alert = {
                    'timestamp': log['timestamp'],
                    'log_entry': log,
                    'anomaly_score': float(anomaly_score),
                    'detected_by_isolation_forest': bool(is_anomaly_if),
                    'detected_by_dbscan': bool(is_anomaly_dbscan),
                    'ai_analysis': ai_analysis,
                    'alert_id': f"ALERT_{len(self.alerts) + 1}"
                }
                
                alerts.append(alert)
                self.alerts.append(alert)
                
        return alerts
    
    def get_alert_summary(self) -> Dict:
        """Get summary of all alerts"""
        if not self.alerts:
            return {"total_alerts": 0}
            
        risk_levels = [alert['ai_analysis'].get('risk_level', 'UNKNOWN') for alert in self.alerts]
        threat_types = [alert['ai_analysis'].get('threat_type', 'Unknown') for alert in self.alerts]
        
        return {
            "total_alerts": len(self.alerts),
            "risk_level_distribution": pd.Series(risk_levels).value_counts().to_dict(),
            "threat_type_distribution": pd.Series(threat_types).value_counts().to_dict(),
            "latest_alert": self.alerts[-1]['timestamp'] if self.alerts else None
        }

# Initialize monitoring system
monitoring_system = LogMonitoringSystem(detector, ai_analyzer)
```

## Step 6: Testing and Demonstration

```python
# Cell 8: Test the System with Sample Anomalous Logs
import asyncio

async def test_monitoring_system():
    """Test the monitoring system with sample logs"""
    
    # Select some anomalous logs for testing
    test_logs = df_logs[df_logs['is_anomaly'] == True].head(5).to_dict('records')
    
    print("🔍 Testing Monitoring System with Anomalous Logs...")
    print("=" * 60)
    
    alerts = await monitoring_system.process_log_batch(test_logs)
    
    for i, alert in enumerate(alerts, 1):
        print(f"\n🚨 ALERT {i}: {alert['alert_id']}")
        print(f"Timestamp: {alert['timestamp']}")
        print(f"User: {alert['log_entry']['user']}")
        print(f"Action: {alert['log_entry']['action']}")
        print(f"Resource: {alert['log_entry']['resource']}")
        print(f"Source IP: {alert['log_entry']['source_ip']}")
        print(f"Anomaly Score: {alert['anomaly_score']:.3f}")
        
        ai_analysis = alert['ai_analysis']
        print(f"\n🤖 AI Analysis:")
        print(f"  Risk Level: {ai_analysis.get('risk_level', 'Unknown')}")
        print(f"  Threat Type: {ai_analysis.get('threat_type', 'Unknown')}")
        print(f"  Explanation: {ai_analysis.get('explanation', 'No explanation')}")
        print(f"  Recommendations: {', '.join(ai_analysis.get('recommendations', []))}")
        print("-" * 60)
    
    # Get summary
    summary = monitoring_system.get_alert_summary()
    print(f"\n📊 Alert Summary:")
    print(f"Total Alerts: {summary['total_alerts']}")
    if summary['total_alerts'] > 0:
        print(f"Risk Level Distribution: {summary['risk_level_distribution']}")
        print(f"Threat Type Distribution: {summary['threat_type_distribution']}")

# Run the test
await test_monitoring_system()
```

## Step 7: Visualization Dashboard

```python
# Cell 9: Create Visualization Dashboard
import matplotlib.pyplot as plt
import seaborn as sns

class LogVisualizationDashboard:
    def __init__(self, df):
        self.df = df
        
    def plot_anomaly_detection_results(self):
        """Plot anomaly detection results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Anomaly scores distribution
        axes[0, 0].hist(self.df['anomaly_score'], bins=50, alpha=0.7, color='skyblue')
        axes[0, 0].axvline(x=-0.5, color='red', linestyle='--', label='Threshold')
        axes[0, 0].set_title('Distribution of Anomaly Scores')
        axes[0, 0].set_xlabel('Anomaly Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        
        # 2. Anomalies by hour
        hourly_anomalies = self.df.groupby('hour')['is_anomaly'].sum()
        axes[0, 1].bar(hourly_anomalies.index, hourly_anomalies.values, alpha=0.7, color='orange')
        axes[0, 1].set_title('Anomalies by Hour of Day')
        axes[0, 1].set_xlabel('Hour')
        axes[0, 1].set_ylabel('Number of Anomalies')
        
        # 3. Anomalies by user
        user_anomalies = self.df.groupby('user')['is_anomaly'].sum().sort_values(ascending=False)
        axes[1, 0].bar(user_anomalies.index, user_anomalies.values, alpha=0.7, color='lightcoral')
        axes[1, 0].set_title('Anomalies by User')
        axes[1, 0].set_xlabel('User')
        axes[1, 0].set_ylabel('Number of Anomalies')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Session duration vs Data size (colored by anomaly)
        normal_data = self.df[self.df['is_anomaly'] == False]
        anomaly_data = self.df[self.df['is_anomaly'] == True]
        
        axes[1, 1].scatter(normal_data['session_duration'], normal_data['data_size'], 
                          alpha=0.6, label='Normal', color='blue', s=20)
        axes[1, 1].scatter(anomaly_data['session_duration'], anomaly_data['data_size'], 
                          alpha=0.8, label='Anomaly', color='red', s=30)
        axes[1, 1].set_title('Session Duration vs Data Size')
        axes[1, 1].set_xlabel('Session Duration (seconds)')
        axes[1, 1].set_ylabel('Data Size (bytes)')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.show()
    
    def plot_model_performance(self):
        """Plot model performance metrics"""
        from sklearn.metrics import confusion_matrix, classification_report
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Confusion matrix for Isolation Forest
        cm_if = confusion_matrix(self.df['is_anomaly'], self.df['pred_isolation_forest'])
        sns.heatmap(cm_if, annot=True, fmt='d', cmap='Blues', ax=axes[0])
        axes[0].set_title('Isolation Forest - Confusion Matrix')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('Actual')
        
        # Confusion matrix for DBSCAN
        cm_db = confusion_matrix(self.df['is_anomaly'], self.df['pred_dbscan'])
        sns.heatmap(cm_db, annot=True, fmt='d', cmap='Greens', ax=axes[1])
        axes[1].set_title('DBSCAN - Confusion Matrix')
        axes[1].set_xlabel('Predicted')
        axes[1].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.show()
        
        # Print classification reports
        print("Isolation Forest Classification Report:")
        print(classification_report(self.df['is_anomaly'], self.df['pred_isolation_forest']))
        
        print("\nDBSCAN Classification Report:")
        print(classification_report(self.df['is_anomaly'], self.df['pred_dbscan']))

# Create and display visualizations
dashboard = LogVisualizationDashboard(df_processed)
dashboard.plot_anomaly_detection_results()
dashboard.plot_model_performance()
```

## Step 8: Continuous Monitoring Loop

```python
# Cell 10: Continuous Monitoring Simulation
import asyncio
from datetime import datetime, timedelta
import time

class ContinuousMonitor:
    def __init__(self, monitoring_system, log_generator):
        self.monitoring_system = monitoring_system
        self.log_generator = log_generator
        self.is_running = False
        
    async def simulate_real_time_monitoring(self, duration_minutes=5, batch_size=10):
        """Simulate real-time log monitoring"""
        print(f"🔄 Starting continuous monitoring for {duration_minutes} minutes...")
        print("Press Ctrl+C to stop monitoring")
        
        self.is_running = True
        start_time = time.time()
        batch_count = 0
        
        try:
            while self.is_running and (time.time() - start_time) < (duration_minutes * 60):
                # Generate new logs (mix of normal and some anomalous)
                normal_batch = self.log_generator.generate_normal_logs(batch_size - 2)
                anomalous_batch = self.log_generator.generate_anomalous_logs(2)
                new_logs = normal_batch + anomalous_batch
                
                # Process the batch
                alerts = await self.monitoring_system.process_log_batch(new_logs)
                
                batch_count += 1
                current_time = datetime.now().strftime("%H:%M:%S")
                
                if alerts:
                    print(f"⚠️  [{current_time}] Batch {batch_count}: {len(alerts)} ALERTS generated!")
                    for alert in alerts:
                        ai_analysis = alert['ai_analysis']
                        print(f"    - {ai_analysis.get('risk_level', 'UNKNOWN')} risk: {ai_analysis.get('threat_type', 'Unknown')}")
                else:
                    print(f"✅ [{current_time}] Batch {batch_count}: No anomalies detected")
                
                # Wait before next batch
                await asyncio.sleep(10)  # Process every 10 seconds
                
        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped by user")
        finally:
            self.is_running = False
            
        # Final summary
        summary = self.monitoring_system.get_alert_summary()
        print(f"\n📈 Final Summary:")
        print(f"Total batches processed: {batch_count}")
        print(f"Total alerts generated: {summary['total_alerts']}")
        if summary['total_alerts'] > 0:
            print(f"Risk distribution: {summary['risk_level_distribution']}")

# Initialize and run continuous monitoring
continuous_monitor = ContinuousMonitor(monitoring_system, log_generator)

# Uncomment to run continuous monitoring (will run for 2 minutes)
# await continuous_monitor.simulate_real_time_monitoring(duration_minutes=2, batch_size=5)
```

## Step 9: Advanced Pattern Detection

```python
# Cell 11: Advanced Pattern Detection and Analysis
class AdvancedPatternDetector:
    def __init__(self, df):
        self.df = df
        
    def detect_privilege_escalation_patterns(self) -> List[Dict]:
        """Detect privilege escalation patterns"""
        escalation_alerts = []
        
        # Group by user and look for privilege escalation patterns
        for user in self.df['user'].unique():
            user_logs = self.df[self.df['user'] == user].sort_values('timestamp')
            
            # Look for privilege escalation followed by sensitive actions
            escalation_indices = user_logs[user_logs['action'] == 'privilege_escalation'].index
            
            for esc_idx in escalation_indices:
                # Check actions within next 1 hour
                esc_time = user_logs.loc[esc_idx, 'timestamp']
                next_hour_logs = user_logs[
                    (user_logs['timestamp'] > esc_time) & 
                    (user_logs['timestamp'] <= esc_time + pd.Timedelta(hours=1))
                ]
                
                # Check for suspicious activities
                suspicious_actions = ['data_export', 'system_command', 'file_access']
                if any(action in next_hour_logs['action'].values for action in suspicious_actions):
                    alert = {
                        'user': user,
                        'escalation_time': esc_time,
                        'subsequent_actions': next_hour_logs['action'].tolist(),
                        'risk_score': len(next_hour_logs) * 0.3,
                        'pattern': 'privilege_escalation_followed_by_sensitive_actions'
                    }
                    escalation_alerts.append(alert)
        
        return escalation_alerts
    
    def detect_unusual_access_patterns(self) -> List[Dict]:
        """Detect unusual access patterns"""
        unusual_patterns = []
        
        # Detect off-hours access
        off_hours_logs = self.df[
            (self.df['hour'] < 6) | (self.df['hour'] > 22) |
            (self.df['is_weekend'] == 1)
        ]
        
        if len(off_hours_logs) > 0:
            for _, log in off_hours_logs.iterrows():
                pattern = {
                    'user': log['user'],
                    'timestamp': log['timestamp'],
                    'pattern': 'off_hours_access',
                    'risk_score': 0.6 if log['is_weekend'] else 0.4,
                    'details': f"Access at {log['hour']}:00 on {'weekend' if log['is_weekend'] else 'weekday'}"
                }
                unusual_patterns.append(pattern)
        
        # Detect rapid successive logins
        for user in self.df['user'].unique():
            user_logins = self.df[
                (self.df['user'] == user) & 
                (self.df['action'] == 'login')
            ].sort_values('timestamp')
            
            if len(user_logins) > 1:
                time_diffs = user_logins['timestamp'].diff().dt.total_seconds()
                rapid_logins = time_diffs < 60  # Less than 1 minute between logins
                
                if rapid_logins.any():
                    pattern = {
                        'user': user,
                        'pattern': 'rapid_successive_logins',
                        'risk_score': 0.7,
                        'count': rapid_logins.sum(),
                        'details': f"{rapid_logins.sum()} rapid logins detected"
                    }
                    unusual_patterns.append(pattern)
        
        return unusual_patterns
    
    def detect_data_exfiltration_indicators(self) -> List[Dict]:
        """Detect potential data exfiltration"""
        exfiltration_alerts = []
        
        # Large data transfers
        large_transfers = self.df[
            (self.df['action'] == 'data_export') & 
            (self.df['data_size'] > self.df['data_size'].quantile(0.95))
        ]
        
        for _, transfer in large_transfers.iterrows():
            alert = {
                'user': transfer['user'],
                'timestamp': transfer['timestamp'],
                'pattern': 'large_data_transfer',
                'data_size': transfer['data_size'],
                'risk_score': min(1.0, transfer['data_size'] / self.df['data_size'].max()),
                'details': f"Transfer of {transfer['data_size']} bytes"
            }
            exfiltration_alerts.append(alert)
        
        return exfiltration_alerts

# Run advanced pattern detection
pattern_detector = AdvancedPatternDetector(df_processed)

print("🔍 Advanced Pattern Detection Results:")
print("=" * 50)

# Detect privilege escalation patterns
escalation_patterns = pattern_detector.detect_privilege_escalation_patterns()
print(f"\n🚨 Privilege Escalation Patterns: {len(escalation_patterns)}")
for pattern in escalation_patterns[:3]:  # Show first 3
    print(f"  - User: {pattern['user']}, Risk: {pattern['risk_score']:.2f}")
    print(f"    Pattern: {pattern['pattern']}")

# Detect unusual access patterns
unusual_patterns = pattern_detector.detect_unusual_access_patterns()
print(f"\n⚠️  Unusual Access Patterns: {len(unusual_patterns)}")
for pattern in unusual_patterns[:3]:  # Show first 3
    print(f"  - User: {pattern['user']}, Risk: {pattern['risk_score']:.2f}")
    print(f"    Details: {pattern['details']}")

# Detect data exfiltration indicators
exfiltration_patterns = pattern_detector.detect_data_exfiltration_indicators()
print(f"\n📤 Data Exfiltration Indicators: {len(exfiltration_patterns)}")
for pattern in exfiltration_patterns[:3]:  # Show first 3
    print(f"  - User: {pattern['user']}, Risk: {pattern['risk_score']:.2f}")
    print(f"    Details: {pattern['details']}")
```

## Step 10: Configuration and Deployment

```python
# Cell 12: Configuration Management and Deployment Setup
import json
import yaml
from typing import Dict, Any

class IDSConfiguration:
    def __init__(self):
        self.config = {
            "detection_models": {
                "isolation_forest": {
                    "contamination": 0.1,
                    "random_state": 42,
                    "enabled": True
                },
                "dbscan": {
                    "eps": 0.5,
                    "min_samples": 5,
                    "enabled": True
                }
            },
            "thresholds": {
                "anomaly_score_threshold": -0.5,
                "risk_levels": {
                    "low": 0.3,
                    "medium": 0.6,
                    "high": 0.8,
                    "critical": 0.9
                }
            },
            "monitoring": {
                "batch_size": 10,
                "processing_interval_seconds": 10,
                "alert_retention_days": 30
            },
            "ai_analysis": {
                "model": "gpt-3.5-turbo",
                "max_tokens": 500,
                "temperature": 0.1
            },
            "patterns": {
                "privilege_escalation_window_hours": 1,
                "off_hours": {
                    "start": 22,
                    "end": 6
                },
                "rapid_login_threshold_seconds": 60,
                "large_transfer_percentile": 95
            }
        }
    
    def save_config(self, filepath: str):
        """Save configuration to file"""
        with open(filepath, 'w') as f:
            json.dump(self.config, f, indent=2)
        print(f"✅ Configuration saved to {filepath}")
    
    def load_config(self, filepath: str):
        """Load configuration from file"""
        with open(filepath, 'r') as f:
            self.config = json.load(f)
        print(f"✅ Configuration loaded from {filepath}")
    
    def get_config(self) -> Dict[str, Any]:
        return self.config

# Save configuration
config_manager = IDSConfiguration()
config_manager.save_config("ids_config.json")

# Display current configuration
print("📋 Current IDS Configuration:")
print(json.dumps(config_manager.get_config(), indent=2))
```

## Complete Usage Example

```python
# Cell 13: Complete End-to-End Example
async def complete_ids_demo():
    """Complete demonstration of the IDS system"""
    
    print("🚀 Starting Complete Intrusion Detection System Demo")
    print("=" * 60)
    
    # 1. Initialize components
    print("\n1️⃣ Initializing IDS components...")
    log_gen = LogGenerator()
    preprocessor = LogPreprocessor()
    detector = AnomalyDetector()
    ai_analyzer = IntelligentLogAnalyzer(client)
    
    # 2. Generate and preprocess data
    print("\n2️⃣ Generating and preprocessing log data...")
    normal_logs = log_gen.generate_normal_logs(500)
    anomalous_logs = log_gen.generate_anomalous_logs(50)
    all_logs = normal_logs + anomalous_logs
    
    df = pd.DataFrame(all_logs)
    df['is_anomaly'] = [False] * len(normal_logs) + [True] * len(anomalous_logs)
    
    df_processed = preprocessor.preprocess_logs(df)
    feature_cols = preprocessor.get_feature_columns()
    
    # 3. Train models
    print("\n3️⃣ Training anomaly detection models...")
    X = df_processed[feature_cols].values
    detector.fit(X)
    
    # 4. Initialize monitoring system
    print("\n4️⃣ Setting up monitoring system...")
    monitoring_sys = LogMonitoringSystem(detector, ai_analyzer)
    
    # 5. Process some test logs
    print("\n5️⃣ Processing test batch...")
    test_logs = df[df['is_anomaly'] == True].head(3).to_dict('records')
    alerts = await monitoring_sys.process_log_batch(test_logs)
    
    print(f"Generated {len(alerts)} alerts from {len(test_logs)} test logs")
    
    # 6. Show results
    for i, alert in enumerate(alerts, 1):
        print(f"\n🚨 Alert {i}:")
        print(f"  Risk Level: {alert['ai_analysis'].get('risk_level')}")
        print(f"  Threat: {alert['ai_analysis'].get('threat_type')}")
        print(f"  User: {alert['log_entry']['user']}")
        print(f"  Action: {alert['log_entry']['action']}")
    
    # 7. Pattern analysis
    print("\n6️⃣ Running pattern analysis...")
    pattern_det = AdvancedPatternDetector(df_processed)
    escalation_patterns = pattern_det.detect_privilege_escalation_patterns()
    unusual_patterns = pattern_det.detect_unusual_access_patterns()
    
    print(f"Found {len(escalation_patterns)} privilege escalation patterns")
    print(f"Found {len(unusual_patterns)} unusual access patterns")
    
    print("\n✅ IDS Demo completed successfully!")
    
    return {
        "alerts": alerts,
        "escalation_patterns": escalation_patterns,
        "unusual_patterns": unusual_patterns,
        "total_logs_processed": len(all_logs)
    }

# Run the complete demo
demo_results = await complete_ids_demo()
```

## Summary and Next Steps

This comprehensive Intrusion Detection System provides:

### Key Features:
1. **Multi-layered Detection**: Uses Isolation Forest and DBSCAN for anomaly detection
2. **AI-Powered Analysis**: OpenAI integration for intelligent threat assessment
3. **Pattern Recognition**: Advanced pattern detection for privilege escalation and unusual access
4. **Real-time Monitoring**: Continuous log processing and alerting
5. **Visualization**: Comprehensive dashboards for security monitoring

### To Deploy in Production:

1. **Database Integration**: Connect to your actual log sources (Syslog, Windows Event Logs, etc.)
2. **Scaling**: Implement distributed processing for high-volume logs
3. **Alert Management**: Integrate with SIEM systems or notification platforms
4. **Model Updates**: Implement continuous learning and model retraining
5. **Security**: Add authentication, encryption, and secure API endpoints

### Configuration Files:
- Save the `ids_config.json` for deployment settings
- Customize thresholds based on your environment
- Set up monitoring intervals appropriate for your log volume

This system provides a solid foundation for log-based intrusion detection with AI enhancement!
