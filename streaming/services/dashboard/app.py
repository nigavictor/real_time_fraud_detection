#!/usr/bin/env python3
"""
Real-Time Fraud Detection Dashboard
==================================

A Flask-based web dashboard for monitoring real-time fraud detection metrics,
alerts, and system performance.

Features:
- Live transaction and alert monitoring  
- Performance metrics visualization
- Model performance tracking
- Alert investigation interface
- System health monitoring
"""

import os
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any

import redis
import psycopg2
from psycopg2.extras import RealDictCursor
from flask import Flask, render_template, jsonify, request
from kafka import KafkaConsumer
import threading
import queue

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-key-change-in-production')

# Configuration
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
POSTGRES_HOST = os.getenv('POSTGRES_HOST', 'localhost')
POSTGRES_USER = os.getenv('POSTGRES_USER', 'fraud_user')
POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD', 'fraud_password')
POSTGRES_DB = os.getenv('POSTGRES_DB', 'fraud_detection')
KAFKA_BOOTSTRAP_SERVERS = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')

# Global connections
redis_client = None
postgres_conn = None
alert_queue = queue.Queue(maxsize=1000)

def initialize_connections():
    """Initialize database connections."""
    global redis_client, postgres_conn
    
    try:
        # Redis connection
        redis_client = redis.Redis(
            host=REDIS_HOST,
            port=6379,
            decode_responses=True,
            socket_connect_timeout=5
        )
        redis_client.ping()
        print("✓ Redis connection established")
        
        # PostgreSQL connection
        postgres_conn = psycopg2.connect(
            host=POSTGRES_HOST,
            port=5432,
            database=POSTGRES_DB,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD
        )
        print("✓ PostgreSQL connection established")
        
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        raise

def kafka_alert_consumer():
    """Background thread to consume fraud alerts from Kafka."""
    try:
        consumer = KafkaConsumer(
            'fraud-alerts',
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            value_deserializer=lambda x: json.loads(x.decode('utf-8')),
            group_id='dashboard-consumer',
            auto_offset_reset='latest'
        )
        
        print("✓ Kafka alert consumer started")
        
        for message in consumer:
            try:
                alert = message.value
                alert['received_at'] = datetime.now().isoformat()
                
                # Add to queue for real-time updates
                if not alert_queue.full():
                    alert_queue.put(alert)
                
            except Exception as e:
                print(f"Error processing alert: {e}")
                
    except Exception as e:
        print(f"Kafka consumer error: {e}")

# Routes
@app.route('/')
def dashboard():
    """Main dashboard page."""
    return render_template('dashboard.html')

@app.route('/api/metrics/overview')
def metrics_overview():
    """Get overview metrics for the dashboard."""
    try:
        # Get metrics from Redis
        total_predictions = int(redis_client.get('fraud_detector:prediction_count') or 0)
        avg_inference_time = float(redis_client.get('fraud_detector:avg_inference_time') or 0.0)
        
        # Get recent alerts from PostgreSQL
        with postgres_conn.cursor(cursor_factory=RealDictCursor) as cursor:
            # Total alerts today
            cursor.execute("""
                SELECT COUNT(*) as total_alerts,
                       COUNT(*) FILTER (WHERE fraud_probability > 0.8) as high_confidence,
                       COUNT(*) FILTER (WHERE fraud_probability > 0.5 AND fraud_probability <= 0.8) as medium_confidence,
                       AVG(fraud_probability) as avg_fraud_probability
                FROM fraud_alerts 
                WHERE DATE(created_at) = CURRENT_DATE
            """)
            daily_stats = cursor.fetchone()
            
            # Hourly alert trend (last 24 hours)
            cursor.execute("""
                SELECT DATE_TRUNC('hour', created_at) as hour,
                       COUNT(*) as alert_count
                FROM fraud_alerts 
                WHERE created_at >= NOW() - INTERVAL '24 hours'
                GROUP BY DATE_TRUNC('hour', created_at)
                ORDER BY hour
            """)
            hourly_trends = cursor.fetchall()
            
        # Current system status
        current_time = datetime.now()
        
        return jsonify({
            'status': 'healthy',
            'timestamp': current_time.isoformat(),
            'metrics': {
                'total_predictions_today': total_predictions,
                'avg_inference_time_ms': round(avg_inference_time, 2),
                'total_alerts_today': daily_stats['total_alerts'] or 0,
                'high_confidence_alerts': daily_stats['high_confidence'] or 0,
                'medium_confidence_alerts': daily_stats['medium_confidence'] or 0,
                'avg_fraud_probability': round(float(daily_stats['avg_fraud_probability'] or 0), 3),
                'system_uptime_hours': 24,  # Simplified
                'processing_rate_tps': round(total_predictions / (24 * 3600), 1) if total_predictions > 0 else 0
            },
            'trends': {
                'hourly_alerts': [
                    {
                        'hour': trend['hour'].strftime('%H:%M'),
                        'count': trend['alert_count']
                    } for trend in hourly_trends
                ]
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/alerts/recent')
def recent_alerts():
    """Get recent fraud alerts."""
    try:
        limit = int(request.args.get('limit', 50))
        
        with postgres_conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("""
                SELECT transaction_id, user_id, timestamp, amount, merchant_category,
                       location, fraud_probability, model_used, investigation_status,
                       created_at
                FROM fraud_alerts 
                ORDER BY created_at DESC 
                LIMIT %s
            """, (limit,))
            
            alerts = cursor.fetchall()
            
        # Convert datetime objects to strings
        for alert in alerts:
            if alert['timestamp']:
                alert['timestamp'] = alert['timestamp'].isoformat()
            if alert['created_at']:
                alert['created_at'] = alert['created_at'].isoformat()
                
        return jsonify({
            'alerts': alerts,
            'count': len(alerts)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/alerts/live')
def live_alerts():
    """Get live alerts from the queue."""
    try:
        alerts = []
        
        # Get up to 10 alerts from queue without blocking
        while len(alerts) < 10 and not alert_queue.empty():
            try:
                alert = alert_queue.get_nowait()
                alerts.append(alert)
            except queue.Empty:
                break
        
        return jsonify({
            'alerts': alerts,
            'count': len(alerts)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/performance/models')
def model_performance():
    """Get model performance metrics."""
    try:
        with postgres_conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("""
                SELECT model_name, total_predictions, fraud_detections,
                       false_positives, true_positives, avg_response_time_ms,
                       last_updated
                FROM model_performance
                ORDER BY total_predictions DESC
            """)
            
            performance_data = cursor.fetchall()
            
        # Convert datetime objects to strings and calculate metrics
        for perf in performance_data:
            if perf['last_updated']:
                perf['last_updated'] = perf['last_updated'].isoformat()
                
            # Calculate derived metrics
            total_pred = perf['total_predictions'] or 1
            fraud_det = perf['fraud_detections'] or 0
            false_pos = perf['false_positives'] or 0
            true_pos = perf['true_positives'] or 0
            
            perf['detection_rate'] = round(fraud_det / total_pred * 100, 1)
            perf['false_positive_rate'] = round(false_pos / total_pred * 100, 1)
            perf['precision'] = round(true_pos / (true_pos + false_pos) * 100, 1) if (true_pos + false_pos) > 0 else 0
        
        return jsonify({
            'models': performance_data
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/system/health')
def system_health():
    """Get system health status."""
    try:
        health_status = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'healthy',
            'components': {}
        }
        
        # Check Redis
        try:
            redis_client.ping()
            health_status['components']['redis'] = {
                'status': 'healthy',
                'response_time_ms': 1.0  # Simplified
            }
        except Exception as e:
            health_status['components']['redis'] = {
                'status': 'unhealthy',
                'error': str(e)
            }
            health_status['overall_status'] = 'degraded'
        
        # Check PostgreSQL
        try:
            with postgres_conn.cursor() as cursor:
                cursor.execute('SELECT 1')
            health_status['components']['postgres'] = {
                'status': 'healthy',
                'response_time_ms': 2.0  # Simplified
            }
        except Exception as e:
            health_status['components']['postgres'] = {
                'status': 'unhealthy', 
                'error': str(e)
            }
            health_status['overall_status'] = 'degraded'
        
        # Check Kafka (simplified - check if recent predictions exist)
        try:
            recent_predictions = int(redis_client.get('fraud_detector:prediction_count') or 0)
            if recent_predictions > 0:
                health_status['components']['kafka'] = {
                    'status': 'healthy',
                    'recent_messages': recent_predictions
                }
            else:
                health_status['components']['kafka'] = {
                    'status': 'warning',
                    'message': 'No recent activity'
                }
        except Exception as e:
            health_status['components']['kafka'] = {
                'status': 'unknown',
                'error': str(e)
            }
        
        return jsonify(health_status)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/alerts/<transaction_id>/investigate', methods=['POST'])
def investigate_alert(transaction_id):
    """Update alert investigation status."""
    try:
        data = request.get_json()
        new_status = data.get('status', 'investigating')
        notes = data.get('notes', '')
        
        with postgres_conn.cursor() as cursor:
            cursor.execute("""
                UPDATE fraud_alerts 
                SET investigation_status = %s,
                    updated_at = CURRENT_TIMESTAMP
                WHERE transaction_id = %s
            """, (new_status, transaction_id))
            
        postgres_conn.commit()
        
        return jsonify({
            'success': True,
            'message': f'Alert {transaction_id} updated to {new_status}'
        })
        
    except Exception as e:
        postgres_conn.rollback()
        return jsonify({'error': str(e)}), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

# Template for dashboard HTML
@app.route('/static/dashboard.html')
def dashboard_template():
    """Serve the dashboard template."""
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Real-Time Fraud Detection Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .metric-card {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .metric-value {
            font-size: 2em;
            font-weight: bold;
            color: #333;
        }
        .metric-label {
            color: #666;
            margin-bottom: 10px;
        }
        .alert-list {
            background: white;
            border-radius: 10px;
            padding: 20px;
            max-height: 600px;
            overflow-y: auto;
        }
        .alert-item {
            border-left: 4px solid #ff4757;
            padding: 15px;
            margin: 10px 0;
            background: #f8f9fa;
        }
        .alert-high {
            border-left-color: #ff4757;
        }
        .alert-medium {
            border-left-color: #ffa726;
        }
        .alert-low {
            border-left-color: #66bb6a;
        }
        .status-healthy { color: #4caf50; }
        .status-warning { color: #ff9800; }
        .status-error { color: #f44336; }
        #alertChart {
            margin-top: 20px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🛡️ Real-Time Fraud Detection Dashboard</h1>
        <p>Live monitoring of fraud detection system performance and alerts</p>
    </div>

    <div class="metrics-grid">
        <div class="metric-card">
            <div class="metric-label">Total Predictions Today</div>
            <div class="metric-value" id="totalPredictions">--</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Fraud Alerts Today</div>
            <div class="metric-value" id="fraudAlerts">--</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Avg Response Time</div>
            <div class="metric-value" id="responseTime">--</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">System Status</div>
            <div class="metric-value" id="systemStatus">--</div>
        </div>
    </div>

    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
        <div>
            <h3>Alert Trend (Last 24h)</h3>
            <canvas id="alertChart" width="400" height="200"></canvas>
        </div>
        <div>
            <h3>Recent Fraud Alerts</h3>
            <div class="alert-list" id="alertList">
                <div>Loading alerts...</div>
            </div>
        </div>
    </div>

    <script>
        let alertChart;

        // Initialize dashboard
        function initDashboard() {
            // Initialize chart
            const ctx = document.getElementById('alertChart').getContext('2d');
            alertChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Fraud Alerts',
                        data: [],
                        borderColor: '#ff4757',
                        tension: 0.1
                    }]
                },
                options: {
                    responsive: true,
                    scales: {
                        y: {
                            beginAtZero: true
                        }
                    }
                }
            });

            // Start data refresh
            refreshDashboard();
            setInterval(refreshDashboard, 5000); // Refresh every 5 seconds
        }

        // Refresh dashboard data
        async function refreshDashboard() {
            try {
                // Get overview metrics
                const metricsResponse = await fetch('/api/metrics/overview');
                const metrics = await metricsResponse.json();

                // Update metrics
                document.getElementById('totalPredictions').textContent = metrics.metrics.total_predictions_today.toLocaleString();
                document.getElementById('fraudAlerts').textContent = metrics.metrics.total_alerts_today.toLocaleString();
                document.getElementById('responseTime').textContent = metrics.metrics.avg_inference_time_ms + 'ms';

                // Get system health
                const healthResponse = await fetch('/api/system/health');
                const health = await healthResponse.json();
                const statusElement = document.getElementById('systemStatus');
                statusElement.textContent = health.overall_status.toUpperCase();
                statusElement.className = 'metric-value status-' + health.overall_status;

                // Update chart
                if (metrics.trends && metrics.trends.hourly_alerts) {
                    const labels = metrics.trends.hourly_alerts.map(t => t.hour);
                    const data = metrics.trends.hourly_alerts.map(t => t.count);
                    
                    alertChart.data.labels = labels;
                    alertChart.data.datasets[0].data = data;
                    alertChart.update();
                }

                // Get recent alerts
                const alertsResponse = await fetch('/api/alerts/recent?limit=10');
                const alertsData = await alertsResponse.json();

                const alertList = document.getElementById('alertList');
                if (alertsData.alerts && alertsData.alerts.length > 0) {
                    alertList.innerHTML = alertsData.alerts.map(alert => `
                        <div class="alert-item alert-${getAlertPriority(alert.fraud_probability)}">
                            <div><strong>Transaction:</strong> ${alert.transaction_id}</div>
                            <div><strong>User:</strong> ${alert.user_id}</div>
                            <div><strong>Amount:</strong> $${alert.amount}</div>
                            <div><strong>Probability:</strong> ${(alert.fraud_probability * 100).toFixed(1)}%</div>
                            <div><strong>Model:</strong> ${alert.model_used}</div>
                            <div><strong>Status:</strong> ${alert.investigation_status}</div>
                            <div><strong>Time:</strong> ${new Date(alert.created_at).toLocaleString()}</div>
                        </div>
                    `).join('');
                } else {
                    alertList.innerHTML = '<div>No recent alerts</div>';
                }

            } catch (error) {
                console.error('Error refreshing dashboard:', error);
            }
        }

        function getAlertPriority(probability) {
            if (probability >= 0.8) return 'high';
            if (probability >= 0.5) return 'medium';
            return 'low';
        }

        // Initialize when page loads
        window.addEventListener('load', initDashboard);
    </script>
</body>
</html>
    """
    return html_content

# Create templates directory and dashboard template
def create_dashboard_template():
    """Create the dashboard template."""
    template_dir = 'templates'
    os.makedirs(template_dir, exist_ok=True)
    
    template_path = os.path.join(template_dir, 'dashboard.html')
    with open(template_path, 'w') as f:
        f.write(dashboard_template())

if __name__ == '__main__':
    print("🚀 Starting Real-Time Fraud Detection Dashboard...")
    
    # Initialize connections
    initialize_connections()
    
    # Create dashboard template
    create_dashboard_template()
    
    # Start Kafka consumer in background thread
    consumer_thread = threading.Thread(target=kafka_alert_consumer, daemon=True)
    consumer_thread.start()
    
    # Run Flask app
    app.run(
        host='0.0.0.0',
        port=int(os.getenv('PORT', 8080)),
        debug=os.getenv('FLASK_DEBUG', 'false').lower() == 'true'
    )