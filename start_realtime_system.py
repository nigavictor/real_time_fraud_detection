#!/usr/bin/env python3
"""
Real-Time Fraud Detection System Launcher
=========================================

This script sets up and launches the complete real-time fraud detection system
including Kafka, transaction generator, fraud detector, and monitoring dashboard.

Usage:
    python start_realtime_system.py [--mode=demo|full] [--transactions-per-second=10]
"""

import os
import sys
import subprocess
import time
import argparse
import signal
from datetime import datetime

def check_dependencies():
    """Check if required dependencies are installed."""
    print("🔍 Checking dependencies...")
    
    missing_commands = []
    
    # Check Docker
    try:
        subprocess.run(['docker', '--version'], 
                     stdout=subprocess.DEVNULL, 
                     stderr=subprocess.DEVNULL, 
                     check=True)
        print("✓ docker is available")
    except (subprocess.CalledProcessError, FileNotFoundError):
        missing_commands.append('docker')
        print("❌ docker is not available")
    
    # Check Docker Compose (new syntax)
    try:
        subprocess.run(['docker', 'compose', 'version'], 
                     stdout=subprocess.DEVNULL, 
                     stderr=subprocess.DEVNULL, 
                     check=True)
        print("✓ docker compose is available")
    except (subprocess.CalledProcessError, FileNotFoundError):
        missing_commands.append('docker compose')
        print("❌ docker compose is not available")
    
    if missing_commands:
        print(f"\n❌ Missing dependencies: {missing_commands}")
        print("Please install Docker and Docker Compose:")
        print("  - Docker: https://docs.docker.com/get-docker/")
        print("  - Docker Compose: https://docs.docker.com/compose/install/")
        return False
    
    print("✅ All dependencies are available!")
    return True

def check_models():
    """Check if trained models are available."""
    print("🤖 Checking for trained models...")
    
    model_dirs = []
    models_path = "data/models"
    
    if os.path.exists(models_path):
        model_dirs = [d for d in os.listdir(models_path) if d.startswith('trained_models_')]
    
    if not model_dirs:
        print("❌ No trained models found!")
        print("Please run the training pipeline first:")
        print("  python demo_pipeline.py")
        print("  # or")  
        print("  python main.py --step train")
        return False
    
    latest_model_dir = sorted(model_dirs)[-1]
    print(f"✓ Found trained models: {latest_model_dir}")
    
    # Check if required model files exist
    model_files = [
        'xgboost_model.joblib',
        'random_forest_model.joblib',
        'ensemble_model.joblib'
    ]
    
    model_path = os.path.join(models_path, latest_model_dir)
    missing_files = []
    
    for model_file in model_files:
        if not os.path.exists(os.path.join(model_path, model_file)):
            missing_files.append(model_file)
    
    if missing_files:
        print(f"❌ Missing model files: {missing_files}")
        return False
    
    print("✅ All required model files are available!")
    return True

def update_docker_compose_config(transactions_per_second=10, fraud_rate=0.05):
    """Update docker-compose environment variables."""
    print(f"⚙️  Configuring system: {transactions_per_second} TPS, {fraud_rate*100:.1f}% fraud rate")
    
    # Update environment variables in docker-compose.yml
    # This is a simplified approach - in production you'd use environment files
    env_vars = {
        'TRANSACTIONS_PER_SECOND': str(transactions_per_second),
        'FRAUD_RATE': str(fraud_rate)
    }
    
    return env_vars

def start_infrastructure():
    """Start Kafka, Redis, and PostgreSQL infrastructure."""
    print("🚀 Starting infrastructure services...")
    
    try:
        # Start infrastructure services only
        result = subprocess.run([
            'docker', 'compose', 'up', '-d',
            'zookeeper', 'kafka', 'redis', 'postgres'
        ], check=True, capture_output=True, text=True)
        
        print("✓ Infrastructure services started")
        
        # Wait for services to be ready
        print("⏳ Waiting for services to initialize...")
        time.sleep(20)
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start infrastructure: {e.stderr}")
        return False

def start_fraud_system():
    """Start fraud detection components."""
    print("🛡️ Starting fraud detection system...")
    
    try:
        # Start all fraud detection services
        result = subprocess.run([
            'docker', 'compose', 'up', '-d',
            'fraud-detector', 'transaction-generator', 'dashboard'
        ], check=True, capture_output=True, text=True)
        
        print("✓ Fraud detection system started")
        time.sleep(5)
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start fraud system: {e.stderr}")
        return False

def show_system_status():
    """Show system status and access URLs."""
    print("\n" + "="*60)
    print("🎉 REAL-TIME FRAUD DETECTION SYSTEM RUNNING!")
    print("="*60)
    
    print("\n📊 Access Points:")
    print("├── 🖥️  Fraud Detection Dashboard: http://localhost:8080")
    print("├── 📈 Kafka UI (Monitoring): http://localhost:8090") 
    print("├── 🔄 Redis: localhost:6379")
    print("└── 🗄️  PostgreSQL: localhost:5432")
    
    print("\n🔍 System Components:")
    try:
        result = subprocess.run(['docker', 'compose', 'ps'], 
                              capture_output=True, text=True, check=True)
        print(result.stdout)
    except subprocess.CalledProcessError:
        print("❌ Could not get system status")
    
    print("\n📝 Log Monitoring:")
    print("├── All services: docker-compose logs -f")
    print("├── Generator: docker-compose logs -f transaction-generator") 
    print("├── Detector: docker-compose logs -f fraud-detector")
    print("└── Dashboard: docker-compose logs -f dashboard")
    
    print("\n⚡ System Features:")
    print("├── Real-time transaction generation")
    print("├── <10ms fraud detection inference") 
    print("├── Live monitoring dashboard")
    print("├── Fraud alert generation")
    print("└── Performance metrics tracking")
    
    print(f"\n🕐 System started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def monitor_system():
    """Monitor system logs in real-time."""
    print("\n🔍 Monitoring system logs (Ctrl+C to stop)...")
    
    try:
        subprocess.run(['docker', 'compose', 'logs', '-f', '--tail=50'])
    except KeyboardInterrupt:
        print("\n⏹️  Log monitoring stopped")

def stop_system():
    """Stop the entire system."""
    print("\n🛑 Stopping real-time fraud detection system...")
    
    try:
        subprocess.run(['docker', 'compose', 'down'], check=True)
        print("✓ System stopped successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error stopping system: {e}")

def signal_handler(signum, frame):
    """Handle interrupt signals gracefully."""
    print(f"\n\n🛑 Received signal {signum}, shutting down...")
    stop_system()
    sys.exit(0)

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Launch Real-Time Fraud Detection System')
    parser.add_argument('--mode', choices=['demo', 'full'], default='demo',
                       help='System mode: demo (fast) or full (production-like)')
    parser.add_argument('--transactions-per-second', type=int, default=10,
                       help='Number of transactions per second to generate')
    parser.add_argument('--fraud-rate', type=float, default=0.05,
                       help='Fraud rate (0.05 = 5%%)')
    parser.add_argument('--monitor', action='store_true',
                       help='Monitor system logs after startup')
    parser.add_argument('--stop', action='store_true',
                       help='Stop the running system')
    
    args = parser.parse_args()
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Handle stop command
    if args.stop:
        stop_system()
        return
    
    print("🛡️ REAL-TIME FRAUD DETECTION SYSTEM LAUNCHER")
    print("="*60)
    print(f"Mode: {args.mode}")
    print(f"Transaction Rate: {args.transactions_per_second} TPS")
    print(f"Fraud Rate: {args.fraud_rate*100:.1f}%")
    print()
    
    # Pre-flight checks
    if not check_dependencies():
        sys.exit(1)
    
    if not check_models():
        print("\n💡 Tip: Run the demo pipeline first to train models:")
        print("   python demo_pipeline.py")
        sys.exit(1)
    
    # Configure system
    env_vars = update_docker_compose_config(
        args.transactions_per_second, 
        args.fraud_rate
    )
    
    # Set environment variables
    for key, value in env_vars.items():
        os.environ[key] = value
    
    try:
        # Start system components
        if not start_infrastructure():
            sys.exit(1)
            
        if not start_fraud_system():
            sys.exit(1)
        
        # Show system status
        show_system_status()
        
        # Monitor logs if requested
        if args.monitor:
            monitor_system()
        else:
            print("\n✅ System is running! Press Ctrl+C to stop or use --monitor to view logs")
            print("💡 Visit http://localhost:8080 to view the fraud detection dashboard")
            
            # Keep main process running
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                pass
                
    except KeyboardInterrupt:
        pass
    finally:
        stop_system()

if __name__ == "__main__":
    main()