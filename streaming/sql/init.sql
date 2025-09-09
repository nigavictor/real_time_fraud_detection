-- Initialize database for fraud detection real-time system

-- Create tables for storing fraud alerts and metrics
CREATE TABLE IF NOT EXISTS fraud_alerts (
    id SERIAL PRIMARY KEY,
    transaction_id VARCHAR(255) NOT NULL,
    user_id VARCHAR(255) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    amount DECIMAL(10,2) NOT NULL,
    merchant_category VARCHAR(100),
    location VARCHAR(255),
    fraud_probability DECIMAL(5,4) NOT NULL,
    model_used VARCHAR(50) NOT NULL,
    is_confirmed_fraud BOOLEAN DEFAULT NULL,
    investigation_status VARCHAR(50) DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create table for real-time metrics
CREATE TABLE IF NOT EXISTS real_time_metrics (
    id SERIAL PRIMARY KEY,
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(10,4) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create table for model performance tracking
CREATE TABLE IF NOT EXISTS model_performance (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(50) NOT NULL,
    total_predictions INTEGER DEFAULT 0,
    fraud_detections INTEGER DEFAULT 0,
    false_positives INTEGER DEFAULT 0,
    true_positives INTEGER DEFAULT 0,
    avg_response_time_ms DECIMAL(8,2) DEFAULT 0,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_fraud_alerts_timestamp ON fraud_alerts(timestamp);
CREATE INDEX IF NOT EXISTS idx_fraud_alerts_user_id ON fraud_alerts(user_id);
CREATE INDEX IF NOT EXISTS idx_fraud_alerts_fraud_probability ON fraud_alerts(fraud_probability);
CREATE INDEX IF NOT EXISTS idx_real_time_metrics_timestamp ON real_time_metrics(timestamp);
CREATE INDEX IF NOT EXISTS idx_model_performance_model_name ON model_performance(model_name);

-- Insert initial model performance records
INSERT INTO model_performance (model_name, total_predictions, fraud_detections, false_positives, true_positives, avg_response_time_ms)
VALUES 
    ('xgboost', 0, 0, 0, 0, 0.0),
    ('ensemble', 0, 0, 0, 0, 0.0),
    ('random_forest', 0, 0, 0, 0, 0.0),
    ('logistic_regression', 0, 0, 0, 0, 0.0)
ON CONFLICT DO NOTHING;

-- Create function to update timestamp on update
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create trigger for fraud_alerts table
CREATE TRIGGER update_fraud_alerts_updated_at 
    BEFORE UPDATE ON fraud_alerts 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

-- Create view for fraud detection summary
CREATE OR REPLACE VIEW fraud_detection_summary AS
SELECT 
    DATE(timestamp) as detection_date,
    COUNT(*) as total_alerts,
    COUNT(*) FILTER (WHERE fraud_probability > 0.8) as high_confidence_alerts,
    COUNT(*) FILTER (WHERE fraud_probability > 0.5 AND fraud_probability <= 0.8) as medium_confidence_alerts,
    COUNT(*) FILTER (WHERE fraud_probability <= 0.5) as low_confidence_alerts,
    AVG(fraud_probability) as avg_fraud_probability,
    SUM(amount) as total_flagged_amount
FROM fraud_alerts
GROUP BY DATE(timestamp)
ORDER BY detection_date DESC;