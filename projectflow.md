Batch data  → flights.csv
Static data → airports.csv, airlines.csv
API data    → weather.py
Generated   → holidays_data.py

JOINING ALL DATA SOURCES

This is:

💥 Hardest part
💥 Most important part
💥 What makes your project “10x MLOps”

You are about to learn:

Temporal joins (flight time ↔ weather time)
Multi-source joins (airport ↔ metadata)
Feature engineering


Excellent project scope! This is a production-grade MLOps system with real complexity. Let me summarize what you're building:

🎯 Your Project Architecture:
Data Sources (4 heterogeneous types):
Technical Challenges You'll Solve:
Multi-schema validation

4 different data formats
Schema drift detection
Complex joins:

Spatial: Flight → Airport (by IATA code)
Temporal: Flight date → Holiday date
Spatio-temporal: Flight + Airport location → Weather station + time
API constraints:

Rate limiting (NOAA: 1000/day)
Incremental updates vs full reload
Feature engineering:

Time-series: rolling averages, lag features
Derived: hour_of_day, airport_congestion, historical_delay_rate
Production MLOps:

Data drift detection
Auto-retraining triggers
Model monitoring (target: 85%+ accuracy)
Expected Resume Impact:
✅ Multi-source data ingestion
✅ Schema validation & versioning
✅ Temporal/spatial joins
✅ Feature engineering pipeline
✅ Drift detection
✅ Automated retraining
✅ Production deployment (K8s target)