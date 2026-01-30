import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

# Generate 61 days of data (like the tutorial)
start_date = datetime(2024, 10, 1)
dates = [start_date + timedelta(days=i) for i in range(61)]

data = []
for i, date in enumerate(dates):
    # Base sales with weekly pattern
    day_of_week = date.weekday()
    
    # Base around $5000 with variation
    base_sales = 5000
    
    # Weekly pattern (weekends slightly higher)
    if day_of_week == 5:  # Saturday
        base_sales *= 1.15
    elif day_of_week == 6:  # Sunday
        base_sales *= 1.10
    elif day_of_week == 0:  # Monday (usually lower)
        base_sales *= 0.90
    
    # Add random variation (+/- 25%)
    variation = np.random.uniform(0.75, 1.25)
    sales = base_sales * variation
    
    # Add some trend (slight growth over time)
    trend = 1 + (i * 0.002)  # 0.2% daily growth
    sales *= trend
    
    # Round to 2 decimal places
    sales = round(sales, 2)
    
    data.append({
        'date': date.strftime('%Y-%m-%d'),
        'store_id': 'store_001',
        'sales': sales
    })

df = pd.DataFrame(data)

# Save to CSV
output_path = 'sample_sales_data.csv'
df.to_csv(output_path, index=False)

print(f"Generated {len(df)} records")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")
print(f"Sales range: ${df['sales'].min():,.2f} - ${df['sales'].max():,.2f}")
print(f"Sales mean: ${df['sales'].mean():,.2f}")
print(f"\nSample data:")
print(df.head(10).to_string(index=False))
print(f"\nSaved to {output_path}")
