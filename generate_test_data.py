import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)
start_date = datetime(2026, 1, 30)
dates = [start_date + timedelta(days=i) for i in range(30)]

data = []
for date in dates:
    day_of_week = date.weekday()
    is_weekend = day_of_week >= 5
    
    for store_id in [1, 2, 3]:
        for product_id in [101, 102, 103]:
            # Base sales between 2000-6000 with normal distribution
            base_sales = np.random.normal(4000, 800)  # Mean 4000, std 800
            
            # Store performance multiplier
            store_multiplier = {1: 1.0, 2: 0.9, 3: 1.1}[store_id]
            
            # Product performance multiplier
            product_multiplier = {101: 1.0, 102: 0.85, 103: 1.15}[product_id]
            
            # Weekend boost
            weekend_boost = 1.2 if is_weekend else 1.0
            
            # Daily random variation
            daily_variation = np.random.uniform(0.9, 1.1)
            
            # Calculate final sales
            sales = base_sales * store_multiplier * product_multiplier * weekend_boost * daily_variation
            
            # Occasional outliers (5% chance each)
            if np.random.random() > 0.95:
                sales = np.random.uniform(5500, 6500)  # High performer
            elif np.random.random() > 0.95:
                sales = np.random.uniform(1500, 2000)  # Low day
            else:
                sales = np.clip(sales, 2000, 6000)
            
            sales = round(sales, 2)
            
            # Calculate related values
            quantity = max(1, int(sales / np.random.uniform(30, 60)))
            unit_price = round(sales / quantity, 2)
            discount = round(np.random.uniform(0, 20), 2)
            revenue = sales
            cost = round(sales * np.random.uniform(0.55, 0.65), 2)
            profit = round(revenue - cost, 2)
            
            data.append({
                'date': date.strftime('%Y-%m-%d'),
                'store_id': store_id,
                'product_id': product_id,
                'category': np.random.choice(['Electronics', 'Clothing', 'Groceries']),
                'quantity_sold': quantity,
                'unit_price': unit_price,
                'discount_percent': discount,
                'revenue': revenue,
                'cost': cost,
                'profit': profit,
                'sales': sales,
                'customer_traffic': np.random.randint(100, 500),
                'competitor_price': round(np.random.uniform(20, 80), 2)
            })

df = pd.DataFrame(data)

# Save to CSV
output_path = 'test_dataset_v2.csv'
df.to_csv(output_path, index=False)

print(f"Generated {len(df)} rows")
print(f"Sales range: ${df['sales'].min():,.2f} - ${df['sales'].max():,.2f}")
print(f"Per-row average: ${df['sales'].mean():,.2f}")
print(f"Daily total average: ${df.groupby('date')['sales'].sum().mean():,.2f}")
print(f"\nSaved to {output_path}")
