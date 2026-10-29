
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Project folders
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "logistics_data.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
CHART_DIR = OUTPUT_DIR / "charts"

OUTPUT_DIR.mkdir(exist_ok=True)
CHART_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load dataset
df = pd.read_csv(DATA_FILE)

# 2. Check and clean essential columns
required = [
    "Order_ID", "Shipping_Cost", "Delivery_Days",
    "Expected_Days", "Distance_km", "Weather", "Traffic_Level"
]

missing = [col for col in required if col not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

df = df.dropna(subset=required).copy()

if df.empty:
    raise ValueError("No valid order records found.")

# 3. Calculate KPIs
total_orders = len(df)
avg_shipping_cost = df["Shipping_Cost"].mean()
avg_delivery_days = df["Delivery_Days"].mean()
on_time_orders = (df["Delivery_Days"] <= df["Expected_Days"]).sum()
on_time_rate = on_time_orders / total_orders * 100

print("\n=== LOGISTICS DATA ANALYSIS ===")
print(f"Total orders: {total_orders}")
print(f"Average shipping cost: {avg_shipping_cost:.2f}")
print(f"Average delivery time: {avg_delivery_days:.2f} days")
print(f"On-time orders: {on_time_orders}")
print(f"On-time delivery rate: {on_time_rate:.2f}%")

# 4. Explore delivery time by weather and traffic
weather_analysis = df.groupby("Weather")["Delivery_Days"].mean()
traffic_analysis = df.groupby("Traffic_Level")["Delivery_Days"].mean()

print("\nAverage delivery days by weather:")
print(weather_analysis.round(2))

print("\nAverage delivery days by traffic:")
print(traffic_analysis.round(2))

# 5. Save analysis results
summary = pd.DataFrame({
    "Metric": [
        "Total Orders",
        "Average Shipping Cost",
        "Average Delivery Days",
        "On-Time Orders",
        "On-Time Delivery Rate (%)"
    ],
    "Value": [
        total_orders,
        round(avg_shipping_cost, 2),
        round(avg_delivery_days, 2),
        int(on_time_orders),
        round(on_time_rate, 2)
    ]
})

summary.to_csv(OUTPUT_DIR / "week1_summary.csv", index=False)
weather_analysis.round(2).to_csv(OUTPUT_DIR / "delivery_by_weather.csv")
traffic_analysis.round(2).to_csv(OUTPUT_DIR / "delivery_by_traffic.csv")

# 6. Create a chart
traffic_analysis.plot(kind="bar", title="Average Delivery Days by Traffic Level")
plt.xlabel("Traffic Level")
plt.ylabel("Average Delivery Days")
plt.tight_layout()
plt.savefig(CHART_DIR / "delivery_by_traffic.png")
plt.close()

print("\nAnalysis completed successfully!")
print(f"Results saved in: {OUTPUT_DIR}")
