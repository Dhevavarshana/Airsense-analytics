"""AirSense Analytics - Synthetic Flight Data Generator
Generates realistic flight operations dataset with delay & cancellation patterns.
"""

import os
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# Set reproducible random seed
np.random.seed(42)

def generate_flight_data(num_records: int = 50000) -> pd.DataFrame:
    """Generate synthetic flight records with weather, hour, and airline correlations."""
    print(f"Generating {num_records:,} flight records...")

    # Reference entities
    airlines = ["AA", "DL", "UA", "WN", "B6", "AS", "NK"]
    airline_delay_bias = {
        "DL": -2.0,   # High operational reliability
        "AS": -1.0,
        "WN": 1.0,
        "AA": 3.0,
        "UA": 4.0,
        "B6": 8.0,
        "NK": 12.0    # Higher baseline delay
    }

    airports = ["ATL", "ORD", "DFW", "DEN", "CLT", "LAX", "IAH", "PHX", "MCO", "SEA", "MIA", "JFK", "BOS", "SFO"]
    weather_types = ["Clear", "Cloudy", "Rain", "Thunderstorm", "Snow", "Fog"]
    weather_weights = [0.50, 0.25, 0.12, 0.05, 0.05, 0.03]

    # Date range: 2024 full calendar year
    start_date = datetime(2024, 1, 1)
    random_days = np.random.randint(0, 366, size=num_records)
    random_minutes = np.random.randint(0, 24 * 60, size=num_records)

    dates = [start_date + timedelta(days=int(d)) for d in random_days]
    scheduled_departures = [
        d + timedelta(minutes=int(m)) for d, m in zip(dates, random_minutes)
    ]

    # Temporal features
    day_of_week = [d.weekday() for d in scheduled_departures]  # 0=Monday, 6=Sunday
    month = [d.month for d in scheduled_departures]
    hour = [d.hour for d in scheduled_departures]
    date_strs = [d.strftime("%Y-%m-%d") for d in dates]
    sched_dep_strs = [d.strftime("%Y-%m-%d %H:%M:%S") for d in scheduled_departures]

    # Airline and flight number
    chosen_airlines = np.random.choice(airlines, size=num_records)
    flight_numbers = [f"{a}{np.random.randint(100, 9999)}" for a in chosen_airlines]

    # Origin and destination pairs
    origins = np.random.choice(airports, size=num_records)
    destinations = []
    for orig in origins:
        avail = [a for a in airports if a != orig]
        destinations.append(np.random.choice(avail))

    # Distance based on routes (approximated between 200 and 2600 miles)
    distances = np.random.choice(
        [240, 450, 680, 850, 1100, 1380, 1720, 2150, 2580],
        size=num_records,
        p=[0.10, 0.15, 0.20, 0.20, 0.15, 0.10, 0.05, 0.03, 0.02]
    ) + np.random.randint(-30, 30, size=num_records)

    # Weather conditions
    chosen_weather = np.random.choice(weather_types, size=num_records, p=weather_weights)

    # Weather metrics correlated with weather condition and month
    temperatures = []
    wind_speeds = []
    visibilities = []

    for w, m in zip(chosen_weather, month):
        # Base temperature varies by season (month)
        is_winter = m in [12, 1, 2]
        is_summer = m in [6, 7, 8]
        base_temp = 5.0 if is_winter else (28.0 if is_summer else 18.0)
        temp = np.random.normal(base_temp, 6.0)

        if w == "Snow":
            temp = min(temp, np.random.uniform(-10.0, 1.0))
            wind = np.random.uniform(15.0, 35.0)
            vis = np.random.uniform(0.5, 3.0)
        elif w == "Thunderstorm":
            temp = max(temp, np.random.uniform(18.0, 34.0))
            wind = np.random.uniform(20.0, 48.0)
            vis = np.random.uniform(1.0, 4.0)
        elif w == "Rain":
            wind = np.random.uniform(10.0, 28.0)
            vis = np.random.uniform(3.0, 6.5)
        elif w == "Fog":
            wind = np.random.uniform(2.0, 10.0)
            vis = np.random.uniform(0.2, 1.8)
        elif w == "Cloudy":
            wind = np.random.uniform(5.0, 20.0)
            vis = np.random.uniform(7.0, 10.0)
        else:  # Clear
            wind = np.random.uniform(2.0, 15.0)
            vis = 10.0

        temperatures.append(round(temp, 1))
        wind_speeds.append(round(wind, 1))
        visibilities.append(round(vis, 1))

    # Calculate delays correlated with weather, hour, and airline
    departure_delays = []
    arrival_delays = []
    cancellations = []

    weather_delay_impact = {
        "Clear": 0.0,
        "Cloudy": 2.0,
        "Rain": 16.0,
        "Thunderstorm": 42.0,
        "Snow": 48.0,
        "Fog": 28.0
    }

    for i in range(num_records):
        w = chosen_weather[i]
        h = hour[i]
        air = chosen_airlines[i]
        wind = wind_speeds[i]
        vis = visibilities[i]

        # 1. Hour effect: cascading delays peak in the evening (16:00 - 21:00)
        if 5 <= h < 9:
            hour_effect = -3.0
        elif 9 <= h < 14:
            hour_effect = 2.0
        elif 14 <= h < 18:
            hour_effect = 10.0
        elif 18 <= h < 22:
            hour_effect = 18.0
        else:
            hour_effect = 6.0

        # 2. Weather effect + wind/visibility severity
        w_effect = weather_delay_impact[w]
        if wind > 25.0:
            w_effect += (wind - 25.0) * 1.5
        if vis < 3.0:
            w_effect += (3.0 - vis) * 4.0

        # 3. Airline effect
        air_effect = airline_delay_bias[air]

        # Base flight operational delay (exponential tail for severe delays)
        is_delayed_event = np.random.rand() < (0.20 + (w_effect / 100.0))
        if is_delayed_event:
            delay_shock = np.random.exponential(scale=22.0)
        else:
            delay_shock = np.random.normal(loc=-2.0, scale=4.0)

        dep_delay = round(hour_effect + w_effect + air_effect + delay_shock)
        dep_delay = max(-20, dep_delay)  # At most 20 mins early departure

        # Enroute adjustment: longer flights can absorb small delays; buffer noise
        enroute_noise = np.random.normal(loc=0.0, scale=6.0)
        arr_delay = round(dep_delay + enroute_noise)
        arr_delay = max(-30, arr_delay)

        # 4. Cancellation correlation (severe storms, blizzards, extreme delays)
        cancel_prob = 0.012  # Baseline 1.2%
        if w in ["Thunderstorm", "Snow"]:
            cancel_prob += 0.16
        if wind > 35.0:
            cancel_prob += 0.12
        if vis < 1.0:
            cancel_prob += 0.10
        if dep_delay > 90:
            cancel_prob += 0.15

        cancelled = 1 if np.random.rand() < cancel_prob else 0

        # If cancelled, flight never operated
        if cancelled == 1:
            arr_delay = np.nan
            dep_delay = np.nan

        departure_delays.append(dep_delay)
        arrival_delays.append(arr_delay)
        cancellations.append(cancelled)

    # Build DataFrame with exact specified columns
    df = pd.DataFrame({
        "date": date_strs,
        "airline": chosen_airlines,
        "flight_number": flight_numbers,
        "origin": origins,
        "destination": destinations,
        "scheduled_departure": sched_dep_strs,
        "distance": distances,
        "departure_delay": departure_delays,
        "arrival_delay": arrival_delays,
        "cancelled": cancellations,
        "weather_condition": chosen_weather,
        "temperature": temperatures,
        "wind_speed": wind_speeds,
        "visibility": visibilities,
        "day_of_week": day_of_week,
        "month": month,
        "hour": hour
    })

    return df

def main():
    # Target output file path
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_csv = os.path.join(output_dir, "flights.csv")

    df = generate_flight_data(num_records=50000)
    df.to_csv(output_csv, index=False)

    print(f"Dataset successfully created at: {output_csv}")
    print(f"Shape: {df.shape}")
    print(f"Cancelled count: {df['cancelled'].sum()} ({df['cancelled'].mean():.2%})")
    non_cancelled = df[df['cancelled'] == 0]
    print(f"Mean departure delay (non-cancelled): {non_cancelled['departure_delay'].mean():.2f} min")
    print(f"Mean arrival delay (non-cancelled): {non_cancelled['arrival_delay'].mean():.2f} min")

if __name__ == "__main__":
    main()
