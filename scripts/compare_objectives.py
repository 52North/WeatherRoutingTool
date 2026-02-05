from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker


def time_formatter(x, pos):
    """Converts total minutes to HH:MM format."""
    hours = int(x) // 60
    minutes = int(x) % 60
    return f'{hours:02d}:{minutes:02d}'


def time_str_convertert(time_str, days):
    t = datetime.strptime(time_str, '%H:%M:%S')
    delta = timedelta(days=days, hours=t.hour, minutes=t.minute, seconds=t.second)
    t_min = delta.total_seconds() * 1. / 60
    print('t: ', delta)
    print('t_min: ', t_min)
    return t_min


def compare_time_obj():
    # Sample data
    labels = ['fuel opt.', 'time opt.', 'fuel:time opt. 1:1']
    values = [
        time_str_convertert(time_str="8:36:36", days=1),
        time_str_convertert(time_str="10:12:00", days=1),
        time_str_convertert(time_str="8:58:25", days=1),
    ]  # Time in minutes

    departure_time = datetime.strptime("2025-12-05T13:48Z", '%Y-%m-%dT%H:%MZ')
    arrival_time = datetime.strptime("2025-12-07T00:00Z", '%Y-%m-%dT%H:%MZ')
    optimal_travel_time = (arrival_time - departure_time).total_seconds() * 1. / 60

    arrival_time = 140  # Arrival time in minutes

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Generate the bar plot
    x_pos = np.arange(len(labels))
    ax.bar(x_pos, values, color='skyblue', edgecolor='navy', alpha=0.7, label='Recorded Time')

    # Set manual labels for x-axis
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)

    # Add the constant 'arrival time' line
    ax.axhline(y=optimal_travel_time, color='red', linestyle='--', linewidth=2, label='Arrival Time')

    # Add the shaded area (+/- 30 minutes)
    ax.fill_between([-0.5, len(labels) - 0.5],
                    optimal_travel_time - 30,
                    optimal_travel_time + 30,
                    color='gray', alpha=0.2, label='Arrival Window ($\pm 30$ min)')

    # Apply the custom formatter
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(time_formatter))

    # Set tick frequency to every 30 minutes for clarity
    ax.yaxis.set_major_locator(ticker.MultipleLocator(30))
    ax.set_ylim(25 * 60, 35 * 60)

    # General formatting
    ax.set_ylabel('Time (HH:MM)')
    ax.set_xlabel('Run ID')
    ax.legend(loc='upper left')
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(
        "/home/kdemmich/1_Projekte/TwinShip/5_Results/260203_feature-15-add-time-objective/Summary/arrival_time.png")


def compare_fuel_obj():
    # Sample data
    labels = ['fuel opt.', 'time opt.', 'fuel:time opt. 1:1']
    values = [
        26567.46247030554,
        27755.58059942486,
        26742.834078860997,
    ]  # Time in minutes

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Generate the bar plot
    x_pos = np.arange(len(labels))
    ax.bar(x_pos, values, color='skyblue', edgecolor='navy', alpha=0.7, label='fuel consumption')

    # Set manual labels for x-axis
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_ylim(20000, 30000)

    # General formatting
    ax.set_ylabel('fuel consumption (kg)')
    ax.set_xlabel('Run ID')
    ax.legend(loc='upper left')
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(
        "/home/kdemmich/1_Projekte/TwinShip/5_Results/260203_feature-15-add-time-objective/Summary/fuel_consumption.png")


if __name__ == "__main__":
    # Compare variations of resistances for specific routes
    compare_time_obj()
    compare_fuel_obj()
