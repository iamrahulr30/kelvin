import ast
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def parse_timing_data(file_path):
    data = {}
    with open(file_path, 'r') as f:
        for line in f:
            line = line.replace("[", "  [").split("  ")
            core_type, no_cores, num_gpus = line[0].split(",", 2)
            no_cores = int(no_cores)
            num_gpus = int(num_gpus.replace(",", ""))
            
            # Parse the timing data list of tuples
            timings = ast.literal_eval(line[1].strip())

            # Store data in dictionary with (core_type, no_cores, num_gpus) as key
            config_key = (core_type, no_cores, num_gpus)
            data[config_key] = timings
    return data

def calculate_metrics(data):
    results = []
    for config_key, timings in data.items():
        core_type, no_cores, num_gpus = config_key
        # Extract frame counts and elapsed times
        frame_counts, elapsed_times = zip(*timings)
        
        # Calculate speedup, efficiency, and throughput
        base_time = elapsed_times[0]  # Assuming first entry is single-threaded for comparison
        speedup = [base_time / time for time in elapsed_times]
        efficiency = [s / no_cores for s in speedup]
        throughput = [frames / time for frames, time in zip(frame_counts, elapsed_times)]

        results.append((core_type, no_cores, num_gpus, frame_counts, elapsed_times, speedup, efficiency, throughput))

    return results

def display_results(results):
    # Create a DataFrame for better visualization
    rows = []
    for result in results:
        core_type, no_cores, num_gpus, frame_counts, elapsed_times, speedup, efficiency, throughput = result
        for i in range(len(frame_counts)):
            rows.append((core_type, no_cores, num_gpus, frame_counts[i], elapsed_times[i], speedup[i], efficiency[i], throughput[i]))
    
    df = pd.DataFrame(rows, columns=['Core Type', 'Number of Cores', 'Number of GPUs', 'Frames', 'Elapsed Time', 'Speedup', 'Efficiency', 'Throughput'])
    print(df)

    # Set the Seaborn style
    sns.set(style="whitegrid")

    # Plotting Speedup and Efficiency using Seaborn
    plt.figure(figsize=(12, 6))
    
    # Speedup Plot
    sns.lineplot(data=df, x='Frames', y='Speedup', hue='Core Type', marker='o')
    plt.title('Speedup vs Frames Processed')
    plt.xlabel('Frames Processed')
    plt.ylabel('Speedup')
    plt.legend(title='Core Type')
    plt.grid()
    plt.show()
    
    # Efficiency Plot
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x='Frames', y='Efficiency', hue='Core Type', marker='x')
    plt.title('Efficiency vs Frames Processed')
    plt.xlabel('Frames Processed')
    plt.ylabel('Efficiency')
    plt.legend(title='Core Type')
    plt.grid()
    plt.show()

# File path to the timing data
file_path = 'timings.txt'

# Parse the timing data
timing_data = parse_timing_data(file_path)

# Calculate metrics
metrics_results = calculate_metrics(timing_data)

# Display results
display_results(metrics_results)
