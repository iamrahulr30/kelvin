import matplotlib.pyplot as plt
import ast

# Function to parse and store timing data
def parse_timing_data(file_path):
    data = {}
    with open(file_path, 'r') as f:
        for line in f:
            # Preprocess line for splitting
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

# Function to plot timing data
def plot_timing_data(data):
    plt.figure(figsize=(10, 8))
    
    # Assign a unique color to each configuration
    color_map = plt.cm.get_cmap('tab10', len(data))
    
    for i, (config, frame_times) in enumerate(data.items()):
        frames, times = zip(*frame_times)  # Unzip frame and time values
        label = f"{config[0]}, {config[1]} cores, {config[2]} GPU(s)"
        
        # Plot each line with a unique color and markers
        plt.plot(frames, times, marker='o', color=color_map(i), label=label, linewidth=0.5)

    # Customize plot appearance
    plt.xlabel('Total Number of Frames')
    plt.ylabel('Seconds to Run')
    plt.title('Performance Metrics by Configuration')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# Load and plot data
file_path = 'timings.txt'  # Replace with the actual path if needed
data = parse_timing_data(file_path)
plot_timing_data(data)
