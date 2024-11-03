import ast
from tabulate import tabulate

def parse_timing_data(file_path):
    data = {}
    with open(file_path, 'r') as f:
        for line in f:
            line = line.replace("[", "  [").split("  ")
            core_type, no_cores, num_gpus = line[0].split(",", 2)
            no_cores = int(no_cores)
            num_gpus = int(num_gpus.replace(",", ""))
            timings = ast.literal_eval(line[1].strip())
            config_key = (core_type, no_cores, num_gpus)
            data[config_key] = timings
    return data

def calculate_speedup_efficiency(data, sequential_time_data):
    results = {}
    for config, timings in data.items():
        core_type, no_cores, num_gpus = config
        results[config] = []
        for task_size, parallel_time in timings:
            if task_size in sequential_time_data:
                sequential_time = sequential_time_data[task_size]
                speedup = sequential_time / parallel_time
                efficiency = speedup / no_cores
                results[config].append((task_size, speedup, efficiency))
    return results

def calculate_average_speedup_efficiency(results):
    averages = {}
    for config, values in results.items():
        total_speedup = sum(speedup for _, speedup, _ in values)
        total_efficiency = sum(efficiency for _, _, efficiency in values)
        num_tasks = len(values)
        avg_speedup = total_speedup / num_tasks
        avg_efficiency = total_efficiency / num_tasks
        averages[config] = (avg_speedup, avg_efficiency)
    return averages

sequential_time_data = {
    100: 41.47, 200: 81.76, 300: 122.72, 400: 162.66, 
    500: 201.68, 600: 248.45, 700: 288.07, 800: 327.88
}

file_path = "timings.txt"
data = parse_timing_data(file_path)
results = calculate_speedup_efficiency(data, sequential_time_data)
averages = calculate_average_speedup_efficiency(results)

table = []
for config, (avg_speedup, avg_efficiency) in averages.items():
    core_type, no_cores, num_gpus = config
    table.append([core_type, no_cores, num_gpus, f"{avg_speedup:.2f}", f"{avg_efficiency:.2f}"])

headers = ["Core Type", "Cores", "GPUs", "Average Speedup", "Average Efficiency"]
print(tabulate(table, headers, tablefmt="grid"))
