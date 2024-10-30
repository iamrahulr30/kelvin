import ray

ray.init()
resources = ray.cluster_resources()

num_cpus = resources.get("CPU", 0)  
num_gpus = resources.get("GPU", 0)  

print(f"Number of CPUs: {num_cpus}")
print(f"Number of GPUs: {num_gpus}")

# Shutdown Ray
ray.shutdown()
