import ray

ray.init()
resources = ray.cluster_resources()
num_cpus = resources.get("CPU", 0)  
num_gpus = resources.get("GPU", 0)  
print(f"Number of CPUs: {num_cpus}  GPUs: {num_gpus}")
ray.shutdown()
