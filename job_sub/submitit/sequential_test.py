import submitit
import time

def slow_task(x):
    import time
    time.sleep(10)  # 10 second delay
    return f"Done with {x}"

executor = submitit.AutoExecutor(folder="logs_test")
executor.update_parameters(timeout_min=5, slurm_partition="wzt_20250411,intel-sc3")

print("About to submit jobs...")
start_time = time.time()

# This returns IMMEDIATELY (even though jobs take 10 seconds each)
jobs = executor.map_array(slow_task, [1, 2, 3])
submit_time = time.time()

print(f"Jobs submitted in {submit_time - start_time:.2f} seconds")
print("Jobs are now running in the background...")
print("About to wait for results...")

# THIS is where we wait
results = [job.result() for job in jobs]
end_time = time.time()

print(f"All jobs completed in {end_time - start_time:.2f} seconds total")
print("Results:", results)