import htcondor
from pathlib import Path

# Job bidding constants
JOB_BID_SINGLE = 15

def launch_analyzejsonl_job(
        input_path,
        output_dir,
        JOB_MEMORY=16,
        JOB_CPUS=4,
        JOB_BID=JOB_BID_SINGLE,
):
    # Name/prefix for cluster logs related to this job
    LOG_PATH = "/fast/sgoel/logs/forecasting/news/analyze_jsons/"

    CLUSTER_LOGS_SAVE_DIR = Path(LOG_PATH)
    CLUSTER_LOGS_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    
    cluster_job_log_name = str(
        CLUSTER_LOGS_SAVE_DIR
        / f"$(Cluster).$(Process)"
    )

    executable = 'src/analysis/launch_analyzejsonl.sh'

    # Arguments to pass to the shell script
    arguments = f"{input_path} {output_dir}"

    # Construct job description
    job_settings = {
        "executable": executable,
        "arguments": arguments,
        
        "output": f"{cluster_job_log_name}.out",
        "error": f"{cluster_job_log_name}.err",
        "log": f"{cluster_job_log_name}.log",
        
        "request_cpus": f"{JOB_CPUS}",  # how many CPU cores we want
        "request_memory": f"{JOB_MEMORY}GB",  # how much memory we want
        
        "jobprio": f"{JOB_BID - 1000}",
        "notify_user": "shashwat.goel@tuebingen.mpg.de",
        "notification": "error",
    }

    job_description = htcondor.Submit(job_settings)

    # Submit job to scheduler
    schedd = htcondor.Schedd()  # get the Python representation of the scheduler
    submit_result = schedd.submit(job_description)  # submit the job

    print(
        f"Launched analyzejsonl job with cluster-ID={submit_result.cluster()}, "
        f"proc-ID={submit_result.first_proc()}")
    
    return submit_result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Launch JSONL analysis job")
    
    # Data parameters
    parser.add_argument("--input", type=str, 
                        default="/fast/sgoel/forecasting/news/tokenized_data/news/deduped/",
                        help="Path to JSONL file or directory containing JSONL files")
    parser.add_argument("--output", type=str, default="./plots",
                        help="Directory to save plots")
    
    # Job parameters
    parser.add_argument("--job_memory", type=int, default=None,
                        help="Memory in GB")
    parser.add_argument("--job_cpus", type=int, default=4,
                        help="Number of CPUs")
    
    args = parser.parse_args()
    
    if args.job_memory is None:
        args.job_memory = args.job_cpus * 16
    
    # Launch the job
    launch_analyzejsonl_job(
        input_path=args.input,
        output_dir=args.output,
        JOB_MEMORY=args.job_memory,
        JOB_CPUS=args.job_cpus,
    )