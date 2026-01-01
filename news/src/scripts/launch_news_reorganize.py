import os
import argparse
from pathlib import Path
import htcondor

JOB_BID_SINGLE = 15

def launch_analyze_jsons_job(
        news_dir=None,
        recursive=False,
        JOB_MEMORY=32,
        JOB_CPUS=8,
        JOB_GPUS=0,  # This job doesn't require GPUs
        JOB_BID=JOB_BID_SINGLE,
        processes=None,
):
    # Name/prefix for cluster logs related to this job
    LOG_PATH = "/fast/sgoel/logs/forecasting/news/analyze_jsons"
    
    CLUSTER_LOGS_SAVE_DIR = Path(LOG_PATH)
    os.makedirs(CLUSTER_LOGS_SAVE_DIR, exist_ok=True)
    
    cluster_job_log_name = str(
        CLUSTER_LOGS_SAVE_DIR
        / f"$(Cluster).$(Process)"
    )

    executable = '/is/cluster/sgoel/forecasting-rl/news/src/scripts/launch_news_reorganize.sh'

    # If recursive mode is enabled and news_dir is provided, submit a job for each subdirectory
    if recursive and news_dir:
        print(f"Launching jobs recursively for subdirectories in {news_dir}")
        subdirs = [d for d in Path(news_dir).iterdir() if d.is_dir()]
        
        if not subdirs:
            print(f"No subdirectories found in {news_dir}")
            return
            
        for subdir in subdirs:
            # Construct job description for this subdirectory
            job_settings = {
                "executable": executable,
                "arguments": f"{subdir} {processes}",
                "output": f"{cluster_job_log_name}.out",
                "error": f"{cluster_job_log_name}.err",
                "log": f"{cluster_job_log_name}.log",
                
                "request_cpus": f"{JOB_CPUS}",
                "request_memory": f"{JOB_MEMORY}GB",
                "request_disk": f"{JOB_MEMORY}GB",
                
                "jobprio": f"{JOB_BID - 1000}",
                "notify_user": "nikhil.chandak@tuebingen.mpg.de",  # Change this to your email
                "notification": "error",
            }

            # Only request GPUs if needed
            if JOB_GPUS > 0:
                job_settings["request_gpus"] = f"{JOB_GPUS}"

            job_description = htcondor.Submit(job_settings)

            # Submit job to scheduler
            schedd = htcondor.Schedd()
            submit_result = schedd.submit(job_description)

            print(
                f"Launched job for {subdir} with cluster-ID={submit_result.cluster()}, "
                f"proc-ID={submit_result.first_proc()}")
    else:
        # Construct arguments string for single job
        args_str = f"{news_dir}" if news_dir else ""

        # Construct job description
        job_settings = {
            "executable": executable,
            "arguments": args_str,
            "output": f"{cluster_job_log_name}.out",
            "error": f"{cluster_job_log_name}.err",
            "log": f"{cluster_job_log_name}.log",
            
            "request_cpus": f"{JOB_CPUS}",
            "request_memory": f"{JOB_MEMORY}GB",
            "request_disk": f"{JOB_MEMORY}GB",
            
            "jobprio": f"{JOB_BID - 1000}",
            "notify_user": "nikhil.chandak@tuebingen.mpg.de",  # Change this to your email
            "notification": "error",
        }

        # Only request GPUs if needed
        if JOB_GPUS > 0:
            job_settings["request_gpus"] = f"{JOB_GPUS}"

        job_description = htcondor.Submit(job_settings)

        # Submit job to scheduler
        schedd = htcondor.Schedd()
        submit_result = schedd.submit(job_description)

        print(
            f"Launched JSON analysis job with cluster-ID={submit_result.cluster()}, "
            f"proc-ID={submit_result.first_proc()}")
        if news_dir:
            print(f"News directory: {news_dir}")
        else:
            print(f"Using default news directory in the script")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch job to analyze news JSON files")
    
    parser.add_argument('--news_dir', type=str, 
                       help="Directory containing the news JSON files (optional)")
    
    parser.add_argument('--recursive', action='store_true',
                       help="Process all subdirectories at depth 1 in the news directory")
    
    parser.add_argument('--job_memory', type=int, default=16,
                       help="Job memory request in GB")
    
    parser.add_argument('--job_cpus', type=int, default=1,
                       help="Number of CPUs to request")
    
    parser.add_argument('--processes', type=int, default=None,
                        help="Number of parallel processes to use (default: number of CPU cores)")
    
    args = parser.parse_args()
    
    # Launch the job
    launch_analyze_jsons_job(
        news_dir=args.news_dir,
        recursive=args.recursive,
        JOB_MEMORY=args.job_memory,
        JOB_CPUS=args.job_cpus,
        processes=args.processes,
    )