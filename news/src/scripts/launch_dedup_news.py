import os
import argparse
from pathlib import Path
import htcondor

JOB_BID_SINGLE = 15

def launch_deduplicate_news_job(
        jsonl_path=None,
        num_workers=48,
        JOB_MEMORY=768,  # 48*16 GB
        JOB_CPUS=48,
        JOB_GPUS=0,  # This job doesn't require GPUs
        JOB_BID=JOB_BID_SINGLE,
):
    # Name/prefix for cluster logs related to this job
    LOG_PATH = "/fast/sgoel/logs/forecasting/news/deduplicate_news"
    
    CLUSTER_LOGS_SAVE_DIR = Path(LOG_PATH)
    os.makedirs(CLUSTER_LOGS_SAVE_DIR, exist_ok=True)
    
    cluster_job_log_name = str(
        CLUSTER_LOGS_SAVE_DIR
        / f"$(Cluster).$(Process)"
    )

    executable = '/is/cluster/sgoel/forecasting-rl/news/src/scripts/launch_dedup_news.sh'

    # Construct arguments string
    args_list = []
    if jsonl_path:
        args_list.append(f"--jsonl_path {jsonl_path}")
    if num_workers:
        args_list.append(f"--num_workers {num_workers}")
    
    args_str = " ".join(args_list)

    # Construct job description
    job_settings = {
        "executable": executable,
        "arguments": args_str,
        "output": f"{cluster_job_log_name}.out",
        "error": f"{cluster_job_log_name}.err",
        "log": f"{cluster_job_log_name}.log",
        
        "request_cpus": f"{JOB_CPUS}",
        "request_memory": f"{JOB_MEMORY}GB",
        "request_disk": f"100GB",  # Increased disk space for large datasets
        
        "jobprio": f"{JOB_BID - 1000}",
        "notify_user": "shashwat.goel@tuebingen.mpg.de",
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
        f"Launched news deduplication job with cluster-ID={submit_result.cluster()}, "
        f"proc-ID={submit_result.first_proc()}")
    print(f"JSONL path: {jsonl_path}")
    print(f"Number of workers: {num_workers}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch job for deduplicating news articles in JSONL files")
    
    parser.add_argument('--jsonl_path', type=str, default='/fast/sgoel/forecasting/news/tokenized_data/news/',
                       help="Directory containing JSONL files to deduplicate")
    
    parser.add_argument('--num_workers', type=int, default=48,
                       help="Number of parallel workers (default: 48)")
    
    parser.add_argument('--job_memory', type=int, default=None,  # CPUS*16 GB
                       help="Job memory request in GB")
    
    parser.add_argument('--job_cpus', type=int, default=48,
                       help="Number of CPUs to request")
    
    args = parser.parse_args()

    if args.job_memory is None:
        args.job_memory = args.job_cpus * 16
    
    # Launch the job
    launch_deduplicate_news_job(
        jsonl_path=args.jsonl_path,
        num_workers=args.num_workers,
        JOB_MEMORY=args.job_memory,
        JOB_CPUS=args.job_cpus,
    )