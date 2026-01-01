import os
import argparse
from pathlib import Path
import htcondor

JOB_BID_SINGLE = 15

def launch_tokenize_job(
        jsonl_path=None,
        output_dir=None,
        num_workers=48,
        delete_original=False,
        JOB_MEMORY=128,
        JOB_CPUS=48,
        JOB_GPUS=0,  # This job doesn't require GPUs
        JOB_BID=JOB_BID_SINGLE,
):
    # Name/prefix for cluster logs related to this job
    LOG_PATH = "/fast/sgoel/logs/forecasting/news/tokenized_data"
    
    CLUSTER_LOGS_SAVE_DIR = Path(LOG_PATH)
    os.makedirs(CLUSTER_LOGS_SAVE_DIR, exist_ok=True)
    
    cluster_job_log_name = str(
        CLUSTER_LOGS_SAVE_DIR
        / f"$(Cluster).$(Process)"
    )

    executable = '/is/cluster/sgoel/forecasting-rl/news/src/scripts/launch_tokenize_for_rag.sh'

    # Construct arguments string
    args_list = []
    if jsonl_path:
        args_list.append(f"--jsonl_path {jsonl_path}")
    if output_dir:
        args_list.append(f"--output_dir {output_dir}")
    if num_workers:
        args_list.append(f"--num_workers {num_workers}")
    if delete_original:
        args_list.append("--delete_original")
    
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
        "request_disk": f"20GB",
        
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
        f"Launched tokenization job with cluster-ID={submit_result.cluster()}, "
        f"proc-ID={submit_result.first_proc()}")
    print(f"JSONL path: {jsonl_path}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch job to tokenize news data for RAG")
    
    parser.add_argument('--jsonl_path', type=str, default="/fast/sgoel/forecasting/news/filtered_cc_articles/jsonl/",
                       help="Path to the JSONL files")
    
    parser.add_argument('--output_dir', type=str, default="/fast/sgoel/forecasting/news/tokenized_data/",
                       help="Output directory for tokenized data")
    
    parser.add_argument('--num_workers', type=int, default=48,
                       help="Number of workers for tokenization")
    
    parser.add_argument('--delete_original', action='store_true',
                       help="Delete original files after tokenization")
    
    parser.add_argument('--job_memory', type=int, default=None,
                       help="Job memory request in GB")
    
    parser.add_argument('--job_cpus', type=int, default=48,
                       help="Number of CPUs to request")
    
    args = parser.parse_args()

    if args.job_memory is None:
        args.job_memory = args.job_cpus * 16
    
    # Launch the job
    launch_tokenize_job(
        jsonl_path=args.jsonl_path,
        output_dir=args.output_dir,
        num_workers=args.num_workers,
        delete_original=args.delete_original,
        JOB_MEMORY=args.job_memory,
        JOB_CPUS=args.job_cpus,
    )