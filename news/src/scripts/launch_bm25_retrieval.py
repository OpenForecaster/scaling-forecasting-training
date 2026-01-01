import os
import argparse
from pathlib import Path
import htcondor

JOB_BID_SINGLE = 15

def launch_bm25_retrieval_job(
        jsonl_paths=None,
        metaculus_dataset=None,
        output_file=None,
        top_k=5,
        days_before=30,
        d_freshness=1,
        lookback_days=365,
        JOB_MEMORY=768,  # 48*16 GB
        JOB_CPUS=48,
        JOB_GPUS=0,  # This job doesn't require GPUs
        JOB_BID=JOB_BID_SINGLE,
):
    # Name/prefix for cluster logs related to this job
    LOG_PATH = "/fast/sgoel/logs/forecasting/news/bm25_retrieval"
    
    CLUSTER_LOGS_SAVE_DIR = Path(LOG_PATH)
    os.makedirs(CLUSTER_LOGS_SAVE_DIR, exist_ok=True)
    
    cluster_job_log_name = str(
        CLUSTER_LOGS_SAVE_DIR
        / f"$(Cluster).$(Process)"
    )

    executable = '/is/cluster/sgoel/forecasting-rl/news/src/scripts/launch_bm25_retrieval.sh'

    # Construct arguments string
    args_list = []
    if jsonl_paths:
        args_list.append(f"--jsonl_paths {jsonl_paths}")
    if metaculus_dataset:
        args_list.append(f"--metaculus_dataset {metaculus_dataset}")
    if output_file:
        args_list.append(f"--output_file {output_file}")
    if top_k:
        args_list.append(f"--top_k {top_k}")
    if days_before:
        args_list.append(f"--days_before {days_before}")
    if d_freshness:
        args_list.append(f"--d_freshness {d_freshness}")
    if lookback_days:
        args_list.append(f"--lookback_days {lookback_days}")
    
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
        f"Launched BM25 retrieval job with cluster-ID={submit_result.cluster()}, "
        f"proc-ID={submit_result.first_proc()}")
    print(f"JSONL paths: {jsonl_paths}")
    print(f"Metaculus dataset: {metaculus_dataset}")
    print(f"Output file: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch job for BM25 retrieval on news data")
    
    parser.add_argument('--jsonl_paths', type=str, 
                       default=["/fast/sgoel/forecasting/news/tokenized_data/news/apnews.com_tokenized.jsonl"],
                       help="Paths to the tokenized JSONL files")
    
    parser.add_argument('--metaculus_dataset', type=str, 
                       default="/fast/sgoel/forecasting/news/tokenized_data/questions/metaculus-binary_tokenized",
                       help="Path to the Metaculus dataset")
    
    parser.add_argument('--output_file', type=str, 
                       default="/fast/sgoel/forecasting/news/retrieval_results/metaculus_bm25_results",
                       help="Output file for retrieval results")
    
    parser.add_argument('--top_k', type=int, default=5,
                       help="Number of articles to retrieve per question")
    
    parser.add_argument('--days_before', type=int, default=30,
                       help="Number of days before question date to use as cutoff")
    
    parser.add_argument('--d_freshness', type=int, default=1,
                       help="Number of days difference before refreshing the index")
    
    parser.add_argument('--lookback_days', type=int, default=365,
                       help="Number of days to look back from cutoff date for articles")
    
    parser.add_argument('--job_memory', type=int, default=None,  # CPUS*16 GB
                       help="Job memory request in GB")
    
    parser.add_argument('--job_cpus', type=int, default=48,
                       help="Number of CPUs to request")
    
    args = parser.parse_args()

    if args.job_memory is None:
        args.job_memory = args.job_cpus * 16
    
    # Launch the job
    launch_bm25_retrieval_job(
        jsonl_paths=args.jsonl_paths,
        metaculus_dataset=args.metaculus_dataset,
        output_file=args.output_file,
        top_k=args.top_k,
        days_before=args.days_before,
        d_freshness=args.d_freshness,
        lookback_days=args.lookback_days,
        JOB_MEMORY=args.job_memory,
        JOB_CPUS=args.job_cpus,
    )