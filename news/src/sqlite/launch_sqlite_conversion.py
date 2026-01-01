import os
import argparse
from pathlib import Path
import htcondor
import sys

JOB_BID_SINGLE = 15

def launch_sqlite_conversion_job(
        json_dir=None,
        db_path=None,
        workers=None,
        verify=0.01,
        delete=False,
        batch_size=1000,
        JOB_MEMORY=64,
        JOB_CPUS=48,
        JOB_BID=JOB_BID_SINGLE,
):
    # Name/prefix for cluster logs related to this job
    LOG_PATH = "/fast/sgoel/logs/forecasting/news/sqlite_conversion"
    
    CLUSTER_LOGS_SAVE_DIR = Path(LOG_PATH)
    os.makedirs(CLUSTER_LOGS_SAVE_DIR, exist_ok=True)
    
    cluster_job_log_name = str(
        CLUSTER_LOGS_SAVE_DIR
        / f"$(Cluster).$(Process)"
    )

    executable = '/is/cluster/sgoel/forecasting-rl/news/src/launch_sqlite_conversion.sh'

    # Set workers to match CPU count if not specified
    if workers is None:
        workers = JOB_CPUS
    
    # Set default database path if not specified
    if db_path is None:
        db_path = os.path.join(os.path.dirname(json_dir), "news_articles.db")
    
    # Construct arguments string
    args_str = f"{json_dir} {db_path} --workers {workers} --verify {verify} --batch_size {batch_size}"
    if delete:
        args_str += " --delete"

    # Construct job description
    job_settings = {
        "executable": executable,
        "arguments": args_str,
        "output": f"{cluster_job_log_name}.out",
        "error": f"{cluster_job_log_name}.err",
        "log": f"{cluster_job_log_name}.log",
        
        "request_cpus": f"{JOB_CPUS}",
        "request_memory": f"{JOB_MEMORY}GB",
        "request_disk": f"{JOB_MEMORY * 2}GB",  # Request double memory for disk space
        
        "jobprio": f"{JOB_BID - 1000}",
        "notify_user": "nikhil.chandak@tuebingen.mpg.de",  # Change this to your email
        "notification": "error",
    }

    job_description = htcondor.Submit(job_settings)

    # Submit job to scheduler
    schedd = htcondor.Schedd()
    submit_result = schedd.submit(job_description)

    print(
        f"Launched SQLite conversion job with cluster-ID={submit_result.cluster()}, "
        f"proc-ID={submit_result.first_proc()}")
    print(f"Source directory: {json_dir}")
    print(f"Database path: {db_path}")
    print(f"Workers: {workers}, Verify: {verify}%, Delete JSONs: {delete}, Batch size: {batch_size}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch job to convert JSON files to SQLite")
    
    parser.add_argument('--json_dir', type=str, default="/fast/sgoel/forecasting/news/filtered_cc_articles/",
                       help="Directory containing the JSON files to convert")
    
    parser.add_argument('--db_path', type=str, default=None,
                       help="Path to SQLite database file (default: parent of json_dir + /news_articles.db)")
    
    parser.add_argument('--workers', type=int, default=None,
                       help="Number of parallel workers to use (default: number of CPUs)")
    
    parser.add_argument('--verify', type=float, default=0.01,
                       help="Fraction of documents to verify (default: 0.01 = 1%%)")
    
    parser.add_argument('--delete', action='store_true',
                       help="Delete JSON files after successful conversion")
    
    parser.add_argument('--batch_size', type=int, default=1000,
                       help="Batch size for database inserts (default: 1000)")
    
    parser.add_argument('--job_memory', type=int, default=None,
                       help="Job memory request in GB (default: cpus * 16)")
    
    parser.add_argument('--job_cpus', type=int, default=48,
                       help="Number of CPUs to request (default: 48)")
    
    args = parser.parse_args()
    
    if args.job_memory is None:
        args.job_memory = args.job_cpus * 16
    
    # Launch the job
    launch_sqlite_conversion_job(
        json_dir=args.json_dir,
        db_path=args.db_path,
        workers=args.workers,
        verify=args.verify,
        delete=args.delete,
        batch_size=args.batch_size,
        JOB_MEMORY=args.job_memory,
        JOB_CPUS=args.job_cpus,
    )