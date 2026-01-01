import os
import argparse
from pathlib import Path
import htcondor

JOB_BID_SINGLE = 15

def launch_postgres_conversion_job(
        json_dir=None,
        db_name="news_articles",
        db_user="postgres",
        db_password=None,
        db_host="localhost",
        db_port=5432,
        workers=None,
        verify=0.01,
        delete=False,
        batch_size=1000,
        JOB_MEMORY=64,
        JOB_CPUS=48,
        JOB_BID=JOB_BID_SINGLE,
):
    # Name/prefix for cluster logs related to this job
    LOG_PATH = "/fast/sgoel/logs/forecasting/news/postgres_conversion"
    
    CLUSTER_LOGS_SAVE_DIR = Path(LOG_PATH)
    os.makedirs(CLUSTER_LOGS_SAVE_DIR, exist_ok=True)
    
    cluster_job_log_name = str(
        CLUSTER_LOGS_SAVE_DIR
        / f"$(Cluster).$(Process)"
    )

    executable = '/is/cluster/sgoel/forecasting-rl/news/src/launch_postgres_conversion.sh'

    # Set workers to match CPU count if not specified
    if workers is None:
        workers = JOB_CPUS
    
    # Construct arguments string
    args_str = f"{json_dir} --db_name {db_name} --db_user {db_user} --db_password {db_password} "
    args_str += f"--db_host {db_host} --db_port {db_port} --workers {workers} --verify {verify} "
    args_str += f"--batch_size {batch_size}"
    
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
        f"Launched PostgreSQL conversion job with cluster-ID={submit_result.cluster()}, "
        f"proc-ID={submit_result.first_proc()}")
    print(f"Source directory: {json_dir}")
    print(f"Database: {db_name} on {db_host}:{db_port}")
    print(f"Workers: {workers}, Verify: {verify}%, Delete JSONs: {delete}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch job to convert JSON files to PostgreSQL")
    
    parser.add_argument('--json_dir', type=str, default="/fast/sgoel/forecasting/news/filtered_cc_articles/",
                       help="Directory containing the JSON files to convert")
    
    parser.add_argument('--db_name', type=str, default="news_articles",
                       help="PostgreSQL database name (default: news_articles)")
    
    parser.add_argument('--db_user', type=str, default="postgres",
                       help="PostgreSQL user name (default: postgres)")
    
    parser.add_argument('--db_password', type=str, required=True,
                       help="PostgreSQL password")
    
    parser.add_argument('--db_host', type=str, default="localhost",
                       help="PostgreSQL host (default: localhost)")
    
    parser.add_argument('--db_port', type=int, default=5432,
                       help="PostgreSQL port (default: 5432)")
    
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
    launch_postgres_conversion_job(
        json_dir=args.json_dir,
        db_name=args.db_name,
        db_user=args.db_user,
        db_password=args.db_password,
        db_host=args.db_host,
        db_port=args.db_port,
        workers=args.workers,
        verify=args.verify,
        delete=args.delete,
        batch_size=args.batch_size,
        JOB_MEMORY=args.job_memory,
        JOB_CPUS=args.job_cpus,
    )