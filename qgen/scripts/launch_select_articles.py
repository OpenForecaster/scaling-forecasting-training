import os
import argparse
from pathlib import Path
import htcondor

JOB_BID_SINGLE = 15

def launch_select_articles_job(
        article_path=None,
        filter_years=None,
        random_sample=None,
        limit=None,
        output_dir=None,
        balance_yearly=False,
        hard_cutoff=None,
        filter_language='en', # Add default language filter
        JOB_MEMORY=16,  # Default memory
        JOB_CPUS=1,     # Default CPUs
        JOB_GPUS=0,     # This job doesn't require GPUs
        JOB_BID=JOB_BID_SINGLE,
):
    # Name/prefix for cluster logs related to this job
    LOG_PATH = "/fast/sgoel/logs/forecasting/qgen/select_articles"
    
    CLUSTER_LOGS_SAVE_DIR = Path(LOG_PATH)
    os.makedirs(CLUSTER_LOGS_SAVE_DIR, exist_ok=True)
    
    cluster_job_log_name = str(
        CLUSTER_LOGS_SAVE_DIR
        / f"$(Cluster).$(Process)"
    )

    executable = '/is/cluster/sgoel/forecasting-rl/qgen/scripts/launch_select_articles.sh'

    # Construct arguments string
    args_list = []
    if article_path:
        args_list.append(f"--article_path {article_path}")
    if filter_years:
        years_str = " ".join(map(str, filter_years))
        args_list.append(f"--filter_years {years_str}")
    if random_sample:
        # Convert list of sample sizes to space-separated string
        samples_str = " ".join(map(str, random_sample))
        args_list.append(f"--random_sample {samples_str}")
    if limit:
        args_list.append(f"--limit {limit}")
    if output_dir:
        args_list.append(f"--output_dir {output_dir}")
    if balance_yearly:
        args_list.append("--balance_yearly")
    if hard_cutoff:
        args_list.append(f"--hard_cutoff {hard_cutoff}")
    # Add language filter argument, handle None or empty string case
    if filter_language:
        args_list.append(f"--filter_language {filter_language}")
    else:
         # Pass 'None' explicitly if language filter is disabled
        args_list.append("--filter_language None")
    
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
        "request_disk": f"50GB",
        
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
        f"Launched article selection job with cluster-ID={submit_result.cluster()}, "
        f"proc-ID={submit_result.first_proc()}")
    print(f"Article path: {article_path}")
    if filter_years:
        print(f"Filter years: {filter_years}")
    if random_sample:
        print(f"Random sample sizes: {random_sample}")
    if limit:
        print(f"Limit: {limit}")
    if output_dir:
        print(f"Output directory: {output_dir}")
    if balance_yearly:
        print(f"Balance yearly: {balance_yearly}")
    if hard_cutoff:
        print(f"Hard cutoff: {hard_cutoff}")
    print(f"Filter language: {filter_language if filter_language else 'Disabled'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch job for selecting and filtering news articles")
    
    parser.add_argument('--article_path', type=str, 
                       default='/fast/sgoel/forecasting/news/tokenized_data/news/deduped/www.reuters.com_tokenized.jsonl',
                       help="Path to news article file or directory")
    
    parser.add_argument('--filter_years', type=int, nargs='+', default=None,
                       help="Filter articles by year(s) (e.g., --filter_years 2021 2022)")
    
    parser.add_argument('--random_sample', type=int, nargs='+', default=None,
                       help="Number(s) of articles to randomly sample from the filtered list (e.g., --random_sample 100 500 1000)")
    
    parser.add_argument('--limit', type=int, default=None,
                       help="Limit the number of articles to process")
    
    parser.add_argument('--output_dir', type=str, default=None,
                       help="Directory to save selected articles")
    
    parser.add_argument('--balance_yearly', action='store_true',
                       help="Balance article selection across years when filtering or sampling")
    
    parser.add_argument('--hard_cutoff', type=str, default=None,
                       help="Exclude articles published on or after this date (format: YYYY-MM)")
    
    parser.add_argument('--filter_language', type=str, default='en',
                       help="Filter articles by language code (e.g., 'en'). Set to 'None' or empty string to disable.")
    
    parser.add_argument('--job_memory', type=int, default=None,
                       help="Job memory request in GB")
    
    parser.add_argument('--job_cpus', type=int, default=1,
                       help="Number of CPUs to request")
    
    args = parser.parse_args()

    if args.job_memory is None:
        args.job_memory = 16*args.job_cpus
    
    # Launch the job
    launch_select_articles_job(
        article_path=args.article_path,
        filter_years=args.filter_years,
        random_sample=args.random_sample,
        limit=args.limit,
        output_dir=args.output_dir,
        balance_yearly=args.balance_yearly,
        hard_cutoff=args.hard_cutoff,
        filter_language=args.filter_language, # Pass language filter
        JOB_MEMORY=args.job_memory,
        JOB_CPUS=args.job_cpus,
    )