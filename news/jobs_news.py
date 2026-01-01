import os
from pathlib import Path
import htcondor

JOB_BID = 15  # Priority for the job

def launch_news_crawl_job(
        download_dir_warc,
        download_dir_article,
        delete_warc_after_extraction,
        num_extraction_processes,
        job_memory,
        job_cpus,
        job_bid=JOB_BID,
):
    # Name/prefix for cluster logs related to this job
    LOG_PATH = "/fast/nchandak/logs/forecasting/news_crawl"
    
    CLUSTER_LOGS_SAVE_DIR = Path(LOG_PATH)
    os.makedirs(CLUSTER_LOGS_SAVE_DIR, exist_ok=True)
    
    cluster_job_log_name = str(
        CLUSTER_LOGS_SAVE_DIR
        / f"$(Cluster).$(Process)"
    )

    executable = 'launch_news_crawl_job.sh'

    # Construct job description
    job_settings = {
        "executable": executable,
        "arguments": (
            f"{download_dir_warc} "
            f"{download_dir_article} "
            f"{delete_warc_after_extraction} "
            f"{num_extraction_processes}"
        ),
        "output": f"{cluster_job_log_name}.out",
        "error": f"{cluster_job_log_name}.err",
        "log": f"{cluster_job_log_name}.log",
        
        "request_cpus": f"{job_cpus}",  # how many CPU cores we want
        "request_memory": f"{job_memory}GB",  # how much memory we want
        "request_disk": f"{job_memory}GB",
        
        "jobprio": f"{job_bid - 1000}",
        "notify_user": "nikhil.chandak@tuebingen.mpg.de",  # change to your email
        "notification": "error",
    }

    job_description = htcondor.Submit(job_settings)

    # Submit job to scheduler
    schedd = htcondor.Schedd()  # get the Python representation of the scheduler
    submit_result = schedd.submit(job_description)  # submit the job

    print(
        f"Launched news crawl job with cluster-ID={submit_result.cluster()}, "
        f"proc-ID={submit_result.first_proc()}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--download_dir_warc', default="/fast/nchandak/forecasting/newsdata/filtered_cc_warc/", 
                        help="Directory to download WARC files")
    parser.add_argument('--download_dir_article', default="/fast/nchandak/forecasting/newsdata/filtered_cc_articles/", 
                        help="Directory to save extracted articles")
    parser.add_argument('--delete_warc', type=str, choices=["delete", "keep"], default="delete", 
                        help="Whether to delete WARC files after extraction")
    parser.add_argument('--num_processes', type=int, default=1, 
                        help="Number of extraction processes to use")
    parser.add_argument('--job_memory', type=int, default=None, 
                        help="Memory in GB to request for the job")
    parser.add_argument('--job_cpus', type=int, default=1, 
                        help="Number of CPUs to request")
    
    args = parser.parse_args()
    if args.job_memory is None:
        args.job_memory = args.job_cpus * 16
    
    launch_news_crawl_job(
        download_dir_warc=args.download_dir_warc,
        download_dir_article=args.download_dir_article,
        delete_warc_after_extraction=args.delete_warc,
        num_extraction_processes=args.num_processes,
        job_memory=args.job_memory,
        job_cpus=args.job_cpus,
        job_bid=JOB_BID,
    )