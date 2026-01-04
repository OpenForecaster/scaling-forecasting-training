import os
from pathlib import Path

import htcondor

JOB_BID_SINGLE = 100 # 100
JOB_BID_MULTI = 100

def launch_lm_label_job(
        JOB_MEMORY,
        JOB_CPUS,
        JOB_GPUS=1,
        JOB_BID=JOB_BID_SINGLE,
        GPU_MEM=None,
        article_path=None,
        output_dir=None,
        additional_args="",
):
    # Name/prefix for cluster logs related to this job
    LOG_PATH = "/fast/nchandak/logs/forecasting/qgen/"
    os.makedirs(LOG_PATH, exist_ok=True)

    CLUSTER_LOGS_SAVE_DIR=Path(LOG_PATH)
    cluster_job_log_name = str(
        CLUSTER_LOGS_SAVE_DIR
        / f"$(Cluster).$(Process)"
    )

    # model_dir += "/snapshots/model/"

    # executable = 'lm_countdown.sh'
    # executable = 'launch_train_job.sh'
    executable = '/home/nchandak/forecasting/qgen/jobs/launch_qgen_job.sh'

    # Construct job description
    job_settings = {
        "executable": executable,
        
        "output": f"{cluster_job_log_name}.out",
        "error": f"{cluster_job_log_name}.err",
        "log": f"{cluster_job_log_name}.log",
        
        # "request_cpus": f"{JOB_CPUS}",  # how many CPU cores we want
        # "request_gpus": f"{JOB_GPUS}",
        # "request_memory": JOB_MEMORY,  # how much memory we want
        # "request_disk": JOB_MEMORY,
        
        "request_gpus": f"{JOB_GPUS}",
        "request_cpus": f"{max(JOB_CPUS*JOB_GPUS, 32)}",  # how many CPU cores we want
        "request_memory": f"{JOB_MEMORY*JOB_GPUS}GB",  # how much memory we want
        "request_disk": f"{JOB_MEMORY*JOB_GPUS}GB",
        
        "jobprio": f"{JOB_BID - 1000}",
        "notify_user": "nikhil.chandak@tuebingen.mpg.de",  # otherwise one does not notice an you can miss clashes
        "notification": "error",
    }
    
    # Add arguments if provided (article_path and output_dir are required for new pipeline)
    # HTCondor requires proper escaping - use single argument string without nested quotes
    if article_path and output_dir:
        args_list = [article_path, output_dir]
        if additional_args:
            args_list.append(additional_args)
        job_settings["arguments"] = " ".join(args_list)
    elif article_path:
        # Fallback: use article directory as output dir
        article_dir = str(Path(article_path).parent / "output")
        job_settings["arguments"] = f"{article_path} {article_dir}"

    if GPU_MEM is not None:
        job_settings["requirements"] = f"(TARGET.CUDAGlobalMemoryMb >= {GPU_MEM}) && (CUDACapability >= 7.0)"
    else:
        job_settings["requirements"] = "CUDACapability >= 7.0"
        
    
    # job_settings["requirements"] = "CUDACapability >= 9.0"


    job_description = htcondor.Submit(job_settings)

    # Submit job to scheduler
    schedd = htcondor.Schedd()  # get the Python representation of the scheduler
    submit_result = schedd.submit(job_description)  # submit the job

    print(
        f"Launched experiment with cluster-ID={submit_result.cluster()}, "
        f"proc-ID={submit_result.first_proc()}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Launch QGen pipeline job with HTCondor. "
                    "Quality defaults always enabled: freeq, leakage checking, best selection, validation, date updates."
    )
    parser.add_argument('--article_path', type=str, required=True,
                       help="Path to the article file to process")
    parser.add_argument('--output_dir', type=str, required=True,
                       help="Directory to save all output files")
    parser.add_argument('--job_memory', type=int, default=64,
                       help="Memory in GB for the job (default: 64)")
    parser.add_argument('--job_cpus', type=int, default=1,
                       help="Number of CPUs for the job (default: 1)")
    parser.add_argument('--job_gpus', type=int, default=1,
                       help="Number of GPUs for the job (default: 1)")
    parser.add_argument('--gpu_mem', type=int, default=45000,
                       help="Minimum GPU memory in MB (default: 45000)")
    parser.add_argument('--additional_args', type=str, default="",
                       help="Additional arguments to pass to run_pipeline.py (e.g., '--cutoff_date 2025-05-01 --seed 42')")
    
    args = parser.parse_args()
    
    launch_lm_label_job(
        JOB_MEMORY=args.job_memory,
        JOB_CPUS=args.job_cpus,
        JOB_GPUS=args.job_gpus,
        JOB_BID=JOB_BID_MULTI,
        GPU_MEM=args.gpu_mem,
        article_path=args.article_path,
        output_dir=args.output_dir,
        additional_args=args.additional_args,
    ) 