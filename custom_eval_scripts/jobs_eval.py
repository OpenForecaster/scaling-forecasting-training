"""
HTCondor job submission script for model evaluations.
Submits evaluation jobs to HTCondor cluster with configurable resources.
Supports multiple tasks: forecasting, freeform, mmlu-pro, math, simpleqa, retrieval.
Automatically determines output directories based on task type.
Main entry point for launching distributed evaluation jobs.
"""

import os
from pathlib import Path

import htcondor

JOB_BID_SINGLE = 2000 # 100
JOB_BID_MULTI = 2000

def launch_lm_label_job(
        model_dir,
        model_name,
        save_file,
        JOB_MEMORY,
        JOB_CPUS,
        JOB_GPUS=1,
        JOB_BID=JOB_BID_SINGLE,
        GPU_MEM=None,
        NEW_TOKENS=4096,
        DATA_SPLIT="test",
        DATA="halawi",
        NUM_GENERATIONS=1,
        TASK="forecasting",
        NUM_ARTICLES=10,
):
    # Name/prefix for cluster logs related to this job
    # LOG_PATH = "/fast/sgoel/logs/forecasting/evals"
    LOG_PATH = "/fast/nchandak/logs/forecasting/custom_evals_vllm/"
    
    
    CLUSTER_LOGS_SAVE_DIR=Path(LOG_PATH)
    cluster_job_log_name = str(
        CLUSTER_LOGS_SAVE_DIR
        / f"$(Cluster).$(Process)"
    )

    # model_dir += "/snapshots/model/"

    executable = 'launch_custom_eval_job.sh'

    # Construct job description
    job_settings = {
        "executable": executable,
        "arguments": (
            f"{TASK} "
            f"{save_file} "
            f"{model_dir} "
            f"{model_name} "
            f"{NEW_TOKENS} "
            f"{DATA_SPLIT} "
            f"{NUM_GENERATIONS} "
            f"{DATA} "
            f"{NUM_ARTICLES} "
        ),
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

    if GPU_MEM is not None:
        job_settings["requirements"] = f"(TARGET.CUDAGlobalMemoryMb >= {GPU_MEM}) && (CUDACapability >= 9.0)"
    else:
        job_settings["requirements"] = "CUDACapability >= 9.0"
        
    
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
    parser = argparse.ArgumentParser()

    parser.add_argument('--base_save_dir', type=str, default=None, help="Where to save outputs of the model")
    
    # parser.add_argument('--model_dir', type=str, default="/fast/rolmedo/models/llama-3.1-8b-it", help="Model directory")
    # parser.add_argument('--model', type=str, default="llama-3.1-8b-it", help="Model name")
    
    parser.add_argument('--model_dir', type=str, default="/fast/rolmedo/models/qwen2.5-7b-it", help="Model directory (either checkpoint or directory of the original model)")
    parser.add_argument('--model', type=str, default="qwen2.5-7b-it", help="Model name")

    # Add max_new_tokens arg
    parser.add_argument('--max_new_tokens', type=int, default=4096, help="Maximum number of new tokens for generation")
    
    parser.add_argument('--n_gpus', type=int, default=1, help="Number of GPUs to request")
    
    parser.add_argument('--data_split', type=str, default="test", help="Data split to use (train or test)")
    parser.add_argument('--data', type=str, default=None,
                      help="Which dataset to use (metaculus or halawi or some math dataset)")
    
    parser.add_argument('--num_generations', type=int, default=1, help="Number of generations to use per prompt")
    
    parser.add_argument('--task', type=str, default="freeform",
                      help="Which task to run (forecasting or mmlu-pro or math)")
    
    parser.add_argument('--num_articles', type=int, default=10, help="Number of articles to use per prompt")
    
    args = parser.parse_args()
    
    DATA = args.data
    if args.task == "math":
        output_dir = "/fast/nchandak/forecasting/evals/manual/math/"
        if DATA is None:
            DATA = "DigitalLearningGmbH/MATH-lighteval"
            
    elif args.task == "mmlu-pro":
        output_dir = "/fast/nchandak/forecasting/evals/manual/mmlu-pro/"
        if DATA is None:
            DATA = "TIGER-Lab/MMLU-Pro"
            
    elif "forecasting" in args.task:
        output_dir = "/fast/nchandak/forecasting/evals/manual/"
        if DATA is None:
            DATA = "halawi"
        output_dir += DATA + "/"
        
    elif "freeform" in args.task:
        output_dir = "/fast/nchandak/forecasting/evals/freeform/manual/"
        if DATA is None:
            DATA = "/fast/sgoel/forecasting/news/tokenized_data/news/deduped/recent/qgen/deepseek-chat-v3-0324_dw_21317_free_3.jsonl"

    elif "simpleqa" in args.task:
        output_dir = "/fast/nchandak/forecasting/evals/freeform/SimpleQA/"
        if DATA is None:
            DATA = "basicv8vc/SimpleQA"
    
    elif "retrieval" in args.task:
        output_dir = "/fast/nchandak/forecasting/evals/freeform/manual/"
        if DATA is None:
            DATA = "/fast/nchandak/forecasting/newsdata/theguardian/qgen/cleaned/o4-mini-high_theguardian-retrieval-30_207_free_3_cleaned.jsonl"
            
    else:
        raise ValueError(f"Task {args.task} not supported")
    # else:
        
    #     raise ValueError(f"Task {args.task} not supported")
        
    # if base_save_dir is provided, then use it as the output directory
    if args.base_save_dir is not None:
        output_dir = args.base_save_dir
    
    GPU_MEM = 45000
    launch_lm_label_job(
        model_dir=args.model_dir,
        model_name=args.model,
        save_file=output_dir,
        JOB_MEMORY=64,
        JOB_CPUS=args.n_gpus,
        JOB_GPUS=args.n_gpus,
        JOB_BID=JOB_BID_SINGLE,
        GPU_MEM=GPU_MEM,
        NEW_TOKENS=args.max_new_tokens,
        DATA_SPLIT=args.data_split,
        DATA=args.data,
        NUM_GENERATIONS=args.num_generations,
        TASK=args.task,
        NUM_ARTICLES=args.num_articles,
    )
        # break 
