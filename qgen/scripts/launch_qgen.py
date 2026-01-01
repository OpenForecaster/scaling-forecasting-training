import os
from pathlib import Path
import htcondor

JOB_BID_SINGLE = 25  # Default bid for single-GPU jobs

def launch_article_question_job(
        JOB_MEMORY=16,
        JOB_CPUS=4,
        JOB_GPUS=1,
        JOB_BID=JOB_BID_SINGLE,
        GPU_MEM=None,
        model_path=None,
        use_openrouter=False,
        openrouter_model="meta-llama/llama-3.1-8b-instruct:free",
        article_path="/fast/sgoel/forecasting/news/tokenized_data/news/www.apnews.com_tokenized.jsonl",
        output_path="debug/generated_questions.json",
        max_tokens=4096,
        temperature=0.7,
        batch_size=5,
        regenerate=False
):
    # Name/prefix for cluster logs related to this job
    LOG_PATH = "/fast/sgoel/logs/forecasting/qgen/openrouter/"
    os.makedirs(LOG_PATH, exist_ok=True)

    CLUSTER_LOGS_SAVE_DIR = Path(LOG_PATH)
    cluster_job_log_name = str(
        CLUSTER_LOGS_SAVE_DIR
        / f"$(Cluster).$(Process)"
    )

    executable = '/is/cluster/sgoel/forecasting-rl/qgen/scripts/launch_qgen.sh'

    # Construct arguments to pass to the shell script
    arguments = (
        f"{model_path if model_path else 'None'} "
        f"{int(use_openrouter)} "
        f"{openrouter_model} "
        f"{article_path} "
        f"{output_path} "
        f"{max_tokens} "
        f"{temperature} "
        f"{batch_size} "
        f"{int(regenerate)}"
    )

    # Construct job description
    job_settings = {
        "executable": executable,
        "arguments": arguments,
        
        "output": f"{cluster_job_log_name}.out",
        "error": f"{cluster_job_log_name}.err",
        "log": f"{cluster_job_log_name}.log",
        
        "request_gpus": f"{JOB_GPUS}",
        "request_cpus": f"{min(JOB_CPUS, 48)}",  # how many CPU cores we want
        "request_memory": f"{JOB_MEMORY*JOB_CPUS}GB",  # how much memory we want
        
        "jobprio": f"{JOB_BID - 1000}",
        "notify_user": "shashwat.goel@tuebingen.mpg.de",
        "notification": "error",
    }

    if GPU_MEM is not None:
        job_settings["requirements"] = f"(TARGET.CUDAGlobalMemoryMb >= {GPU_MEM}) && (CUDACapability >= 8.0) && (TARGET.Machine != \"g125.internal.cluster.is.localnet\") && (TARGET.Machine != \"g147.internal.cluster.is.localnet\") && (TARGET.Machine != \"g136.internal.cluster.is.localnet\")"
    else:
        job_settings["requirements"] = "(CUDACapability >= 8.0) && (TARGET.Machine != \"g125.internal.cluster.is.localnet\") && (TARGET.Machine != \"g147.internal.cluster.is.localnet\") && (TARGET.Machine != \"g136.internal.cluster.is.localnet\")"

    job_description = htcondor.Submit(job_settings)

    # Submit job to scheduler
    schedd = htcondor.Schedd()  # get the Python representation of the scheduler
    submit_result = schedd.submit(job_description)  # submit the job

    print(
        f"Launched question generation job with cluster-ID={submit_result.cluster()}, "
        f"proc-ID={submit_result.first_proc()}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Launch question generation job on cluster")
    parser.add_argument('--model_path', type=str, default="/fast/rolmedo/models/qwen2.5-14b-it/snapshots/model", help="Path to local model")
    parser.add_argument('--use_openrouter', action='store_true', help="Use OpenRouter API")
    parser.add_argument('--openrouter_model', type=str, default="meta-llama/llama-3.1-8b-instruct:free", 
                        help="OpenRouter model to use")
    parser.add_argument('--article_path', type=str, 
                        default="/fast/sgoel/forecasting/news/tokenized_data/news/deduped/www.reuters.com_selected1000.jsonl",
                        help="Path to article file or directory")
    parser.add_argument('--output_path', type=str, default="/fast/sgoel/forecasting/qgen/reuters/selected1000_2021-2022_qwen2.5-14b",
                        help="Path to save generated questions")
    parser.add_argument('--max_tokens', type=int, default=4096, help="Maximum tokens for generation")
    parser.add_argument('--temperature', type=float, default=0.7, help="Temperature for generation")
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size for processing")
    parser.add_argument('--gpu_mem', type=int, default=48000, help="Minimum GPU memory required in MB")
    parser.add_argument('--job_memory', type=int, default=16, help="Memory per CPU in GB")
    parser.add_argument('--job_cpus', type=int, default=4, help="Number of CPUs")
    parser.add_argument('--job_gpus', type=int, default=1, help="Number of GPUs")
    parser.add_argument('--regenerate', action='store_true', help="Regenerate questions, ignoring existing output file.")
    
    args = parser.parse_args()
    
    launch_article_question_job(
        JOB_MEMORY=args.job_memory,
        JOB_CPUS=args.job_cpus,
        JOB_GPUS=args.job_gpus,
        GPU_MEM=args.gpu_mem,
        model_path=args.model_path,
        use_openrouter=args.use_openrouter,
        openrouter_model=args.openrouter_model,
        article_path=args.article_path,
        output_path=args.output_path,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        batch_size=args.batch_size,
        regenerate=args.regenerate
    )