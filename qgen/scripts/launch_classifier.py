import htcondor
from pathlib import Path

# Job bidding constants
JOB_BID_SINGLE = 25  # 100
JOB_BID_MULTI = 400

def launch_classifier_job(
        data_path,
        model_name,
        output_dir,
        wandb_project,
        wandb_run_name,
        train_ratio=0.6,
        test_ratio=0.3,
        num_train_epochs=60,
        freeze_base_model=False,
        random_init=False,
        gradacc_steps=2,
        JOB_MEMORY=32,
        JOB_CPUS=4,
        JOB_GPUS=1,
        JOB_BID=JOB_BID_SINGLE,
        GPU_MEM=None,
):
    # Name/prefix for cluster logs related to this job
    LOG_PATH = "/fast/sgoel/logs/forecasting/qgen/classifier/"

    CLUSTER_LOGS_SAVE_DIR = Path(LOG_PATH)
    CLUSTER_LOGS_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    
    cluster_job_log_name = str(
        CLUSTER_LOGS_SAVE_DIR
        / f"$(Cluster).$(Process)"
    )

    executable = 'scripts/launch_classifier.sh'

    # Arguments to pass to the shell script
    freeze_flag = "--freeze_base_model" if freeze_base_model else ""
    random_init_flag = "--random_init" if random_init else ""
    arguments = f"{data_path} {model_name} {output_dir} {wandb_project} {wandb_run_name} {train_ratio} {test_ratio} {num_train_epochs} {gradacc_steps} {freeze_flag} {random_init_flag}"

    # Construct job description
    job_settings = {
        "executable": executable,
        "arguments": arguments,
        
        "output": f"{cluster_job_log_name}.out",
        "error": f"{cluster_job_log_name}.err",
        "log": f"{cluster_job_log_name}.log",
        
        "request_gpus": f"{JOB_GPUS}",
        "request_cpus": f"{max(JOB_CPUS*JOB_GPUS, 16)}",  # how many CPU cores we want
        "request_memory": f"{JOB_MEMORY*JOB_GPUS}GB",  # how much memory we want
        
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
        f"Launched classifier job with cluster-ID={submit_result.cluster()}, "
        f"proc-ID={submit_result.first_proc()}")
    
    return submit_result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Launch classifier training job")
    
    # Data parameters
    parser.add_argument("--data_path", type=str, 
                        default="/is/cluster/fast/sgoel/forecasting/qgen/reuters/mcq/selected10_2021-2022_deepseekv3024",
                        help="Path to the JSON data file")
    parser.add_argument("--train_ratio", type=float, default=0.6,
                        help="Proportion of data for training")
    parser.add_argument("--test_ratio", type=float, default=0.3,
                        help="Proportion of data for testing")
    
    # Model parameters
    parser.add_argument("--model_name", type=str, default="microsoft/deberta-v3-large",
                        help="Name of the pre-trained model to use")
    parser.add_argument("--output_dir", type=str, default="./results/dsv3-mcq-1k/test",
                        help="Directory to save model outputs")
    parser.add_argument("--num_train_epochs", type=int, default=60,
                        help="Number of training epochs")
    parser.add_argument("--freeze_base_model", action="store_true",
                        help="If set, freeze base model and train only the classification head")
    parser.add_argument("--random_init", action="store_true",
                        help="If set, initialize model with random weights instead of pretrained")
    parser.add_argument("--gradacc_steps", type=int, default=2,
                        help="Number of gradient accumulation steps")
    
    # Wandb parameters
    parser.add_argument("--wandb_project", type=str, default="mcq-classifier",
                        help="Wandb project name")
    parser.add_argument("--wandb_run_name", type=str, default="dsv3-mcq-1k",
                        help="Wandb run name")
    
    # Job parameters
    parser.add_argument("--job_memory", type=int, default=None,
                        help="Memory in GB per GPU")
    parser.add_argument("--job_cpus", type=int, default=4,
                        help="Number of CPUs per GPU")
    parser.add_argument("--job_gpus", type=int, default=1,
                        help="Number of GPUs")
    parser.add_argument("--gpu_mem", type=int, default=48000,
                        help="Minimum GPU memory in MB")
    
    args = parser.parse_args()

    if args.job_memory is None:
        args.job_memory = 16 * args.job_cpus

    # Dynamically update wandb run name and output dir
    run_name = args.wandb_run_name
    output_dir = args.output_dir # <-- Get base output dir
    if args.freeze_base_model:
        run_name += "_freeze"
        output_dir += "_freeze" # <-- Append to output dir
    if args.random_init:
        run_name += "_rnginit"
        output_dir += "_rnginit" # <-- Append to output dir

    # Launch the job
    launch_classifier_job(
        data_path=args.data_path,
        model_name=args.model_name,
        output_dir=output_dir, # <-- Use the updated output_dir
        wandb_project=args.wandb_project,
        wandb_run_name=run_name, # <-- Use the updated run_name
        train_ratio=args.train_ratio,
        test_ratio=args.test_ratio,
        num_train_epochs=args.num_train_epochs,
        freeze_base_model=args.freeze_base_model,
        random_init=args.random_init,
        gradacc_steps=args.gradacc_steps,
        JOB_MEMORY=args.job_memory,
        JOB_CPUS=args.job_cpus,
        JOB_GPUS=args.job_gpus,
        GPU_MEM=args.gpu_mem,
    )