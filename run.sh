#!/usr/bin/env bash
# set -x

#######################################################################
#                          PART 1  Logging                             #
#######################################################################
# Log format
log_time=$(date "+%Y-%m-%d %H:%M:%S")
log_format="[$log_time] [INFO] [$0]"

#######################################################################
#                          PART 2  Project Config                      #
#######################################################################
# Directory
script_dir="$(realpath "$(dirname "$0")")"
if [[ -z "${root_dir:-}" ]]; then
    if [[ -d "$script_dir/xsam" ]]; then
        root_dir="$script_dir"
    else
        root_dir="$(realpath "$script_dir/../")"
    fi
fi

# Force runtime to use the project virtual environment.
venv_dir="${VENV_DIR:-}"
if [[ -z "$venv_dir" ]]; then
    probe_dir="$script_dir"
    while true; do
        if [[ -x "$probe_dir/.venv/bin/python" ]]; then
            venv_dir="$probe_dir/.venv"
            break
        fi
        parent_dir="$(dirname "$probe_dir")"
        if [[ "$parent_dir" == "$probe_dir" ]]; then
            break
        fi
        probe_dir="$parent_dir"
    done
fi
if [[ -z "$venv_dir" || ! -x "$venv_dir/bin/python" ]]; then
    echo -e "$log_format .venv python not found. Set VENV_DIR or create .venv first."
    exit 1
fi
python_cmd="$venv_dir/bin/python"
if [[ -x "$venv_dir/bin/torchrun" ]]; then
    torchrun_cmd=("$venv_dir/bin/torchrun")
else
    torchrun_cmd=("$python_cmd" "-m" "torch.distributed.run")
fi
export PATH="$venv_dir/bin:$PATH"

code_name="xsam"
code_dir="$root_dir/$code_name"
src_code_dir="$code_dir"
data_dir="$root_dir/data"
init_dir="$root_dir/inits"
work_dir="$root_dir/runs"
export ROOT_DIR="$root_dir/"
export DATA_DIR="$data_dir/"
export INIT_DIR="$init_dir/"
export WORK_DIR="$work_dir/"
export LMUData="$data_dir/LMUData"
export HF_HOME="$init_dir/huggingface"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false
# Torch>=2.6 defaults torch.load(weights_only=True), which breaks resuming
# checkpoints that contain ConfigDict metadata.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export XTUNER_DATASET_TIMEOUT=120
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
if [[ -z "${GPU_PER_NODE:-}" ]]; then
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        IFS=',' read -ra _visible_gpu_ids <<< "$CUDA_VISIBLE_DEVICES"
        if [[ "${#_visible_gpu_ids[@]}" -gt 0 ]]; then
            export GPU_PER_NODE="${#_visible_gpu_ids[@]}"
        else
            export GPU_PER_NODE=1
        fi
    elif command -v nvidia-smi >/dev/null 2>&1; then
        _gpu_count=$(nvidia-smi -L | wc -l | tr -d ' ')
        if [[ -n "$_gpu_count" && "$_gpu_count" -ge 1 ]]; then
            export GPU_PER_NODE="$_gpu_count"
        else
            export GPU_PER_NODE=1
        fi
    else
        export GPU_PER_NODE=1
    fi
fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export NCCL_NET_GDR_LEVEL=2
export MKL_NUM_THREADS=4
export OMP_NUM_THREADS=4
export NCCL_TIMEOUT=36000

#######################################################################
#                          PART 3  Run Config                          #
#######################################################################
# Default modes
default_modes=("train" "segeval" "vlmeval" "visualize")

# Parse command line arguments
modes=()
config_file=""
suffix=""
segeval_resume=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --modes|-m)
            shift
            # Parse modes from comma-separated string or space-separated arguments
            if [[ -z "${1:-}" || "$1" == -* ]]; then
                echo "Error: --modes requires a value (comma-separated or space-separated)."
                exit 1
            fi
            if [[ $1 == *","* ]]; then
                IFS=',' read -ra modes <<< "$1"
            else
                # If no comma, treat the next argument as a single mode
                modes+=("$1")
            fi
            ;;
        --config|-c)
            shift
            config_file="$1"
            ;;
        --suffix|-s)
            shift
            suffix="$1"
            ;;
        --work-dir|-w)
            shift
            work_dir="$1"
            ;;
        --resume)
            segeval_resume=1
            ;;
        --help|-h)
            echo "Usage: $0 [--modes MODE1,MODE2,...] --config CONFIG_FILE [--work-dir WORK_DIR] [--suffix SUFFIX] [--resume] [--help]"
            echo "Available modes: train, segeval, vlmeval, visualize, demo"
            echo "Arguments:"
            echo "  --modes, -m          Specify modes to run (comma-separated or space-separated)"
            echo "  --config, -c         Specify config file path (REQUIRED)"
            echo "  --work-dir, -w       Specify work directory path (optional)"
            echo "  --suffix, -s         Specify suffix for work directory (optional)"
            echo "  --resume             Enable segeval resume (skip datasets with complete predictions)"
            echo "  --help, -h           Show this help message"
            echo "Examples:"
            echo "  $0 --config path/to/config.py                    # Run all modes with specified config"
            echo "  $0 --config config.py --modes train             # Run only training"
            echo "  $0 --config config.py --modes train,segeval     # Run training and segmentation evaluation"
            echo "  $0 --config config.py --work-dir /path/to/work   # Run with custom work directory"
            echo "  $0 --config config.py --suffix test             # Run with suffix 'test'"
            echo "  $0 --config config.py --modes segeval --resume  # Resume segmentation evaluation"
            echo "  $0 --config config.py --modes demo --work-dir /path/to/work  # Launch local Gradio demo (requires checkpoint in work-dir)"
            exit 0
            ;;
        *)
            # If no recognized flag, treat as mode
            modes+=("$1")
            ;;
    esac
    shift
done

# Validate config_file is provided
if [ -z "$config_file" ]; then
    echo "Error: --config/-c parameter is required. Please specify a config file."
    echo "Usage: $0 [--modes MODE1,MODE2,...] --config CONFIG_FILE [--work-dir WORK_DIR] [--suffix SUFFIX] [--help]"
    exit 1
fi

# Extract prefix from config file path
if [ -n "$config_file" ]; then
    # Extract the stage name (s1, s2, s3, etc.) from config file path
    prefix=$(echo "$config_file" | grep -o 's[0-9]_[^/]*' | head -1)
    if [ -z "$prefix" ]; then
        # Fallback to default if no stage found in path
        prefix="s3_mixed_finetune"
    fi
else
    prefix="s3_mixed_finetune"
fi

# If no modes specified, use defaults
if [ ${#modes[@]} -eq 0 ]; then
    modes=("${default_modes[@]}")
fi
model_name=$(basename "$config_file" .py)

# Set vlm_name based on config_file content
if [[ "$config_file" == *"llava"* ]]; then
    vlm_name="llava-phi3-siglip2-ft"
else
    vlm_name="xsam-phi3-siglip2-sam-l-mft"
fi

if [[ "$work_dir" == "$root_dir/runs" || "$work_dir" == "$root_dir/runs/" ]]; then
    suffix_norm=""
    if [[ -n "$suffix" ]]; then
        if [[ "$suffix" == -* ]]; then
            suffix_norm="$suffix"
        else
            suffix_norm="_$suffix"
        fi
    fi
    work_dir="$work_dir/$prefix/$model_name$suffix_norm"
fi

# Normalize to absolute path so later `cd` operations won't break path checks.
work_dir="$(realpath -m "$work_dir")"

ckpt_file="$work_dir/pytorch_model.bin"

# Validate modes
valid_modes=("train" "segeval" "vlmeval" "visualize" "demo")
for mode in "${modes[@]}"; do
    valid=0
    for valid_mode in "${valid_modes[@]}"; do
        if [ "$mode" = "$valid_mode" ]; then
            valid=1
            break
        fi
    done
    if [ $valid -eq 0 ]; then
        echo "Error: Invalid mode '$mode'. Valid modes are: ${valid_modes[*]}"
        exit 1
    fi
done

echo -e "$log_format Running modes: ${modes[*]}"

gpu_per_node="${GPU_PER_NODE:-}"
if [[ -z "$gpu_per_node" ]]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        gpu_per_node=$(nvidia-smi -L | wc -l | tr -d ' ')
        [[ -z "$gpu_per_node" || "$gpu_per_node" -lt 1 ]] && gpu_per_node=1
    else
        gpu_per_node=1
    fi
fi
master_addr="${MASTER_ADDR:-localhost}"
master_port="${MASTER_PORT:-29500}"
node_rank="${NODE_RANK:-0}"
deepspeed_cfg="${DEEPSPEED_CFG:-}"
if [[ -z "$deepspeed_cfg" ]]; then
    is_wsl=0
    if [[ -r /proc/version ]] && grep -qi microsoft /proc/version; then
        is_wsl=1
    fi
    if [[ "$gpu_per_node" -le 1 ]]; then
        if [[ "$is_wsl" -eq 1 ]]; then
            deepspeed_cfg="deepspeed_zero2"
        else
            deepspeed_cfg="deepspeed_zero2_offload"
        fi
    else
        deepspeed_cfg="deepspeed_zero2"
    fi
fi
echo -e "$log_format DeepSpeed config: $deepspeed_cfg"

# Run
for mode in "${modes[@]}"
do
    cd $root_dir
    echo -e "$log_format Mode: $mode."
    if [ "$mode" = "segeval" ] && [ "$segeval_resume" = "1" ]; then
        echo -e "$log_format segeval resume: enabled."
    fi
    time=$(date "+%Y%m%d-%H%M%S")
    if [ $mode = "train" ] && [ ! -d "$work_dir" ] && [ $node_rank = 0 ]; then
        mkdir -p $work_dir
        cp -rf $(realpath $0) $code_dir $work_dir
        find "$work_dir/$code_name" -name "*.crc" -type f -delete
        find "$work_dir/$code_name" -type f -exec chmod 664 {} +
        find "$work_dir/$code_name" -type d -exec chmod 775 {} +
    fi
    if [ -d "$work_dir/$code_name" ]; then
        code_dir="$work_dir/$code_name"
        cp $(realpath $0) $work_dir
    fi

    # Always sync latest source config into work_dir snapshot to avoid stale copied configs.
    source_cfg_abs="$root_dir/$config_file"
    [ -f "$source_cfg_abs" ] || source_cfg_abs="$root_dir/${config_file#$code_name/}"
    if [ -d "$work_dir/$code_name" ] && [ -f "$source_cfg_abs" ]; then
        if [[ "$source_cfg_abs" == "$src_code_dir/"* ]]; then
            cfg_rel="${source_cfg_abs#$src_code_dir/}"
        else
            cfg_rel="${config_file#$code_name/}"
        fi
        mkdir -p "$code_dir/$(dirname "$cfg_rel")"
        cp -f "$source_cfg_abs" "$code_dir/$cfg_rel"
        config_file="$cfg_rel"
        echo -e "$log_format Synced config to snapshot: $cfg_rel"
    fi

    cd $code_dir
    export CODE_DIR="$code_dir/"
    echo -e "$log_format code_dir: $code_dir"

    [ -f "$config_file" ] || config_file="${config_file#$code_name/}"
    [ -f "$config_file" ] || { echo -e "$log_format Config file not found: $config_file" >&2; exit 1; }
    
    # mode: train
    trained_flag=0
    if [ $mode = "train" ]; then
        echo -e "$log_format Training $model_name."
        PYTHONPATH="$(realpath $code_dir)":$PYTHONPATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
            "${torchrun_cmd[@]}" --master_addr=$master_addr --master_port=$master_port --nproc_per_node=$gpu_per_node \
            $code_dir/xsam/tools/train.py \
            $config_file \
            --work-dir $work_dir \
            --resume auto \
            --launcher pytorch \
            --deepspeed $deepspeed_cfg \
            --seed 1024 | { [ $node_rank = "0" ] && tee $work_dir/${mode}-${time}.log || cat; }
    fi
    # Check if training completed successfully
    if [ -f $ckpt_file ]; then
        trained_flag=1
    fi
    # mode: segeval
    if [ $mode = "segeval" ] && [ $trained_flag = 1 ]; then
        echo -e "$log_format Evaluating Seg: $model_name."
        [ $node_rank -ne 0 ] && sleep 60
        segeval_resume_args=()
        if [ "$segeval_resume" = "1" ]; then
            segeval_resume_args+=(--resume)
        fi
        PYTHONPATH="$(realpath $code_dir)":$PYTHONPATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
            "${torchrun_cmd[@]}" --master_addr=$master_addr --master_port=$master_port --nproc_per_node=$gpu_per_node \
            $code_dir/xsam/tools/eval.py \
            $config_file \
            --launcher pytorch \
            --work-dir $work_dir \
            --seed 0 \
            --pth_model latest \
            "${segeval_resume_args[@]}" | { [ $node_rank = "0" ] && tee $work_dir/${mode}-${time}.log || cat; }
    fi
    # mode: vlmeval
    if [ $mode = "vlmeval" ] && [ $trained_flag = 1 ]; then
        if [ $node_rank = 0 ] && [ ! -d "$work_dir/xtuner_model" ]; then
            echo -e "$log_format Converting $model_name to HF format."
            PYTHONPATH="$(realpath $code_dir)":$PYTHONPATH \
                "$python_cmd" $code_dir/xsam/tools/model_tools/pth_to_hf.py \
                $code_dir/$config_file \
                $work_dir
        fi
        # Remove existing target and create/refresh symlink safely
        rm -rf "$init_dir/$vlm_name"
        ln -sfn "$work_dir/xtuner_model" "$init_dir/$vlm_name"
        while [ ! -d "$work_dir/xtuner_model" ]; do
            echo -e "$log_format Waiting for $model_name to be converted to HF format."
            sleep 5
        done
        if [ -d "$work_dir/xtuner_model" ]; then
            echo -e "$log_format Evaluating VLM: $model_name."
            [ $node_rank -ne 0 ] && sleep 30
            PYTHONPATH="$(realpath $code_dir)":$PYTHONPATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
                "${torchrun_cmd[@]}" --master_addr=$master_addr --master_port=$master_port --nproc_per_node=$gpu_per_node \
                $code_dir/xsam/evaluation/vlmeval/run.py \
                --data MME MMBench_DEV_EN SEEDBench_IMG POPE AI2D_TEST \
                --model $vlm_name \
                --work-dir $work_dir/vlmeval_results | { [ "$node_rank" = "0" ] && tee "$work_dir/${mode}-${time}.log" || cat; }
        fi
    fi
    # mode: visualize
    if [ $mode = "visualize" ] && [ $trained_flag = 1 ] && [ $node_rank = 0 ]; then
        echo -e "$log_format Visualizing $model_name."
        PYTHONPATH="$(realpath $code_dir)":$PYTHONPATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
            "$python_cmd" $code_dir/xsam/tools/visualize.py \
            $config_file \
            --work-dir $work_dir \
            --seed 0 \
            --pth_model latest | { [ $node_rank = "0" ] && tee $work_dir/${mode}-${time}.log || cat; }
    fi
    # mode: demo
    if [ $mode = "demo" ] && [ $trained_flag = 1 ] && [ $node_rank = 0 ]; then
        echo -e "$log_format Demoing $model_name."
        mkdir -p "$work_dir/app_logs"
        PYTHONPATH="$(realpath $code_dir)":$PYTHONPATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
            "$python_cmd" $code_dir/xsam/demo/app.py \
            $config_file \
            --work-dir $work_dir \
            --log-dir $work_dir/app_logs \
            --pth_model latest \
            --seed 0 \
            --port 7862 | { [ $node_rank = "0" ] && tee $work_dir/${mode}-${time}.log || cat; }
    fi
    rm -rf /tmp/xsam_cache > /dev/null 2>&1
done
