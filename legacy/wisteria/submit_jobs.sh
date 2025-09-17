#!/bin/bash
# Legacy helper: moved from experiments/wisteria/

# Job Submission Helper Script for LGD Training
# Usage: ./submit_jobs.sh [job_type] [options...]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

show_help() {
    echo "LGD Job Submission Helper"
    echo ""
    echo "Usage: $0 <job_type> [options...]"
    echo ""
    echo "Job Types:"
    echo "  zinc-encoder                  - Submit ZINC encoder pretraining job"
    echo "  zinc-diffusion [checkpoint]   - Submit ZINC diffusion training job"
    echo "  qm9-encoder <property>        - Submit QM9 encoder pretraining job"
    echo "  qm9-diffusion <property> [checkpoint] - Submit QM9 diffusion training job"
    echo "  general <dataset> <task>      - Submit general training job"
    echo "  test                          - Submit test job (quick validation)"
    echo ""
    echo "QM9 Properties:"
    echo "  mu, alpha, e_HOMO, e_LUMO, delta_e, cv"
    echo ""
    echo "Examples:"
    echo "  $0 zinc-encoder"
    echo "  $0 zinc-diffusion auto"
    echo "  $0 qm9-encoder mu"
    echo "  $0 qm9-diffusion mu auto"
    echo "  $0 general physics encoder"
    echo "  $0 general photo diffusion /path/to/checkpoint.ckpt"
    echo "  $0 test"
    echo ""
}

check_script_exists() {
    local script_path="$1"
    if [ ! -f "$script_path" ]; then
        echo "ERROR: Script not found: $script_path"
        exit 1
    fi
    if [ ! -x "$script_path" ]; then
        chmod +x "$script_path"
        echo "Made script executable: $script_path"
    fi
}

submit_job() {
    local script_path="$1"
    shift
    local args="$@"
    
    check_script_exists "$script_path"
    
    echo "Submitting job: $(basename "$script_path") $args"
    echo "Script path: $script_path"
    
    if [ -n "$args" ]; then
        # For scripts that need arguments, we need to create a wrapper
        local wrapper_script="${script_path%.sh}_wrapper_$$.sh"
        cat > "$wrapper_script" << EOF
#!/bin/bash
exec "$script_path" $args
EOF
        chmod +x "$wrapper_script"
        pjsub "$wrapper_script"
        # Clean up wrapper after short delay
        (sleep 10; rm -f "$wrapper_script") &
    else
        pjsub "$script_path"
    fi
    
    echo "Job submitted successfully!"
    echo "Check status with: pjstat"
}

# Main logic
case "$1" in
    "zinc-encoder")
        submit_job "$SCRIPT_DIR/zinc_pretrain_encoder.sh"
        ;;
    "zinc-diffusion")
        checkpoint="${2:-auto}"
        submit_job "$SCRIPT_DIR/zinc_train_diffusion.sh" "$checkpoint"
        ;;
    "qm9-encoder")
        property="${2:-mu}"
        if [[ ! "$property" =~ ^(mu|alpha|e_HOMO|e_LUMO|delta_e|cv)$ ]]; then
            echo "ERROR: Invalid QM9 property. Use: mu, alpha, e_HOMO, e_LUMO, delta_e, cv"
            exit 1
        fi
        submit_job "$SCRIPT_DIR/qm9_pretrain_encoder.sh" "$property"
        ;;
    "qm9-diffusion")
        property="${2:-mu}"
        checkpoint="${3:-auto}"
        if [[ ! "$property" =~ ^(mu|alpha|e_HOMO|e_LUMO|delta_e|cv)$ ]]; then
            echo "ERROR: Invalid QM9 property. Use: mu, alpha, e_HOMO, e_LUMO, delta_e, cv"
            exit 1
        fi
        submit_job "$SCRIPT_DIR/qm9_train_diffusion.sh" "$property" "$checkpoint"
        ;;
    "general")
        dataset="$2"
        task="$3"
        checkpoint="$4"
        if [ -z "$dataset" ] || [ -z "$task" ]; then
            echo "ERROR: General job requires dataset and task type"
            echo "Usage: $0 general <dataset> <encoder|diffusion> [checkpoint]"
            exit 1
        fi
        if [ "$task" = "diffusion" ] && [ -n "$checkpoint" ]; then
            submit_job "$SCRIPT_DIR/general_pretrain_template.sh" "$dataset" "$task" "$checkpoint"
        else
            submit_job "$SCRIPT_DIR/general_pretrain_template.sh" "$dataset" "$task"
        fi
        ;;
    "test")
        submit_job "$SCRIPT_DIR/lgd_test.sh"
        ;;
    "help"|"-h"|"--help"|"")
        show_help
        ;;
    *)
        echo "ERROR: Unknown job type: $1"
        echo ""
        show_help
        exit 1
        ;;
esac

echo ""
echo "To monitor your jobs, use:"
echo "  pjstat                    # Show all your jobs"
echo "  pjstat -v                 # Verbose job status"
echo "  $SCRIPT_DIR/monitor_jobs.sh     # Monitor LGD jobs specifically"
