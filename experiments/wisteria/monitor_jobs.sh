#!/bin/bash

# Job Monitoring Script for LGD Training
# Usage: ./monitor_jobs.sh [options]

ROOT=/work/gp15/q25030
RUNS=$ROOT/LatentGraphDiffusion/runs

show_help() {
    echo "LGD Job Monitoring Script"
    echo ""
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -s, --status      Show current job status"
    echo "  -r, --results     Show completed experiments and results"
    echo "  -l, --logs        Show recent log entries"
    echo "  -c, --checkpoints Show available checkpoints"
    echo "  -w, --watch       Continuously monitor (refresh every 30s)"
    echo "  -h, --help        Show this help message"
    echo ""
}

show_job_status() {
    echo "=== Current PJM Job Status ==="
    pjstat -v | grep -E "(JOB_ID|JOB_NAME|STATE|ELAPSE_TIME|zinc|qm9|lgd)" || echo "No LGD jobs found in queue"
    echo ""
}

show_results() {
    echo "=== Completed Experiments ==="
    if [ -d "$RUNS" ]; then
        for exp_dir in "$RUNS"/*/; do
            if [ -d "$exp_dir" ]; then
                exp_name=$(basename "$exp_dir")
                if [ -f "$exp_dir/job_completed.txt" ]; then
                    echo "✓ $exp_name - COMPLETED"
                    cat "$exp_dir/job_completed.txt" | sed 's/^/  /'
                elif [ -f "$exp_dir"/*.log ]; then
                    echo "⧗ $exp_name - IN PROGRESS"
                    # Check if there are any recent log updates
                    latest_log=$(find "$exp_dir" -name "*.log" -type f -exec ls -t {} + | head -1)
                    if [ -n "$latest_log" ]; then
                        last_update=$(stat -f "%Sm" -t "%Y-%m-%d %H:%M:%S" "$latest_log" 2>/dev/null || date -r "$latest_log" "+%Y-%m-%d %H:%M:%S" 2>/dev/null || echo "unknown")
                        echo "  Last update: $last_update"
                    fi
                else
                    echo "? $exp_name - UNKNOWN STATUS"
                fi
                echo ""
            fi
        done
    else
        echo "No experiments directory found at $RUNS"
    fi
}

show_logs() {
    echo "=== Recent Log Entries ==="
    if [ -d "$RUNS" ]; then
        # Find the most recent log files
        latest_logs=$(find "$RUNS" -name "*.log" -type f -exec ls -t {} + | head -3)
        if [ -n "$latest_logs" ]; then
            for log_file in $latest_logs; do
                exp_name=$(basename $(dirname "$log_file"))
                log_name=$(basename "$log_file")
                echo "--- $exp_name/$log_name ---"
                tail -10 "$log_file"
                echo ""
            done
        else
            echo "No log files found"
        fi
    else
        echo "No experiments directory found at $RUNS"
    fi
}

show_checkpoints() {
    echo "=== Available Checkpoints ==="
    if [ -d "$RUNS" ]; then
        find "$RUNS" -name "*.ckpt" -type f | while read -r ckpt; do
            rel_path=${ckpt#$RUNS/}
            size=$(du -h "$ckpt" | cut -f1)
            mod_time=$(stat -f "%Sm" -t "%Y-%m-%d %H:%M:%S" "$ckpt" 2>/dev/null || date -r "$ckpt" "+%Y-%m-%d %H:%M:%S" 2>/dev/null || echo "unknown")
            echo "$rel_path ($size, $mod_time)"
        done | sort
    else
        echo "No experiments directory found at $RUNS"
    fi
}

watch_mode() {
    echo "Monitoring LGD jobs (Ctrl+C to exit)..."
    echo "Refreshing every 30 seconds..."
    echo ""
    
    while true; do
        clear
        echo "LGD Job Monitor - $(date)"
        echo "=================================================================================="
        show_job_status
        echo ""
        show_results
        echo ""
        echo "Press Ctrl+C to exit"
        sleep 30
    done
}

# Parse command line arguments
case "$1" in
    -s|--status)
        show_job_status
        ;;
    -r|--results)
        show_results
        ;;
    -l|--logs)
        show_logs
        ;;
    -c|--checkpoints)
        show_checkpoints
        ;;
    -w|--watch)
        watch_mode
        ;;
    -h|--help|"")
        show_help
        ;;
    *)
        echo "Unknown option: $1"
        show_help
        exit 1
        ;;
esac
