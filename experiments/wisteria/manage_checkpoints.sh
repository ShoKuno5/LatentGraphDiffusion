#!/bin/bash

# Checkpoint Management Script for LGD Training
# Usage: ./manage_checkpoints.sh [command] [options]

ROOT=/work/gp15/q25030
RUNS=$ROOT/LatentGraphDiffusion/runs

show_help() {
    echo "LGD Checkpoint Management Script"
    echo ""
    echo "Usage: $0 <command> [options]"
    echo ""
    echo "Commands:"
    echo "  list [pattern]           List all checkpoints (optionally filter by pattern)"
    echo "  find <dataset> <type>    Find latest checkpoint for dataset/type"
    echo "  info <checkpoint_path>   Show detailed checkpoint information"
    echo "  clean [days]             Clean old checkpoints (default: 30 days)"
    echo "  backup <checkpoint_path> Backup checkpoint to archive directory"
    echo "  link <src> <dest>        Create symbolic link for checkpoint"
    echo ""
    echo "Examples:"
    echo "  $0 list zinc"
    echo "  $0 find qm9 encoder"
    echo "  $0 info /path/to/checkpoint.ckpt"
    echo "  $0 clean 7"
    echo "  $0 backup /path/to/best_checkpoint.ckpt"
    echo ""
}

list_checkpoints() {
    local pattern="${1:-.*}"
    echo "=== Available Checkpoints ==="
    echo "Pattern: $pattern"
    echo ""
    
    if [ -d "$RUNS" ]; then
        find "$RUNS" -name "*.ckpt" -type f | grep -E "$pattern" | while read -r ckpt; do
            rel_path=${ckpt#$RUNS/}
            size=$(du -h "$ckpt" | cut -f1)
            mod_time=$(stat -f "%Sm" -t "%Y-%m-%d %H:%M:%S" "$ckpt" 2>/dev/null || date -r "$ckpt" "+%Y-%m-%d %H:%M:%S" 2>/dev/null || echo "unknown")
            
            # Extract experiment info from path
            exp_name=$(echo "$rel_path" | cut -d'/' -f1)
            checkpoint_name=$(basename "$ckpt")
            
            printf "%-50s %8s %s\n" "$exp_name/$checkpoint_name" "$size" "$mod_time"
        done | sort -k3r  # Sort by modification time (newest first)
    else
        echo "No experiments directory found at $RUNS"
    fi
}

find_latest_checkpoint() {
    local dataset="$1"
    local task_type="$2"
    
    if [ -z "$dataset" ] || [ -z "$task_type" ]; then
        echo "ERROR: Please specify dataset and task type"
        echo "Usage: $0 find <dataset> <task_type>"
        return 1
    fi
    
    echo "=== Finding Latest Checkpoint ==="
    echo "Dataset: $dataset"
    echo "Task Type: $task_type"
    echo ""
    
    # Look for checkpoints matching the pattern
    pattern="${dataset}.*${task_type}"
    latest=$(find "$RUNS" -path "*${dataset}*${task_type}*/ckpt/*.ckpt" -type f -exec ls -t {} + | head -1)
    
    if [ -n "$latest" ]; then
        echo "Latest checkpoint found:"
        echo "$latest"
        echo ""
        # Show checkpoint info
        show_checkpoint_info "$latest"
    else
        echo "No checkpoints found matching pattern: $pattern"
        echo ""
        echo "Available experiments:"
        ls -la "$RUNS" | grep -E "$dataset|$task_type" || echo "None found"
    fi
}

show_checkpoint_info() {
    local ckpt_path="$1"
    
    if [ -z "$ckpt_path" ]; then
        echo "ERROR: Please specify checkpoint path"
        return 1
    fi
    
    if [ ! -f "$ckpt_path" ]; then
        echo "ERROR: Checkpoint file not found: $ckpt_path"
        return 1
    fi
    
    echo "=== Checkpoint Information ==="
    echo "Path: $ckpt_path"
    echo "Size: $(du -h "$ckpt_path" | cut -f1)"
    echo "Modified: $(stat -f "%Sm" -t "%Y-%m-%d %H:%M:%S" "$ckpt_path" 2>/dev/null || date -r "$ckpt_path" "+%Y-%m-%d %H:%M:%S" 2>/dev/null || echo "unknown")"
    echo ""
    
    # Try to extract information from the checkpoint using Python
    python3 -c "
import torch
import sys
try:
    ckpt = torch.load('$ckpt_path', map_location='cpu')
    print('Checkpoint Keys:')
    for key in ckpt.keys():
        if key == 'state_dict':
            print(f'  {key}: model weights ({len(ckpt[key])} parameters)')
        elif key == 'optimizer':
            print(f'  {key}: optimizer state')
        elif key == 'lr_scheduler':
            print(f'  {key}: learning rate scheduler state')
        elif key == 'epoch':
            print(f'  {key}: {ckpt[key]}')
        elif key == 'global_step':
            print(f'  {key}: {ckpt[key]}')
        else:
            print(f'  {key}: {type(ckpt[key])}')
    
    if 'epoch' in ckpt:
        print(f'\\nTraining Epoch: {ckpt[\"epoch\"]}')
    if 'global_step' in ckpt:
        print(f'Global Step: {ckpt[\"global_step\"]}')
        
except Exception as e:
    print(f'Error reading checkpoint: {e}')
    sys.exit(1)
" 2>/dev/null || echo "Could not read checkpoint metadata (Python/PyTorch not available)"
}

clean_old_checkpoints() {
    local days="${1:-30}"
    echo "=== Cleaning Old Checkpoints ==="
    echo "Removing checkpoints older than $days days..."
    echo ""
    
    # Find old checkpoints
    old_checkpoints=$(find "$RUNS" -name "*.ckpt" -type f -mtime +$days)
    
    if [ -n "$old_checkpoints" ]; then
        echo "Checkpoints to be removed:"
        echo "$old_checkpoints" | while read -r ckpt; do
            size=$(du -h "$ckpt" | cut -f1)
            mod_time=$(stat -f "%Sm" -t "%Y-%m-%d %H:%M:%S" "$ckpt" 2>/dev/null || date -r "$ckpt" "+%Y-%m-%d %H:%M:%S" 2>/dev/null)
            echo "  $ckpt ($size, $mod_time)"
        done
        
        echo ""
        read -p "Are you sure you want to delete these checkpoints? (y/N): " confirm
        if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
            echo "$old_checkpoints" | xargs rm -f
            echo "Checkpoints removed."
        else
            echo "Operation cancelled."
        fi
    else
        echo "No checkpoints older than $days days found."
    fi
}

backup_checkpoint() {
    local ckpt_path="$1"
    
    if [ -z "$ckpt_path" ]; then
        echo "ERROR: Please specify checkpoint path"
        return 1
    fi
    
    if [ ! -f "$ckpt_path" ]; then
        echo "ERROR: Checkpoint file not found: $ckpt_path"
        return 1
    fi
    
    # Create backup directory
    backup_dir="$RUNS/backups"
    mkdir -p "$backup_dir"
    
    # Generate backup filename with timestamp
    backup_name="$(basename "$ckpt_path" .ckpt)_backup_$(date +%Y%m%d_%H%M%S).ckpt"
    backup_path="$backup_dir/$backup_name"
    
    echo "=== Backing up Checkpoint ==="
    echo "Source: $ckpt_path"
    echo "Backup: $backup_path"
    echo ""
    
    cp "$ckpt_path" "$backup_path"
    if [ $? -eq 0 ]; then
        echo "Checkpoint backed up successfully!"
        echo "Backup size: $(du -h "$backup_path" | cut -f1)"
    else
        echo "ERROR: Failed to backup checkpoint"
        return 1
    fi
}

create_checkpoint_link() {
    local src="$1"
    local dest="$2"
    
    if [ -z "$src" ] || [ -z "$dest" ]; then
        echo "ERROR: Please specify source and destination paths"
        echo "Usage: $0 link <source_checkpoint> <destination_link>"
        return 1
    fi
    
    if [ ! -f "$src" ]; then
        echo "ERROR: Source checkpoint not found: $src"
        return 1
    fi
    
    echo "=== Creating Checkpoint Link ==="
    echo "Source: $src"
    echo "Link: $dest"
    echo ""
    
    ln -sf "$src" "$dest"
    if [ $? -eq 0 ]; then
        echo "Symbolic link created successfully!"
        ls -la "$dest"
    else
        echo "ERROR: Failed to create symbolic link"
        return 1
    fi
}

# Main command dispatch
case "$1" in
    "list")
        list_checkpoints "$2"
        ;;
    "find")
        find_latest_checkpoint "$2" "$3"
        ;;
    "info")
        show_checkpoint_info "$2"
        ;;
    "clean")
        clean_old_checkpoints "$2"
        ;;
    "backup")
        backup_checkpoint "$2"
        ;;
    "link")
        create_checkpoint_link "$2" "$3"
        ;;
    "help"|"-h"|"--help"|"")
        show_help
        ;;
    *)
        echo "ERROR: Unknown command: $1"
        echo ""
        show_help
        exit 1
        ;;
esac
