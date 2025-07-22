#!/bin/bash
# Setup script for Weights & Biases integration

echo "Setting up Weights & Biases for LatentGraphDiffusion..."

# Check if wandb is installed
if ! command -v wandb &> /dev/null; then
    echo "wandb not found. Installing..."
    pip install wandb
fi

# Option 1: Set API key as environment variable
if [ -z "$WANDB_API_KEY" ]; then
    echo ""
    echo "WANDB_API_KEY is not set."
    echo "Please choose one of the following options:"
    echo ""
    echo "1. Set environment variable (recommended for scripts):"
    echo "   export WANDB_API_KEY=your_api_key_here"
    echo "   # Add this to your ~/.bashrc or ~/.zshrc for persistence"
    echo ""
    echo "2. Login via wandb CLI (interactive):"
    echo "   wandb login"
    echo ""
    echo "3. Create API key file in project directory:"
    echo "   echo 'your_api_key_here' > .wandb_api_key"
    echo ""
    read -p "Enter your WandB API key (or press Enter to skip): " api_key
    
    if [ ! -z "$api_key" ]; then
        export WANDB_API_KEY="$api_key"
        echo "export WANDB_API_KEY=\"$api_key\"" >> ~/.bashrc
        echo "WANDB_API_KEY set and added to ~/.bashrc"
        echo ""
        echo "Don't forget to also set your wandb entity in the config file:"
        echo "   wandb.entity: \"your_username_or_team\""
    else
        echo "Skipping API key setup. You can set it later using one of the methods above."
    fi
else
    echo "WANDB_API_KEY is already set ✓"
fi

echo ""
echo "Setup complete! Remember to:"
echo "1. Set your wandb entity in cfg/zinc-encoder-fast.yaml"
echo "2. Run: python pretrain.py --cfg cfg/zinc-encoder-fast.yaml"
echo ""
echo "Your runs will be logged at: https://wandb.ai/your_entity/LatentGraphDiffusion-ZINC"