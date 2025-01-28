ml gcc/8 
# Initialize Conda for the script
eval "$(conda shell.bash hook)"

# Activate the Conda environment
conda activate LipFD

# Debug: Confirm the active environment
echo "Active Conda environment:"
echo "Python executable: $(which python)"

python preprocess_lavdf.py
deactivate