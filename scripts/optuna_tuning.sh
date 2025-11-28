#!/bin/bash

# ==============================================================================
# This script runs a Python script for specific combinations of arguments.
# It can be run in three modes:
#
# 1. OPTIMIZATION MODE (default):
#    - Runs the Optuna hyperparameter search for each combination.
#    - USAGE: ./scripts/optuna_tuning.sh
#
# 2. FINAL RUN MODE:
#    - Runs the final training using the best parameters found in the DB.
#    - This mode adds the '--run_best_trial' flag to the python command.
#    - USAGE: ./scripts/optuna_tuning.sh --run-best
#
# 3. DEFAULT RUN MODE:
#    - Runs a training using the default parameters found in the config.
#    - This mode adds the '--run_default' flag to the python command.
#    - USAGE: ./scripts/optuna_tuning.sh --run-default
# ==============================================================================

# --- Configuration ---

PYTHON_SCRIPT="scripts/optuna_tuning.py"
PRETRAINED_EMBEDDINGS_OPTIONS=(
    ""
    # "--use_pretrained_embeddings_for_input_layer"
)

PREDICTION_HORIZONS=(
    # "7"
    "30"
    "60"
    "90"
    # "365"
)


# Define the specific pairs of cutoff days for train and valid sets.
# FORMAT: "<train_cutoff_days>;<valid_cutoff_days>"
# - Use a space to separate multiple values (e.g., "0 7 30").
# - To use the script's None, leave the part for train or valid empty.
#   Example: ";30" runs with None train cutoff and valid cutoff of 30.
#   None means full sequences are taken
CUTOFF_PAIRS=(
    ";"  # -> this means None and None, i.e., all possible training + evaluation sequences
    # ";0"
    # ";30"
    # ";120"
    # ";365"
    # "1000;1000"
    # ";1000"
)

# Add any other arguments that should be constant across all runs.
OTHER_ARGS=()


# --- Script Mode ---

# Check for a command-line argument to determine the run mode.
MODE_FLAG="" # More generic variable name
if [[ "$1" == "--run-best" || "$1" == "-rb" ]]; then
    MODE_FLAG="--run_best_trial"
    echo "================================================================================"
    echo " SCRIPT MODE: FINAL RUN"
    echo " Will execute the final training using the best hyper-parameters for each combo."
    echo "================================================================================"
elif [[ "$1" == "--run-default" || "$1" == "-rd" ]]; then
    # Handle the default run mode
    MODE_FLAG="--run_default"
    echo "================================================================================"
    echo " SCRIPT MODE: DEFAULT RUN"
    echo " Will execute training using the default config parameters for each combo."
    echo "================================================================================"
else
    # Default case is optimization
    echo "================================================================================"
    echo " SCRIPT MODE: OPTIMIZATION"
    echo " Will run Optuna hyper-parameter search for each combination."
    echo "================================================================================"
fi


# --- Execution ---

echo "Starting runs for specified combinations..."

# Loop through the primary combinations
for pe_option in "${PRETRAINED_EMBEDDINGS_OPTIONS[@]}"; do
    for ph_option in "${PREDICTION_HORIZONS[@]}"; do
        # Loop through the specific pairs of cutoff days
        for pair in "${CUTOFF_PAIRS[@]}"; do

            # Split the pair into train and valid options based on the semicolon
            ct_option="${pair%;*}"
            cv_option="${pair#*;}"

            # Start building the command in an array for robustness
            CMD=(python "$PYTHON_SCRIPT")

            # Add the --use_pretrained_embeddings_for_input_layer flag if the option is not empty
            if [[ -n "$pe_option" ]]; then
                CMD+=("$pe_option")
            fi

            # Add --prediction_horizon
            CMD+=("--prediction_horizon" "$ph_option")

            # Add --cutoff_days_train if the option is not empty
            if [[ -n "$ct_option" ]]; then
                CMD+=("--cutoff_days_train" $ct_option)
            fi

            # Add --cutoff_days_valid if the option is not empty
            if [[ -n "$cv_option" ]]; then
                CMD+=("--cutoff_days_valid" $cv_option)
            fi

            # Add any other constant arguments
            if [ ${#OTHER_ARGS[@]} -gt 0 ]; then
                CMD+=("${OTHER_ARGS[@]}")
            fi

            # Add the mode-specific flag if it was set
            if [[ -n "$MODE_FLAG" ]]; then
                CMD+=("$MODE_FLAG")
            fi

            # Print the command that is about to be executed
            echo "################################################################################"
            echo "#"
            echo "# RUNNING NEW COMBINATION"
            echo "# Command: ${CMD[@]}"
            echo "#"
            echo "################################################################################"

            # Execute the command
            "${CMD[@]}"

            # Check the exit code of the last command
            if [ $? -ne 0 ]; then
                echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
                echo "WARNING: The last command failed with a non-zero exit code."
                echo "Command was: ${CMD[@]}"
                echo "Continuing with the next combination..."
                echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            fi

        done
    done
done

echo "================================================================================"
echo "All combinations have been executed."
echo "================================================================================"