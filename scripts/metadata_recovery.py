import yaml
from pathlib import Path
from src.data.patient_dataset import load_hf_data_and_metadata

def main():
    # Paths copied directly from your analysis log configuration
    DATA_DIR = Path("/home/shares/ds4dh/aiidkit_project/data_new/processed/v3.6/teav")
    CONFIG_PATH = Path("configs/discriminative_training.yaml")
    
    # Load config to get structural mappings
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    
    for data_split_type in ["random_split", "center_split", "temporal_split"]:
        target_data_dir = DATA_DIR / data_split_type       
        print(f"\n--- Triggering Metadata Re-generation for: {target_data_dir} ---")
        
        # Passing fup_train=None triggers the 'is_pretraining=True' block 
        # which calculates and writes vocab.pkl and bin_intervals.pkl
        _, bin_intervals, vocab = load_hf_data_and_metadata(
            data_dir=target_data_dir,
            fup_train=None,   # important: this forces pre-training scan mode
            fup_valid=None,   
            fup_test=None,    
            time_mapping=config["data_collator"].get("time_mapping", None),
            eav_mappings=config["data_collator"].get("eav_mappings", None),
        )
        
        print("\n[SUCCESS] Pretraining metadata has been successfully rebuilt!")
        print(f"Vocabulary size: {len(vocab)}")
        print(f"Binned attributes count: {len(bin_intervals)}")

if __name__ == "__main__":
    main()