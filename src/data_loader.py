import os
import pandas as pd
from datasets import load_dataset

class AgencyDataLoader:
    """
    Handles data ingestion for the pipeline, supporting both instant text injection
    and automatic retrieval/downloading of production datasets via Hugging Face.
    """
    def __init__(self, local_dir="local_datasets"):
        self.local_dir = local_dir
        if not os.path.exists(self.local_dir):
            os.makedirs(self.local_dir)

    def load_raw_real_life_example(self) -> str:
        """
        Returns the specific real-life example summary specified in the project brief.
        Simulates the extracted manual data from NotebookLM.
        """
        example_context = (
            "Source 1 (Clinical Study): Long-term use of Metformin is heavily correlated with "
            "malabsorption of Vitamin B12. Over 15 years, up to 30% of patients experience a drop, "
            "leading to fatigue, anemia, and peripheral neuropathy.\n"
            "Source 2 (Gastrointestinal Tracker): Gastrointestinal side effects like diarrhea, nausea, "
            "and abdominal cramping are common during initiation, but in a small subset of chronic users, "
            "these symptoms persist long-term or worsen due to changes in gut microbiota.\n"
            "Source 3 (Renal Guidelines): Metformin is excreted unchanged by the kidneys. While it doesn't cause "
            "kidney disease, long-term monitoring is vital because declining renal function reduces clearance, "
            "heightening other toxicities.\n"
            "Source 4 (Critical Risks): Lactic acidosis is a rare but life-threatening complication. Long-term risk "
            "factors include hidden liver impairment, hypoxia, or binge drinking while on the medication."
        )
        return example_context

    def auto_download_hf_dataset(self, dataset_repo: str = "imdb") -> pd.DataFrame:
        """
        Automatically downloads a dataset from Hugging Face for the purpose of scaffolding 
        fine-tuning data or fetching agency historical templates.
        
        Args:
            dataset_repo: Hugging Face dataset repository path.
        Returns:
            A Pandas DataFrame wrapper of the text elements.
        """
        try:
            # Using a tiny subset configuration for generic text examples if repository is generic
            dataset = load_dataset(dataset_repo, split='train', streaming=True)
            # Take a small sample of 5 items for pipeline validation without bloating memory
            sample_data = list(dataset.take(5))
            df = pd.DataFrame(sample_data)
            df.to_csv(os.path.join(self.local_dir, "downloaded_pipeline_templates.csv"), index=False)
            return df
        except Exception as e:
            # Fallback wrapper to keep production code execution alive
            print(f"Error fetching Hugging Face dataset: {e}")
            return pd.DataFrame({"text": ["Fallback example placeholder for fine-tuning configuration."]})