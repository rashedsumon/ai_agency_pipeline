import pandas as pd
from typing import List, Dict, Any

def convert_production_sheet_to_dataframe(production_sheet: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Converts structured graph outputs into flat pandas DataFrames
    tailored for direct CSV export into Canva Bulk Create or CapCut metadata sheets.
    """
    if not production_sheet:
        return pd.DataFrame()
    return pd.DataFrame(production_sheet)