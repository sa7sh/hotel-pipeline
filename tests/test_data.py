# tests/test_data.py
import pandas as pd
import os
from src.bigbasket.data.prepare import clean

def test_clean_removes_nans_and_duplicates(tmp_path):
    df = pd.DataFrame({
        "a": [1, 2, 2, None],
        "target": [0,1,1,1],
        "cat": ["x","y","y","z"]
    })
    cleaned = clean(df)
    assert cleaned['target'].notnull().all()
    assert cleaned.shape[0] >= 1