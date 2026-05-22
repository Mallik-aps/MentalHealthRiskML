import pandas as pd
from src.validation import cronbach_alpha

def test_cronbach_alpha():
    df = pd.DataFrame({"a": [1,2,3,4], "b": [1,2,3,4], "c": [1,2,3,4]})
    assert cronbach_alpha(df) > 0.9
