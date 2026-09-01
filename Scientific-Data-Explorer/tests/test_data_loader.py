import pandas as pd

from backend.src.data_loader import clean_data, extract_numeric_matrix, load_dataset


def test_clean_data_accepts_fill_value():
    df = pd.DataFrame({"a": [1.0, None], "b": ["x", "y"]})

    cleaned = clean_data(df, fill_value=0.0)

    assert cleaned.loc[1, "a"] == 0.0
    assert cleaned["b"].tolist() == ["x", "y"]


def test_extract_numeric_matrix_ignores_non_numeric_columns(tmp_path):
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text("a,b,c\n1,hello,3\n2,world,4\n")

    df = load_dataset(csv_path)
    matrix = extract_numeric_matrix(df)

    assert matrix.tolist() == [[1.0, 3.0], [2.0, 4.0]]
