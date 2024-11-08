import pytest
import os
import pandas as pd
import readcsvturbo as rct
from io import StringIO

# Define the sample CSV data as a global variable
SAMPLE_CSV_DATA = """col1,col2,col3
1,2,3
4,5,6
7,8,9
10,11,12
13,14,15
"""

@pytest.fixture
def sample_csv():
    # Create a sample CSV file for testing
    sample_path = 'sample.csv'
    with open(sample_path, 'w') as f:
        f.write(SAMPLE_CSV_DATA)
    yield sample_path
    # Clean up
    os.remove(sample_path)

@pytest.fixture
def expected_df():
    # Create a DataFrame from the sample data for comparison
    df = pd.read_csv(StringIO(SAMPLE_CSV_DATA))
    return df

def test_read_csv_head_with_extra_rows(sample_csv, expected_df):
    df_head = rct.read_csv_head(sample_csv, header=True, n_rows=10)
    # Check that all data rows are returned
    pd.testing.assert_frame_equal(df_head, expected_df)

def test_read_csv_head_with_exact_rows(sample_csv, expected_df):
    df_head = rct.read_csv_head(sample_csv, header=True, n_rows=3)
    expected_head = expected_df.iloc[:3]
    pd.testing.assert_frame_equal(df_head, expected_head)

def test_read_csv_tail_with_extra_rows(sample_csv, expected_df):
    df_tail = rct.read_csv_tail(sample_csv, header=True, n_rows=10)
    # Check that all data rows are returned
    pd.testing.assert_frame_equal(df_tail, expected_df)

def test_read_csv_tail_with_exact_rows(sample_csv, expected_df):
    df_tail = rct.read_csv_tail(sample_csv, header=True, n_rows=2)
    expected_tail = expected_df.iloc[-2:]
    pd.testing.assert_frame_equal(df_tail.reset_index(drop=True), expected_tail.reset_index(drop=True))

def test_read_csv_headtail_with_extra_head_rows(sample_csv, expected_df):
    df_headtail = rct.read_csv_headtail(
        sample_csv, header=True, n_rows_head=10, n_rows_tail=2
    )
    expected_head = expected_df
    expected_tail = expected_df.iloc[-2:]
    expected_combined = pd.concat([expected_head, expected_tail]).drop_duplicates()
    pd.testing.assert_frame_equal(df_headtail.reset_index(drop=True), expected_combined.reset_index(drop=True))

def test_read_csv_headtail_with_extra_tail_rows(sample_csv, expected_df):
    df_headtail = rct.read_csv_headtail(
        sample_csv, header=True, n_rows_head=2, n_rows_tail=10
    )
    expected_head = expected_df.iloc[:2]
    expected_tail = expected_df
    expected_combined = pd.concat([expected_head, expected_tail]).drop_duplicates()
    pd.testing.assert_frame_equal(df_headtail.reset_index(drop=True), expected_combined.reset_index(drop=True))

def test_read_csv_line_with_extra_rows_after(sample_csv, expected_df):
    df_line = rct.read_csv_line_range(sample_csv, n=2, header=True, rows_after_n=10)
    expected_lines = expected_df.iloc[1:]
    pd.testing.assert_frame_equal(df_line.reset_index(drop=True), expected_lines.reset_index(drop=True))

def test_read_csv_line_with_exact_row_count(sample_csv, expected_df):
    df_line = rct.read_csv_line_range(sample_csv, n=2, header=True, rows_after_n=2)
    expected_lines = expected_df.iloc[1:4]
    pd.testing.assert_frame_equal(df_line.reset_index(drop=True), expected_lines.reset_index(drop=True))

# Additional test cases covering more scenarios

def test_read_csv_head_no_header(sample_csv):
    df_head = rct.read_csv_head(sample_csv, header=False, n_rows=3)
    expected_df_no_header = pd.read_csv(StringIO(SAMPLE_CSV_DATA), header=None)
    expected_head = expected_df_no_header.iloc[:3]
    pd.testing.assert_frame_equal(df_head, expected_head)

def test_read_csv_tail_no_header(sample_csv):
    df_tail = rct.read_csv_tail(sample_csv, header=False, n_rows=2)
    expected_df_no_header = pd.read_csv(StringIO(SAMPLE_CSV_DATA), header=None)
    expected_tail = expected_df_no_header.iloc[-2:]
    pd.testing.assert_frame_equal(df_tail.reset_index(drop=True), expected_tail.reset_index(drop=True))

def test_read_csv_head_skip_rows(sample_csv, expected_df):
    df_head = rct.read_csv_head(sample_csv, header=True, skip_n_first_rows=2, n_rows=2)
    expected_head = expected_df.iloc[2:4]
    pd.testing.assert_frame_equal(df_head.reset_index(drop=True), expected_head.reset_index(drop=True))

def test_read_csv_tail_skip_rows(sample_csv, expected_df):
    df_tail = rct.read_csv_tail(sample_csv, header=True, skip_n_first_rows=1, n_rows=2)
    expected_tail = expected_df.iloc[1:-1]
    pd.testing.assert_frame_equal(df_tail.reset_index(drop=True), expected_tail.reset_index(drop=True))

def test_read_csv_line_range_skip_rows(sample_csv, expected_df):
    df_line = rct.read_csv_line_range(sample_csv, n=1, header=True, skip_n_first_rows=1, rows_after_n=2)
    expected_lines = expected_df.iloc[2:5]
    pd.testing.assert_frame_equal(df_line.reset_index(drop=True), expected_lines.reset_index(drop=True))

def test_read_csv_headtail_no_header(sample_csv):
    df_headtail = rct.read_csv_headtail(
        sample_csv, header=False, n_rows_head=2, n_rows_tail=2
    )
    expected_df_no_header = pd.read_csv(StringIO(SAMPLE_CSV_DATA), header=None)
    expected_head = expected_df_no_header.iloc[:2]
    expected_tail = expected_df_no_header.iloc[-2:]
    expected_combined = pd.concat([expected_head, expected_tail]).drop_duplicates()
    pd.testing.assert_frame_equal(df_headtail.reset_index(drop=True), expected_combined.reset_index(drop=True))

def test_read_csv_line_range_no_header(sample_csv):
    df_line = rct.read_csv_line_range(sample_csv, n=2, header=False, rows_after_n=2)
    expected_df_no_header = pd.read_csv(StringIO(SAMPLE_CSV_DATA), header=None)
    expected_lines = expected_df_no_header.iloc[1:4]
    pd.testing.assert_frame_equal(df_line.reset_index(drop=True), expected_lines.reset_index(drop=True))

def test_read_csv_line_range_invalid_n(sample_csv):
    with pytest.raises(ValueError):
        rct.read_csv_line_range(sample_csv, n=10, header=True)

def test_read_csv_line_range_negative_n(sample_csv):
    with pytest.raises(ValueError):
        rct.read_csv_line_range(sample_csv, n=0, header=True)

def test_read_csv_line_range_negative_rows_after(sample_csv, expected_df):
    df_line = rct.read_csv_line_range(sample_csv, n=2, rows_after_n=-1, header=True)
    expected_lines = expected_df.iloc[1:2]
    pd.testing.assert_frame_equal(df_line.reset_index(drop=True), expected_lines.reset_index(drop=True))

def test_read_csv_head_empty_file():
    # Create an empty CSV file
    empty_csv_path = 'empty.csv'
    with open(empty_csv_path, 'w') as f:
        f.write('')
    df_head = rct.read_csv_head(empty_csv_path, header=False, n_rows=10)
    assert df_head.empty, "Expected empty DataFrame"
    os.remove(empty_csv_path)

def test_read_csv_tail_empty_file():
    # Create an empty CSV file
    empty_csv_path = 'empty.csv'
    with open(empty_csv_path, 'w') as f:
        f.write('')
    df_tail = rct.read_csv_tail(empty_csv_path, header=False, n_rows=10)
    assert df_tail.empty, "Expected empty DataFrame"
    os.remove(empty_csv_path)

def test_read_csv_nonexistent_file():
    with pytest.raises(FileNotFoundError):
        rct.read_csv_head('nonexistent.csv', header=True, n_rows=5)

def test_read_csv_head_large_n_rows(sample_csv, expected_df):
    # Request more rows than are available
    df_head = rct.read_csv_head(sample_csv, header=True, n_rows=100)
    pd.testing.assert_frame_equal(df_head, expected_df)

def test_read_csv_tail_large_n_rows(sample_csv, expected_df):
    # Request more rows than are available
    df_tail = rct.read_csv_tail(sample_csv, header=True, n_rows=100)
    pd.testing.assert_frame_equal(df_tail, expected_df)

def test_read_csv_headtail_large_n_rows(sample_csv, expected_df):
    # Request more rows than are available for both head and tail
    df_headtail = rct.read_csv_headtail(
        sample_csv, header=True, n_rows_head=100, n_rows_tail=100
    )
    expected_combined = pd.concat([expected_df, expected_df]).drop_duplicates()
    pd.testing.assert_frame_equal(df_headtail.reset_index(drop=True), expected_combined.reset_index(drop=True))

def test_read_csv_head_with_different_sep():
    # Create a sample CSV file with semicolon as separator
    sample_data = """col1;col2;col3
1;2;3
4;5;6
"""
    sample_path = 'sample_semicolon.csv'
    with open(sample_path, 'w') as f:
        f.write(sample_data)
    df_head = rct.read_csv_head(sample_path, header=True, n_rows=2, sep=';')
    expected_df = pd.read_csv(StringIO(sample_data), sep=';')
    pd.testing.assert_frame_equal(df_head, expected_df)
    os.remove(sample_path)

def test_read_csv_special_characters():
    # Create a sample CSV file with special characters
    sample_data = """col1,col2,col3
"1,2","3,4","5,6"
"7,8","9,10","11,12"
"""
    sample_path = 'sample_special.csv'
    with open(sample_path, 'w') as f:
        f.write(sample_data)
    df_head = rct.read_csv_head(sample_path, header=True, n_rows=2)
    expected_df = pd.read_csv(StringIO(sample_data))
    pd.testing.assert_frame_equal(df_head, expected_df)
    os.remove(sample_path)
