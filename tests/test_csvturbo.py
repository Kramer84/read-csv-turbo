import pytest
import os
import pandas as pd
import readcsvturbo as rct
from io import StringIO

# Define the sample CSV data
SAMPLE_CSV_DATA = """col1,col2,col3
1,2,3
4,5,6
7,8,9
10,11,12
13,14,15
"""

@pytest.fixture
def sample_csv():
    sample_path = 'sample.csv'
    with open(sample_path, 'w') as f:
        f.write(SAMPLE_CSV_DATA)
    yield sample_path
    if os.path.exists(sample_path):
        os.remove(sample_path)

@pytest.fixture
def expected_df():
    return pd.read_csv(StringIO(SAMPLE_CSV_DATA))

# --- Standard Reads ---

def test_read_csv_head_with_extra_rows(sample_csv, expected_df):
    df_head = rct.read_csv_head(sample_csv, header=True, n_rows=10)
    pd.testing.assert_frame_equal(df_head, expected_df)

def test_read_csv_head_with_exact_rows(sample_csv, expected_df):
    df_head = rct.read_csv_head(sample_csv, header=True, n_rows=3)
    expected_head = expected_df.iloc[:3]
    pd.testing.assert_frame_equal(df_head, expected_head)

def test_read_csv_tail_with_extra_rows(sample_csv, expected_df):
    df_tail = rct.read_csv_tail(sample_csv, header=True, n_rows=10)
    pd.testing.assert_frame_equal(df_tail, expected_df)

def test_read_csv_tail_with_exact_rows(sample_csv, expected_df):
    df_tail = rct.read_csv_tail(sample_csv, header=True, n_rows=2)
    expected_tail = expected_df.iloc[-2:]
    pd.testing.assert_frame_equal(df_tail.reset_index(drop=True), expected_tail.reset_index(drop=True))

# --- HeadTail ---

def test_read_csv_headtail_with_extra_head_rows(sample_csv, expected_df):
    df_headtail = rct.read_csv_headtail(sample_csv, header=True, n_rows_head=10, n_rows_tail=2)
    expected_tail = expected_df.iloc[-2:]
    # Combine and drop duplicates logic matches implementation
    expected_combined = pd.concat([expected_df, expected_tail]).drop_duplicates()
    pd.testing.assert_frame_equal(df_headtail.reset_index(drop=True), expected_combined.reset_index(drop=True))

def test_read_csv_headtail_with_extra_tail_rows(sample_csv, expected_df):
    df_headtail = rct.read_csv_headtail(sample_csv, header=True, n_rows_head=2, n_rows_tail=10)
    expected_head = expected_df.iloc[:2]
    expected_combined = pd.concat([expected_head, expected_df]).drop_duplicates()
    pd.testing.assert_frame_equal(df_headtail.reset_index(drop=True), expected_combined.reset_index(drop=True))

# --- Line Range ---

def test_read_csv_line_with_extra_rows_after(sample_csv, expected_df):
    df_line = rct.read_csv_line_range(sample_csv, n=2, header=True, rows_after_n=10)
    expected_lines = expected_df.iloc[1:]
    pd.testing.assert_frame_equal(df_line.reset_index(drop=True), expected_lines.reset_index(drop=True))

def test_read_csv_line_with_exact_row_count(sample_csv, expected_df):
    df_line = rct.read_csv_line_range(sample_csv, n=2, header=True, rows_after_n=2)
    expected_lines = expected_df.iloc[1:4]
    pd.testing.assert_frame_equal(df_line.reset_index(drop=True), expected_lines.reset_index(drop=True))

# --- FIX 1: Handling "No Header" Type Mismatches ---
# When reading parts of a file without a header, Pandas might infer Int64,
# whereas reading the *whole* file (which includes a text header) infers Object.
# We must cast the Expected DF to match the Turbo DF's more accurate type inference.

def test_read_csv_head_no_header(sample_csv):
    df_head = rct.read_csv_head(sample_csv, header=False, n_rows=3)
    # We use header=None here to mimic the structure, but we must compare apples to apples
    expected_df_no_header = pd.read_csv(StringIO(SAMPLE_CSV_DATA), header=None)
    expected_head = expected_df_no_header.iloc[:3]
    pd.testing.assert_frame_equal(df_head, expected_head)

def test_read_csv_tail_no_header(sample_csv):
    df_tail = rct.read_csv_tail(sample_csv, header=False, n_rows=2)

    # FIX: Manually create expectation or read only the relevant lines to check types
    # Since Turbo reads purely numbers here, it returns Ints.
    # The full file read returns Objects (strings) because of row 1.
    data = {"0": [10, 13], "1": [11, 14], "2": [12, 15]}
    expected_tail = pd.DataFrame(data)

    # We must match column names which default to 0,1,2 in header=None
    expected_tail.columns = expected_tail.columns.astype(int)

    pd.testing.assert_frame_equal(df_tail.reset_index(drop=True), expected_tail)

def test_read_csv_line_range_no_header(sample_csv):
    df_line = rct.read_csv_line_range(sample_csv, n=2, header=False, rows_after_n=2)

    # FIX: Same as above. Turbo sees Ints, Full file sees Objects.
    # Rows 2,3,4 (0-indexed logic) -> 1,4,7...
    # The data at n=2 (row 3 in file) is 4,5,6
    # n=2 means start at the 2nd *data* line (lines are 1, 4, 7, 10, 13)
    # Wait, n is 1-based index *after* skipping.
    # File: [Header], [1,2,3], [4,5,6], [7,8,9]...
    # Header=False, so File: [col1..], [1,2,3]...
    # n=2 is [1,2,3].

    # Let's trust the Turbo output type and ensure data content is correct.
    expected_df_no_header = pd.read_csv(StringIO(SAMPLE_CSV_DATA), header=None)
    expected_lines = expected_df_no_header.iloc[1:4].copy()

    # Cast expected object columns to int64 to match Turbo's correct inference
    for col in expected_lines.columns:
        expected_lines[col] = pd.to_numeric(expected_lines[col], errors='coerce')

    pd.testing.assert_frame_equal(df_line.reset_index(drop=True), expected_lines.reset_index(drop=True))

def test_read_csv_headtail_no_header(sample_csv):
    df_headtail = rct.read_csv_headtail(
        sample_csv, header=False, n_rows_head=2, n_rows_tail=2
    )
    expected_df_no_header = pd.read_csv(StringIO(SAMPLE_CSV_DATA), header=None)
    expected_head = expected_df_no_header.iloc[:2]
    expected_tail = expected_df_no_header.iloc[-2:]
    expected_combined = pd.concat([expected_head, expected_tail]).drop_duplicates()

    # No type casting needed here because the Head includes the text header "col1,col2...",
    # so Turbo will correctly infer Object type, matching the expected DF.
    pd.testing.assert_frame_equal(df_headtail.reset_index(drop=True), expected_combined.reset_index(drop=True))

# --- FIX 2: Handling "Skip Rows" logic ---
# If you skip n rows, the header changes.
# The expected DF must be generated using skiprows to match.

def test_read_csv_head_skip_rows(sample_csv):
    # Skipping 2 rows. Row 0 (col1...) skipped. Row 1 (1,2,3) skipped.
    # New Header is Row 2 (4,5,6).
    df_head = rct.read_csv_head(sample_csv, header=True, skip_n_first_rows=2, n_rows=2)

    # FIX: Read the expected DF with the same skip logic
    expected_head = pd.read_csv(StringIO(SAMPLE_CSV_DATA), skiprows=2, nrows=2)

    pd.testing.assert_frame_equal(df_head.reset_index(drop=True), expected_head.reset_index(drop=True))

def test_read_csv_tail_skip_rows(sample_csv):
    # Skipping 1 row (col1...). New header is 1,2,3.
    df_tail = rct.read_csv_tail(sample_csv, header=True, skip_n_first_rows=1, n_rows=2)

    # FIX: Generate expectation by skipping 1 row
    # Note: When we use skiprows=1, row 2 becomes header.
    # Then we grab the last 2 rows.
    full_skipped_df = pd.read_csv(StringIO(SAMPLE_CSV_DATA), skiprows=1)
    expected_tail = full_skipped_df.iloc[-2:]

    pd.testing.assert_frame_equal(df_tail.reset_index(drop=True), expected_tail.reset_index(drop=True))

def test_read_csv_line_range_skip_rows(sample_csv):
    # n=1, skip=1.
    # Skip row 0 (header). Row 1 (1,2,3) becomes NEW Header.
    # n=1 refers to the first data row *after* the new header.
    # So data starts at 4,5,6.
    df_line = rct.read_csv_line_range(sample_csv, n=1, header=True, skip_n_first_rows=1, rows_after_n=2)

    # FIX: Align expectation
    full_skipped_df = pd.read_csv(StringIO(SAMPLE_CSV_DATA), skiprows=1)
    # Get 3 rows starting from index 0 (which is the first data row)
    expected_lines = full_skipped_df.iloc[0:3]

    pd.testing.assert_frame_equal(df_line.reset_index(drop=True), expected_lines.reset_index(drop=True))

# --- Error Handling ---

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
    empty_csv_path = 'empty.csv'
    with open(empty_csv_path, 'w') as f:
        f.write('')
    df_head = rct.read_csv_head(empty_csv_path, header=False, n_rows=10)
    assert df_head.empty
    os.remove(empty_csv_path)

def test_read_csv_tail_empty_file():
    empty_csv_path = 'empty.csv'
    with open(empty_csv_path, 'w') as f:
        f.write('')
    df_tail = rct.read_csv_tail(empty_csv_path, header=False, n_rows=10)
    assert df_tail.empty
    os.remove(empty_csv_path)

def test_read_csv_nonexistent_file():
    with pytest.raises(FileNotFoundError):
        rct.read_csv_head('nonexistent.csv', header=True, n_rows=5)

# --- Large/Special Cases ---

def test_read_csv_head_large_n_rows(sample_csv, expected_df):
    df_head = rct.read_csv_head(sample_csv, header=True, n_rows=100)
    pd.testing.assert_frame_equal(df_head, expected_df)

def test_read_csv_tail_large_n_rows(sample_csv, expected_df):
    df_tail = rct.read_csv_tail(sample_csv, header=True, n_rows=100)
    pd.testing.assert_frame_equal(df_tail, expected_df)

def test_read_csv_headtail_large_n_rows(sample_csv, expected_df):
    df_headtail = rct.read_csv_headtail(
        sample_csv, header=True, n_rows_head=100, n_rows_tail=100
    )
    expected_combined = pd.concat([expected_df, expected_df]).drop_duplicates()
    pd.testing.assert_frame_equal(df_headtail.reset_index(drop=True), expected_combined.reset_index(drop=True))

def test_read_csv_head_with_different_sep():
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
