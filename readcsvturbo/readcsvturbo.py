import os
import pandas as pd
import platform
import subprocess
import shlex
from io import StringIO
import concurrent.futures

__all__ = [
    "read_csv_head",
    "read_csv_tail",
    "read_csv_headtail",
    "read_csv_line_range",
]

def check_file_exists(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"The file '{path}' does not exist.")

def get_total_lines(path):
    path_quoted = shlex.quote(str(path))
    if platform.system().lower().startswith('win'):
        cmd = [
            'powershell',
            '-Command',
            f"(Get-Content -Path {path_quoted} | Measure-Object -Line).Lines"
        ]
    else:
        cmd = ['wc', '-l', '--', path]
    output = subprocess.check_output(cmd).decode().strip()
    total_lines = output.strip().split()[0]
    return int(total_lines)

def csv_header(path, skip_n_first_rows=0):
    path_quoted = shlex.quote(str(path))
    skip_lines = skip_n_first_rows

    if platform.system().lower().startswith("win"):
        cmd = [
            'powershell',
            '-Command',
            f"Get-Content -Path '{path}' | Select-Object -Skip {skip_lines} -First 1"
        ]
    else:
        start_line = 1 + skip_lines
        cmd = ['sed', '-n', f'{start_line}p', path]
    output = subprocess.check_output(cmd).decode('utf-8').strip()
    return output

def csv_head(path, total_lines, header=True, skip_n_first_rows=0, n_rows=1):
    skip_lines = skip_n_first_rows + (1 if header else 0)
    n_rows = min(n_rows, total_lines - skip_lines)
    if n_rows <= 0:
        return ''
    path_quoted = shlex.quote(str(path))

    if platform.system().lower().startswith('win'):
        cmd = [
            'powershell',
            '-Command',
            f"Get-Content -Path {path_quoted} | Select-Object -Skip {skip_lines} -First {n_rows}"
        ]
    else:
        start_line = skip_lines + 1
        end_line = start_line + n_rows - 1
        cmd = ['sed', '-n', f'{start_line},{end_line}p', path]
    output = subprocess.check_output(cmd).decode('utf-8').strip()
    return output

def csv_tail(path, total_lines, header=True, skip_n_first_rows=0, n_rows=1):
    skip_lines = skip_n_first_rows + (1 if header else 0)
    n_available_lines = total_lines - skip_lines
    n_rows = min(n_rows, n_available_lines)
    if n_rows <= 0:
        return ''
    path_quoted = shlex.quote(str(path))

    if platform.system().lower().startswith('win'):
        # On Windows
        cmd = [
            'powershell',
            '-Command',
            f"Get-Content -Path {path_quoted} | Select-Object -Skip {skip_lines}"
        ]
        output = subprocess.check_output(cmd).decode('utf-8').splitlines()
        data_lines = output[-n_rows:]  # Get the last n_rows from the remaining lines
        output = '\n'.join(data_lines)
    else:
        # On Unix-like systems
        # Skip the first 'skip_lines' lines, then get the last 'n_rows' lines
        cmd = ['tail', '-n', f'+{skip_lines + 1}', path]
        tail_proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        output = subprocess.check_output(['tail', '-n', f'{n_rows}'], stdin=tail_proc.stdout).decode('utf-8').strip()
        tail_proc.stdout.close()
        tail_proc.wait()
    return output

def csv_line_range(path, total_lines, n, rows_after_n=0, header=True, skip_n_first_rows=0):
    skip_lines = (1 if header else 0) + skip_n_first_rows
    available_lines = total_lines - skip_lines

    if n < 1 or n > available_lines:
        raise ValueError("Requested starting line exceeds the available number of data lines in the file.")

    rows_after_n = min(rows_after_n, available_lines - n)
    num_lines = rows_after_n + 1  # Total number of lines to retrieve

    path_quoted = shlex.quote(str(path))

    # Map the data line numbers to file line numbers
    start_file_line = skip_lines + n
    end_file_line = start_file_line + rows_after_n

    if platform.system().lower().startswith('win'):
        # Windows implementation
        skip_lines_cmd = start_file_line - 1
        cmd = [
            'powershell',
            '-Command',
            f"Get-Content -Path {path_quoted} | Select-Object -Skip {skip_lines_cmd} -First {num_lines}"
        ]
    else:
        # Unix-like systems implementation
        cmd = ['sed', '-n', f'{start_file_line},{end_file_line}p', path]
    output = subprocess.check_output(cmd).decode('utf-8').strip()
    return output

def parse_csv_content(header_str, data_str, header=True, **kwargs):
    sep = kwargs.pop('sep', ',')
    # Strip whitespace to accurately check for emptiness
    header_str = header_str.strip() if header_str else ''
    data_str = data_str.strip() if data_str else ''

    if header:
        if not header_str:
            # No header line found
            if not data_str:
                # No header and no data
                return pd.DataFrame()
            else:
                # No header but data present
                string_data = StringIO(data_str)
                return pd.read_csv(string_data, sep=sep, header=None, **kwargs)
        else:
            if not data_str:
                # Header present but no data
                string_data = StringIO(header_str)
                return pd.read_csv(string_data, sep=sep, header=0, **kwargs)
            else:
                # Both header and data present
                string_data = StringIO(f'{header_str}\n{data_str}')
                return pd.read_csv(string_data, sep=sep, header=0, **kwargs)
    else:
        if not data_str:
            # No data and no header
            return pd.DataFrame()
        else:
            string_data = StringIO(data_str)
            return pd.read_csv(string_data, sep=sep, header=None, **kwargs)

def read_csv_head(path, header=True, skip_n_first_rows=0, n_rows=1, **kwargs):
    """
    Read the first `n_rows` of a CSV file into a pandas DataFrame.

    Parameters
    ----------
    path : str
        The file path to the CSV file.
    header : bool, optional
        Whether the CSV file contains a header row. Default is True.
    skip_n_first_rows : int, optional
        Number of initial data rows to skip before reading. Does not count the header row if `header` is True. Default is 0.
    n_rows : int, optional
        Number of data rows to read after skipping. Default is 1.
    **kwargs
        Additional keyword arguments passed to `pandas.read_csv`.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the requested rows from the CSV file.

    Notes
    -----
    - If the CSV file has fewer rows than requested, all available data rows are returned.
    - The function efficiently reads only the necessary lines, making it suitable for large files.
    - The `header` parameter controls whether the header is read and used as column names.
    - Use `sep` in `**kwargs` to specify a different delimiter if the CSV uses one.
    """
    check_file_exists(path)
    total_lines = get_total_lines(path)
    data_str = csv_head(path, total_lines, header, skip_n_first_rows, n_rows)
    header_str = csv_header(path, skip_n_first_rows) if header else ''
    return parse_csv_content(header_str, data_str, header=header, **kwargs)

def read_csv_tail(path, header=True, skip_n_first_rows=0, n_rows=1, **kwargs):
    """
    Read the last `n_rows` of a CSV file into a pandas DataFrame.

    Parameters
    ----------
    path : str
        The file path to the CSV file.
    header : bool, optional
        Whether the CSV file contains a header row. Default is True.
    skip_n_first_rows : int, optional
        Number of initial data rows to skip before reading. Does not count the header row if `header` is True. Default is 0.
    n_rows : int, optional
        Number of data rows to read from the end of the file. Default is 1.
    **kwargs
        Additional keyword arguments passed to `pandas.read_csv`.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the requested rows from the end of the CSV file.

    Notes
    -----
    - If the CSV file has fewer rows than requested, all available data rows are returned.
    - The function efficiently reads only the necessary lines from the end of the file.
    - The `header` parameter controls whether the header is read and used as column names.
    - Use `sep` in `**kwargs` to specify a different delimiter if the CSV uses one.
    """
    check_file_exists(path)
    total_lines = get_total_lines(path)
    data_str = csv_tail(path, total_lines, header, skip_n_first_rows, n_rows=n_rows)
    header_str = csv_header(path, skip_n_first_rows) if header else ''
    return parse_csv_content(header_str, data_str, header=header, **kwargs)

def read_csv_headtail(path, header=True, skip_n_first_rows=0, n_rows_head=1, n_rows_tail=1, **kwargs):
    """
    Read both the first `n_rows_head` and the last `n_rows_tail` of a CSV file into a pandas DataFrame.

    Parameters
    ----------
    path : str
        The file path to the CSV file.
    header : bool, optional
        Whether the CSV file contains a header row. Default is True.
    skip_n_first_rows : int, optional
        Number of initial data rows to skip before reading the head and tail. Does not count the header row if `header` is True. Default is 0.
    n_rows_head : int, optional
        Number of data rows to read from the start of the file after skipping. Default is 1.
    n_rows_tail : int, optional
        Number of data rows to read from the end of the file. Default is 1.
    **kwargs
        Additional keyword arguments passed to `pandas.read_csv`.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the combined head and tail rows from the CSV file, without duplicates.

    Notes
    -----
    - If the total available data rows are fewer than `n_rows_head + n_rows_tail`, overlapping rows are included only once.
    - The function efficiently reads only the necessary lines, making it suitable for large files.
    - The `header` parameter controls whether the header is read and used as column names.
    - Use `sep` in `**kwargs` to specify a different delimiter if the CSV uses one.

    Example
    -------
    >>> df = read_csv_headtail('data.csv', n_rows_head=2, n_rows_tail=2)
    >>> print(df)
    """
    check_file_exists(path)
    total_lines = get_total_lines(path)
    skip_lines = skip_n_first_rows + (1 if header else 0)
    available_lines = total_lines - skip_lines

    if available_lines <= 0:
        # No data available
        header_str = csv_header(path, skip_n_first_rows) if header else ''
        return parse_csv_content(header_str, '', header=header, **kwargs)

    # Adjust n_rows_head and n_rows_tail if they exceed available lines
    n_rows_head = min(n_rows_head, available_lines)

    n_rows_tail = min(n_rows_tail, available_lines)

    # Calculate overlap
    overlap = n_rows_head + n_rows_tail - available_lines
    if overlap > 0:
        # Reduce n_rows_tail to avoid overlap
        n_rows_tail -= overlap

    # Concurrently retrieve header, head, and tail data
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_header = executor.submit(csv_header, path, skip_n_first_rows) if header else None
        future_head = executor.submit(csv_head, path, total_lines, header, skip_n_first_rows, n_rows_head)
        future_tail = executor.submit(csv_tail, path, total_lines, header, skip_n_first_rows, n_rows_tail)

        header_str = future_header.result() if future_header else ''
        head_str = future_head.result()
        tail_str = future_tail.result() if n_rows_tail > 0 else ''

    # Combine head and tail data
    data_str = '\n'.join(filter(None, [head_str.strip(), tail_str.strip()]))

    return parse_csv_content(header_str, data_str, header=header, **kwargs)

def read_csv_line_range(path, n, rows_after_n=0, header=True, skip_n_first_rows=0, **kwargs):
    """
    Read a specific range of lines from a CSV file into a pandas DataFrame.

    Parameters
    ----------
    path : str
        The file path to the CSV file.
    n : int
        The starting line number (1-based index after skipping initial rows).
    rows_after_n : int, optional
        Number of additional data rows to read after the starting line. Default is 0.
    header : bool, optional
        Whether the CSV file contains a header row. Default is True.
    skip_n_first_rows : int, optional
        Number of initial data rows to skip before counting line `n`. Does not count the header row if `header` is True. Default is 0.
    **kwargs
        Additional keyword arguments passed to `pandas.read_csv`.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the specified range of rows from the CSV file.

    Notes
    -----
    - Line numbering starts after skipping the initial rows and the header.
    - If the CSV file has fewer rows than requested, all available data rows starting from line `n` are returned.
    - The function efficiently reads only the necessary lines.
    - The `header` parameter controls whether the header is read and used as column names.
    - Use `sep` in `**kwargs` to specify a different delimiter if the CSV uses one.

    Raises
    ------
    ValueError
        If the starting line `n` is less than 1 or exceeds the number of available data rows.

    Example
    -------
    >>> df = read_csv_line_range('data.csv', n=5, rows_after_n=2)
    >>> print(df)
    """
    check_file_exists(path)
    total_lines = get_total_lines(path)
    data_str = csv_line_range(path, total_lines, n, rows_after_n, header, skip_n_first_rows)
    header_str = csv_header(path, skip_n_first_rows) if header else ''
    return parse_csv_content(header_str, data_str, header=header, **kwargs)
