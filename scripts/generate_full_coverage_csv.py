import re
import sys

import pandas as pd


def txt_to_csv(txt_file_path, csv_file_path):
    """
    Convert a coverage report in text format to a CSV file.
    Args:
        txt_file_path (str): Path to the input text file containing the coverage report
        csv_file_path (str): Path to the output CSV file where the converted data will be saved
        Returns:
            None
    Raises:
        Exception: If there is an error reading or writing the files
    """
    rows = []
    total_row = None

    # Read all lines from the coverage report
    with open(txt_file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        line = line.rstrip()
        if not line:
            continue

        # Skip table headers and separator lines
        if line.startswith("Name") or set(line.strip()) == set("-"):
            continue

        # Match the TOTAL line (e.g., TOTAL + numbers + percent)
        m_total = re.match(r"^TOTAL\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+([\d.]+)%\s*(.*)$", line)
        if m_total:
            stmts, miss, branch, brpart, cover, missing = m_total.groups()
            total_row = {
                "File": "TOTAL",
                "Stmts": int(stmts),
                "Miss": int(miss),
                "Branch": int(branch),
                "BrPart": int(brpart),
                "Cover(%)": float(cover) if "." in cover else int(cover),
                "Missing": missing.strip(),
            }
            continue

        # Match regular file lines: filename, Stmts, Miss, Branch, BrPart, Cover%, Missing
        # File path may contain non-space characters
        m = re.match(r"^(\S+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+([\d.]+)%\s*(.*)$", line)
        if m:
            filename, stmts, miss, branch, brpart, cover, missing = m.groups()
            rows.append(
                {
                    "File": filename,
                    "Stmts": int(stmts),
                    "Miss": int(miss),
                    "Branch": int(branch),
                    "BrPart": int(brpart),
                    "Cover(%)": float(cover) if "." in cover else int(cover),
                    "Missing": missing.strip() if missing.strip() else "",
                }
            )
        else:
            continue

    # Sort by coverage percentage (ascending), excluding TOTAL — add TOTAL at the end
    if rows:
        df = pd.DataFrame(rows)
        df.sort_values("Cover(%)", inplace=True)
    else:
        df = pd.DataFrame(columns=["File", "Stmts", "Miss", "Branch", "BrPart", "Cover(%)", "Missing"])

    # Append TOTAL row at the end if it exists
    if total_row:
        df_total = pd.DataFrame([total_row])
        df = pd.concat([df, df_total], ignore_index=True)

    # Save the final CSV
    df.to_csv(csv_file_path, index=False, encoding="utf-8")
    print("✅ Saved coverage CSV: {}".format(csv_file_path))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python generate_full_coverage_csv.py <input_txt_path> <output_csv_path>")
        sys.exit(1)

    txt_file_path = sys.argv[1]
    csv_file_path = sys.argv[2]

    txt_to_csv(txt_file_path, csv_file_path)
