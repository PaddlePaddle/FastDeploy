import re
import sys

import pandas as pd


def txt_to_excel(txt_file_path, excel_file_path):
    rows = []
    total_row = None

    with open(txt_file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        line = line.rstrip()
        if not line:
            continue

        # Skip headers and separators
        if line.startswith("Name") or set(line.strip()) == set("-"):
            continue

        # Match TOTAL line (e.g., "TOTAL  123  4  10  2  95%  missing_info")
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

        # Match regular file rows
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

    # Sort by coverage ascending (excluding TOTAL)
    if rows:
        df = pd.DataFrame(rows)
        df.sort_values("Cover(%)", inplace=True)
    else:
        df = pd.DataFrame(columns=["File", "Stmts", "Miss", "Branch", "BrPart", "Cover(%)", "Missing"])

    # Append TOTAL at the bottom
    if total_row:
        df_total = pd.DataFrame([total_row])
        df = pd.concat([df, df_total], ignore_index=True)

    # Save as Excel file
    df.to_excel(excel_file_path, index=False, engine="openpyxl")
    print("✅ Saved coverage Excel file: {}".format(excel_file_path))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python generate_all_coverage_xlsx.py <input_txt_path> <output_excel_path>")
        sys.exit(1)

    txt_file_path = sys.argv[1]
    excel_file_path = sys.argv[2]

    txt_to_excel(txt_file_path, excel_file_path)
