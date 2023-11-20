# get diff rate bewteen two files
def get_diff(file1, file2):
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        lines1 = f1.read().splitlines()
        lines2 = f2.read().splitlines()
    assert len(lines1) == len(lines2)
    total_lines = len(lines1)
    diff_lines = 0

    for i in range(total_lines):
        if lines1[i] != lines2[i]:
            diff_lines = diff_lines + 1

    diff_rate = f"{diff_lines}/{total_lines}"
    if diff_lines == 0:
        is_diff = 1
    else:
        is_diff = 0
    return is_diff, diff_rate
