import re
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict


def get_changed_lines_from_file(diff_txt_path):
    """Parse diff.txt to get changed lines per file"""
    file_changes = defaultdict(set)
    current_file = None

    with open(diff_txt_path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("+++ b/"):
                current_file = line[6:].strip()
            elif line.startswith("@@"):
                match = re.search(r"\+(\d+)(?:,(\d+))?", line)
                if match and current_file:
                    start_line = int(match.group(1))
                    line_count = int(match.group(2) or "1")
                    for i in range(start_line, start_line + line_count):
                        file_changes[current_file].add(i)
    return file_changes


def generate_diff_coverage(original_xml, diff_lines, output_xml):
    """Generate a new coverage.xml containing only changed lines"""
    tree = ET.parse(original_xml)
    root = tree.getroot()

    for package in root.findall(".//packages/package"):
        classes = package.find("classes")
        new_classes = ET.Element("classes")

        for cls in classes.findall("class"):
            filename = cls.attrib["filename"]
            if filename not in diff_lines:
                continue

            lines = cls.find("lines")
            new_lines = ET.Element("lines")

            for line in lines.findall("line"):
                line_num = int(line.attrib["number"])
                if line_num in diff_lines[filename]:
                    new_lines.append(line)

            if len(new_lines) > 0:
                new_cls = ET.Element("class", cls.attrib)
                new_cls.append(new_lines)
                new_classes.append(new_cls)

        package.remove(classes)
        package.append(new_classes)

    ET.indent(tree, space="  ")
    tree.write(output_xml, encoding="utf-8", xml_declaration=True)
    print(f"Generated diff coverage file: {output_xml}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python generate_diff_coverage_xml.py diff.txt coverage.xml")
        sys.exit(1)

    diff_path = sys.argv[1]
    coverage_path = sys.argv[2]
    output_path = "diff_coverage.xml"

    diff_lines = get_changed_lines_from_file(diff_path)
    generate_diff_coverage(coverage_path, diff_lines, output_path)
