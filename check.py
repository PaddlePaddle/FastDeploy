import xml.etree.ElementTree as ET
import argparse
import os

# 预设的覆盖率 XML 文件路径（可以在这里修改）
XML_FILE = "python_coverage_all (26).xml"
# 预设的输出文件名
OUTPUT_FILE = "result.txt"
# 预设的要分析的文件名（可以在这里修改）
TARGET_FILE = "engine/common_engine.py"
# api_server.py
#"common_engine.py"
# spec_decode/mtp.py
#mtp.py
# openai/api_server.py

def analyze_coverage(xml_path, target_file=None):
    # 用于收集所有输出的缓冲区
    output_lines = []
    
    def output(text):
        """同时输出到控制台和缓冲区"""
        print(text)
        output_lines.append(text)
    
    if not os.path.exists(xml_path):
        error_msg = f"错误: 找不到文件 '{xml_path}'"
        output(error_msg)
        _write_to_file(OUTPUT_FILE, output_lines)
        return

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError:
        error_msg = "错误: 无法解析 XML 文件，请确认格式正确。"
        output(error_msg)
        _write_to_file(OUTPUT_FILE, output_lines)
        return

    # 查找所有的 class 节点 (对应具体的代码文件)
    # 使用 .//class 可以查找任意深度的 class 标签
    classes = root.findall(".//class")
    
    found_target = False

    output(f"{'-'*60}")
    output(f"正在分析覆盖率报告: {xml_path}")
    output(f"{'-'*60}")

    for cls in classes:
        filename = cls.get('filename')
        
        # 如果指定了文件名进行过滤，则跳过不匹配的文件
        if target_file and (target_file not in filename):
            continue
            
        found_target = True
        output(f"[文件] {filename}")
        
        # 获取该文件下的所有行
        lines = cls.findall("lines/line")
        
        missed_lines = []
        partial_lines = []

        for line in lines:
            line_num = line.get('number')
            hits = int(line.get('hits', 0))
            
            # 检查完全未覆盖 (hits="0")
            if hits == 0:
                missed_lines.append(line_num)
            
            # 检查分支覆盖不全 (虽然 hits>0，但是 branch="true" 且 condition-coverage < 100%)
            elif line.get('branch') == 'true':
                coverage_str = line.get('condition-coverage', '0%')
                try:
                    # 提取百分比数字，例如 "50% (1/2)" -> 50
                    if '%' in coverage_str:
                        percent = int(coverage_str.split('%')[0])
                    else:
                        # 如果没有 % 符号，尝试直接转换为数字
                        percent = int(coverage_str) if coverage_str.isdigit() else 0
                    if percent < 100:
                        missing_branches = line.get('missing-branches', '未知')
                        partial_lines.append(f"行 {line_num} (覆盖率 {coverage_str}, 缺失分支: {missing_branches})")
                except (ValueError, IndexError):
                    # 如果解析失败，跳过这行
                    pass

        # --- 输出结果 ---
        if not missed_lines and not partial_lines:
            output("   [OK] 全覆盖！(没有发现未覆盖的行)")
        else:
            if missed_lines:
                output(f"   [X] 完全未覆盖行 (hits=0): {', '.join(missed_lines)}")
            
            if partial_lines:
                output(f"   [!] 分支覆盖不全:")
                for pl in partial_lines:
                    output(f"      - {pl}")
        
        output("-" * 60)

    if target_file and not found_target:
        output(f"[警告] 未在报告中找到包含 '{target_file}' 的文件。")
    
    # 写入预设的输出文件（会覆盖已存在的文件）
    _write_to_file(OUTPUT_FILE, output_lines)

def _write_to_file(output_file, output_lines):
    """将输出写入文件，使用UTF-8编码"""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(output_lines))
        print(f"\n结果已保存到: {output_file}")
    except Exception as e:
        print(f"\n错误: 无法写入文件 '{output_file}': {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="解析 XML 覆盖率报告，找出未覆盖的行。")
    parser.add_argument("xml_file", nargs='?', default=XML_FILE, 
                        help=f"覆盖率 XML 文件的路径 (默认: {XML_FILE})")
    parser.add_argument("--filter", "-f", help=f"可选：只查看特定文件名 (默认: {TARGET_FILE})", default=TARGET_FILE)

    args = parser.parse_args()
    
    # 使用预设的目标文件名（如果命令行指定了 --filter 则使用命令行参数）
    analyze_coverage(args.xml_file, args.filter)