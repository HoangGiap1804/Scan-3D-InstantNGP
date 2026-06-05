"""
export_diagrams.py
==================
Trích xuất tất cả sơ đồ Mermaid từ file báo cáo và render thành ảnh PNG.

Cách dùng:
    python export_diagrams.py

Yêu cầu:
    npm install -g @mermaid-js/mermaid-cli   (chạy lần đầu)
"""

import os
import re
import subprocess
import sys

# ── Cấu hình ──────────────────────────────────────────────────────────────────

REPORT_MD = os.path.expanduser(
    "~/.gemini/antigravity-ide/brain/"
    "5e82575f-a8a8-4010-8725-01234decfd41/usecase_diagrams.md"
)
OUTPUT_DIR = os.path.join(os.path.dirname(REPORT_MD), "diagrams")

# Tên tương ứng cho từng sơ đồ (theo thứ tự xuất hiện trong file)
DIAGRAM_NAMES = [
    "01_actors",
    "02_usecase_overview",
    "03_system_architecture",
    "04_data_flow",
    "05_sequence_training",
    "06_sequence_blender_live_render",
    "07_sequence_video_processing",
    "08_component_diagram",
    "09_deployment_diagram",
    "10_state_machine_blender",
    "11_class_diagram",
]

# ── Mermaid CLI config ────────────────────────────────────────────────────────

MMDC_CONFIG = """{
  "theme": "default",
  "themeVariables": {
    "fontSize": "16px"
  }
}"""


def ensure_mmdc():
    """Kiểm tra mmdc có sẵn không, nếu không thì cài."""
    result = subprocess.run(
        ["npx", "--yes", "mmdc", "--version"],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        print(f"[OK] mmdc version: {result.stdout.strip()}")
        return True
    print("[ERR] Không thể chạy mmdc.")
    return False


def extract_mermaid_blocks(md_path):
    """Trích xuất tất cả block ```mermaid ... ``` từ file markdown."""
    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()
    pattern = r"```mermaid\n(.*?)```"
    blocks = re.findall(pattern, content, re.DOTALL)
    return blocks


def render_diagram(mmd_content, output_png, config_path):
    """Render một sơ đồ Mermaid thành ảnh PNG."""
    # Lưu tạm file .mmd
    mmd_path = output_png.replace(".png", ".mmd")
    with open(mmd_path, "w", encoding="utf-8") as f:
        f.write(mmd_content)

    cmd = [
        "npx", "--yes", "mmdc",
        "-i", mmd_path,
        "-o", output_png,
        "-c", config_path,
        "-b", "white",
        "--width", "1400",
        "--height", "900",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"  [OK]  {os.path.basename(output_png)}")
    else:
        print(f"  [ERR] {os.path.basename(output_png)}")
        print(f"        {result.stderr.strip()[:200]}")
    return result.returncode == 0


def main():
    # Tạo thư mục đầu ra
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Lưu file config Mermaid
    config_path = os.path.join(OUTPUT_DIR, "mermaid_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(MMDC_CONFIG)

    print(f"\n=== Trích xuất sơ đồ từ: {REPORT_MD}")
    print(f"=== Lưu ảnh vào:         {OUTPUT_DIR}\n")

    # Kiểm tra mmdc
    if not ensure_mmdc():
        print("\n[!] Hãy cài mmdc trước:")
        print("    npm install -g @mermaid-js/mermaid-cli")
        sys.exit(1)

    # Trích xuất blocks
    blocks = extract_mermaid_blocks(REPORT_MD)
    print(f"\n[INFO] Tìm thấy {len(blocks)} sơ đồ Mermaid\n")

    # Render từng block
    success = 0
    for i, block in enumerate(blocks):
        # Lấy tên (dùng tên mặc định nếu không đủ)
        if i < len(DIAGRAM_NAMES):
            name = DIAGRAM_NAMES[i]
        else:
            name = f"diagram_{i+1:02d}"

        output_png = os.path.join(OUTPUT_DIR, f"{name}.png")
        ok = render_diagram(block, output_png, config_path)
        if ok:
            success += 1

    print(f"\n=== Hoàn thành: {success}/{len(blocks)} sơ đồ được xuất thành công")
    print(f"=== Thư mục ảnh: {OUTPUT_DIR}\n")


if __name__ == "__main__":
    main()
