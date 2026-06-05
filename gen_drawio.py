"""
gen_drawio.py
Tạo file usecase_diagram.drawio đúng định dạng draw.io (deflate + base64).
Chạy: python3 gen_drawio.py
"""
import zlib, base64, urllib.parse, os

# ── Nội dung sơ đồ (mxGraphModel XML) ────────────────────────────────────────
GRAPH_XML = """\
<mxGraphModel dx="1600" dy="1000" grid="0" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="0" pageScale="1" pageWidth="1654" pageHeight="1169" math="0" shadow="0">
  <root>
    <mxCell id="0"/>
    <mxCell id="1" parent="0"/>
    <mxCell id="sys" value="He Thong Scan-3D-InstantNGP" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#f5f5f5;strokeColor=#555;fontColor=#333;fontSize=15;fontStyle=1;verticalAlign=top;arcSize=2;strokeWidth=2;" vertex="1" parent="1"><mxGeometry x="180" y="30" width="980" height="1000" as="geometry"/></mxCell>
    <mxCell id="aND" value="Nguoi Dung" style="shape=mxgraph.uml.actor;whiteSpace=wrap;html=1;fillColor=#dae8fc;strokeColor=#6c8ebf;fontStyle=1;fontSize=11;" vertex="1" parent="1"><mxGeometry x="45" y="290" width="60" height="90" as="geometry"/></mxCell>
    <mxCell id="aBA" value="Blender Artist" style="shape=mxgraph.uml.actor;whiteSpace=wrap;html=1;fillColor=#d5e8d4;strokeColor=#82b366;fontStyle=1;fontSize=11;" vertex="1" parent="1"><mxGeometry x="1215" y="450" width="60" height="90" as="geometry"/></mxCell>
    <mxCell id="aCOL" value="COLMAP" style="shape=mxgraph.uml.actor;whiteSpace=wrap;html=1;fillColor=#fff2cc;strokeColor=#d6b656;fontStyle=1;fontSize=11;" vertex="1" parent="1"><mxGeometry x="45" y="560" width="60" height="90" as="geometry"/></mxCell>
    <mxCell id="aGPU" value="GPU/CUDA" style="shape=mxgraph.uml.actor;whiteSpace=wrap;html=1;fillColor=#f8cecc;strokeColor=#b85450;fontStyle=1;fontSize=11;" vertex="1" parent="1"><mxGeometry x="45" y="730" width="60" height="90" as="geometry"/></mxCell>
    <mxCell id="uc01" value="UC01&#xa;Xu Ly Video / Anh" style="ellipse;whiteSpace=wrap;html=1;fillColor=#dae8fc;strokeColor=#6c8ebf;fontSize=12;" vertex="1" parent="1"><mxGeometry x="210" y="120" width="220" height="70" as="geometry"/></mxCell>
    <mxCell id="uc02" value="UC02&#xa;Huan Luyen NeRF" style="ellipse;whiteSpace=wrap;html=1;fillColor=#dae8fc;strokeColor=#6c8ebf;fontSize=12;" vertex="1" parent="1"><mxGeometry x="555" y="120" width="220" height="70" as="geometry"/></mxCell>
    <mxCell id="uc03" value="UC03&#xa;Xem 3D Qua GUI" style="ellipse;whiteSpace=wrap;html=1;fillColor=#dae8fc;strokeColor=#6c8ebf;fontSize=12;" vertex="1" parent="1"><mxGeometry x="875" y="120" width="220" height="70" as="geometry"/></mxCell>
    <mxCell id="uc10" value="UC10&#xa;Danh Gia Mo Hinh" style="ellipse;whiteSpace=wrap;html=1;fillColor=#fff2cc;strokeColor=#d6b656;fontSize=12;" vertex="1" parent="1"><mxGeometry x="210" y="300" width="220" height="70" as="geometry"/></mxCell>
    <mxCell id="uc09" value="UC09&#xa;Luu / Tai Checkpoint" style="ellipse;whiteSpace=wrap;html=1;fillColor=#fff2cc;strokeColor=#d6b656;fontSize=12;" vertex="1" parent="1"><mxGeometry x="555" y="300" width="220" height="70" as="geometry"/></mxCell>
    <mxCell id="uc04" value="UC04&#xa;Xuat Mesh 3D" style="ellipse;whiteSpace=wrap;html=1;fillColor=#dae8fc;strokeColor=#6c8ebf;fontSize=12;" vertex="1" parent="1"><mxGeometry x="875" y="300" width="220" height="70" as="geometry"/></mxCell>
    <mxCell id="uc08" value="UC08&#xa;Model Manager UI" style="ellipse;whiteSpace=wrap;html=1;fillColor=#dae8fc;strokeColor=#6c8ebf;fontSize=12;" vertex="1" parent="1"><mxGeometry x="210" y="490" width="220" height="70" as="geometry"/></mxCell>
    <mxCell id="uc05" value="UC05&#xa;Khoi Chay Render Server" style="ellipse;whiteSpace=wrap;html=1;fillColor=#dae8fc;strokeColor=#6c8ebf;fontSize=12;" vertex="1" parent="1"><mxGeometry x="555" y="490" width="220" height="70" as="geometry"/></mxCell>
    <mxCell id="uc06" value="UC06&#xa;Ket Noi Blender Add-on" style="ellipse;whiteSpace=wrap;html=1;fillColor=#d5e8d4;strokeColor=#82b366;fontSize=12;" vertex="1" parent="1"><mxGeometry x="875" y="490" width="220" height="70" as="geometry"/></mxCell>
    <mxCell id="uc07" value="UC07&#xa;Depth Compositing" style="ellipse;whiteSpace=wrap;html=1;fillColor=#d5e8d4;strokeColor=#82b366;fontSize=12;" vertex="1" parent="1"><mxGeometry x="875" y="700" width="220" height="70" as="geometry"/></mxCell>
    <mxCell id="e01" style="endArrow=none;html=1;strokeWidth=1.5;" edge="1" source="aND" target="uc01" parent="1"><mxGeometry relative="1" as="geometry"/></mxCell>
    <mxCell id="e02" style="endArrow=none;html=1;strokeWidth=1.5;" edge="1" source="aND" target="uc02" parent="1"><mxGeometry relative="1" as="geometry"/></mxCell>
    <mxCell id="e03" style="endArrow=none;html=1;strokeWidth=1.5;" edge="1" source="aND" target="uc03" parent="1"><mxGeometry relative="1" as="geometry"/></mxCell>
    <mxCell id="e04" style="endArrow=none;html=1;strokeWidth=1.5;" edge="1" source="aND" target="uc04" parent="1"><mxGeometry relative="1" as="geometry"/></mxCell>
    <mxCell id="e05" style="endArrow=none;html=1;strokeWidth=1.5;" edge="1" source="aND" target="uc05" parent="1"><mxGeometry relative="1" as="geometry"/></mxCell>
    <mxCell id="e06" style="endArrow=none;html=1;strokeWidth=1.5;" edge="1" source="aND" target="uc08" parent="1"><mxGeometry relative="1" as="geometry"/></mxCell>
    <mxCell id="e07" style="endArrow=none;html=1;strokeWidth=1.5;" edge="1" source="aND" target="uc10" parent="1"><mxGeometry relative="1" as="geometry"/></mxCell>
    <mxCell id="e08" style="endArrow=none;html=1;strokeWidth=1.5;" edge="1" source="aBA" target="uc06" parent="1"><mxGeometry relative="1" as="geometry"/></mxCell>
    <mxCell id="e09" style="endArrow=none;html=1;strokeWidth=1.5;" edge="1" source="aBA" target="uc07" parent="1"><mxGeometry relative="1" as="geometry"/></mxCell>
    <mxCell id="e10" style="endArrow=none;html=1;strokeWidth=1.5;" edge="1" source="aCOL" target="uc01" parent="1"><mxGeometry relative="1" as="geometry"/></mxCell>
    <mxCell id="e11" style="endArrow=none;html=1;strokeWidth=1.5;" edge="1" source="aGPU" target="uc02" parent="1"><mxGeometry relative="1" as="geometry"/></mxCell>
    <mxCell id="e12" style="endArrow=none;html=1;strokeWidth=1.5;" edge="1" source="aGPU" target="uc03" parent="1"><mxGeometry relative="1" as="geometry"/></mxCell>
    <mxCell id="e13" style="endArrow=none;html=1;strokeWidth=1.5;" edge="1" source="aGPU" target="uc05" parent="1"><mxGeometry relative="1" as="geometry"/></mxCell>
    <mxCell id="e14" value="&lt;&lt;include&gt;&gt;" style="endArrow=open;endFill=0;dashed=1;html=1;fontSize=10;strokeColor=#555;" edge="1" source="uc02" target="uc09" parent="1"><mxGeometry relative="1" as="geometry"/></mxCell>
    <mxCell id="e15" value="&lt;&lt;include&gt;&gt;" style="endArrow=open;endFill=0;dashed=1;html=1;fontSize=10;strokeColor=#555;" edge="1" source="uc04" target="uc02" parent="1"><mxGeometry relative="1" as="geometry"/></mxCell>
    <mxCell id="e16" value="&lt;&lt;include&gt;&gt;" style="endArrow=open;endFill=0;dashed=1;html=1;fontSize=10;strokeColor=#555;" edge="1" source="uc08" target="uc02" parent="1"><mxGeometry relative="1" as="geometry"/></mxCell>
    <mxCell id="e17" value="&lt;&lt;include&gt;&gt;" style="endArrow=open;endFill=0;dashed=1;html=1;fontSize=10;strokeColor=#555;" edge="1" source="uc08" target="uc05" parent="1"><mxGeometry relative="1" as="geometry"/></mxCell>
    <mxCell id="e18" value="&lt;&lt;extend&gt;&gt;" style="endArrow=open;endFill=0;dashed=1;html=1;fontSize=10;strokeColor=#666;" edge="1" source="uc03" target="uc02" parent="1"><mxGeometry relative="1" as="geometry"/></mxCell>
    <mxCell id="e19" value="&lt;&lt;extend&gt;&gt;" style="endArrow=open;endFill=0;dashed=1;html=1;fontSize=10;strokeColor=#666;" edge="1" source="uc10" target="uc02" parent="1"><mxGeometry relative="1" as="geometry"/></mxCell>
    <mxCell id="e20" value="&lt;&lt;extend&gt;&gt;" style="endArrow=open;endFill=0;dashed=1;html=1;fontSize=10;strokeColor=#666;" edge="1" source="uc07" target="uc06" parent="1"><mxGeometry relative="1" as="geometry"/></mxCell>
    <mxCell id="e21" value="uses" style="endArrow=open;endFill=0;dashed=1;html=1;fontSize=10;strokeColor=#999;" edge="1" source="uc05" target="uc06" parent="1"><mxGeometry relative="1" as="geometry"/></mxCell>
  </root>
</mxGraphModel>"""

def encode_drawio(xml: str) -> str:
    """Chuyển XML thành chuỗi base64(rawDeflate(uriEncoded(xml))) — đúng format draw.io."""
    uri_encoded = urllib.parse.quote(xml, safe="")
    raw_bytes    = uri_encoded.encode("utf-8")
    # zlib.compress cho ra [2-byte header][deflate data][4-byte checksum]
    # draw.io cần raw deflate (không có header/checksum)
    compressed   = zlib.compress(raw_bytes, level=9)[2:-4]
    return base64.b64encode(compressed).decode("utf-8")

def build_drawio_file(encoded: str) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<mxfile host="app.diagrams.net" version="24.0.0">\n'
        '  <diagram id="uc_scan3d" name="Use Case Diagram">\n'
        f'    {encoded}\n'
        '  </diagram>\n'
        '</mxfile>\n'
    )

if __name__ == "__main__":
    out_dir  = os.path.join(os.path.dirname(__file__), "diagrams")
    out_path = os.path.join(out_dir, "usecase_diagram.drawio")
    os.makedirs(out_dir, exist_ok=True)

    encoded = encode_drawio(GRAPH_XML)
    content = build_drawio_file(encoded)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"[OK] File da duoc tao: {out_path}")
    print(f"     Kich thuoc: {os.path.getsize(out_path):,} bytes")
