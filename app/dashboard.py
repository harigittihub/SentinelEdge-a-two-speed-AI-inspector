import tempfile
from pathlib import Path

import streamlit as st
from pipeline import ROOT, categories, inspect_image

st.set_page_config(page_title="SentinelEdge", layout="wide")
st.title("SentinelEdge: a two-speed AI inspector")
st.caption("PyTorch vision -> Jev (fast typed decision) -> Claude (only for ambiguous cases "
           "in categories where it proved reliable)")

cat = st.sidebar.selectbox("Part category", categories())
escalate = st.sidebar.checkbox("Allow Claude escalation", value=True)
source = st.sidebar.radio("Image source", ["Pick a test image", "Upload"])

path, truth = None, None
if source == "Pick a test image":
    base = ROOT / "data" / cat / "test"
    names = [f.relative_to(base).as_posix() for f in sorted(base.rglob("*.png"))]
    choice = st.sidebar.selectbox("Image", names)
    path, truth = str(base / choice), choice.split("/")[0]
else:
    up = st.sidebar.file_uploader("Image", type=["png", "jpg", "jpeg"])
    if up:
        tmp = Path(tempfile.mkdtemp()) / up.name
        tmp.write_bytes(up.getvalue())
        path = str(tmp)

if path and st.sidebar.button("Inspect", type="primary"):
    with st.spinner("Inspecting..."):
        res = inspect_image(path, cat, escalate)

    left, right = st.columns(2)
    left.image(res["image"], caption="Input", use_container_width=True)
    right.image(res["heatmap"], caption="Anomaly heatmap (red = unusual)", use_container_width=True)
    if truth:
        st.info(f"Ground truth (demo only): {truth}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Final decision", res["final"])
    c2.metric("Jev action", res["jev_action"])
    c3.metric("Severity (0-3)", f"{res['severity']:.2f}")
    c4.metric("Stop-line probability", f"{res['stop_prob']:.2f}")
    st.caption(f"Jev latency: {res['jev_ms']:.0f} ms | vision: {res['vision']}")
    if res["jev_probs"]:
        st.bar_chart(res["jev_probs"])

    if res["claude"]:
        st.subheader("Claude escalation")
        st.write(f"Verdict: **{res['claude'].get('verdict')}** "
                 f"({res['claude']['latency_ms']:.0f} ms)")
        st.write(res["claude"].get("incident_report", ""))
    elif res["jev_action"] == "human_review":
        st.warning("Ambiguous case routed to a human inspector.")