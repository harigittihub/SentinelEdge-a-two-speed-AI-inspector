import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from pipeline import categories, inspect_image

app = FastAPI(title="SentinelEdge")


@app.get("/categories")
def list_categories():
    return categories()


@app.post("/inspect")
def inspect(category: str = Form(...), escalate: bool = Form(True), file: UploadFile = File(...)):
    tmp = Path(tempfile.mkdtemp()) / (file.filename or "upload.png")
    tmp.write_bytes(file.file.read())
    res = inspect_image(str(tmp), category, escalate)
    res.pop("image")
    res.pop("heatmap")
    return res