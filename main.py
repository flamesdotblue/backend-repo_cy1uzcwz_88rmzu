import os
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from database import db, create_document, get_documents

app = FastAPI(title="AI Data Analytics Platform API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class IngestPayload(BaseModel):
    name: str = Field(..., description="Human-friendly dataset name")
    header: List[str]
    rows: List[List[str]] = Field(..., description="Sample rows (first 100)")
    total_rows: Optional[int] = None


class ChatRequest(BaseModel):
    dataset_id: str
    question: str


class VizRequest(BaseModel):
    dataset_id: str
    chart_type: str
    x: Optional[str] = None
    y: Optional[str] = None


class ExecRequest(BaseModel):
    dataset_id: str
    language: str = Field(..., description="Only 'python' is accepted")
    code: str


@app.get("/")
def read_root():
    return {"message": "Backend running", "docs": "/docs"}


@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set",
        "database_name": os.getenv("DATABASE_NAME") or "❌ Not Set",
        "connection_status": "Not Connected",
        "collections": [],
    }

    try:
        if db is not None:
            response["database"] = "✅ Available"
            response["connection_status"] = "Connected"
            response["collections"] = db.list_collection_names()[:10]
        else:
            response["database"] = "⚠️ Available but not initialized"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:120]}"
    return response


@app.post("/api/ingest")
async def ingest_dataset_json(payload: IngestPayload):
    if db is None:
        raise HTTPException(status_code=500, detail="Database not configured")

    if not payload.header or not payload.rows:
        raise HTTPException(status_code=400, detail="Header and rows are required")

    doc = {
        "name": payload.name,
        "header": payload.header,
        "sample_rows": payload.rows[:100],
        "total_rows": payload.total_rows or len(payload.rows),
    }
    dataset_id = create_document("dataset", doc)
    return {"dataset_id": dataset_id}


@app.post("/api/ingest/file")
async def ingest_dataset_file(file: UploadFile = File(...)):
    if db is None:
        raise HTTPException(status_code=500, detail="Database not configured")

    content = (await file.read()).decode("utf-8", errors="ignore")
    name = file.filename or "uploaded_file"

    # simple CSV parse
    lines = [ln for ln in content.splitlines() if ln.strip()]
    if not lines:
        raise HTTPException(status_code=400, detail="Empty file")

    header = [h.strip() for h in lines[0].split(",")]
    rows = [[c.strip() for c in ln.split(",")] for ln in lines[1:101]]

    doc = {
        "name": name,
        "header": header,
        "sample_rows": rows,
        "total_rows": len(lines) - 1,
    }
    dataset_id = create_document("dataset", doc)
    return {"dataset_id": dataset_id}


def infer_types(header: List[str], rows: List[List[str]]) -> List[str]:
    types: List[str] = []
    for i, _ in enumerate(header):
        samples = [r[i] for r in rows if len(r) > i and r[i] not in (None, "")][:50]
        is_numeric = len(samples) > 0 and all(_is_number(s) for s in samples)
        if is_numeric:
            types.append("numeric")
            continue
        is_date = len(samples) > 0 and all(_is_date(s) for s in samples)
        if is_date:
            types.append("datetime")
            continue
        types.append("categorical")
    return types


def _is_number(v: str) -> bool:
    try:
        float(v)
        return True
    except Exception:
        return False


def _is_date(v: str) -> bool:
    from dateutil.parser import parse  # type: ignore

    try:
        parse(v)
        return True
    except Exception:
        return False


@app.get("/api/profile/{dataset_id}")
def profile_dataset(dataset_id: str):
    if db is None:
        raise HTTPException(status_code=500, detail="Database not configured")

    from bson import ObjectId

    doc = db["dataset"].find_one({"_id": ObjectId(dataset_id)})
    if not doc:
        raise HTTPException(status_code=404, detail="Dataset not found")

    header: List[str] = doc.get("header", [])
    rows: List[List[str]] = doc.get("sample_rows", [])
    types = infer_types(header, rows)

    # basic per-column null counts
    nulls = []
    for i, col in enumerate(header):
        n_null = sum(1 for r in rows if i >= len(r) or r[i] in (None, ""))
        nulls.append({"column": col, "nulls": n_null})

    return {
        "name": doc.get("name"),
        "columns": header,
        "types": types,
        "rows_preview": rows[:10],
        "null_summary": nulls,
        "total_rows": doc.get("total_rows", len(rows)),
    }


@app.post("/api/chat")
def chat(req: ChatRequest):
    if db is None:
        raise HTTPException(status_code=500, detail="Database not configured")

    # This is a stubbed response. In production, call an LLM using your provider.
    answer = (
        "Here's a quick summary based on the preview: "
        "You can ask for top categories, trends over time, and outlier detection."
    )
    suggestions = [
        "Top 5 categories by total",
        "Monthly trend for the last year",
        "Correlation between two numeric fields",
    ]

    code_example = (
        "# pandas example\n"
        "import pandas as pd\n"
        "# df = pd.read_csv('your_file.csv')\n"
        "# df.groupby('category')['amount'].sum().sort_values(ascending=False).head()\n"
    )

    return {
        "answer": answer,
        "suggestions": suggestions,
        "code": code_example,
    }


@app.post("/api/visualize")
def visualize(req: VizRequest):
    if db is None:
        raise HTTPException(status_code=500, detail="Database not configured")

    # Return a lightweight Vega-Lite spec that the frontend can render or preview
    spec: Dict[str, Any] = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "mark": req.chart_type.lower(),
        "encoding": {},
        "data": {"values": []},
    }

    if req.x:
        spec["encoding"]["x"] = {"field": req.x, "type": "quantitative"}
    if req.y:
        spec["encoding"]["y"] = {"field": req.y, "type": "quantitative"}

    # Provide a tiny fake dataset to visualize shape
    spec["data"]["values"] = [
        {req.x or "x": i, req.y or "y": (i * 2) % 7 + 1} for i in range(1, 8)
    ]

    return {"spec": spec}


@app.post("/api/report/{dataset_id}")
def report(dataset_id: str):
    if db is None:
        raise HTTPException(status_code=500, detail="Database not configured")

    from bson import ObjectId

    doc = db["dataset"].find_one({"_id": ObjectId(dataset_id)})
    if not doc:
        raise HTTPException(status_code=404, detail="Dataset not found")

    header = doc.get("header", [])
    total_rows = doc.get("total_rows", 0)

    md = [
        f"# Dataset Report: {doc.get('name', '')}",
        "",
        f"- Columns: {len(header)}",
        f"- Rows (approx): {total_rows}",
        "",
        "## Columns",
        "",
    ]
    md.extend([f"- {c}" for c in header])

    return {"markdown": "\n".join(md)}


FORBIDDEN_KEYWORDS = [
    "import os", "import sys", "open(", "subprocess", "socket", "shutil", "pathlib",
    "requests", "urllib", "eval(", "exec(", "__import__", "globals()", "locals()",
]


@app.post("/api/execute")
def execute(req: ExecRequest):
    if req.language.lower() != "python":
        raise HTTPException(status_code=400, detail="Only python is supported")

    lower = req.code.lower()
    if any(k in lower for k in FORBIDDEN_KEYWORDS):
        raise HTTPException(status_code=400, detail="Disallowed code detected")

    # Extremely restricted sandbox – do not provide builtins or modules
    safe_globals: Dict[str, Any] = {"__builtins__": {}}
    safe_locals: Dict[str, Any] = {}

    try:
        exec(req.code, safe_globals, safe_locals)
    except Exception as e:
        return {"ok": False, "error": str(e)}

    # Return whatever safe variables were produced (limited)
    preview = {
        k: v for k, v in safe_locals.items() if isinstance(v, (int, float, str, list, dict))
    }
    return {"ok": True, "result": preview}


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
