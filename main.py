from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import shutil
import os

from starlette.responses import JSONResponse

from rag_utils import save_file_to_uploads, process_and_store, list_uploaded_files, get_all_sources, get_rag_answer

app = FastAPI()
templates = Jinja2Templates(directory="templates")
os.makedirs("uploads", exist_ok=True)


@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    try:
        saved_path = save_file_to_uploads(file.file, file.filename)
        process_and_store(saved_path)
        return {"status": "ok", "filename": file.filename}
    except Exception as e:
        return JSONResponse(content={"status": "error", "detail": str(e)}, status_code=400)


@app.get("/files")
async def get_files():
    return {"files": list_uploaded_files()}

@app.get("/sources")
async def get_files():
    return {"sources": get_all_sources()}


@app.post("/ask")
async def ask_question(
    question: str = Form(...),
    source: str = Form(...)
):
    print("ask input", question, source)

    answer = get_rag_answer(question, source if source else None)
    return {"answer": answer}
