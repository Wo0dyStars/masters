# Run cd ../backend && venv\\Scripts\\activate && uvicorn main:app --reload

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

from systems.naive import router as naive_router
from systems.advanced import router as advanced_router
from systems.agentic import router as agentic_router

app.include_router(naive_router, prefix="/api/v1", tags=["Naive RAG"])
app.include_router(advanced_router, prefix="/api/v1", tags=["Advanced RAG"])
app.include_router(agentic_router, prefix="/api/v1", tags=["Agentic RAG"])