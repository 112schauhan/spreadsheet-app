import asyncio
import json
import logging
from typing import Dict, Set
import uvicorn
from fastapi import FastAPI, Request, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

logger = logging.getLogger(__name__)


app = FastAPI(title="Spreadsheet API", version="1.0.0",
              description="Real-time collaborative spreadsheet application", docs_url="/api/docs", redoc_url="/api/redoc")


@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "1.0.0"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000,
                log_level="info", access_log=true)
