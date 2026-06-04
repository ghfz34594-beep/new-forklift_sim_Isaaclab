"""
Web 服务模块：FastAPI + WebSocket。

由仿真主线程在后台 asyncio 线程中启动。
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
from pathlib import Path
from typing import Optional, Set

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

logger = logging.getLogger("forklift.web")

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Forklift Web Control", version="1.0.0")

# ---------- 全局控制器引用（由 main.py 注入）----------
_controller = None

def set_controller(ctrl) -> None:
    global _controller
    _controller = ctrl


# ---------- 数据模型 ----------

class CommandRequest(BaseModel):
    drive: float = Field(default=0.0, ge=-1.0, le=1.0, description="前进/后退，[-1,1]")
    steer: float = Field(default=0.0, ge=-1.0, le=1.0, description="左转/右转，[-1,1]")
    lift: float = Field(default=0.0, ge=-1.0, le=1.0, description="起升/下降，[-1,1]")


# ---------- REST 端点 ----------

@app.post("/api/command")
async def post_command(req: CommandRequest):
    """发送叉车控制命令。"""
    if _controller is None:
        return JSONResponse({"error": "仿真尚未就绪"}, status_code=503)
    _controller.set_command(req.drive, req.steer, req.lift)
    return {"ok": True}


@app.get("/api/state")
async def get_state():
    """获取叉车当前状态。"""
    if _controller is None:
        return JSONResponse({"error": "仿真尚未就绪"}, status_code=503)
    state = _controller.get_state()
    return state.to_dict()


@app.get("/api/ready")
async def get_ready():
    """查询仿真是否完成初始化。"""
    ready = _controller is not None and _controller.is_ready()
    return {"ready": ready}


# ---------- WebSocket ----------

_ws_clients: Set[WebSocket] = set()
_ws_lock = asyncio.Lock()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    async with _ws_lock:
        _ws_clients.add(websocket)
    logger.info("WebSocket 客户端已连接，当前连接数: %d", len(_ws_clients))
    try:
        while True:
            # 保持连接；客户端发送的消息暂时忽略
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        async with _ws_lock:
            _ws_clients.discard(websocket)
        logger.info("WebSocket 客户端已断开，当前连接数: %d", len(_ws_clients))


async def _broadcast_state_loop(interval: float = 0.1) -> None:
    """后台协程：每 interval 秒向所有 WebSocket 客户端推送叉车状态。"""
    while True:
        await asyncio.sleep(interval)
        if not _ws_clients or _controller is None:
            continue
        try:
            state = _controller.get_state()
            payload = json.dumps(state.to_dict())
        except Exception:
            continue

        dead: list[WebSocket] = []
        async with _ws_lock:
            clients = list(_ws_clients)
        for ws in clients:
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(ws)
        if dead:
            async with _ws_lock:
                for ws in dead:
                    _ws_clients.discard(ws)


# ---------- 静态文件 ----------

@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


# 挂载 static 目录（用于可能的 JS/CSS 资源）
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ---------- 启动事件 ----------

@app.on_event("startup")
async def on_startup():
    asyncio.create_task(_broadcast_state_loop(interval=0.1))
    logger.info("状态广播协程已启动（10 Hz）")


# ---------- 在后台线程中启动服务 ----------

def start_web_server(host: str = "0.0.0.0", port: int = 8080) -> threading.Thread:
    """
    在独立的 daemon 线程中启动 uvicorn。
    返回线程对象，主线程无需等待。
    """
    def _run():
        uvicorn.run(
            app,
            host=host,
            port=port,
            log_level="warning",
            access_log=False,
        )

    t = threading.Thread(target=_run, name="forklift-web", daemon=True)
    t.start()
    logger.info("Web 服务已在后台启动：http://%s:%d", host, port)
    return t
