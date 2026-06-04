#!/usr/bin/env python3
"""Codex Orchestrator

在 codex app-server 执行一个 goal 的过程中，由"裁判模型"周期性审查进展，
检测到死胡同/反复/偏题时调用 turn/steer 注入引导消息；
同时提供 REPL 让人手动随时介入。

依赖：仅 Python 3.10+ 标准库 + 本机已安装 `codex` CLI（0.130+ 含 app-server）。

使用示例（最简）：
    export JUDGE_BASE_URL="https://gmliv.top:8443/v1"
    export JUDGE_API_KEY="sk-..."          # 与 codex 同一 endpoint 可复用
    export JUDGE_MODEL="claude-opus-4.8"           # 第二个、与主模型不同的裁判
    python codex_orchestrator.py \
        --cwd /data/jianshi/projects/forklift_sim_exp9 \
        --goal "把 v311 训练复现到 model1999 并打通离线评测" \
        --task "请按照 docs/toyota_reference_curve_exploration_20260602.md 的方案开始第一步"

启动后输入：
    /steer <文本>    立刻向当前 turn 注入一条引导消息
    /judge           立刻触发一次裁判审查（不等定时器）
    /interrupt       打断当前 turn
    /status          打印当前状态
    /quit            退出（终止 codex 子进程）
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from datetime import datetime
from typing import Any, Callable, Optional


# ============================== Codex app-server 客户端 ==============================


class CodexAppServer:
    """通过 stdio 与 `codex app-server` 通信的最小 JSON-RPC 客户端。"""

    def __init__(self, binary: str, extra_args: Optional[list[str]] = None,
                 env: Optional[dict[str, str]] = None) -> None:
        self.proc = subprocess.Popen(
            [binary, "app-server", "--listen", "stdio://"] + (extra_args or []),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            env=env,
        )
        self._next_id = 0
        self._send_lock = threading.Lock()
        self._pending: dict[int, queue.Queue] = {}
        self._notification_handler: Optional[Callable[[dict], None]] = None
        self._server_request_handler: Optional[Callable[[dict], dict]] = None
        self._closed = threading.Event()
        threading.Thread(target=self._reader_loop, daemon=True).start()
        threading.Thread(target=self._stderr_loop, daemon=True).start()

    def on_notification(self, handler: Callable[[dict], None]) -> None:
        self._notification_handler = handler

    def on_server_request(self, handler: Callable[[dict], dict]) -> None:
        """handler(msg) 返回 result dict；若返回 None 则发送 method-not-found error。"""
        self._server_request_handler = handler

    def call(self, method: str, params: Any, timeout: float = 120.0) -> Any:
        with self._send_lock:
            self._next_id += 1
            req_id = self._next_id
        q: queue.Queue = queue.Queue()
        self._pending[req_id] = q
        self._write({"jsonrpc": "2.0", "id": req_id, "method": method, "params": params})
        try:
            resp = q.get(timeout=timeout)
        except queue.Empty:
            self._pending.pop(req_id, None)
            raise TimeoutError(f"call timeout: {method}")
        if "error" in resp and resp["error"] is not None:
            raise RuntimeError(f"{method} failed: {resp['error']}")
        return resp.get("result")

    def _write(self, obj: dict) -> None:
        data = (json.dumps(obj) + "\n").encode("utf-8")
        with self._send_lock:
            try:
                self.proc.stdin.write(data)
                self.proc.stdin.flush()
            except (BrokenPipeError, OSError):
                self._closed.set()

    def _reader_loop(self) -> None:
        for raw in iter(self.proc.stdout.readline, b""):
            if not raw:
                break
            try:
                msg = json.loads(raw.decode("utf-8", "replace"))
            except Exception as e:
                sys.stderr.write(f"[parse-err] {e}: {raw!r}\n")
                continue
            if "id" in msg and ("result" in msg or "error" in msg):
                q = self._pending.pop(msg["id"], None)
                if q is not None:
                    q.put(msg)
            elif "method" in msg and "id" in msg:
                self._dispatch_server_request(msg)
            elif "method" in msg:
                self._dispatch_notification(msg)
        self._closed.set()

    def _dispatch_notification(self, msg: dict) -> None:
        if self._notification_handler is None:
            return
        try:
            self._notification_handler(msg)
        except Exception as e:
            sys.stderr.write(f"[notif-handler-err] {e}\n")

    def _dispatch_server_request(self, msg: dict) -> None:
        result: Optional[dict] = None
        error: Optional[dict] = None
        if self._server_request_handler is not None:
            try:
                result = self._server_request_handler(msg)
            except Exception as e:
                error = {"code": -32603, "message": f"handler error: {e}"}
        if result is None and error is None:
            error = {"code": -32601, "message": f"method not handled: {msg.get('method')}"}
        reply = {"jsonrpc": "2.0", "id": msg["id"]}
        if error is not None:
            reply["error"] = error
        else:
            reply["result"] = result
        self._write(reply)

    def _stderr_loop(self) -> None:
        for line in iter(self.proc.stderr.readline, b""):
            try:
                sys.stderr.write(f"[codex-err] {line.decode('utf-8', 'replace').rstrip()}\n")
            except Exception:
                pass

    def close(self) -> None:
        self._closed.set()
        try:
            self.proc.terminate()
            self.proc.wait(timeout=3)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass


# ============================== 裁判模型客户端 ==============================


class JudgeClient:
    """OpenAI 兼容 /chat/completions 调用裁判模型。"""

    def __init__(self, base_url: str, api_key: str, model: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model

    def review(self, system_prompt: str, user_prompt: str, timeout: float = 90.0) -> str:
        url = f"{self.base_url}/chat/completions"
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.2,
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        try:
            return payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            raise RuntimeError(f"unexpected judge response: {payload}") from e


# ============================== 编排器 ==============================


JUDGE_SYSTEM_PROMPT = (
    "你是 Codex 执行裁判。输入包含：一个 goal、当前已运行时长、最近一段执行轨迹"
    "（命令、文件改动、agent 输出、计划、推理摘要）。"
    "你要判断 codex 当前状态属于以下哪一种：\n"
    "  - continue: 正常推进，不需要打断。\n"
    "  - steer:    出现死胡同/重复尝试同一思路/偏轨但还能挽回，需要点拨。\n"
    "  - interrupt:严重偏离目标或正在做破坏性/不可逆的错误动作，需要立刻打断。\n"
    "判定原则：宁可 continue 也不要乱 steer；仅在你有明确依据时才出手。\n"
    "输出严格 JSON，单一对象，字段如下：\n"
    "  verdict     : \"continue\" | \"steer\" | \"interrupt\"\n"
    "  confidence  : 0.0-1.0\n"
    "  reason      : 一句话说明判定依据（中文）\n"
    "  message     : 仅当 verdict=steer 或 interrupt 时填，给 codex 看的中文引导，"
    "               必须具体、可执行（指出问题 + 建议下一步），不要空话。"
)


class Orchestrator:
    def __init__(self, codex: CodexAppServer, judge: JudgeClient, args: argparse.Namespace) -> None:
        self.codex = codex
        self.judge = judge
        self.args = args

        self.thread_id: Optional[str] = None
        self.current_turn_id: Optional[str] = None
        self.start_ts: float = time.time()
        self.recent_items: deque = deque(maxlen=args.max_recent)
        self.items_since_review: int = 0
        self.last_review_at: float = time.time()
        self.review_lock = threading.Lock()
        self._stop = threading.Event()

        self._approval_mode = args.auto_approve  # "all" | "none" | "prompt"

    # -------- 事件处理 --------

    def on_notification(self, msg: dict) -> None:
        method = msg.get("method", "")
        params = msg.get("params") or {}
        if method == "thread/started":
            self.thread_id = params.get("thread", {}).get("id")
            self._log(f"[thread] started id={self.thread_id}")
        elif method == "turn/started":
            self.current_turn_id = params.get("turn", {}).get("id")
            self.start_ts = time.time()
            self.last_review_at = time.time()
            self.items_since_review = 0
            self._log(f"[turn] started id={self.current_turn_id}")
        elif method == "turn/completed":
            turn = params.get("turn", {}) or {}
            self._log(f"[turn] completed status={turn.get('status')} duration_ms={turn.get('durationMs')}")
            self.current_turn_id = None
        elif method == "item/completed":
            self._record_item(params.get("item", {}) or {}, completed=True)
        elif method == "item/started":
            self._record_item(params.get("item", {}) or {}, completed=False)
        elif method == "error":
            self._log(f"[error] {params}")
        # 其余事件忽略（delta 噪声大）

    def on_server_request(self, msg: dict) -> Optional[dict]:
        method = msg.get("method", "")
        params = msg.get("params") or {}
        decision = self._decide_approval(method, params)
        if method in ("execCommandApproval", "applyPatchApproval"):
            # v1 协议：ReviewDecision
            mapping = {"accept": "approved", "decline": "denied", "cancel": "abort"}
            self._log(f"[approval/{method}] -> {decision}")
            return {"decision": mapping.get(decision, "denied")}
        if method == "item/commandExecution/requestApproval":
            self._log(f"[approval/cmd] {self._summarize_approval_cmd(params)} -> {decision}")
            return {"decision": decision}
        if method == "item/fileChange/requestApproval":
            self._log(f"[approval/file] reason={params.get('reason')} -> {decision}")
            return {"decision": decision}
        # 其余反向请求不处理（permissions/elicitation/dynamicToolCall...）
        return None

    def _decide_approval(self, method: str, params: dict) -> str:
        mode = self._approval_mode
        if mode == "all":
            return "accept"
        if mode == "none":
            return "decline"
        # prompt: 同步阻塞问人。简单做：打印到 stderr，从 stdin 读 y/n（5s 超时则 decline）
        return self._prompt_approval(method, params)

    def _prompt_approval(self, method: str, params: dict) -> str:
        self._log(f"[approval?] {method} params={json.dumps(params, ensure_ascii=False)[:500]}")
        sys.stderr.write("    y=accept / n=decline (默认 decline, 5s 超时): ")
        sys.stderr.flush()
        # 注意 stdin 同时被 REPL 使用，避免争用 - 这里简化只在 prompt 模式下用
        import select
        rlist, _, _ = select.select([sys.stdin], [], [], 5.0)
        if not rlist:
            return "decline"
        line = sys.stdin.readline().strip().lower()
        return "accept" if line in ("y", "yes") else "decline"

    @staticmethod
    def _summarize_approval_cmd(params: dict) -> str:
        cmd = params.get("command") or ""
        cwd = params.get("cwd") or ""
        return f"$ {cmd[:200]}  (cwd={cwd})"

    def _record_item(self, item: dict, completed: bool) -> None:
        line = self._summarize_item(item, completed)
        if not line:
            return
        self.recent_items.append({
            "ts": time.time(),
            "completed": completed,
            "type": item.get("type"),
            "line": line,
        })
        if completed:
            self.items_since_review += 1
        if self.args.verbose:
            self._log(line)

    @staticmethod
    def _summarize_item(item: dict, completed: bool) -> Optional[str]:
        t = item.get("type")
        tag = "✓" if completed else "→"
        if t == "agentMessage":
            text = (item.get("text") or "").strip().replace("\n", " ")
            return f"{tag} [agent] {text[:300]}"
        if t == "commandExecution":
            cmd = (item.get("command") or "").strip().replace("\n", " ")
            status = item.get("status")
            ec = item.get("exitCode")
            out = (item.get("aggregatedOutput") or "").strip()
            tail = out[-200:].replace("\n", " ⏎ ") if out else ""
            return f"{tag} [cmd:{status} ec={ec}] $ {cmd[:160]}  | {tail}"
        if t == "fileChange":
            paths = []
            for c in (item.get("changes") or [])[:6]:
                p = c.get("path") or c.get("targetPath") or c.get("filePath")
                if p:
                    paths.append(str(p))
            return f"{tag} [edit:{item.get('status')}] {', '.join(paths)}"
        if t == "plan":
            text = (item.get("text") or "").replace("\n", " ")
            return f"{tag} [plan] {text[:300]}"
        if t == "reasoning":
            summary = " | ".join((item.get("summary") or [])[:3])
            if not summary:
                return None
            return f"{tag} [reasoning] {summary[:300]}"
        if t == "mcpToolCall":
            return f"{tag} [mcp] {item.get('server')}/{item.get('tool')} status={item.get('status')}"
        if t == "webSearch":
            return f"{tag} [web] {(item.get('query') or '')[:160]}"
        return None

    # -------- 审查 / 注入 --------

    def reviewer_loop(self) -> None:
        while not self._stop.is_set():
            time.sleep(2.0)
            if not self.current_turn_id:
                continue
            elapsed = time.time() - self.last_review_at
            if elapsed >= self.args.review_interval_sec or \
                    self.items_since_review >= self.args.review_item_count:
                self.run_review(reason="auto")

    def run_review(self, reason: str = "manual") -> None:
        with self.review_lock:
            if reason == "auto" and not self.current_turn_id:
                return
            snapshot = list(self.recent_items)
            self.last_review_at = time.time()
            self.items_since_review = 0
            elapsed_sec = int(time.time() - self.start_ts)

        if not snapshot:
            self._log(f"[judge:{reason}] 无可审查内容，跳过")
            return

        lines = [it["line"] for it in snapshot[-self.args.review_item_count * 2:]]
        history = "\n".join(lines)
        # 截断防爆
        if len(history) > self.args.history_chars:
            history = history[-self.args.history_chars:]

        user_prompt = json.dumps({
            "goal": self.args.goal,
            "elapsed_sec_in_turn": elapsed_sec,
            "active_turn_id": self.current_turn_id,
            "recent_activity": history,
        }, ensure_ascii=False, indent=2)

        t0 = time.time()
        try:
            raw = self.judge.review(JUDGE_SYSTEM_PROMPT, user_prompt)
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")[:500]
            self._log(f"[judge:err] HTTP {e.code} {body}")
            return
        except Exception as e:
            self._log(f"[judge:err] {type(e).__name__}: {e}")
            return
        dt = time.time() - t0

        try:
            verdict = json.loads(raw)
        except Exception:
            self._log(f"[judge:bad-json {dt:.1f}s] {raw[:400]}")
            return

        v = verdict.get("verdict")
        conf = verdict.get("confidence")
        why = verdict.get("reason", "")
        self._log(f"[judge:{reason} {dt:.1f}s] verdict={v} conf={conf} reason={why}")

        if v == "steer":
            msg = (verdict.get("message") or "").strip()
            if msg and self.current_turn_id:
                self.do_steer(msg)
            else:
                self._log("[judge] 想 steer 但 message 为空或 turn 已结束，跳过")
        elif v == "interrupt":
            if self.current_turn_id:
                self.do_interrupt()
                msg = (verdict.get("message") or "").strip()
                if msg:
                    self._log(f"[judge] interrupt reason -> 将作为下一轮提示: {msg}")
                    # 打断后启动一个新 turn 把 judge 的纠正传进去
                    self._start_followup_turn(msg)

    def do_steer(self, text: str) -> None:
        if not self.thread_id or not self.current_turn_id:
            self._log("[steer] 当前没有活跃 turn")
            return
        try:
            res = self.codex.call("turn/steer", {
                "threadId": self.thread_id,
                "expectedTurnId": self.current_turn_id,
                "input": [{"type": "text", "text": text, "text_elements": []}],
            })
            self._log(f"[steer-ok] turnId={res.get('turnId')} text={text[:160]}")
        except Exception as e:
            self._log(f"[steer-err] {e}")

    def do_interrupt(self) -> None:
        if not self.thread_id or not self.current_turn_id:
            self._log("[interrupt] 当前没有活跃 turn")
            return
        try:
            self.codex.call("turn/interrupt", {
                "threadId": self.thread_id,
                "turnId": self.current_turn_id,
            })
            self._log("[interrupt-ok]")
        except Exception as e:
            self._log(f"[interrupt-err] {e}")

    def _start_followup_turn(self, message: str) -> None:
        if not self.thread_id:
            return
        try:
            res = self.codex.call("turn/start", {
                "threadId": self.thread_id,
                "input": [{"type": "text", "text": message, "text_elements": []}],
            })
            self._log(f"[followup-turn] id={res.get('turn', {}).get('id')}")
        except Exception as e:
            self._log(f"[followup-err] {e}")

    # -------- REPL --------

    def repl(self) -> None:
        self._log("REPL 就绪。输入：/steer <文本> | /judge | /interrupt | /status | /quit")
        while not self._stop.is_set():
            try:
                line = sys.stdin.readline()
            except KeyboardInterrupt:
                break
            if not line:
                break
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith("/steer "):
                text = line[len("/steer "):].strip()
                if text:
                    self.do_steer(text)
            elif line == "/judge":
                threading.Thread(target=self.run_review, args=("manual",), daemon=True).start()
            elif line == "/interrupt":
                self.do_interrupt()
            elif line == "/status":
                self._print_status()
            elif line in ("/quit", "/exit"):
                self._stop.set()
                break
            elif line.startswith("/say "):
                # 直接开新 turn 把消息当作 user message
                msg = line[len("/say "):].strip()
                if msg:
                    self._start_followup_turn(msg)
            else:
                self._log("未知命令；可用 /steer /judge /interrupt /say /status /quit")

    def _print_status(self) -> None:
        self._log(
            f"[status] thread={self.thread_id} turn={self.current_turn_id} "
            f"items_buf={len(self.recent_items)} items_since_review={self.items_since_review} "
            f"sec_since_review={int(time.time() - self.last_review_at)}"
        )

    # -------- 工具 --------

    @staticmethod
    def _log(msg: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        sys.stderr.write(f"{ts} {msg}\n")
        sys.stderr.flush()

    def stop(self) -> None:
        self._stop.set()


# ============================== main ==============================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Codex 编排器：第二个模型做裁判，周期审查 codex 进展并按需注入引导。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--cwd", default=os.getcwd(), help="codex thread 的工作目录")
    p.add_argument("--goal", required=True, help="高层目标（写入 thread/goal）")
    p.add_argument("--task", required=True, help="第一轮 turn 的具体指令")

    p.add_argument("--codex-bin", default="codex", help="codex 可执行文件路径")
    p.add_argument("--model", default=None, help="覆盖主模型（默认走 ~/.codex/config.toml）")

    p.add_argument("--sandbox", default="workspace-write",
                   choices=["read-only", "workspace-write", "danger-full-access"])
    p.add_argument("--approval-policy", default="never",
                   choices=["never", "on-request", "on-failure", "untrusted"])
    p.add_argument("--auto-approve", default="none", choices=["none", "all", "prompt"],
                   help="当 codex 仍发起 approval 反向请求时如何应答")

    p.add_argument("--judge-base-url", default=os.environ.get("JUDGE_BASE_URL"),
                   help="裁判模型的 OpenAI 兼容 base url，如 https://gmliv.top:8443/v1")
    p.add_argument("--judge-api-key", default=os.environ.get("JUDGE_API_KEY"))
    p.add_argument("--judge-model", default=os.environ.get("JUDGE_MODEL", "claude-opus-4.8"))

    p.add_argument("--review-interval-sec", type=float, default=90.0, help="周期审查时间间隔（秒）")
    p.add_argument("--review-item-count", type=int, default=15, help="周期审查的 item 数阈值")
    p.add_argument("--max-recent", type=int, default=60, help="本地保留的最近 item 上限")
    p.add_argument("--history-chars", type=int, default=8000, help="发给裁判的历史最大字符数")

    p.add_argument("--verbose", action="store_true", help="打印每条 item 摘要")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.judge_base_url or not args.judge_api_key:
        sys.stderr.write("必须设置 --judge-base-url 和 --judge-api-key（或环境变量 JUDGE_BASE_URL / JUDGE_API_KEY）\n")
        return 2

    codex = CodexAppServer(args.codex_bin)
    judge = JudgeClient(args.judge_base_url, args.judge_api_key, args.judge_model)
    orch = Orchestrator(codex, judge, args)
    codex.on_notification(orch.on_notification)
    codex.on_server_request(orch.on_server_request)

    try:
        info = codex.call("initialize", {
            "clientInfo": {"name": "codex-orchestrator", "title": "Codex Orchestrator", "version": "0.1.0"},
            "capabilities": None,
        })
        Orchestrator._log(f"[init] codex={info.get('userAgent')} home={info.get('codexHome')}")

        thread_params: dict[str, Any] = {
            "cwd": args.cwd,
            "approvalPolicy": args.approval_policy,
            "sandbox": args.sandbox,
            "experimentalRawEvents": False,
            "persistExtendedHistory": False,
        }
        if args.model:
            thread_params["model"] = args.model
        start_resp = codex.call("thread/start", thread_params)
        orch.thread_id = start_resp["thread"]["id"]
        Orchestrator._log(f"[thread] id={orch.thread_id} model={start_resp.get('model')} sandbox={start_resp.get('sandbox')}")

        try:
            codex.call("thread/goal/set", {
                "threadId": orch.thread_id,
                "objective": args.goal,
                "status": "active",
            })
            Orchestrator._log(f"[goal] set: {args.goal}")
        except Exception as e:
            Orchestrator._log(f"[goal:warn] 设定失败（不致命）: {e}")

        turn = codex.call("turn/start", {
            "threadId": orch.thread_id,
            "input": [{"type": "text", "text": args.task, "text_elements": []}],
        })
        orch.current_turn_id = turn["turn"]["id"]
        orch.start_ts = time.time()
        Orchestrator._log(f"[turn] start id={orch.current_turn_id}")

        threading.Thread(target=orch.reviewer_loop, daemon=True).start()
        orch.repl()
    except KeyboardInterrupt:
        Orchestrator._log("收到中断，关闭中...")
    except Exception as e:
        Orchestrator._log(f"[fatal] {type(e).__name__}: {e}")
        return 1
    finally:
        try:
            if orch.thread_id and orch.current_turn_id:
                orch.do_interrupt()
        except Exception:
            pass
        codex.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
