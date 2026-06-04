"""Build side-by-side videos and an HTML index for visual rollout evals."""

from __future__ import annotations

import argparse
import html
import json
import subprocess
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval_dir", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--panel_width", type=int, default=720)
    parser.add_argument("--panel_height", type=int, default=456)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSansMono.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _fit_image(image: Image.Image, width: int, height: int) -> Image.Image:
    image = image.convert("RGB")
    scale = min(width / image.width, height / image.height)
    new_size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
    resized = image.resize(new_size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), (238, 240, 242))
    canvas.paste(resized, ((width - new_size[0]) // 2, (height - new_size[1]) // 2))
    return canvas


def _draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str) -> None:
    draw.text(xy, text, fill=(232, 235, 238), font=_font(18))


def _make_video(frame_dir: Path, output_path: Path, fps: int) -> bool:
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-framerate",
        str(fps),
        "-i",
        str(frame_dir / "frame_%06d.png"),
        "-pix_fmt",
        "yuv420p",
        "-vcodec",
        "libx264",
        str(output_path),
    ]
    return subprocess.run(cmd, check=False).returncode == 0 and output_path.is_file()


def _combine_episode(
    ep_dir: Path,
    fps: int,
    panel_width: int,
    panel_height: int,
    force: bool,
) -> Path | None:
    top_dir = ep_dir / "topdown_frames"
    if not top_dir.is_dir():
        top_dir = ep_dir / "global_frames"
    dual_dir = ep_dir / "dual_camera_frames"
    if not top_dir.is_dir() or not dual_dir.is_dir():
        return None

    output_video = ep_dir / "side_by_side_topdown_dual_camera.mp4"
    if output_video.is_file() and not force:
        return output_video

    top_frames = sorted(top_dir.glob("frame_*.png"))
    dual_frames = sorted(dual_dir.glob("frame_*.png"))
    frame_count = min(len(top_frames), len(dual_frames))
    if frame_count <= 0:
        return None

    out_dir = ep_dir / "side_by_side_frames"
    out_dir.mkdir(parents=True, exist_ok=True)
    for old_frame in out_dir.glob("frame_*.png"):
        old_frame.unlink()

    gutter = 10
    label_h = 34
    width = panel_width * 2 + gutter
    height = panel_height + label_h
    title_font = _font(18)
    for idx in range(frame_count):
        left = _fit_image(Image.open(top_frames[idx]), panel_width, panel_height)
        right = _fit_image(Image.open(dual_frames[idx]), panel_width, panel_height)
        canvas = Image.new("RGB", (width, height), (20, 22, 24))
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((0, 0, panel_width, label_h), fill=(18, 18, 18))
        draw.rectangle((panel_width + gutter, 0, width, label_h), fill=(18, 18, 18))
        draw.text((10, 7), "third-person topdown", fill=(232, 235, 238), font=title_font)
        draw.text((panel_width + gutter + 10, 7), "forklift left/right cameras", fill=(232, 235, 238), font=title_font)
        canvas.paste(left, (0, label_h))
        canvas.paste(right, (panel_width + gutter, label_h))
        canvas.save(out_dir / f"frame_{idx:06d}.png")

    if not _make_video(out_dir, output_video, fps):
        return None
    return output_video


def _rel(path: Path | None, root: Path) -> str:
    if path is None:
        return ""
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _fmt_float(value: Any, ndigits: int = 3) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.{ndigits}f}"
    except Exception:
        return html.escape(str(value))


def _bool_cell(value: Any) -> str:
    return "yes" if bool(value) else "no"


def _row_class(ep: dict[str, Any]) -> str:
    if ep.get("visual_clean_success"):
        return "ok"
    if ep.get("ever_dirty_insert") or ep.get("hard_lateral_high_disp"):
        return "warn"
    return "fail"


def _build_index(eval_dir: Path, side_by_side: dict[int, Path | None]) -> Path:
    summary = _read_json(eval_dir / "summary.json")
    episodes = list(summary.get("episodes", []))
    aggregate = summary.get("aggregate", {})
    steer_health = summary.get("steer_health", {})
    cfg = summary.get("eval_config", {})
    dual_cfg = cfg.get("dual_camera", {})

    cards = [
        ("episodes", len(episodes)),
        ("success", _fmt_float(aggregate.get("success_rate"), 3)),
        ("visual clean", _fmt_float(aggregate.get("visual_clean_success_rate"), 3)),
        ("insert", _fmt_float(aggregate.get("insert_rate"), 3)),
        ("dirty insert", _fmt_float(aggregate.get("dirty_insert_rate"), 3)),
        ("mean pallet disp", f"{_fmt_float(aggregate.get('mean_max_pallet_disp_xy_m'), 3)} m"),
        ("same-sign steer", _bool_cell(steer_health.get("same_sign_steer_collapse"))),
    ]

    rows = []
    for ep in episodes:
        ep_num = int(ep.get("episode", -1))
        ep_dir = eval_dir / f"episode_{ep_num:03d}"
        side_video = side_by_side.get(ep_num)
        links = []
        if side_video is not None:
            links.append(f'<a href="{html.escape(_rel(side_video, eval_dir))}">side-by-side</a>')
        for key, label in [
            ("kinematic_check_topdown_video", "topdown"),
            ("dual_camera_video", "dual camera"),
            ("global_video", "global"),
            ("metrics_csv", "metrics"),
            ("frame_metadata_jsonl", "metadata"),
        ]:
            value = ep.get(key)
            if value:
                links.append(f'<a href="{html.escape(_rel(Path(value), eval_dir))}">{html.escape(label)}</a>')
        rows.append(
            "<tr class=\"{row_class}\">"
            "<td>{episode:03d}</td>"
            "<td>{status}</td>"
            "<td>{steps}</td>"
            "<td>{init_y}</td>"
            "<td>{init_yaw}</td>"
            "<td>{steer}</td>"
            "<td>{insert}</td>"
            "<td>{disp}</td>"
            "<td>{links}</td>"
            "</tr>".format(
                row_class=_row_class(ep),
                episode=ep_num,
                status=html.escape(str(ep.get("done_reason", ""))),
                steps=int(ep.get("steps", 0)),
                init_y=_fmt_float(ep.get("init_signed_lateral_err_m"), 3),
                init_yaw=_fmt_float(ep.get("init_yaw_err_signed_deg"), 1),
                steer=_fmt_float(ep.get("mean_env_steer"), 3),
                insert=_fmt_float(ep.get("max_insert_depth_m"), 3),
                disp=_fmt_float(ep.get("max_pallet_disp_xy_m"), 3),
                links=" | ".join(links),
            )
        )

    cards_html = "\n".join(
        f"<div class=\"metric\"><span>{html.escape(str(name))}</span><strong>{html.escape(str(value))}</strong></div>"
        for name, value in cards
    )
    camera_text = html.escape(
        "HFOV={hfov} far={far} left_pos={left_pos} right_pos={right_pos} "
        "left_rpy={left_rpy} right_rpy={right_rpy} third_person={mode}".format(
            hfov=dual_cfg.get("hfov_deg"),
            far=dual_cfg.get("far_clip_m"),
            left_pos=dual_cfg.get("left_pos_local"),
            right_pos=dual_cfg.get("right_pos_local"),
            left_rpy=dual_cfg.get("left_rpy_local_deg"),
            right_rpy=dual_cfg.get("right_rpy_local_deg"),
            mode=cfg.get("third_person_mode"),
        )
    )
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>v5 visual rollout eval</title>
  <style>
    :root {{
      color-scheme: light;
      --text: #171a1f;
      --muted: #5f6876;
      --line: #d8dde4;
      --bg: #f6f7f9;
      --ok: #e8f4ec;
      --warn: #fff4df;
      --fail: #fdebec;
    }}
    body {{
      margin: 0;
      font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--text);
      background: var(--bg);
    }}
    main {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 28px 20px 44px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 24px;
      font-weight: 720;
      letter-spacing: 0;
    }}
    p {{
      margin: 4px 0 20px;
      color: var(--muted);
    }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
      gap: 10px;
      margin: 18px 0 20px;
    }}
    .metric {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px 12px;
      background: white;
    }}
    .metric span {{
      display: block;
      color: var(--muted);
      font-size: 12px;
    }}
    .metric strong {{
      display: block;
      margin-top: 3px;
      font-size: 17px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: white;
      border: 1px solid var(--line);
    }}
    th, td {{
      padding: 8px 10px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
      white-space: nowrap;
    }}
    th {{
      font-size: 12px;
      color: var(--muted);
      background: #eef1f4;
    }}
    tr.ok td {{ background: var(--ok); }}
    tr.warn td {{ background: var(--warn); }}
    tr.fail td {{ background: var(--fail); }}
    a {{
      color: #1f5fbf;
      text-decoration: none;
    }}
    a:hover {{ text-decoration: underline; }}
    .scroll {{ overflow-x: auto; }}
  </style>
</head>
<body>
<main>
  <h1>v5 visual rollout eval</h1>
  <p>{camera_text}</p>
  <div class="metrics">
    {cards_html}
  </div>
  <div class="scroll">
    <table>
      <thead>
        <tr>
          <th>episode</th>
          <th>done reason</th>
          <th>steps</th>
          <th>init y m</th>
          <th>init yaw deg</th>
          <th>mean steer</th>
          <th>max insert m</th>
          <th>max pallet disp m</th>
          <th>videos / data</th>
        </tr>
      </thead>
      <tbody>
        {"".join(rows)}
      </tbody>
    </table>
  </div>
</main>
</body>
</html>
"""
    output = eval_dir / "index.html"
    output.write_text(html_text, encoding="utf-8")
    return output


def main() -> None:
    args = _parse_args()
    eval_dir = args.eval_dir.resolve()
    if not eval_dir.is_dir():
        raise SystemExit(f"eval_dir does not exist: {eval_dir}")

    side_by_side: dict[int, Path | None] = {}
    for ep_dir in sorted(eval_dir.glob("episode_*")):
        if not ep_dir.is_dir():
            continue
        try:
            ep_num = int(ep_dir.name.split("_")[-1])
        except ValueError:
            continue
        side_by_side[ep_num] = _combine_episode(
            ep_dir,
            fps=int(args.fps),
            panel_width=int(args.panel_width),
            panel_height=int(args.panel_height),
            force=bool(args.force),
        )
    index = _build_index(eval_dir, side_by_side)
    print(f"wrote {index}")
    print(f"side-by-side videos: {sum(1 for value in side_by_side.values() if value is not None)}")


if __name__ == "__main__":
    main()
