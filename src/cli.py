import sys
import os
import base64
import asyncio
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

console = Console()


def _supports_images() -> bool:
    """Check if the terminal supports iTerm2 inline image protocol."""
    # iTerm2, VS Code, WezTerm
    if os.environ.get("TERM_PROGRAM") in ("iTerm.app", "WezTerm"):
        return True
    if os.environ.get("TERM_PROGRAM_VERSION"):
        return True  # iTerm2 sets this
    # VS Code integrated terminal also supports it
    if "VSCODE" in os.environ.get("TERM_PROGRAM", ""):
        return True
    if os.environ.get("ITERM_SESSION_ID"):
        return True
    return False


def _render_image(path: str) -> str | None:
    """Render an image in the terminal. Returns the escape sequence or None if unsupported."""
    if not _supports_images():
        return None

    try:
        with open(path, "rb") as f:
            raw = f.read()
        encoded = base64.b64encode(raw).decode()
        name = base64.b64encode(os.path.basename(path).encode()).decode()
        size = len(raw)

        # iTerm2 inline image protocol
        return f"\n\x1b]1337;File=name={name};size={size};inline=1;width=600px:{encoded}\x07\n"
    except Exception:
        return None


def _build_config() -> dict:
    import uuid

    return {"configurable": {"thread_id": f"session-{uuid.uuid4().hex[:8]}"}}


async def run_cli(agent) -> None:
    config = _build_config()
    has_image_support = _supports_images()

    console.print(
        Panel.fit(
            "[bold cyan]数据分析 AI Agent[/bold cyan] v2\n"
            "[dim]DeepSeek + LangChain | Python REPL | 文件分析 | 图表展示 | 数据导出[/dim]\n\n"
            "[dim]工具: python_repl | read_file | list_files | show_image | export_data | search[/dim]\n"
            f"[dim]图片展示: {'支持' if has_image_support else '不支持（请用 iTerm2）'}[/dim]",
            border_style="cyan",
        )
    )
    console.print("[dim]输入 q 退出, 输入 /clear 清空对话记忆[/dim]\n")

    while True:
        try:
            user_input = input("📊 请输入分析需求: ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]已退出。[/dim]")
            break

        if user_input.lower() == "q":
            console.print("[dim]已退出。[/dim]")
            break
        if user_input == "/clear":
            config = _build_config()
            console.print("[dim]✓ 对话记忆已清空[/dim]\n")
            continue
        if not user_input:
            continue

        console.print()

        tool_phase = False
        pending_images: list[str] = []

        try:
            async for event in agent.astream_events(
                {"messages": [{"role": "user", "content": user_input}]},
                config=config,
                version="v2",
            ):
                kind = event.get("event")

                if kind == "on_chat_model_stream":
                    content = event["data"]["chunk"].content
                    if isinstance(content, str) and content:
                        if tool_phase:
                            tool_phase = False
                            console.print()
                        console.print(content, end="", highlight=False)

                elif kind == "on_tool_end":
                    output = event["data"].get("output", "")
                    if isinstance(output, str) and "@@IMAGE:" in output:
                        # Extract image paths from show_image output
                        for part in output.split("@@"):
                            if part.startswith("IMAGE:") and part.endswith("@@"):
                                pending_images.append(part[6:-1])
                    console.print(f" [dim]✓[/dim]")

            console.print()

            # Render any images generated this turn
            for img_path in pending_images:
                img_render = _render_image(img_path)
                if img_render:
                    console.print(img_render, highlight=False)
                else:
                    console.print(f"[dim]📈 图表已保存: {img_path}[/dim]")
            if pending_images:
                console.print()

        except Exception as e:
            console.print(f"\n[red]错误: {e}[/red]\n")


def start_cli(agent) -> None:
    asyncio.run(run_cli(agent))
