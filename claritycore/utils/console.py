# Copyright (c) Aman Urumbekov and other contributors.
"""Rich-powered console utilities for ClarityCore."""

from typing import Any

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.theme import Theme

# ═══════════════════════════════════════════════════════════════════════════════
# THEME & CONSOLE
# ═══════════════════════════════════════════════════════════════════════════════

# Sci-fi color palette: Electric cyan, neon magenta, terminal green
CLARITY_THEME = Theme(
    {
        # Status colors
        "info": "#00d4ff",
        "warning": "#ffaf00",
        "error": "bold #ff3366",
        "success": "bold #00ff9f",
        "highlight": "bold #ff00ff",
        "muted": "dim white",
        # Data display
        "metric.name": "#00d4ff",
        "metric.value": "bold white",
        "config.key": "#00ff9f",
        "config.value": "white",
        # Banner
        "banner.line1": "bold #00ffff",
        "banner.line2": "bold #00e5ff",
        "banner.line3": "bold #00ccff",
        "banner.line4": "#33bbff",
        "banner.line5": "#6699ff",
        "banner.line6": "#9966ff",
        # Progress
        "bar.complete": "#00d4ff",
        "bar.finished": "#00ff9f",
    }
)

console = Console(theme=CLARITY_THEME, highlight=True, markup=True)


# ═══════════════════════════════════════════════════════════════════════════════
# LOGURU + RICH INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════


def get_rich_handler(level: str = "INFO") -> RichHandler:
    """
    Creates a RichHandler for use with loguru.

    Usage:
        from loguru import logger
        logger.remove()
        logger.add(get_rich_handler(), format="{message}", level="DEBUG")
    """
    return RichHandler(
        console=console,
        show_time=True,
        show_level=True,
        show_path=False,
        rich_tracebacks=True,
        tracebacks_show_locals=True,
        markup=True,
        log_time_format="[%H:%M:%S]",
        level=level,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PRETTY PRINT FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════


def print_info(message: str, **kwargs) -> None:
    """Print an info message."""
    console.print(f"[info]│[/info] {message}", **kwargs)


def print_success(message: str, **kwargs) -> None:
    """Print a success message."""
    console.print(f"[success]✓[/success] {message}", **kwargs)


def print_warning(message: str, **kwargs) -> None:
    """Print a warning message."""
    console.print(f"[warning]⚠[/warning] {message}", **kwargs)


def print_error(message: str, **kwargs) -> None:
    """Print an error message."""
    console.print(f"[error]✗[/error] {message}", **kwargs)


def print_panel(
    content: str,
    title: str | None = None,
    subtitle: str | None = None,
    border_style: str = "#00d4ff",
    **kwargs,
) -> None:
    """Print content in a styled panel."""
    console.print(
        Panel(
            content,
            title=title,
            subtitle=subtitle,
            border_style=border_style,
            padding=(0, 2),
            **kwargs,
        )
    )


def print_rule(title: str = "", style: str = "#00d4ff", **kwargs) -> None:
    """Print a horizontal rule with optional title."""
    console.rule(title, style=style, **kwargs)


# ═══════════════════════════════════════════════════════════════════════════════
# TABLE UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════


def create_table(
    title: str | None = None,
    caption: str | None = None,
    show_header: bool = True,
    header_style: str = "bold #00d4ff",
    border_style: str = "#444444",
    **kwargs,
) -> Table:
    """Create a pre-styled Rich table."""
    return Table(
        title=title,
        caption=caption,
        show_header=show_header,
        header_style=header_style,
        border_style=border_style,
        title_style="bold",
        **kwargs,
    )


def print_metrics(metrics: dict[str, Any], title: str = "◈ Metrics") -> None:
    """Print metrics in a formatted table."""
    table = create_table(title=title, box=None, padding=(0, 2))
    table.add_column("", style="metric.name", no_wrap=True)
    table.add_column("", style="metric.value", justify="right")

    for name, value in metrics.items():
        if isinstance(value, float):
            table.add_row(name, f"{value:.6f}")
        else:
            table.add_row(name, str(value))

    console.print(table)


def print_config(config: dict[str, Any], title: str = "◈ Configuration") -> None:
    """Print configuration in a formatted table."""
    table = create_table(title=title, box=None, padding=(0, 2))
    table.add_column("", style="config.key", no_wrap=True)
    table.add_column("", style="config.value")

    def _flatten(d: dict, parent_key: str = "") -> list[tuple[str, Any]]:
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}.{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(_flatten(v, new_key))
            else:
                items.append((new_key, v))
        return items

    for key, value in _flatten(config):
        table.add_row(key, str(value))

    console.print(table)


# ═══════════════════════════════════════════════════════════════════════════════
# PROGRESS BARS
# ═══════════════════════════════════════════════════════════════════════════════


def create_training_progress() -> Progress:
    """Create a progress bar optimized for training loops."""
    return Progress(
        SpinnerColumn(spinner_name="dots", style="#00d4ff"),
        TextColumn("[bold white]{task.description}"),
        BarColumn(bar_width=40, style="#333333", complete_style="bar.complete", finished_style="bar.finished"),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TextColumn("•", style="dim"),
        TimeElapsedColumn(),
        TextColumn("→", style="dim"),
        TimeRemainingColumn(),
        console=console,
        expand=False,
    )


def create_data_progress(description: str = "Loading") -> Progress:
    """Create a progress bar for data loading/processing."""
    return Progress(
        SpinnerColumn(spinner_name="dots2", style="#ff00ff"),
        TextColumn(f"[bold]{description}"),
        BarColumn(bar_width=30, style="#333333", complete_style="#ff00ff"),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# BANNER
# ═══════════════════════════════════════════════════════════════════════════════

# DOS Rebel font - https://manytools.org/hacker-tools/ascii-banner/
_BANNER_LINES = [
    " ██████╗██╗      █████╗ ██████╗ ██╗████████╗██╗   ██╗ ██████╗ ██████╗ ██████╗ ███████╗",
    "██╔════╝██║     ██╔══██╗██╔══██╗██║╚══██╔══╝╚██╗ ██╔╝██╔════╝██╔═══██╗██╔══██╗██╔════╝",
    "██║     ██║     ███████║██████╔╝██║   ██║    ╚████╔╝ ██║     ██║   ██║██████╔╝█████╗  ",
    "██║     ██║     ██╔══██║██╔══██╗██║   ██║     ╚██╔╝  ██║     ██║   ██║██╔══██╗██╔══╝  ",
    "╚██████╗███████╗██║  ██║██║  ██║██║   ██║      ██║   ╚██████╗╚██████╔╝██║  ██║███████╗",
    " ╚═════╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝   ╚═╝      ╚═╝    ╚═════╝ ╚═════╝ ╚═╝  ╚═╝╚══════╝",
]

# ANSI true color gradient: cyan → blue → purple (sci-fi aesthetic)
_ANSI_GRADIENT = [
    "\033[1m\033[38;2;0;255;255m",    # Bold cyan
    "\033[1m\033[38;2;0;229;255m",    # Bold light cyan
    "\033[38;2;0;204;255m",           # Sky blue
    "\033[38;2;51;187;255m",          # Light blue
    "\033[38;2;102;153;255m",         # Periwinkle
    "\033[38;2;153;102;255m",         # Purple
]
_ANSI_RESET = "\033[0m"
_ANSI_DIM = "\033[2m"


def print_banner(version: str | None = None, subtitle: str | None = None) -> None:
    """
    Print the ClarityCore banner with a sci-fi gradient effect.

    Uses direct ANSI codes for the ASCII art to avoid Rich's Unicode width
    calculation issues with block characters.

    Args:
        version: Optional version string to display.
        subtitle: Optional subtitle/tagline.
    """
    import sys

    print(file=sys.stderr)  # Blank line

    # Print gradient banner using ANSI (bypasses Rich's width issues with Unicode)
    for i, line in enumerate(_BANNER_LINES):
        color = _ANSI_GRADIENT[i % len(_ANSI_GRADIENT)]
        print(f"{color}{line}{_ANSI_RESET}", file=sys.stderr)

    # Tagline with dim styling
    tagline = subtitle or "Next-generation toolkit for Low-level Vision"
    print(file=sys.stderr)
    print(f"{_ANSI_DIM}{tagline}{_ANSI_RESET}", file=sys.stderr)
    print(file=sys.stderr)


def print_startup_info(
    version: str | None = None,
    device: str | None = None,
    config_path: str | None = None,
) -> None:
    """Print startup information in a sci-fi styled panel."""
    from importlib.metadata import version as get_version

    v = version or get_version("claritycore")

    lines = []
    lines.append(f"[#00d4ff]VERSION[/]   [bold white]{v}[/]")

    if device:
        lines.append(f"[#00d4ff]DEVICE[/]    [bold white]{device}[/]")
    if config_path:
        lines.append(f"[#00d4ff]CONFIG[/]    [bold white]{config_path}[/]")

    content = "\n".join(lines)
    console.print(
        Panel(
            content,
            title="[bold #00ffff]◈ CLARITYCORE ◈[/]",
            border_style="#00d4ff",
            padding=(1, 3),
        )
    )


# ═══════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ═══════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Console
    "console",
    "CLARITY_THEME",
    # Logging
    "get_rich_handler",
    # Print functions
    "print_info",
    "print_success",
    "print_warning",
    "print_error",
    "print_panel",
    "print_rule",
    # Tables
    "create_table",
    "print_metrics",
    "print_config",
    # Progress
    "create_training_progress",
    "create_data_progress",
    # Banner
    "print_banner",
    "print_startup_info",
]
