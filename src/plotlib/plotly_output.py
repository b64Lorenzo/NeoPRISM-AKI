"""Shared Plotly output for offline-capable interactive artifacts."""

from pathlib import Path


def plotly_template(theme: str = "light") -> str:
    """Resolve the configured light/dark theme to a supported Plotly template."""
    templates = {"light": "plotly_white", "dark": "plotly_dark"}
    try:
        return templates[theme]
    except KeyError as error:
        raise ValueError("theme must be light or dark") from error


def write_plotly_html(
    figure, path: str | Path, *, post_script: str | list[str] | None = None,
    head: str = "", before_plot: str = "", after_plot: str = "",
) -> Path:
    """Write an interactive HTML plot with one shared local Plotly.js bundle."""
    destination = Path(path)
    if destination.suffix.casefold() != ".html":
        raise ValueError("Plotly output path must end with .html")
    try:
        import plotly.io as pio
    except ImportError as error:
        raise RuntimeError("Plotly charts require Plotly; install dependencies from requirements.txt") from error

    destination.parent.mkdir(parents=True, exist_ok=True)
    pio.write_html(
        figure,
        file=str(destination),
        include_plotlyjs="directory",
        full_html=True,
        auto_open=False,
        config={"responsive": True},
        post_script=post_script,
    )
    if head or before_plot or after_plot:
        document = destination.read_text(encoding="utf-8")
        document = document.replace("</head>", f"{head}</head>", 1)
        document = document.replace("<body>", f"<body>{before_plot}", 1)
        document = document.replace("</body>", f"{after_plot}</body>", 1)
        destination.write_text(document, encoding="utf-8")
    return destination
