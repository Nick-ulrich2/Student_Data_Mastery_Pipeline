from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import streamlit as st

THEMES: dict[str, dict[str, str]] = {
    "light": {
        "background": "#F8FAFC",
        "secondary_background": "#E2E8F0",
        "surface": "#FFFFFF",
        "text": "#172033",
        "muted_text": "#475569",
        "primary": "#1D4ED8",
        "primary_contrast": "#FFFFFF",
        "secondary": "#CBD5E1",
        "border": "#94A3B8",
        "success": "#166534",
        "warning": "#92400E",
        "error": "#991B1B",
        "chart_grid": "#CBD5E1",
        "chart_blue": "#1D4ED8",
        "chart_orange": "#C2410C",
        "chart_green": "#15803D",
        "heatmap_cmap": "Blues",
    },
    "dark": {
        "background": "#0F172A",
        "secondary_background": "#1E293B",
        "surface": "#172033",
        "text": "#F8FAFC",
        "muted_text": "#CBD5E1",
        "primary": "#60A5FA",
        "primary_contrast": "#0F172A",
        "secondary": "#334155",
        "border": "#64748B",
        "success": "#86EFAC",
        "warning": "#FCD34D",
        "error": "#FCA5A5",
        "chart_grid": "#475569",
        "chart_blue": "#60A5FA",
        "chart_orange": "#FB923C",
        "chart_green": "#4ADE80",
        "heatmap_cmap": "mako",
    },
}


def get_theme_name() -> str:
    return st.session_state.get("theme", "light")


def set_theme(theme_name: str) -> None:
    st.session_state.theme = theme_name if theme_name in THEMES else "light"


def get_theme(theme_name: str | None = None) -> dict[str, str]:
    return THEMES[theme_name or get_theme_name()]


def apply_theme(theme_name: str | None = None) -> None:
    theme = get_theme(theme_name)
    st.markdown(
        f"""
        <style>
        :root {{
            --app-background: {theme['background']};
            --app-secondary-background: {theme['secondary_background']};
            --app-surface: {theme['surface']};
            --app-text: {theme['text']};
            --app-muted-text: {theme['muted_text']};
            --app-primary: {theme['primary']};
            --app-border: {theme['border']};
            --app-success: {theme['success']};
            --app-warning: {theme['warning']};
            --app-error: {theme['error']};
        }}
        .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {{
            background-color: var(--app-background);
            color: var(--app-text);
        }}
        [data-testid="stSidebar"], [data-testid="stSidebarContent"] {{
            background-color: var(--app-secondary-background);
        }}
        h1, h2, h3, h4, h5, h6, p, label, span, div, li {{
            color: var(--app-text);
        }}
        [data-testid="stCaptionContainer"], .stCaption {{
            color: var(--app-muted-text) !important;
        }}
        [data-testid="stDivider"] {{
            border-color: var(--app-border) !important;
        }}
        div[data-testid="stButton"] button {{
            background-color: var(--app-surface);
            color: var(--app-text);
            border: 1px solid var(--app-border);
        }}
        div[data-testid="stButton"] button:hover {{
            background-color: var(--app-primary);
            color: {theme['primary_contrast']};
            border-color: var(--app-primary);
        }}
        div[data-testid="stButton"] button:disabled {{
            background-color: var(--app-secondary-background);
            color: var(--app-muted-text);
            border-color: var(--app-border);
            opacity: 1;
        }}
        input, textarea, [data-baseweb="select"] > div, [data-baseweb="popover"] {{
            background-color: var(--app-surface) !important;
            color: var(--app-text) !important;
            border-color: var(--app-border) !important;
        }}
        [data-testid="stMetric"] {{
            background-color: var(--app-surface);
            border: 1px solid var(--app-border);
            border-radius: 0.5rem;
            padding: 0.75rem;
        }}
        [data-testid="stAlert"] {{
            background-color: var(--app-surface);
            border-color: var(--app-border);
        }}
        [data-testid="stDataFrame"] {{
            border: 1px solid var(--app-border);
        }}
        [data-testid="stRadio"] label, [data-testid="stSelectbox"] label {{
            color: var(--app-text) !important;
        }}
        @media (max-width: 640px) {{
            [data-testid="stSidebarContent"] {{ padding: 0.75rem; }}
            div[data-testid="stButton"] button {{ min-height: 2.5rem; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def configure_matplotlib_theme(theme_name: str | None = None) -> None:
    theme = get_theme(theme_name)
    plt.rcParams.update(
        {
            "figure.facecolor": theme["background"],
            "axes.facecolor": theme["surface"],
            "axes.edgecolor": theme["border"],
            "axes.labelcolor": theme["text"],
            "axes.titlecolor": theme["text"],
            "xtick.color": theme["text"],
            "ytick.color": theme["text"],
            "text.color": theme["text"],
            "grid.color": theme["chart_grid"],
            "legend.facecolor": theme["surface"],
            "legend.edgecolor": theme["border"],
        }
    )


def theme_chart_colors(theme_name: str | None = None) -> dict[str, Any]:
    theme = get_theme(theme_name)
    return {
        "blue": theme["chart_blue"],
        "orange": theme["chart_orange"],
        "green": theme["chart_green"],
        "grid": theme["chart_grid"],
        "text": theme["text"],
        "heatmap_cmap": theme["heatmap_cmap"],
    }
