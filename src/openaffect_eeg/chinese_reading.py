"""Render the Chinese manuscript sources into a readable XeLaTeX document."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path


class ChineseReadingError(RuntimeError):
    """Raised when the local Chinese reading PDF cannot be generated."""


_COMMENT = re.compile(r"<!--.*?-->")
_CITATION = re.compile(r"\[@([^\]]+)\]")
_BOLD = re.compile(r"\*\*(.+?)\*\*")
_CODE = re.compile(r"`([^`]+)`")
_PAREN_MATH = re.compile(re.escape(r"\(") + r"(.+?)" + re.escape(r"\)"))
_DOLLAR_MATH = re.compile(r"\$([^$]+)\$")
_FIGURE = re.compile(r"(?:^|[^\u4e00-\u9fff])图\s*([1-5])\s*[：:]")
_TABLE_DIVIDER = re.compile(r"^:?-{3,}:?$")


def _reading_only(markdown: str, *, supplement: bool) -> str:
    marker = r"\n### J\.\d+ 补充图片审批" if supplement else r"\n## 稿件内部证据说明"
    return re.split(marker, markdown, maxsplit=1)[0]


def _escape_latex(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in value)


def _inline_tex(value: str) -> str:
    """Translate the small, controlled Markdown subset used by the Chinese draft."""

    protected: list[str] = []

    def protect(replacement: str) -> str:
        token = f"@@OPENAFFECT{len(protected)}@@"
        protected.append(replacement)
        return token

    value = _COMMENT.sub("", value)
    value = _PAREN_MATH.sub(
        lambda match: protect("$" + match.group(1) + "$"), value
    )
    value = _DOLLAR_MATH.sub(
        lambda match: protect("$" + match.group(1) + "$"), value
    )
    value = _CITATION.sub(
        lambda match: protect(
            r"[\textup{文献: }\texttt{"
            + _escape_latex(match.group(1).replace("@", ""))
            + "}]"
        ),
        value,
    )

    def render_code(match: re.Match[str]) -> str:
        code = match.group(1)
        if "/" in code:
            return protect(r"\url{" + code + "}")
        escaped = _escape_latex(code)
        if len(code) > 18:
            escaped = escaped.replace(r"\_", r"\_\allowbreak{}").replace(
                "-", r"-\allowbreak{}"
            )
        return protect(r"\texttt{" + escaped + "}")

    value = _CODE.sub(
        render_code,
        value,
    )
    value = _BOLD.sub(
        lambda match: protect(r"\textbf{" + _escape_latex(match.group(1)) + "}"),
        value,
    )
    value = _escape_latex(value)
    for index, replacement in enumerate(protected):
        value = value.replace(f"@@OPENAFFECT{index}@@", replacement)
    return value


def _table_tex(lines: list[str]) -> str:
    rows = [
        [cell.strip() for cell in line.strip().strip("|").split("|")]
        for line in lines
    ]
    if len(rows) < 3 or not all(_TABLE_DIVIDER.fullmatch(cell) for cell in rows[1]):
        return "\n".join(_inline_tex(line) + r"\\" for line in lines)

    widths = [0.90 / len(rows[0])] * len(rows[0])
    if rows[0] == ["数据集", "指标", "全局选择", "条件选择", "差值 [区间]"]:
        widths = [0.18, 0.08, 0.12, 0.12, 0.40]
    elif rows[0] == ["任务", "设置", "EC-P", "PC-P", "EC-PC", "配对 percentile 区间"]:
        widths = [0.14, 0.18, 0.11, 0.11, 0.11, 0.25]
    columns = "".join(r">{\raggedright\arraybackslash}p{" + f"{width:.3f}" + r"\linewidth}"
                      for width in widths)
    output = [
        r"\small",
        r"\begin{longtable}{@{}" + columns + r"@{}}",
        r"\toprule",
    ]
    output.append(" & ".join(r"\textbf{" + _inline_tex(cell) + "}" for cell in rows[0]) + r"\\")
    output.extend([r"\midrule", r"\endfirsthead", r"\toprule"])
    output.append(" & ".join(r"\textbf{" + _inline_tex(cell) + "}" for cell in rows[0]) + r"\\")
    output.extend([r"\midrule", r"\endhead"])
    output.extend(" & ".join(_inline_tex(cell) for cell in row) + r"\\" for row in rows[2:])
    output.extend([r"\bottomrule", r"\end{longtable}", r"\normalsize"])
    return "\n".join(output)


def _body_tex(
    markdown: str,
    *,
    supplement: bool,
    figure_directory: Path,
) -> tuple[str, str | None]:
    lines = markdown.replace("\r\n", "\n").split("\n")
    output: list[str] = []
    title: str | None = None
    index = 0
    list_environment: str | None = None

    def close_list() -> None:
        nonlocal list_environment
        if list_environment is not None:
            output.append(r"\end{" + list_environment + "}")
            list_environment = None

    while index < len(lines):
        line = lines[index].strip()
        if not line:
            close_list()
            index += 1
            continue
        if line.startswith("|") and line.endswith("|"):
            close_list()
            table: list[str] = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                table.append(lines[index].strip())
                index += 1
            output.append(_table_tex(table))
            continue
        if line == r"\[":
            close_list()
            display = [line]
            index += 1
            while index < len(lines):
                display.append(lines[index])
                if lines[index].strip() == r"\]":
                    index += 1
                    break
                index += 1
            output.append("\n".join(display))
            continue
        heading = re.fullmatch(r"(#{1,3})\s+(.+)", line)
        if heading is not None:
            close_list()
            level, text = heading.groups()
            if level == "#" and not supplement and title is None:
                title = text
            elif level == "#":
                output.extend([r"\clearpage", r"\section*{" + _inline_tex(text) + "}"])
                output.append(r"\addcontentsline{toc}{section}{" + _inline_tex(text) + "}")
            else:
                command = "section*" if level == "##" else "subsection*"
                output.append("\\" + command + "{" + _inline_tex(text) + "}")
                if level == "##":
                    output.append(r"\addcontentsline{toc}{section}{" + _inline_tex(text) + "}")
            index += 1
            continue
        if line.startswith(">"):
            close_list()
            quote_lines: list[str] = []
            while index < len(lines) and lines[index].strip().startswith(">"):
                quote_lines.append(lines[index].strip().lstrip(">").strip())
                index += 1
            output.extend([r"\begin{quote}\small", r"\\".join(_inline_tex(item) for item in quote_lines), r"\end{quote}"])
            continue
        ordered = re.fullmatch(r"\d+\.\s+(.+)", line)
        unordered = re.fullmatch(r"[-*]\s+(.+)", line)
        if ordered is not None or unordered is not None:
            environment = "enumerate" if ordered is not None else "itemize"
            if list_environment != environment:
                close_list()
                output.append(r"\begin{" + environment + "}")
                list_environment = environment
            output.append(r"\item " + _inline_tex((ordered or unordered).group(1)))
            index += 1
            continue

        close_list()
        figure = _FIGURE.search(line)
        if figure is not None:
            output.extend(
                [
                    r"\begin{figure}[htbp]",
                    r"\centering",
                    r"\includegraphics[width=\linewidth]{"
                    + (figure_directory / f"figure{figure.group(1)}.pdf").as_posix()
                    + "}",
                    r"\par\small\raggedright " + _inline_tex(line),
                    r"\end{figure}",
                ]
            )
        else:
            output.append(_inline_tex(line) + "\n")
        index += 1
    close_list()
    return "\n\n".join(output), title


def render_chinese_reading_tex(
    main_markdown: str,
    supplement_markdown: str,
    *,
    figure_directory: Path,
) -> str:
    """Create the self-contained LaTeX source for the Chinese reading PDF."""

    main, title = _body_tex(
        _reading_only(main_markdown, supplement=False),
        supplement=False,
        figure_directory=figure_directory,
    )
    supplement, _ = _body_tex(
        _reading_only(supplement_markdown, supplement=True),
        supplement=True,
        figure_directory=figure_directory,
    )
    if title is None:
        raise ChineseReadingError("Chinese main manuscript must begin with one H1 title")
    main = main.replace(
        "OPENAFFECTMATCHEDGRIDFIGURE",
        r"\begin{figure}[ht]\centering"
        r"\includegraphics[width=0.97\linewidth]{../generated/reviewer_revision_v12/matched_increment_heatmap.pdf}"
        r"\caption{完整匹配 EEG 增量网格。数值与色标均为 CCC 差值乘以 1000，四舍五入的零不表示精确相等。"
        r"全部 12 个网格和 300 个单元均保留；每格为五个身份块内配对差值的平均。完整条件区间见随附 CSV。}"
        r"\end{figure}",
    )

    preamble = r"""\documentclass[UTF8,11pt,a4paper]{ctexart}
\usepackage[margin=2.25cm]{geometry}
\usepackage{amsmath,amssymb,booktabs,longtable,array,graphicx,url,hyperref}
\setlength{\parindent}{2em}
\setlength{\parskip}{0.45em}
\setlength{\tabcolsep}{2pt}
\clubpenalty=10000
\widowpenalty=10000
\renewcommand{\arraystretch}{1.22}
\sloppy
\hypersetup{colorlinks=true,linkcolor=black,urlcolor=blue,citecolor=black}
\begin{document}
\pagestyle{plain}
\title{"""
    return (
        preamble
        + _inline_tex(title)
        + r"""}
\author{中文阅读版：与匿名英文投稿稿同步}
\date{\today}
\maketitle
\begingroup
\small
\setlength{\parskip}{0pt}
\tableofcontents
\endgroup
\clearpage
"""
        + main
        + "\n"
        + supplement
        + "\n\\end{document}\n"
    )


def build_chinese_reading_pdf(
    project_root: str | Path,
    *,
    output: str | Path | None = None,
) -> Path:
    """Compile three passes so a new contents page has stable page references."""

    root = Path(project_root).resolve()
    paper = root / "paper"
    destination = (
        Path(output).resolve()
        if output is not None
        else paper / "chinese_reading" / "OpenAffect-EEG-zh.pdf"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    tex_path = destination.with_suffix(".tex")
    manuscript = (paper / "manuscript_zh.md").read_text(encoding="utf-8")
    supplement = (paper / "supplement_zh.md").read_text(encoding="utf-8")
    generated = {
        "<!-- resource-decomposition-v10-table -->": "generated/evidence_argument_v10/decomposition_zh.md",
        "<!-- primary-v10-table -->": "generated/evidence_argument_v10/primary_zh.md",
        "<!-- full-sensitivity-v9-tables -->": "generated/final_closure_v9/tables_zh.md",
    }
    for marker, relative in generated.items():
        if marker in manuscript or marker in supplement:
            table = (paper / relative).read_text(encoding="utf-8")
            manuscript = manuscript.replace(marker, table)
            supplement = supplement.replace(marker, table)
    marker = "<!-- paired-added-value-table -->"
    if marker in manuscript:
        paired = paper / "generated/identity_exposure_v2/complete/paired_added_value_reading_zh.md"
        manuscript = manuscript.replace(marker, paired.read_text(encoding="utf-8"))
    marker = "<!-- revision-v3-calibration-table -->"
    if marker in manuscript:
        revised = paper / "generated/competitive_revision_v3/calibration_sensitivity_zh.md"
        manuscript = manuscript.replace(marker, revised.read_text(encoding="utf-8"))
    marker = "<!-- deployment-utility-v5-table -->"
    if marker in manuscript:
        utility = paper / "generated/deployment_utility_v5/selection_utility_zh.md"
        manuscript = manuscript.replace(marker, utility.read_text(encoding="utf-8"))
    tex_path.write_text(
        render_chinese_reading_tex(
            manuscript,
            supplement,
            figure_directory=Path("../neurips2026/figures"),
        ),
        encoding="utf-8",
    )
    xelatex = shutil.which("xelatex")
    if xelatex is None:
        raise ChineseReadingError("xelatex is required to build the Chinese reading PDF")
    command = [
        xelatex,
        "-interaction=nonstopmode",
        "-halt-on-error",
        "-output-directory",
        str(destination.parent),
        tex_path.name,
    ]
    for _ in range(3):
        completed = subprocess.run(
            command,
            cwd=destination.parent,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if completed.returncode != 0:
            tail = (completed.stdout + completed.stderr)[-3000:]
            raise ChineseReadingError(f"XeLaTeX failed while building {tex_path}:\n{tail}")
    if not destination.is_file():
        raise ChineseReadingError("XeLaTeX completed without producing the Chinese PDF")
    return destination
