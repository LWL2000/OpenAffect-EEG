from pathlib import Path

from openaffect_eeg.chinese_reading import render_chinese_reading_tex


def test_render_chinese_reading_tex_keeps_structure_tables_and_figures() -> None:
    main = """# 中文主稿标题

## 摘要

带有 **强调** 与 `code_name` 的正文。<!-- claims: C01 -->

**图 1：评测契约。** 图注说明。

| 项目 | 结果 |
|---|---:|
| 样本 | 279 |
"""
    supplement = """# 中文补充材料

## A. 补充方法

1. 第一项
2. 第二项
"""

    rendered = render_chinese_reading_tex(
        main,
        supplement,
        figure_directory=Path("../neurips2026/figures"),
    )

    assert "\\title{中文主稿标题}" in rendered
    assert "\\section*{摘要}" in rendered
    assert "\\addcontentsline{toc}{section}{摘要}" in rendered
    assert "\\addcontentsline{toc}{section}{A. 补充方法}" in rendered
    assert "\\textbf{强调}" in rendered
    assert "\\texttt{code\\_name}" in rendered
    assert "\\includegraphics[width=\\linewidth]{../neurips2026/figures/figure1.pdf}" in rendered
    assert "\\begin{longtable}" in rendered
    assert "279" in rendered
    assert "claims:" not in rendered
    assert "\\begin{enumerate}" in rendered
    figure = rendered.split(r"\begin{figure}", 1)[1].split(r"\end{figure}", 1)[0]
    assert "图注说明" in figure
    assert rendered.count("图注说明") == 1


def test_render_chinese_reading_tex_removes_every_citation_marker() -> None:
    rendered = render_chinese_reading_tex(
        "# 标题\n\n带参考文献 [@first2026; @second2025] 的正文。",
        "# 补充\n\n说明。",
        figure_directory=Path("../neurips2026/figures"),
    )

    assert "@first2026" not in rendered
    assert "@second2025" not in rendered
    assert "first2026; second2025" in rendered


def test_render_chinese_reading_tex_preserves_inline_math() -> None:
    rendered = render_chinese_reading_tex(
        "# 标题\n\n残差为 \\(\\mathbf{r}_{i,s}\\)。",
        "# 补充\n\n说明。",
        figure_directory=Path("../neurips2026/figures"),
    )

    assert "$\\mathbf{r}_{i,s}$" in rendered
    assert "\\textbackslash{}(\\textbackslash{}mathbf" not in rendered


def test_render_chinese_reading_tex_allows_long_code_paths_to_wrap() -> None:
    rendered = render_chinese_reading_tex(
        "# 标题\n\n见 `paper/generated/table_external_validation.csv`。",
        "# 补充\n\n说明。",
        figure_directory=Path("../neurips2026/figures"),
    )

    assert "\\url{paper/generated/table_external_validation.csv}" in rendered


def test_render_chinese_reading_tex_allows_long_identifier_to_wrap() -> None:
    rendered = render_chinese_reading_tex(
        "# 标题\n\n状态为 `blocked_by_source_metadata`。",
        "# 补充\n\n说明。",
        figure_directory=Path("../neurips2026/figures"),
    )

    assert "\\_\\allowbreak" in rendered


def test_render_chinese_reading_tex_excludes_internal_workflow_notes() -> None:
    rendered = render_chinese_reading_tex(
        "# 标题\n\n正文。\n\n## 稿件内部证据说明\n\n不应出现在阅读版。",
        "# 补充\n\n正文。\n\n### J.8 补充图片审批\n\n内部流程。",
        figure_directory=Path("../neurips2026/figures"),
    )

    assert "不应出现在阅读版" not in rendered
    assert "内部流程" not in rendered


def test_resource_table_allocates_space_for_paired_interval() -> None:
    rendered = render_chinese_reading_tex(
        "# 标题\n\n| 任务 | 设置 | EC-P | PC-P | EC-PC | 配对 percentile 区间 |\n"
        "|---|---|---|---|---|---|\n"
        "| Emo | Band-power R1 | 0.1312 | 0.1420 | -0.0108 | [-0.0429, +0.0219] |",
        "# 补充\n\n说明。",
        figure_directory=Path("../neurips2026/figures"),
    )
    assert r"p{0.250\linewidth}" in rendered
    assert "[-0.0429, +0.0219]" in rendered


def test_current_chinese_manuscript_embeds_all_five_final_figures():
    root = Path(__file__).resolve().parents[1]
    rendered = render_chinese_reading_tex(
        (root / "paper/manuscript_zh.md").read_text(encoding="utf-8"),
        (root / "paper/supplement_zh.md").read_text(encoding="utf-8"),
        figure_directory=Path("../neurips2026/figures"))
    for number in range(1, 6):
        assert rendered.count(f"\\includegraphics[width=\\linewidth]{{../neurips2026/figures/figure{number}.pdf}}") == 1
