#!/usr/bin/env python3
"""Create isolated, identical evidence packages for human coders A and B."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import zipfile

import pandas as pd


README = """# Independent human coding packet

Work alone and do not inspect or discuss the other coder's file before both
locks exist. Read `literature_resource_coding_manual_v14.md`, inspect the PDF
and any cited supplement or code, then complete every field in the two CSVs.
The JSON evidence packet is a search aid only; verify every judgment against the
PDF page. Use `unclear` when the document does not establish an access fact.

After coding, run the lock command shown in `LOCK_COMMAND.txt`. Do not edit the
CSVs after the lock file is created.
"""

README_ZH = """# 独立人工编码包

请独立完成编码，在 A、B 两份锁定文件都生成前，不查看或讨论另一位编码者的逐篇判断。
先阅读 `literature_resource_coding_manual_v14.md`，再核对 PDF、补充材料和论文代码。
JSON 只用于定位候选段落，每个结论都必须回到 PDF 页码核实；原文不足时填写
`unclear`，不得根据常见做法猜测。完成两个 CSV 后，运行 `LOCK_COMMAND.txt`
中的命令。锁定后不要再修改 CSV。
"""

DECLARATION = """coder_id:
coder_name_or_study_identifier:
affiliation_or_role:
relationship_to_manuscript_authors:
prior_involvement_in_this_project:
coding_started_at:
coding_completed_at:
statement: I coded independently and did not inspect the other coder's judgments before both locks.
signature_or_typed_name:
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cohort_root", type=Path)
    parser.add_argument("full_text_root", type=Path)
    parser.add_argument("manual", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    cohort = pd.read_csv(args.cohort_root / "paper_set.csv", keep_default_na=False)
    for coder in ("A", "B"):
        root = args.output / f"coder_{coder}"
        if root.exists():
            shutil.rmtree(root)
        (root / "papers").mkdir(parents=True)
        (root / "evidence_packets").mkdir()
        shutil.copy2(args.manual, root / args.manual.name)
        lock_script = Path(__file__).with_name("lock_literature_coding_v14.py")
        shutil.copy2(lock_script, root / lock_script.name)
        shutil.copy2(args.cohort_root / "paper_set.csv", root / "paper_set.csv")
        shutil.copy2(args.cohort_root / f"coder_{coder}_paper.csv", root / f"coder_{coder}_paper.csv")
        shutil.copy2(args.cohort_root / f"coder_{coder}_evaluations.csv", root / f"coder_{coder}_evaluations.csv")
        for paper_id in cohort.paper_id:
            shutil.copy2(args.full_text_root / f"{paper_id}.pdf", root / "papers" / f"{paper_id}.pdf")
            shutil.copy2(
                args.cohort_root / "evidence_packets" / f"{paper_id}.json",
                root / "evidence_packets" / f"{paper_id}.json",
            )
        (root / "README.md").write_text(README, encoding="utf-8")
        (root / "README_zh.md").write_text(README_ZH, encoding="utf-8")
        (root / "CODER_DECLARATION.txt").write_text(
            DECLARATION.replace("coder_id:\n", f"coder_id: {coder}\n"), encoding="utf-8"
        )
        command = (
            f"python lock_literature_coding_v14.py paper_set.csv "
            f"coder_{coder}_paper.csv coder_{coder}_evaluations.csv coder_{coder}_lock.json\n"
        )
        (root / "LOCK_COMMAND.txt").write_text(command, encoding="utf-8")
        archive = args.output / f"coder_{coder}_packet.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zipped:
            for path in sorted(root.rglob("*")):
                if path.is_file():
                    zipped.write(path, path.relative_to(root))
        print(f"created {archive} ({archive.stat().st_size} bytes)")
        shutil.rmtree(root)


if __name__ == "__main__":
    main()
