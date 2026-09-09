"""Build separate valence/arousal manifests, preserving response-to-epoch lineage."""
import argparse
import json
from pathlib import Path

import pandas as pd

from openaffect_eeg.artifacts import sha256_file
from openaffect_eeg.urban_appraisal import parse_events


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--exclude-subject", action="append", default=[])
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    frames, errors, inputs, exclusions = [], [], {}, []
    paths = sorted(args.root.glob("sub-*/ses-*/eeg/*_events.tsv"))
    if len(paths) != 126:
        raise ValueError("Pinned snapshot requires all 126 event tables before analysis")
    for path in paths:
        relative = path.relative_to(args.root)
        participant, session = relative.parts[:2]
        eeg = str(relative).replace("_events.tsv", "_eeg.vhdr")
        inputs[str(relative)] = sha256_file(path)
        try:
            frame = parse_events(pd.read_csv(path, sep="\t"), participant=participant,
                session=session, eeg_path=eeg, source_path=str(relative), allow_leading_orphans=True)
            exclusions.extend(dict(source=str(relative), **item) for item in frame.attrs["excluded_leading_events"])
            frames.append(frame)
        except (ValueError, KeyError) as exc:
            errors.append({"source": str(relative), "error": str(exc)})
    (args.output/"parse_failures.json").write_text(json.dumps(errors, indent=2)+"\n")
    (args.output/"excluded_events.json").write_text(json.dumps(exclusions, indent=2)+"\n")
    report = {"source_files": len(paths), "source_sha256": inputs, "failed_recordings": len(errors),
        "excluded_leading_events": len(exclusions),
        "source_snapshot": "ds006850/1.0.0",
        "response_orientation": {
            "valence": "response 1=happy, 9=unhappy; target=(9-response)/8",
            "arousal": "response 1=excited, 9=calm; target=(9-response)/8",
            "source_repository": "https://github.com/BeMoBIL/urban_appraisal-experiment",
            "source_commit": "11fd96132f240cb3da759f03c10672d83dfd8e7e",
        },
        "boundary": "Separate prompt-specific trials, not paired valence/arousal labels on one EEG epoch"}
    if frames:
        table = pd.concat(frames, ignore_index=True)
        requested_exclusions = sorted(set(args.exclude_subject))
        available_subjects = set(table["subject_uid"])
        missing_exclusions = sorted(set(requested_exclusions) - available_subjects)
        if missing_exclusions:
            raise ValueError(f"Unknown requested subject exclusions: {missing_exclusions}")
        excluded_rows = table[table["subject_uid"].isin(requested_exclusions)].copy()
        table = table[~table["subject_uid"].isin(requested_exclusions)].copy()
        duplicates = table.duplicated(["subject_uid", "stimulus_uid", "context"], keep=False)
        report.update(parsed_trials=len(table), participants=table.subject_uid.nunique(),
            stimuli=table.stimulus_uid.nunique(), duplicate_records=int(duplicates.sum()),
            tasks=table.groupby("context").size().to_dict(),
            excluded_subjects=requested_exclusions,
            excluded_subject_rows=len(excluded_rows),
            exclusion_boundary=(
                "Subjects are excluded only by an externally documented raw-integrity "
                "audit performed before EEG prediction scores are inspected."
            ))
        if duplicates.any():
            table.loc[duplicates].to_csv(args.output/"duplicate_lineage.tsv", sep="\t", index=False)
        if not errors and not duplicates.any():
            for task, subset in table.groupby("target_name"):
                subset = subset.copy()
                subset[f"target_{task}"] = subset["rating_normalized"]
                subset.to_csv(args.output/f"{task}_trials.tsv.gz", sep="\t", index=False)
            report["status"] = "metadata_validated_no_EEG_performance_evaluated"
        else:
            report["status"] = "blocked_by_source_alignment"
    (args.output/"metadata_audit.json").write_text(json.dumps(report, indent=2)+"\n")
    print({k: v for k, v in report.items() if k != "source_sha256"})
    if errors or report.get("duplicate_records", 0):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
