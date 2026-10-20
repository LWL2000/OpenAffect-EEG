#!/usr/bin/env python3
"""Render the user-selected matched-increment heatmap from frozen aggregates."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
from PIL import Image


def main():
    root = Path(__file__).resolve().parents[1]
    folder = root / "paper/generated/evidence_v14"
    doses = [0, 1, 2, 4, 8]
    sources = []
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 8,
                         "axes.titlesize": 9, "axes.labelsize": 8,
                         "xtick.labelsize": 7.2, "ytick.labelsize": 7.2,
                         "pdf.fonttype": 42, "ps.fonttype": 42}):
        fig, axes = plt.subplots(1, 2, figsize=(5.5, 3.45))
        fig.subplots_adjust(left=0.08, right=0.86, top=0.86, bottom=0.31, wspace=0.36)
        norm = TwoSlopeNorm(vmin=-0.1, vcenter=0, vmax=0.1)
        for ax, key, title in zip(axes, ["faced_eegnet_observed", "faced_labram_observed"], ["EEGNet", "LaBraM tail adaptation"]):
            source = folder / "archive" / key / "cells.csv"
            summary_path = source.with_name("summary.json")
            summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))
            matrix = np.full((5, 5), np.nan)
            with source.open(encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    matrix[doses.index(int(row["participant_dose"])), doses.index(int(row["stimulus_dose"]))] = float(row["ccc_delta"])
            if np.isnan(matrix).any():
                raise ValueError("Incomplete heatmap")
            image = ax.imshow(matrix, origin="lower", cmap="PuOr_r", norm=norm, interpolation="nearest")
            ax.set_xticks(range(5)); ax.set_xticklabels(doses)
            ax.set_yticks(range(5)); ax.set_yticklabels(doses)
            ax.set_xlabel("Stimulus-label dose s")
            ax.set_title(title, pad=10)
            for y in range(5):
                for x in range(5):
                    val = matrix[y, x]
                    color = "white" if abs(val) > 0.067 else "black"
                    ax.text(x, y, "{:+.3f}".format(val), ha="center", va="center", fontsize=7.2, color=color)
            ax.tick_params(length=0)
            for spine in ax.spines.values():
                spine.set_visible(False)
            primary = summary["primary"]
            ci95 = primary["ci_95"]
            ci90 = primary["equivalence_primary"]["decision_interval_90"]
            label = "Uniform mean {:+.4f}\n95% CI [{:+.4f}, {:+.4f}]\n90% CI [{:+.4f}, {:+.4f}]".format(primary["estimate"], ci95[0], ci95[1], ci90[0], ci90[1])
            box = ax.get_position()
            fig.text((box.x0 + box.x1) / 2, 0.205, label, ha="center", va="top", fontsize=7.2, linespacing=1.5)
            sources.extend([source, summary_path])
        axes[0].set_ylabel("Participant-label dose p")
        cax = fig.add_axes([0.89, 0.36, 0.018, 0.44])
        bar = fig.colorbar(image, cax=cax, ticks=[-0.1, -0.05, 0, 0.05, 0.1])
        bar.ax.tick_params(labelsize=7.2)
        bar.ax.set_title("Delta CCC", fontsize=7.5, pad=9)
        fig.text(0.08, 0.965, "FACED: fixed support, five seeds, five refitted rotations", fontsize=8, va="top")
        fig.savefig(str(folder / "faced_matched_heatmap.pdf"), metadata={"CreationDate": None, "ModDate": None})
        fig.savefig(str(folder / "faced_matched_heatmap.png"), dpi=300)
        plt.close(fig)
    preview = Image.open(str(folder / "faced_matched_heatmap.png")).convert("L")
    preview.save(str(folder / "faced_matched_heatmap_grayscale.png"))
    qa = {
        "status": "rendered_pending_visual_inspection", "width_inches": 5.5, "height_inches": 3.45,
        "dpi": 300, "minimum_font_pt": 7.2, "scale": [-0.1, 0.1],
        "redundant_encoding": "Every cell prints its signed value; hue is not the only encoding.",
        "scientific_question": "Does matched EEG added value vary across participant and stimulus label-resource doses?",
        "permitted_conclusion": "The model-specific uniform means and heterogeneous point estimates differ; interval/equivalence inference applies to the mean, not all cells.",
        "unsupported_conclusion": "No individual heatmap cell is certified equivalent or evidence of universal EEG absence.",
        "uncertainty": "Archived joint intervals under each heatmap; pointwise and simultaneous cell bands released in archive/cells.csv.",
        "controls": "All negative, failed positive and exploratory positive controls appear in Table S and the public archive, rather than being overlaid on these observed-data grids.",
        "grayscale": "paper/generated/evidence_v14/faced_matched_heatmap_grayscale.png",
        "sources": {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "whole_page_preview": "paper/generated/evidence_v14/qa/heatmap_page.png",
    }
    (folder / "figure_qa.json").write_text(json.dumps(qa, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
