"""Strict event-state parsing for the ds006850 image-appraisal recording."""
from __future__ import annotations

import csv
import math

import pandas as pd


def event_fields(value):
    fields = {}
    for part in next(csv.reader([str(value)], delimiter=";")):
        key, separator, val = part.partition(":")
        if not separator or not key.strip() or key.strip() in fields:
            raise ValueError("Malformed or duplicate event field")
        fields[key.strip()] = val.strip()
    if "event" not in fields:
        raise ValueError("Missing event type")
    return fields


def parse_events(events, *, participant, session, eeg_path, source_path, allow_leading_orphans=False):
    required = {"onset", "sample", "value"}
    if not required.issubset(events):
        raise ValueError("Missing event columns")
    if not events.onset.astype(float).is_monotonic_increasing:
        raise ValueError("Events are not chronological")
    prompt, pending, rows = None, None, []
    seen_anchor, leading_orphans = False, []
    for index, row in enumerate(events.itertuples(index=False)):
        fields = event_fields(row.value)
        event = fields["event"]
        onset = float(row.onset)
        if not math.isfinite(onset):
            raise ValueError("Non-finite event onset")
        if event == "anchorsPresented":
            seen_anchor = True
            if pending is not None:
                raise ValueError("New prompt before prior response")
            prompt = fields["scale"]
        elif event == "presentedStimulus":
            if pending is not None or prompt is None:
                raise ValueError("Stimulus without a unique priming prompt")
            pending = {"stimulus": fields["stimulus"], "scale": prompt, "onset": onset,
                           "sample": int(row.sample), "event_row": index+2, "scale_onset": None}
        elif event in {"scalePresented", "optionConfirmed"}:
            if allow_leading_orphans and not seen_anchor and pending is None:
                leading_orphans.append({"line": index+2, "event": event,
                    "reason": "No preceding stimulus epoch in this recording"})
                continue
            if pending is None or fields.get("stimulus") != pending["stimulus"] or fields.get("scale") != pending["scale"]:
                raise ValueError("Rating event does not match the preceding image/prompt")
            if event == "scalePresented":
                if pending["scale_onset"] is not None or onset <= pending["onset"]:
                    raise ValueError("Duplicate or out-of-order rating scale")
                pending["scale_onset"] = onset
            else:
                if pending["scale_onset"] is None or onset < pending["scale_onset"]:
                    raise ValueError("Response before scale presentation")
                rating = float(fields["response"])
                if not math.isfinite(rating):
                    raise ValueError("Non-finite confirmed response")
                duration = pending["scale_onset"]-pending["onset"]
                if not 2.9 <= duration <= 3.15:
                    raise ValueError("Image presentation is outside the 3-second protocol tolerance")
                scale = pending["scale"]
                if scale in {"SAM-valence", "SAM-arousal"}:
                    if not 1 <= rating <= 9 or rating != int(rating):
                        raise ValueError("SAM response is not an integer from 1 to 9")
                    uid = f"ds006850:{participant}:{session}:{pending['event_row']}"
                    # The source experiment numbers left-to-right choices 1..9.
                    # Its anchors are happy->unhappy and excited->calm, so high
                    # normalized values represent positive valence/high arousal.
                    normalized = (9-rating)/8
                    rows.append({"trial_uid": uid, "dataset_id": "ds006850", "subject_id": participant,
                        "subject_uid": "ds006850:"+participant, "session_id": session,
                        "stimulus_uid": "ds006850:"+pending["stimulus"], "context": scale,
                        "target_name": scale.removeprefix("SAM-"), "rating_raw": rating,
                        "rating_normalized": normalized, "eeg_path": eeg_path,
                        "eeg_onset_s": pending["onset"], "eeg_sample_1based": pending["sample"],
                        "eeg_duration_s": min(3.0, duration), "eeg_stop_before_s": pending["scale_onset"],
                        "observed_image_duration_s": duration,
                        "response_onset_s": onset, "label_available": True,
                        "source_events_path": source_path, "source_stimulus_line": pending["event_row"],
                        "source_response_line": index+2})
                pending, prompt = None, None
    if pending is not None:
        raise ValueError("Recording ends before the last image response")
    result = pd.DataFrame(rows)
    result.attrs["excluded_leading_events"] = leading_orphans
    return result
