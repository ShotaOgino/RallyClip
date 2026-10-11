"""JSON-ready payloads for engine results and saved matches.

The engine keeps intervals as tuples and paths as Path objects; converting to
client-facing JSON shapes belongs here, not inside the engine.
"""

from __future__ import annotations

from typing import Any, Dict

from rallyclip_core.contracts import RunResult, SavedMatch


def run_result_payload(result: RunResult) -> Dict[str, Any]:
    payload = {
        "pipeline_id": result.diagnostics.get("pipeline_id"),
        "intervals": [
            {"start_s": start, "end_s": end} for start, end in result.intervals_sec
        ],
        "csv_path": str(result.csv_path) if result.csv_path is not None else None,
        "video_path": str(result.video_path) if result.video_path is not None else None,
        "n_segments": len(result.frame_segments),
    }
    # Opt-in player analysis outputs (RunRequest.save_player_tracks); absent by default.
    for key in ("players_npz", "points_json", "player_analysis_error"):
        if result.diagnostics.get(key) is not None:
            payload[key] = result.diagnostics[key]
    return payload


def saved_match_payload(match: SavedMatch) -> Dict[str, Any]:
    return {
        "id": match.id,
        "title": match.title,
        "source_path": str(match.source_path),
        "csv_path": str(match.csv_path),
        "thumbnail_path": str(match.thumbnail_path) if match.thumbnail_path is not None else None,
        "metadata": dict(match.metadata),
    }
