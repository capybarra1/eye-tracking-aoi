"""Read-only progress labels for the import menu; history is not verification."""
from scripts.aoi_slice_timing import effective_manifest
from bisect import bisect_left
import json
from pathlib import Path
import sqlite3


def import_status(entry: dict, project: dict | None) -> dict:
    old = bool(entry.get('annotation_history', {}).get('old_aoi_present'))
    stored = expected = issues = 0
    if project:
        try:
            times = json.loads(Path(project['timestamps']).read_text())
            manifest = json.loads(Path(project['manifest']).read_text()) if project.get('manifest') else entry.get('manifest',[])
            with sqlite3.connect((Path(project['folder'])/'session.sqlite3').resolve().as_uri()+'?mode=ro', uri=True, timeout=1) as db:
                manifest=effective_manifest(manifest,db)
                for row in manifest:
                    first = bisect_left(times, row['start_s'])
                    last = bisect_left(times, row['end_s'])
                    expected += last-first
                    count, flagged = db.execute("SELECT COUNT(*), SUM(CASE WHEN json_array_length(payload,'$.problems')>0 THEN 1 ELSE 0 END) FROM records WHERE segment=? AND frame>=? AND frame<?", (row['segment_id'], first, last)).fetchone()
                    stored += count
                    issues += flagged or 0
        except (OSError, ValueError, KeyError, sqlite3.Error):
            return dict(code='unknown', label='进度待核对', priority=4, old_aoi_present=old)
    if old:
        code, label, priority = 'old', '旧软件已标过', 2
    elif stored and expected and stored >= expected:
        code, label, priority = 'covered', '本工具已覆盖' + ('·待复核' if issues else ''), 3
    elif stored:
        code, label, priority = 'partial', f'部分标注·{min(99,int(stored*100/expected))}%' if expected else '部分标注', 1
    else:
        code, label, priority = 'new', '优先·未标注', 0
    return dict(code=code, label=label, priority=priority, old_aoi_present=old, stored_frames=stored, expected_frames=expected, issue_frames=issues)


def prioritize_imports(videos: list, projects: dict) -> list:
    for row in videos:
        row['annotation_status'] = import_status(row, projects.get(row.get('project_id')))
    # Preserve existing part order within each subject, including supplemental ranges.
    return sorted(videos, key=lambda r:(r['annotation_status']['priority'], r['subject']))
