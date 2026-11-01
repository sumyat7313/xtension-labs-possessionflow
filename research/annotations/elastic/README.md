# Corrected development event annotations

These three JSON documents are sparse **development annotations on already exposed matches**, not fresh holdouts or dense H/A/D labels. Their intended uses are training-only boundary supervision inside the relevant outer training folds and separately reported timing diagnostics. Do not train an evaluated fold on its annotation document.

## Attribution and licence

Data is adapted from [ELASTIC, Hyunsung Kim et al., CIKM 2026](https://github.com/hyunsungkim-ds/elastic), pinned commit `e245224aad7450961eca40954ae2bc85e1e6e696`, specifically `benchmark/gt/J03WMX.parquet`, `J03WN1.parquet`, and `J03WPY.parquet`. [Paper](https://arxiv.org/abs/2608.30227), DOI `10.1145/3799682.3841035`.

The original data is [Bassek, Rein, Weber and Memmert (2025), An integrated dataset of spatiotemporal and event data in elite soccer](https://doi.org/10.1038/s41597-025-04505-y), [Sportec Open DFL dataset](https://doi.org/10.6084/m9.figshare.28196177). These derived annotation files retain the upstream [Creative Commons Attribution 4.0 licence](https://creativecommons.org/licenses/by/4.0/). Credit both publications when using them. The upstream ELASTIC code is MPL-2.0; no upstream code is included here or copied into our importer.

## Our transformations

`tools/data_intake_v8.py` verifies pinned Git blob identities, reads corrected event tables, converts native Sportec frame IDs into our exact half-local cached indices, and splits starts and terminations into sparse point records. It records source hashes and tracking metadata provenance. The original bytes are preserved locally in ignored runs. No interpolation, inferred dead intervals, pseudo-labels or new manual annotations were added.

One inconsistent source row was quarantined: J03WPY row 718 has a reception seven frames before its event. Neither endpoint is used. Other source rows retain their original event IDs in `source_event_id`; `event_id` is the index among accepted rows. A point's `frame_id` is native; `index` is the cached timeline position. The half separator is never a labelled point.

| Match | Source rows | Accepted rows | Restart points | Ball-out points | Reception/touch points |
| --- | ---: | ---: | ---: | ---: | ---: |
| J03WMX | 1,329 | 1,329 | 97 | 52 | 1,048 |
| J03WN1 | 1,030 | 1,030 | 95 | 51 | 779 |
| J03WPY | 1,202 | 1,201 | 90 | 47 | 963 |

## Meaning and limitations

ELASTIC reports median timestamps from three annotators. Individual annotations and uncertainty are not released. A `restart` here is a labelled set-piece action. `ball_out` is an explicit out terminal; a foul remains an ordinary event because a foul timestamp is not necessarily a whistle. `goal` remains distinct. The upstream collapse operation can assign `receive_frame_id` from control or the next incoming/one-touch action, so our `reception` kind does **not** guarantee controlled possession acquisition. Unlisted frames are unknown, not negative examples or D labels.

Use `validated_points(document, match=..., fps=25, blocks=..., n_frames=...)` before training. It verifies scope, source identity, exact index mapping, actor/team semantics and quarantine exclusion. The campaign must additionally freeze complete file hashes. See `summary.json` for all identities and `docs/LABELLED_DATA_V8.md` for acquisition and annotation guidance.
