from __future__ import annotations

from batch_decision.contracts import BatchScorePayload, WindowScore
from batch_decision.decision_engine import decide


def test_decide_patchtst_emits_warn_and_anomaly_with_reasons() -> None:
    payload = BatchScorePayload(
        run_id="decision-single",
        stream="patchtst",
        patchtst_records=[
            WindowScore(
                event_id="patchtst:000000",
                stream="patchtst",
                file_id="fdc_a",
                timestamp="2026-03-11T00:00:00",
                window_index=0,
                score=0.20,
                aux={},
            ),
            WindowScore(
                event_id="patchtst:000001",
                stream="patchtst",
                file_id="fdc_a",
                timestamp="2026-03-11T00:01:00",
                window_index=1,
                score=0.70,
                aux={},
            ),
            WindowScore(
                event_id="patchtst:000002",
                stream="patchtst",
                file_id="fdc_a",
                timestamp="2026-03-11T00:02:00",
                window_index=2,
                score=0.90,
                aux={},
            ),
        ],
    )

    result = decide(
        payload,
        thresholds={"patchtst": {"warn": 0.65, "anomaly": 0.85}},
    )

    assert [event.decision for event in result.events] == ["normal", "warn", "anomaly"]
    assert "warn(0.650000)" in result.events[1].reason
    assert "anomaly(0.850000)" in result.events[2].reason
    assert result.summary["decision_counts"] == {"normal": 1, "warn": 1, "anomaly": 1}
    assert result.chart_payload["thresholds"]["warn"] == 0.65


def test_decide_dual_uses_max_fusion_and_tracks_alignment_metadata() -> None:
    payload = BatchScorePayload(
        run_id="decision-dual",
        stream="dual",
        patchtst_records=[
            WindowScore(
                event_id="patchtst:000000",
                stream="patchtst",
                file_id="fdc_a",
                timestamp="2026-03-11T00:00:00",
                window_index=0,
                score=0.10,
                aux={},
            ),
            WindowScore(
                event_id="patchtst:000001",
                stream="patchtst",
                file_id="fdc_a",
                timestamp="2026-03-11T00:01:00",
                window_index=1,
                score=0.72,
                aux={},
            ),
        ],
        swinmae_records=[
            WindowScore(
                event_id="swinmae:000000",
                stream="swinmae",
                file_id="vib_a",
                timestamp="2026-03-11T00:00:10",
                window_index=0,
                score=0.91,
                aux={},
            ),
        ],
    )

    result = decide(
        payload,
        thresholds={"dual": {"warn": 0.68, "anomaly": 0.88}},
    )

    assert len(result.events) == 1
    assert result.events[0].decision == "anomaly"
    assert result.events[0].fused_score == 0.91
    assert result.events[0].stream_scores == {"patchtst": 0.10, "swinmae": 0.91}
    assert result.metadata["dual_alignment"]["unpaired_patchtst"] == 1
    assert result.metadata["dual_alignment"]["unpaired_swinmae"] == 0
