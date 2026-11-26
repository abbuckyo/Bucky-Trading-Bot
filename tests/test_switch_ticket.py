from bot import CFG, build_switch_ticket_lines, render_executive_summary


def test_quiet_mode_when_delta_below_threshold():
    sleeve_caps = {
        "SCBWORLDE": 0.25,
        "SCBGOLDH": 0.15,
        "SCBSEMI(E)": 0.12,
    }
    prev_weights = {
        "SCBWORLDE": 0.2500,
        "SCBGOLDH": 0.1000,
        "SCBSEMI(E)": 0.0000,
    }
    # Tiny 0.4% change on SCBGOLDH (< 0.5% min_delta_pct) -> Quiet Mode
    decisions = {
        "SCBWORLDE": 0.2500,
        "SCBGOLDH": 0.1040,
        "SCBSEMI(E)": 0.0000,
    }

    lines = build_switch_ticket_lines(
        decisions=decisions,
        sleeve_caps=sleeve_caps,
        prev_weights=prev_weights,
        cash_fund=CFG.CASH_FUND,
    )

    assert lines == ["✅ วันนี้ไม่ต้องทำอะไร (ถือตามเดิม)"]
    for line in lines:
        assert "**" not in line
        assert "`" not in line


def test_buy_only_headline():
    lines = build_switch_ticket_lines(
        decisions={"SCBWORLDE": 0.08, "SCBGOLDH": 0.00},
        sleeve_caps={"SCBWORLDE": 0.24, "SCBGOLDH": 0.18},
        prev_weights={"SCBWORLDE": 0.00, "SCBGOLDH": 0.00},
        cash_fund=CFG.CASH_FUND,
    )

    assert lines[0] == "🟢 วันนี้ต้องซื้อ (ก่อน 15:00 น.)"
    assert lines[1] == f"🟢 สับ {CFG.CASH_FUND} → SCBWORLD(E): +8.0% (8,000 บ./แสน)"


def test_sell_only_headline():
    lines = build_switch_ticket_lines(
        decisions={"SCBGOLDH": 0.06},
        sleeve_caps={"SCBGOLDH": 0.18},
        prev_weights={"SCBGOLDH": 0.18},
        cash_fund=CFG.CASH_FUND,
    )

    assert lines[0] == "🔴 วันนี้ต้องขาย (ก่อน 15:00 น.)"
    assert lines[1] == f"🔴 สับ SCBGOLDH → {CFG.CASH_FUND}: -12.0% (12,000 บ./แสน)"


def test_sells_before_buys_and_rebalance_headline():
    sleeve_caps = {
        "SCBWORLDE": 0.24,
        "SCBGOLDH": 0.18,
        "SCBSEMI(E)": 0.12,
    }
    prev_weights = {
        "SCBWORLDE": 0.00,
        "SCBGOLDH": 0.18,
        "SCBSEMI(E)": 0.04,
    }
    # SCBGOLDH sells -12.0% (12,000 บ./แสน), SCBWORLDE buys +8.0% (8,000 บ./แสน)
    # SCBSEMI(E) changes +0.2% (ignored < 0.5%)
    decisions = {
        "SCBWORLDE": 0.08,
        "SCBGOLDH": 0.06,
        "SCBSEMI(E)": 0.042,
    }

    lines = build_switch_ticket_lines(
        decisions=decisions,
        sleeve_caps=sleeve_caps,
        prev_weights=prev_weights,
        cash_fund=CFG.CASH_FUND,
    )

    assert lines[0] == "🔄 วันนี้ต้องขาย+ซื้อ (ก่อน 15:00 น.)"
    assert len(lines) == 3

    sell_line = lines[1]
    buy_line = lines[2]

    assert sell_line == f"🔴 สับ SCBGOLDH → {CFG.CASH_FUND}: -12.0% (12,000 บ./แสน)"
    assert buy_line == f"🟢 สับ {CFG.CASH_FUND} → SCBWORLD(E): +8.0% (8,000 บ./แสน)"

    for line in lines:
        assert "**" not in line
        assert "`" not in line


def test_render_executive_summary_places_headline_on_line_1():
    summary_rows = [
        ("SCBWORLDE", 80.0, "INVESTED", "T1", "🟢", "detail", 1, "33%"),
        ("SCBGOLDH", 45.0, "CASH PARK", "OUT", "⚪", "detail", 0, "0%"),
    ]
    ticket_lines = build_switch_ticket_lines(
        decisions={"SCBWORLDE": 0.08, "SCBGOLDH": 0.0},
        sleeve_caps={"SCBWORLDE": 0.24, "SCBGOLDH": 0.18},
        prev_weights={"SCBWORLDE": 0.0, "SCBGOLDH": 0.12},
        cash_fund=CFG.CASH_FUND,
    )

    summary = render_executive_summary(
        summary_rows=summary_rows,
        canary_ok=True,
        tip_mom=1.25,
        canary_desc="OK",
        futures_triggered=False,
        futures_desc="ES +0.2%",
        sleeve_caps={"SCBWORLDE": 0.24, "SCBGOLDH": 0.18},
        ticket_lines=ticket_lines,
    )

    out_lines = summary.splitlines()
    assert out_lines[0] == "🔄 วันนี้ต้องขาย+ซื้อ (ก่อน 15:00 น.)"
    assert "━━━━━━━━━━━━━━" in out_lines
    assert out_lines.index("━━━━━━━━━━━━━━") < out_lines.index("⚡ สรุปคำสั่งพอร์ต AlphaShield V8.1")
    assert summary.count("🔄 วันนี้ต้องขาย+ซื้อ (ก่อน 15:00 น.)") == 1
    assert "SCBWORLD(E)" in summary
    assert "บ./แสน" in summary
    assert "**" not in summary
    assert "`" not in summary
