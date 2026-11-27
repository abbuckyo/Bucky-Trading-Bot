from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

from bot import (
    NEWS_SYMBOL_MAP,
    Asset,
    Decision,
    Metrics,
    ScoreResult,
    fetch_macro_news,
    fetch_news,
    is_low_signal_headline,
    render_asset_block,
    render_executive_summary,
)


def test_news_symbol_map_maps_cspx_to_spy():
    assert NEWS_SYMBOL_MAP["CSPX.L"] == "SPY"
    assert NEWS_SYMBOL_MAP["URTH"] == "VT"


def test_is_low_signal_headline_filters_clickbait_and_keeps_macro():
    assert is_low_signal_headline("When Sequence Risk Kicks The Door In") is True
    assert is_low_signal_headline("Is It Really All Clear In The Stock Market?") is True
    assert is_low_signal_headline("Should You Buy These 3 Stocks Before You Retire?") is True
    assert is_low_signal_headline("Opinion: Why I Sold Everything") is True
    assert is_low_signal_headline("Fed holds rates steady as core PCE cools") is False
    assert is_low_signal_headline("Nvidia beats Q3 earnings estimates on AI server demand") is False


def test_fetch_news_maps_cspx_sorts_and_filters_stale_and_clickbait():
    now_ts = int(datetime.now(timezone.utc).timestamp())
    old_ts = int((datetime.now(timezone.utc) - timedelta(hours=72)).timestamp())

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = [
        {"datetime": old_ts, "headline": "Old S&P 500 news from 3 days ago"},
        {"datetime": now_ts - 100, "headline": "S&P 500 rises on cooling inflation"},
        {"datetime": now_ts - 10, "headline": "When Sequence Risk Kicks The Door In"},
        {"datetime": now_ts - 5, "headline": "Fed signals patience on rate path"},
        {"datetime": now_ts - 20, "headline": "S&P 500 rises on cooling inflation"},
    ]

    with patch.dict("bot.SECRETS", {"FINNHUB_API_KEY": "test_key"}), patch(
        "bot.requests.get", return_value=mock_resp
    ) as mock_get:
        heads = fetch_news("CSPX.L")

    assert mock_get.call_args.kwargs["params"]["symbol"] == "SPY"
    assert heads == [
        "Fed signals patience on rate path",
        "S&P 500 rises on cooling inflation",
    ]


def test_fetch_macro_news_sorts_filters_and_handles_error():
    now_ts = int(datetime.now(timezone.utc).timestamp())
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = [
        {"datetime": now_ts - 60, "headline": "Oil prices dip on supply outlook"},
        {"datetime": now_ts - 5, "headline": "Podcast: How to become a millionaire"},
        {"datetime": now_ts - 10, "headline": "Treasury yields steady ahead of PCE data"},
    ]

    with patch.dict("bot.SECRETS", {"FINNHUB_API_KEY": "test_key"}), patch(
        "bot.requests.get", return_value=mock_resp
    ):
        heads = fetch_macro_news(max_items=5)

    assert heads == [
        "Treasury yields steady ahead of PCE data",
        "Oil prices dip on supply outlook",
    ]

    with patch.dict("bot.SECRETS", {"FINNHUB_API_KEY": "test_key"}), patch(
        "bot.requests.get", side_effect=RuntimeError("network down")
    ):
        assert fetch_macro_news() == []


def test_render_asset_block_omits_english_news_and_strips_markdown():
    asset = Asset(fund="SCBWORLDE", yahoo="URTH", stooq="urth.us", label="Global Equity")
    metrics = Metrics(
        price=150.25,
        chg_pct=1.25,
        dist_ema50_pct=2.2,
        dist_ema200_pct=7.3,
        rsi=60.5,
        adx=25.4,
    )
    score_res = ScoreResult(
        score=82.0,
        raw_score=82.0,
        breakdown=False,
        breakdown_reason="",
        chop_capped=False,
        components={},
        notes=[],
    )
    dec = Decision(
        signal="INVESTED",
        position="FULL",
        prev_position="FULL",
        action="[HOLD]",
        icon="🟢",
        action_detail="ถือต่อ",
        target_exposure=1.0,
        tranche_stage=3,
        tranche_label="100% (3/3)",
    )

    block = render_asset_block(
        asset=asset,
        m=metrics,
        sc=score_res,
        dec=dec,
        news=["Some English Headline That Should Not Appear"],
        ai_text="**แนวโน้มขาขึ้นแข็งแกร่ง** ยืนเหนือ `EMA200` ได้ดี",
    )

    assert "📰 ข่าว:" not in block
    assert "Some English Headline" not in block
    assert "**" not in block
    assert "`" not in block
    assert "SCBWORLD(E)" in block
    assert "💬 วิเคราะห์: แนวโน้มขาขึ้นแข็งแกร่ง ยืนเหนือ EMA200 ได้ดี" in block


def test_render_executive_summary_keeps_headline_on_line_1_and_adds_macro_brief():
    summary_rows = [
        ("SCBWORLDE", 75.0, "HOLD", "IN", "🟢", "ถือต่อ", 3, "100% (3/3)"),
    ]
    ticket_lines = [
        "🟢 วันนี้ต้องซื้อ (ก่อน 15:00 น.)",
        "1) ซื้อ SCBWORLD(E) 30.5% (~30,500 บาท)",
    ]

    summary = render_executive_summary(
        summary_rows=summary_rows,
        canary_ok=True,
        tip_mom=1.5,
        canary_desc="TIP OK",
        futures_triggered=False,
        futures_desc="ES +0.2%",
        sleeve_caps={"SCBWORLDE": 0.305},
        ticket_lines=ticket_lines,
        macro_brief="**เฟดคงดอกเบี้ย** ตลาดหุ้นสหรัฐปรับขึ้น\nจับตาเงินเฟ้อ `PCE` คืนนี้",
    )

    lines = summary.splitlines()
    assert lines[0] == "🟢 วันนี้ต้องซื้อ (ก่อน 15:00 น.)"
    assert "━━━━━━━━━━━━━━" in lines
    assert "SCBWORLD(E)" in summary
    assert "**" not in summary
    assert "`" not in summary
    assert "🌍 ตลาดโลก: เฟดคงดอกเบี้ย ตลาดหุ้นสหรัฐปรับขึ้น\nจับตาเงินเฟ้อ PCE คืนนี้" in summary
