from financial_strategy.data.company_events import classify_company_event
def test_periodic_filing_is_classified_as_earnings_only() -> None:
    result = classify_company_event(
        title="Example Corp 10-Q",
        text="The risk factors mention cybersecurity and acquisitions.",
        filing_form="10-Q",
    )

    assert result.event_types == ("earnings",)


def test_company_event_classifier_supports_multiple_high_impact_types() -> None:
    result = classify_company_event(
        title="Quarterly results and corporate updates",
        text=(
            "The company reported quarterly results and raised its guidance. "
            "The board approved a share repurchase program and declared a "
            "quarterly dividend. The company also disclosed a cybersecurity "
            "incident involving unauthorized access."
        ),
        filing_form="8-K",
    )

    assert result.event_types == (
        "earnings",
        "guidance",
        "buyback",
        "dividend",
        "cybersecurity",
    )
