import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def use_temp_hillstrom_csv(monkeypatch, tmp_path):
    rows = []
    segments = ["No E-Mail", "Mens E-Mail", "Womens E-Mail", "Mens E-Mail"]
    zip_codes = ["Urban", "Surburban", "Rural", "Urban"]
    channels = ["Web", "Phone", "Multichannel", "Web"]
    history_segments = [
        "1) $0 - $100",
        "2) $100 - $200",
        "3) $200 - $350",
        "4) $350 - $500",
    ]

    for i in range(100):
        segment = segments[i % len(segments)]
        treatment = int(segment != "No E-Mail")
        conversion = int((i % 5 == 0) or (treatment == 1 and i % 7 == 0))

        rows.append(
            {
                "recency": (i % 12) + 1,
                "history_segment": history_segments[i % len(history_segments)],
                "history": 100.0 + (i * 13.5),
                "mens": int(i % 2 == 0),
                "womens": int(i % 3 != 0),
                "zip_code": zip_codes[i % len(zip_codes)],
                "newbie": int(i % 4 == 0),
                "channel": channels[i % len(channels)],
                "segment": segment,
                "visit": int(i % 3 == 0),
                "conversion": conversion,
                "spend": float(conversion * (20 + (i % 6) * 5)),
            }
        )

    csv_path = tmp_path / "hillstrom.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    monkeypatch.setattr("src.data_loader.find_hillstrom_csv", lambda data_dir=None: csv_path)
    return csv_path
