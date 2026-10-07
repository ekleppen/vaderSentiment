"""
Checks the hospitality analyzer against the labeled guest SMS messages in
vaderTesting.xlsx (columns: id, input, expected, prediction, sentiment,
polarity_score). ``sentiment`` is the reference label, ``prediction`` is the
alert-worthiness prediction the analyzer may lean on for tough calls, and
``polarity_score`` is what stock VADER produced.
"""
import os

import pytest

openpyxl = pytest.importorskip("openpyxl")

from vaderSentiment.hospitality import HospitalitySentimentAnalyzer, extract_rating, label_for
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

SHEET = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vaderTesting.xlsx")
LABELS = {"positive": "positive", "negative": "negative", "not_opinion": "neutral"}

MIN_ALIGNMENT_WITH_ALERT = 0.95
MIN_ALIGNMENT_TEXT_ONLY = 0.90


def load_rows():
    sheet = openpyxl.load_workbook(SHEET, read_only=True).active
    rows = sheet.iter_rows(values_only=True)
    header = next(rows)
    return [dict(zip(header, row)) for row in rows if row[0] is not None]


ROWS = load_rows() if os.path.exists(SHEET) else []
pytestmark = pytest.mark.skipif(not ROWS, reason="vaderTesting.xlsx not found")


@pytest.fixture(scope="module")
def analyzer():
    return HospitalitySentimentAnalyzer()


def alignment(results):
    misses = [r for r in results if r["got"] != r["want"]]
    report = "\n".join("  id={id} want={want} got={got} compound={compound} alert={alert!r}: {text:.120}".format(**m)
                       for m in misses)
    return 1 - len(misses) / float(len(results)), report


def score_sheet(analyzer, use_alert):
    results = []
    for row in ROWS:
        text = str(row["input"])
        scores = analyzer.polarity_scores(text, row["prediction"] if use_alert else None)
        results.append({"id": row["id"], "want": LABELS[row["sentiment"]], "got": scores["label"],
                        "compound": scores["compound"], "alert": row["prediction"], "text": text})
    return results


def test_sheet_polarity_matches_sentiment_label(analyzer):
    rate, report = alignment(score_sheet(analyzer, use_alert=True))
    assert rate >= MIN_ALIGNMENT_WITH_ALERT, "alignment {:.1%}; misaligned rows:\n{}".format(rate, report)


def test_sheet_polarity_matches_label_from_text_alone(analyzer):
    rate, report = alignment(score_sheet(analyzer, use_alert=False))
    assert rate >= MIN_ALIGNMENT_TEXT_ONLY, "alignment {:.1%}; misaligned rows:\n{}".format(rate, report)


def test_beats_stock_vader_polarity_score(analyzer):
    stock_rate = sum(LABELS[r["sentiment"]] == label_for(r["polarity_score"]) for r in ROWS) / float(len(ROWS))
    rate, _ = alignment(score_sheet(analyzer, use_alert=True))
    assert rate - stock_rate >= 0.25, "stock {:.1%} vs hospitality {:.1%}".format(stock_rate, rate)


@pytest.mark.parametrize("row", ROWS, ids=[str(r["id"]) for r in ROWS])
def test_label_always_agrees_with_compound(analyzer, row):
    scores = analyzer.polarity_scores(str(row["input"]), row["prediction"])
    assert scores["label"] == label_for(scores["compound"])


@pytest.mark.parametrize("text, alert", [
    ("0. I never got a response.", "Definitely alert-worthy"),
    ("0 they never answered my question", "Definitely alert-worthy"),
    ('"9" Room Service was literally unavailable on Friday night. On hold for 15 minutes and gave up',
     "Not alert-worthy"),
    ("(1/2) The remote control in my room is not working. Tv is on but chan (2/2) nels cannot be changed.",
     "Not alert-worthy"),
    ("0 we just got bit by bed bugs and had blood on the sheets the day before this", "Definitely alert-worthy"),
    ("Has been a lovely stay but some suggestions we would provide is that you may need some more staff. I know it "
     "is tough to find people these days but we had drinks at the bar last night and we had to clean the table "
     "ourselves and there was food all over the floor. The woman handling service at the pool was very nice, she "
     "deserves kudos, but she seems like she is covering too many people. Thanks for asking and I hope this helps. "
     "All the best,", "Definitely alert-worthy"),
])
def test_complaints_that_stock_vader_misses(analyzer, text, alert):
    assert analyzer.polarity_scores(text, alert)["label"] == "negative"
    assert analyzer.polarity_scores(text)["compound"] < SentimentIntensityAnalyzer().polarity_scores(text)["compound"]


@pytest.mark.parametrize("text", [
    "8, only because our refrigerator has not been working since we arrived. Everything else has been great.",
    "(1/2) Everything is going well. I give it a 9 . My television has been acting up since I arrived.",
    "Towels felt like sand paper, besides that I'd give it an 8",
    "...I do have a question...the A/C doesn't seem to be working in the room...won't come on, room very warm",
    "I had requested that my room be made up today - that didn't happen",
    "It's been great. Thank you. Our shower doesn't drain very well though. Fyi",
    "Place is fantastic. Only one issue, the dryer doesn't have heat.",
    "Checked out already. Wi-Fi was spotty last night.",
    "(1/2) I'm afraid not good. I had some of the chicken pizza from downstairs and I've been throwing it up all night.",
])
def test_softened_complaints_lean_negative(analyzer, text):
    assert analyzer.polarity_scores(text)["label"] == "negative"


@pytest.mark.parametrize("text", [
    "So far so good....last day, seeing the sights",
    "No I'm good thank you very much",
    "(1/2) 10! You were so accommodating to us evacuees i cant even thank you all enough for your hospitality",
    "It's been a 10. Thank you! We set a check out for 1:30. No need for a bellman.",
])
def test_polite_messages_do_not_read_as_complaints(analyzer, text):
    assert analyzer.polarity_scores(text)["label"] != "negative"


@pytest.mark.parametrize("text", [
    "Hello! Thanks so much for checking in. We are having a wonderful time. Things couldn't be better to celebrate. "
    "Can you please pass on to your general manager how happy we are and kudos to Alex and the entire staff.",
    "(1/2) 5 Wen Wen and her partner were excellent, friendly and very courteous. Love them and loved our room.",
])
def test_alert_worthy_praise_stays_positive(analyzer, text):
    assert analyzer.polarity_scores(text, "Definitely alert-worthy")["label"] == "positive"


@pytest.mark.parametrize("text, rating", [
    ("0. I never got a response.", 0),
    ('"3" - I am a Pres Club Member and did not get "upgraded" at all???', 3),
    ("(1/2) 10!!! We have stayed here in the past", 10),
    ("(Re: )4 it would have been a 5 if it was not for the shooting outside our window", 4),
    ("0/5 the staff are terrible", 0),
    ("8 out of 10 Two outlets don't work", 8),
    ("So far, I will say an 8. Im withholding a 10", 8),
    ("-10, it was very nice", 10),
    ("0ur stay was a 10! Thank you", 10),
    ("0 I checked out Monday, I'm not staying there", None),
    ("0.... lol just kidding everything is great...10", None),
    ("*24 on the phone isn't working", None),
    ("We had to wait 45 min for a table", None),
    ("Well...so far a 3. Reason being, we can not watch ABC", 3),
    ("Solid 7. Room service is a bit of an inconvenience", 7),
    ("Six. When I upgraded I was of the opinion that I would be overlooking the river", 6),
    ("So far so good", None),
    ("Our room was 77 degrees this morning", None),
])
def test_extract_rating(text, rating):
    assert extract_rating(text) == rating
