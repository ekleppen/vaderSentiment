"""
Score every message in more_mssc_samples.xlsx (Sheet2) with stock VADER and the
hospitality analyzer, and write the per-message results to an .xlsx file.

    .venv/bin/python score_samples.py [input.xlsx] [output.xlsx]
"""
import sys

import pandas as pd

from vaderSentiment.hospitality import HospitalitySentimentAnalyzer, label_for

LABELS = {"positive": "positive", "negative": "negative", "not_opinion": "neutral"}


def main(src="more_mssc_samples.xlsx", dst="more_mssc_samples_scored.xlsx"):
    df = pd.read_excel(src, sheet_name="Sheet2", header=1)
    df = df[df.msg_text.notna()][["company_id", "convo_id", "msg_id", "msg_text", "sentiment", "polarity_score"]]
    df = df.rename(columns={"polarity_score": "stock_compound"})
    in_test_sheet = set(pd.read_excel("vaderTesting.xlsx").id)

    analyzer = HospitalitySentimentAnalyzer()
    scores = [analyzer.polarity_scores(str(text)) for text in df.msg_text]
    df["expected_label"] = df.sentiment.map(LABELS)
    df["stock_label"] = df.stock_compound.map(label_for)
    df["hosp_compound"] = [s["compound"] for s in scores]
    df["hosp_label"] = [s["label"] for s in scores]
    df["hosp_pos"] = [s["pos"] for s in scores]
    df["hosp_neg"] = [s["neg"] for s in scores]
    df["rating"] = [s["rating"] for s in scores]
    df["stock_matches"] = df.stock_label == df.expected_label
    df["hosp_matches"] = df.hosp_label == df.expected_label
    df["in_test_sheet"] = df.msg_id.isin(in_test_sheet)

    holdout = df[~df.in_test_sheet]
    summary = pd.DataFrame([
        {"expected_label": label, "messages": int((holdout.expected_label == label).sum()),
         "stock_accuracy": holdout.stock_matches[holdout.expected_label == label].mean(),
         "hosp_accuracy": holdout.hosp_matches[holdout.expected_label == label].mean()}
        for label in ["positive", "negative", "neutral"]
    ] + [{"expected_label": "all", "messages": len(holdout),
          "stock_accuracy": holdout.stock_matches.mean(), "hosp_accuracy": holdout.hosp_matches.mean()}])

    with pd.ExcelWriter(dst) as writer:
        summary.to_excel(writer, sheet_name="summary (holdout)", index=False)
        df.to_excel(writer, sheet_name="scored", index=False)
    print(summary.to_string(index=False))
    print("wrote {} rows to {}".format(len(df), dst))


if __name__ == "__main__":
    main(*sys.argv[1:])
