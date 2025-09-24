# app.py
import matplotlib
matplotlib.use("Agg")  # headless backend

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import io
import re
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --------------------------
# App setup
# --------------------------
app = Flask(__name__)
CORS(app)  # allow chrome extension

# --------------------------
# Load model + vectorizer
# --------------------------
def load_model(model_path, vectorizer_path):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

# Adjust paths if needed (they’re relative to where you run app.py)
model, vectorizer = load_model("./lgbm_model.pkl", "./tfidf_vectorizer.pkl")

# --------------------------
# Text preprocessing
# --------------------------
def preprocess_comment(comment: str) -> str:
    try:
        comment = (comment or "").lower().strip()
        comment = re.sub(r"\n", " ", comment)
        comment = re.sub(r"[^A-Za-z0-9\s!?.,]", "", comment)
        keep_words = {"not", "but", "however", "no", "yet"}
        sw = set(stopwords.words("english")) - keep_words
        comment = " ".join(w for w in comment.split() if w not in sw)
        lemma = WordNetLemmatizer()
        comment = " ".join(lemma.lemmatize(w) for w in comment.split())
        return comment
    except Exception:
        return comment or ""

# --------------------------
# Routes
# --------------------------
@app.get("/")
def home():
    return "Welcome to YouTube sentiment API"

@app.post("/predict")
def predict():
    data = request.get_json(force=True) or {}
    comments = data.get("comments", [])
    if not comments:
        return jsonify({"error": "No comments provided"}), 400

    texts = [preprocess_comment(c) for c in comments]
    X = vectorizer.transform(texts)
    try:
        preds = model.predict(X)
    except Exception:
        preds = model.predict(X.toarray())

    return jsonify([{"comment": c, "sentiment": str(p)} for c, p in zip(comments, preds)])

@app.post("/predict_with_timestamps")
def predict_with_timestamps():
    """
    Expects:
      { "comments": [ { "text": "...", "timestamp": "...", "authorId": "..." }, ... ] }
    Returns:
      [ { "comment": "...", "sentiment": "-1|0|1", "timestamp": "..." }, ... ]
    """
    data = request.get_json(force=True) or {}
    items = data.get("comments")
    if not items or not isinstance(items, list):
        return jsonify({"error": "No comments provided"}), 400

    texts = [preprocess_comment(i.get("text", "")) for i in items]
    ts = [i.get("timestamp") for i in items]

    X = vectorizer.transform(texts)
    try:
        preds = model.predict(X)
    except Exception:
        preds = model.predict(X.toarray())
    preds = [str(p) for p in preds]

    # If your model returns label strings, map them:
    mapped = []
    for p in preds:
        pl = p.lower()
        if pl in {"positive", "neutral", "negative"}:
            p = {"positive": "1", "neutral": "0", "negative": "-1"}[pl]
        mapped.append(p)

    return jsonify([
        {"comment": i.get("text", ""), "sentiment": m, "timestamp": t}
        for i, m, t in zip(items, mapped, ts)
    ])

@app.post("/generate_chart")
def generate_chart():
    """Pie chart for sentiment distribution. Body: { sentiment_counts: {"1":n,"0":m,"-1":k} }"""
    data = request.get_json(force=True) or {}
    counts = data.get("sentiment_counts", {})
    if not counts:
        return jsonify({"error": "No sentiment counts provided"}), 400

    labels = ["Positive", "Neutral", "Negative"]
    sizes = [int(counts.get("1", 0)), int(counts.get("0", 0)), int(counts.get("-1", 0))]
    if sum(sizes) == 0:
        return jsonify({"error": "Counts sum to zero"}), 400

    colors = ["#36A2EB", "#C9CBCF", "#FF6384"]
    plt.figure(figsize=(6, 6))
    plt.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=140, textprops={"color": "w"})
    plt.axis("equal")
    buf = io.BytesIO()
    plt.savefig(buf, format="PNG", transparent=True)
    buf.seek(0)
    plt.close()
    return send_file(buf, mimetype="image/png")

@app.post("/generate_wordcloud")
def generate_wordcloud():
    """Word cloud from comments. Body: { comments: ["...","..."] }"""
    from wordcloud import WordCloud
    data = request.get_json(force=True) or {}
    comments = data.get("comments", [])
    if not comments:
        return jsonify({"error": "No comments provided"}), 400

    text = " ".join(preprocess_comment(c) for c in comments)
    wc = WordCloud(width=800, height=400, background_color="black", colormap="Blues",
                   stopwords=set(stopwords.words("english")), collocations=False).generate(text)
    buf = io.BytesIO()
    wc.to_image().save(buf, format="PNG")
    buf.seek(0)
    return send_file(buf, mimetype="image/png")

@app.post("/generate_trend_graph")
def generate_trend_graph():
    """
    Line chart over time. Body:
      { "sentiment_data": [ {"timestamp":"...","sentiment":-1|0|1}, ... ] }
    """
    data = request.get_json(force=True) or {}
    sd = data.get("sentiment_data", [])
    if not sd:
        return jsonify({"error": "No sentiment data provided"}), 400

    df = pd.DataFrame(sd)
    if "timestamp" not in df or "sentiment" not in df:
        return jsonify({"error": "Bad payload"}), 400

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df["sentiment"] = pd.to_numeric(df["sentiment"], errors="coerce").fillna(0).astype(int)

    # resample to hour to smooth
    df = df.set_index("timestamp").resample("H")["sentiment"].mean()

    plt.figure(figsize=(10, 4))
    plt.plot(df.index, df.values, marker="o")
    plt.title("Sentiment Trend")
    plt.ylabel("Avg sentiment (-1..1)")
    plt.grid(True)
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
    plt.xticks(rotation=45)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="PNG")
    buf.seek(0)
    plt.close()
    return send_file(buf, mimetype="image/png")

@app.post("/insights")
def insights():
    """
    Rich analytics panel used by popup.
    Body: { "comments": [ {"text":"...", "timestamp":"...", "authorId":"..."}, ... ] }
    """
    data = request.get_json(force=True) or {}
    comments = data.get("comments", [])
    if not comments:
        return jsonify({"error": "No comments provided"}), 400

    raw_texts = [c.get("text", "") for c in comments]
    texts = [preprocess_comment(t) for t in raw_texts]
    X = vectorizer.transform(texts)
    try:
        preds = model.predict(X)
    except Exception:
        preds = model.predict(X.toarray())
    preds = [str(p) for p in preds]
    # normalize labels to -1/0/1 strings
    preds = [ {"positive":"1","neutral":"0","negative":"-1"}.get(p.lower(), p) for p in preds ]

    total = len(comments)
    unique_commenters = len({c.get("authorId") for c in comments if c.get("authorId")})
    avg_words = float(np.mean([len((c.get("text","")).split()) for c in comments])) if total else 0.0
    avg_score = float(np.mean([int(p) for p in preds])) if preds else 0.0  # -1..1

    summary = {
        "total": total,
        "unique_commenters": unique_commenters,
        "avg_comment_length": round(avg_words, 2),
        "avg_sentiment_score_0_10": round(((avg_score + 1) / 2) * 10, 2),
    }

    dist = dict(Counter(preds))
    ts = pd.to_datetime([c.get("timestamp") for c in comments], errors="coerce")
    by_hour = (
        pd.DataFrame({"h": ts.dt.hour, "y": [int(p) for p in preds]})
        .dropna()
        .groupby("h")["y"].mean()
        .reindex(range(24), fill_value=0.0)
        .round(3)
        .reset_index()
        .to_dict(orient="records")
    )

    # top comments
    scored = list(zip(raw_texts, [int(p) for p in preds]))
    top_positive = [{"text": t, "score": s} for t, s in sorted(scored, key=lambda x: x[1], reverse=True)[:10]]
    top_negative = [{"text": t, "score": s} for t, s in sorted(scored, key=lambda x: x[1])[:10]]

    # keywords / bigrams
    import string as _string
    sw = set(stopwords.words("english"))
    toks = []
    for t in texts:
        toks.extend([w for w in t.split() if w not in sw and w not in _string.punctuation and len(w) > 2])
    top_words = [w for w, _ in Counter(toks).most_common(20)]

    bigrams = []
    for t in texts:
        ws = [w for w in t.split() if w]
        bigrams += list(zip(ws, ws[1:]))
    top_bigrams = [" ".join(bg) for bg, _ in Counter(bigrams).most_common(20)]

    top_commenters = [
        {"authorId": a, "count": c}
        for a, c in Counter([c.get("authorId","Unknown") for c in comments]).most_common(10)
    ]

    return jsonify({
        "summary": summary,
        "distribution": {"-1": dist.get("-1", 0), "0": dist.get("0", 0), "1": dist.get("1", 0)},
        "by_hour": by_hour,
        "top_positive": top_positive,
        "top_negative": top_negative,
        "top_words": top_words,
        "top_bigrams": top_bigrams,
        "top_commenters": top_commenters
    })

# --------------------------
# Entrypoint
# --------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002, debug=True)