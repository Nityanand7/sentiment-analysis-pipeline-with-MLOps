const API_URL = "http://127.0.0.1:5002";
const API_KEY = "*****************************";  // keep restricted

// ======= DOM HELPERS =======
const $ = (sel) => document.querySelector(sel);
const out = () => document.getElementById("output");
const log = (...a) => {
  const el = document.getElementById("log");
  if (el) el.textContent += a.join(" ") + "\n";
  console.log(...a);
};

// ======= MAIN =======
document.addEventListener("DOMContentLoaded", async () => {
  log("popup loaded");

  chrome.tabs.query({ active: true, currentWindow: true }, async (tabs) => {
    if (!tabs?.[0]?.url) {
      out().innerHTML = "<p>No active tab.</p>";
      return;
    }

    const url = tabs[0].url;
    log("tab url:", url);

    const m = url.match(/^https:\/\/(?:www\.)?youtube\.com\/watch\?v=([\w-]{11})/);
    if (!m) {
      out().innerHTML = "<p>Open a YouTube watch page.</p>";
      return;
    }

    const videoId = m[1];
    out().innerHTML = sectionHTML("YouTube Video ID", `<p>${videoId}</p><p>Fetching comments…</p>`);

    // 1) Fetch comments
    const comments = await fetchComments(videoId);
    log("comments:", comments.length);
    if (!comments.length) {
      out().innerHTML += "<p>No comments found (quota / restrictions / no public comments).</p>";
      return;
    }

    out().innerHTML += `<p>Fetched ${comments.length} comments. Performing sentiment analysis…</p>`;

    // 2) Predictions (array: [{comment, sentiment, timestamp}, ...])
    const predictions = await predictWithTimestamps(comments);
    if (!Array.isArray(predictions)) {
      out().innerHTML += "<p>Prediction failed. See log.</p>";
      return;
    }

    // Build counts & sentimentData for charts
    const sentimentCounts = { "1": 0, "0": 0, "-1": 0 };
    const sentimentData = [];
    let totalSentimentScore = 0;
    predictions.forEach((p) => {
      sentimentCounts[p.sentiment] = (sentimentCounts[p.sentiment] || 0) + 1;
      sentimentData.push({ timestamp: p.timestamp, sentiment: parseInt(p.sentiment) });
      totalSentimentScore += parseInt(p.sentiment);
    });

    // 3) Render “Comment Analysis Summary”
    const totalComments = comments.length;
    const uniqueCommenters = new Set(comments.map((c) => c.authorId)).size;
    const totalWords = comments.reduce(
      (sum, c) => sum + c.text.split(/\s+/).filter((w) => w.length > 0).length,
      0
    );
    const avgWordLength = (totalWords / totalComments).toFixed(2);
    const avgSentimentScore = (totalSentimentScore / totalComments).toFixed(2);
    const normalizedSentimentScore = (((parseFloat(avgSentimentScore) + 1) / 2) * 10).toFixed(2);

    out().innerHTML += sectionHTML(
      "Comment Analysis Summary",
      `
      <div class="metrics-container">
        <div class="metric"><div class="metric-title">Total Comments</div><div class="metric-value">${totalComments}</div></div>
        <div class="metric"><div class="metric-title">Unique Commenters</div><div class="metric-value">${uniqueCommenters}</div></div>
        <div class="metric"><div class="metric-title">Avg Comment Length</div><div class="metric-value">${avgWordLength} words</div></div>
        <div class="metric"><div class="metric-title">Avg Sentiment Score</div><div class="metric-value">${normalizedSentimentScore}/10</div></div>
      </div>
    `
    );

    // 4) Sentiment distribution (pie) via /generate_chart
    out().innerHTML += sectionHTML(
      "Sentiment Analysis Results",
      `<p>See the pie chart below for sentiment distribution.</p><div id="chart-container"></div>`
    );
    await fetchAndDisplayChart(sentimentCounts);

    // 5) Sentiment trend graph via /generate_trend_graph
    out().innerHTML += sectionHTML(
      "Sentiment Trend Over Time",
      `<div id="trend-graph-container"></div>`
    );
    await fetchAndDisplayTrendGraph(sentimentData);

    // 6) Word cloud
    out().innerHTML += sectionHTML(
      "Comment Wordcloud",
      `<div id="wordcloud-container"></div>`
    );
    await fetchAndDisplayWordCloud(comments.map((c) => c.text));

    // 7) Top comments list (quick view)
    out().innerHTML += sectionHTML(
      "Top 25 Comments with Sentiments",
      `<ul class="comment-list">
        ${predictions.slice(0, 25).map((item, i) =>
          `<li class="comment-item"><span>${i + 1}. ${escapeHTML(item.comment)}</span><br>
           <span class="comment-sentiment">Sentiment: ${item.sentiment}</span></li>`).join("")}
       </ul>`
    );

    // 8) Deeper Insights (NEW) via /insights
    const insights = await fetchInsights(comments);
    renderInsights(insights);
  });
});

// ======= RENDER HELPERS =======
function sectionHTML(title, inner) {
  return `
    <div class="section">
      <div class="section-title">${title}</div>
      ${inner}
    </div>
  `;
}

function escapeHTML(s) {
  return (s || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

// ======= BACKEND CALLS =======
async function fetchComments(videoId) {
  try {
    let comments = [];
    let pageToken = "";
    while (comments.length < 500) {
      const url =
        `https://www.googleapis.com/youtube/v3/commentThreads` +
        `?part=snippet&videoId=${videoId}&maxResults=100&pageToken=${pageToken}&key=${API_KEY}`;
      const r = await fetch(url);
      if (!r.ok) {
        log("YT status", r.status);
        break;
      }
      const j = await r.json();
      (j.items || []).forEach((it) => {
        const s = it.snippet.topLevelComment.snippet;
        comments.push({
          text: s.textOriginal,
          timestamp: s.publishedAt,
          authorId: s.authorChannelId?.value || "Unknown",
        });
      });
      pageToken = j.nextPageToken || "";
      if (!pageToken) break;
    }
    return comments;
  } catch (e) {
    log("YT error", e);
    return [];
  }
}

async function predictWithTimestamps(comments) {
  try {
    const r = await fetch(`${API_URL}/predict_with_timestamps`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ comments }),
    });
    log("predict status", r.status);
    const j = await r.json().catch(() => ({}));
    if (!r.ok) log("predict err", JSON.stringify(j));
    return r.ok ? j : null;
  } catch (e) {
    log("predict exception", e);
    return null;
  }
}

async function fetchAndDisplayChart(sentimentCounts) {
  try {
    const r = await fetch(`${API_URL}/generate_chart`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sentiment_counts: sentimentCounts }),
    });
    if (!r.ok) throw new Error("chart fetch failed");
    const blob = await r.blob();
    const imgURL = URL.createObjectURL(blob);
    const img = document.createElement("img");
    img.src = imgURL;
    img.style.width = "100%";
    img.style.marginTop = "12px";
    $("#chart-container").appendChild(img);
  } catch (e) {
    log("chart error", e);
  }
}

async function fetchAndDisplayWordCloud(comments) {
  try {
    const r = await fetch(`${API_URL}/generate_wordcloud`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ comments }),
    });
    if (!r.ok) throw new Error("wordcloud fetch failed");
    const blob = await r.blob();
    const imgURL = URL.createObjectURL(blob);
    const img = document.createElement("img");
    img.src = imgURL;
    img.style.width = "100%";
    img.style.marginTop = "12px";
    $("#wordcloud-container").appendChild(img);
  } catch (e) {
    log("wordcloud error", e);
  }
}

async function fetchAndDisplayTrendGraph(sentimentData) {
  try {
    const r = await fetch(`${API_URL}/generate_trend_graph`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sentiment_data: sentimentData }),
    });
    if (!r.ok) throw new Error("trend fetch failed");
    const blob = await r.blob();
    const imgURL = URL.createObjectURL(blob);
    const img = document.createElement("img");
    img.src = imgURL;
    img.style.width = "100%";
    img.style.marginTop = "12px";
    $("#trend-graph-container").appendChild(img);
  } catch (e) {
    log("trend error", e);
  }
}

// ======= NEW INSIGHTS =======
async function fetchInsights(comments) {
  try {
    const r = await fetch(`${API_URL}/insights`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ comments }),
    });
    if (!r.ok) {
      log("insights status", r.status);
      return null;
    }
    return r.json();
  } catch (e) {
    log("insights exception", e);
    return null;
  }
}

function renderInsights(ins) {
  if (!ins) return;
  const s = ins.summary;

  // Summary + keywords/bigrams/commenters + top pos/neg
  const html = `
    <div class="metrics-container">
      <div class="metric"><div class="metric-title">Unique Commenters</div><div class="metric-value">${s.unique_commenters}</div></div>
      <div class="metric"><div class="metric-title">Avg Words/Comment</div><div class="metric-value">${s.avg_comment_length}</div></div>
      <div class="metric"><div class="metric-title">Avg Sentiment</div><div class="metric-value">${s.avg_sentiment_score_0_10}/10</div></div>
    </div>

    <div style="margin-top:8px">
      <b>Top keywords</b>: ${ins.top_words.slice(0,10).map(escapeHTML).join(", ")}<br/>
      <b>Top bigrams</b>: ${ins.top_bigrams.slice(0,10).map(escapeHTML).join(", ")}
    </div>

    <div style="margin-top:12px">
      <b>Top commenters</b>:
      <ul>${ins.top_commenters.map(c => `<li>${escapeHTML(c.authorId)} — ${c.count}</li>`).join("")}</ul>
    </div>

    <div style="margin-top:12px">
      <b>Best comments</b>:
      <ul>${ins.top_positive.slice(0,5).map(c => `<li>👍 ${escapeHTML(c.text)}</li>`).join("")}</ul>
      <b>Critical comments</b>:
      <ul>${ins.top_negative.slice(0,5).map(c => `<li>👎 ${escapeHTML(c.text)}</li>`).join("")}</ul>
    </div>
  `;
  out().innerHTML += sectionHTML("Deeper Insights", html);

  // Tiny inline hourly sparkline (optional, no lib)
  const pts = (ins.by_hour || []).map((p, i) => `${(i / 23) * 300},${50 - (p.y * 50)}`).join(" ");
  const svg = `<svg width="300" height="60"><polyline fill="none" stroke="#4fc3f7" stroke-width="2" points="${pts}"/></svg>`;
  out().innerHTML += sectionHTML("Hourly Sentiment (avg)", svg);
}