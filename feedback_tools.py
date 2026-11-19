from pathlib import Path
import csv


BASE_DIR = Path(__file__).resolve().parent
FEEDBACK_FILE = BASE_DIR / "data" / "student_feedback.csv"

# Transparent keyword-based theme detection.
# These rules identify topics, not statistically representative opinions.
THEMES = {
    "Hostel mess food": ["food", "mess", "meal", "meals"],
    "Hostel Wi-Fi": ["wi-fi", "wifi", "internet"],
    "Library timings": ["library", "opening hours", "library hours"],
    "Teaching support": [
        "professor", "professors", "teacher", "teachers",
        "faculty", "supportive",
    ],
    "University events": [
        "tech fest", "cultural fest", "event", "events", "sessions",
    ],
}

CONCERN_THEMES = [
    "Hostel mess food",
    "Hostel Wi-Fi",
    "Library timings",
]
POSITIVE_THEMES = [
    "Teaching support",
    "University events",
]
RECOMMENDATIONS = {
    "Hostel mess food":
        "Review mess food quality and collect more specific student feedback.",
    "Hostel Wi-Fi":
        "Check hostel network performance and investigate connectivity issues.",
    "Library timings":
        "Assess demand for extended library opening hours.",
    "Teaching support":
        "Continue supporting effective teaching and student-faculty interaction.",
    "University events":
        "Continue gathering feedback to improve future university events.",
}
def load_feedback():
    """Load non-empty feedback comments from the project CSV file."""
    if not FEEDBACK_FILE.exists():
        raise FileNotFoundError(
            f"Feedback file not found: {FEEDBACK_FILE}. "
            "Make sure data/student_feedback.csv exists."
        )

    with FEEDBACK_FILE.open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:
        reader = csv.DictReader(file)

        if not reader.fieldnames or "Feedback" not in reader.fieldnames:
            raise ValueError(
                "The feedback CSV must contain a column named 'Feedback'."
            )

        comments = [
            (row.get("Feedback") or "").strip()
            for row in reader
            if (row.get("Feedback") or "").strip()
        ]

    if not comments:
        raise ValueError(
            "No non-empty student feedback comments were found in the CSV."
        )

    return comments

def analyze_feedback():
    """
    Match comments to configured themes using case-insensitive keywords.

    A comment can match more than one theme. This is rule-based topic
    matching; it is not an LLM-based sentiment classifier.
    """
    comments = load_feedback()
    theme_comments = {theme: [] for theme in THEMES}
    unclassified_comments = []

    for comment in comments:
        normalized = comment.casefold()
        matched_any_theme = False

        for theme, keywords in THEMES.items():
            if any(keyword.casefold() in normalized for keyword in keywords):
                theme_comments[theme].append(comment)
                matched_any_theme = True

        if not matched_any_theme:
            unclassified_comments.append(comment)

    return comments, theme_comments, unclassified_comments

def _append_theme_section(lines, heading, themes, theme_comments):
    """Append a theme section and its original comment evidence."""
    lines.extend(["", heading])

    found = False
    for theme in themes:
        matching = theme_comments[theme]
        if not matching:
            continue

        found = True
        lines.append(f"- **{theme}:** {len(matching)} matching comment(s).")
        for comment in matching:
            lines.append(f'  - Evidence: "{comment}"')

    if not found:
        lines.append("- No comments matched these configured themes.")

def summarize_feedback():
    """Return a concise, evidence-based summary of the feedback CSV."""
    comments, theme_comments, unclassified = analyze_feedback()

    lines = [
        "### Student Feedback Summary",
        "",
        f"**Feedback comments analyzed:** {len(comments)}",
        "",
        "Themes are identified using configured keyword rules. "
        "A comment may match more than one theme, so theme counts "
        "may not add up to the total number of comments.",
    ]

    _append_theme_section(
        lines, "#### Main Concern Themes",
        CONCERN_THEMES, theme_comments
    )
    _append_theme_section(
        lines, "#### Positive Feedback Themes",
        POSITIVE_THEMES, theme_comments
    )

    lines.extend(["", "#### Other Comments"])
    if unclassified:
        for comment in unclassified:
            lines.append(f'- "{comment}"')
    else:
        lines.append("- All comments matched at least one configured theme.")

    lines.extend([
        "",
        "#### Limitations",
        "- Theme detection is keyword-based, not semantic or LLM-based.",
        "- Keyword matching may miss relevant wording or match a word "
        "without understanding its context.",
        "- The categories are configured in code and may need to be "
        "updated for a different feedback dataset.",
        "- The results describe only the comments in this CSV and may "
        "not represent the wider student population.",
    ])

    return "\n".join(lines)

def generate_feedback_report():
    """Generate a structured report using measured counts and original comments."""
    comments, theme_comments, unclassified = analyze_feedback()

    lines = [
        "# CampusAI Student Feedback Report",
        "",
        f"**Feedback comments analyzed:** {len(comments)}",
        "",
        "## 1. Positive Feedback",
    ]

    found_positive = False
    for theme in POSITIVE_THEMES:
        matching = theme_comments[theme]
        if matching:
            found_positive = True
            lines.append(f"### {theme} — {len(matching)} matching comment(s)")
            for comment in matching:
                lines.append(f'- "{comment}"')

    if not found_positive:
        lines.append("No comments matched the configured positive feedback themes.")

    lines.extend(["", "## 2. Main Concerns"])

    found_concern = False
    for theme in CONCERN_THEMES:
        matching = theme_comments[theme]
        if matching:
            found_concern = True
            lines.append(f"### {theme} — {len(matching)} matching comment(s)")
            for comment in matching:
                lines.append(f'- "{comment}"')

    if not found_concern:
        lines.append("No comments matched the configured concern themes.")

    lines.extend([
        "",
        "## 3. Suggested Actions",
        "",
        "These are general suggestions based on the themes detected. "
        "They are not actions explicitly requested by every student.",
    ])

    any_recommendations = False
    for theme, recommendation in RECOMMENDATIONS.items():
        if theme_comments[theme]:
            any_recommendations = True
            lines.append(f"- **{theme}:** {recommendation}")

    if not any_recommendations:
        lines.append(
            "- Review the submitted comments and gather more detailed feedback."
        )

    lines.extend([
        "",
        "## 4. Other Comments",
    ])
    if unclassified:
        lines.extend(f'- "{comment}"' for comment in unclassified)
    else:
        lines.append("No unclassified comments.")

    lines.extend([
        "",
        "## 5. Overall Observations",
        "",
        "This report describes only the submitted comments. A theme "
        "appearing once is an individual observation, not evidence of "
        "a recurring or widespread problem. Theme counts can overlap "
        "because one comment may match multiple themes. No overall "
        "satisfaction score or percentage is calculated.",
        "",
        "## 6. Limitations",
        "",
        "- Theme identification is based on keyword matching, not "
        "LLM-based semantic analysis.",
        "- Keyword matching may miss context, negation, sarcasm, or "
        "alternative wording.",
        "- The dataset may not represent the wider student population.",
        "- No academic year is assigned because the dataset does not "
        "specify one.",
    ])

    return "\n".join(lines)


