import streamlit as st

from agent import ask_campusai
from feedback_tools import (
    summarize_feedback,
    generate_feedback_report,
)


# --------------------------------------------------
# 1. Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="CampusAI",
    page_icon="🎓",
    layout="wide",
)

st.title("🎓 CampusAI")
st.subheader("University Student Support Assistant")

st.write(
    "An AI-powered university support assistant for document-based "
    "questions, student feedback analysis, and report generation."
)

st.info(
    "University Q&A is handled by the CampusAI agent using the RAG "
    "knowledge base and locally hosted Qwen through Ollama. "
    "Feedback summaries and reports currently use transparent "
    "keyword-based analysis."
)

# --------------------------------------------------
# 2. Navigation
# --------------------------------------------------

feature = st.sidebar.radio(
    "Choose a feature",
    [
        "CampusAI Agent",
        "Summarize Student Feedback",
        "Generate Student Report",
    ],
)

st.divider()

# --------------------------------------------------
# 3. Main agent interface
# --------------------------------------------------

if feature == "CampusAI Agent":
    st.header("Chat with CampusAI Agent")

    st.write(
        "Ask questions about university rules and information, "
        "request a feedback summary, or ask the agent to generate "
        "a student feedback report. The agent selects the relevant tool."
    )

    st.caption(
        "Examples: What are the responsibilities of students? "
        "| Summarize student feedback. | Generate a student feedback report."
    )

    with st.form("campusai_agent_form"):
        user_message = st.text_area(
            "Your question",
            placeholder="What would you like help with?",
            height=100,
        )
        submitted = st.form_submit_button(
            "Ask CampusAI Agent",
            type="primary",
        )

    if submitted:
        if not user_message.strip():
            st.warning("Please enter a question first.")
        else:
            try:
                with st.spinner(
                    "CampusAI Agent is processing your request..."
                ):
                    answer = ask_campusai(user_message.strip())

                if answer and str(answer).strip():
                    st.markdown("### CampusAI Agent's Response")
                    st.markdown(str(answer))
                else:
                    st.warning("The agent returned an empty response.")

            except Exception as error:
                st.error(
                    "The CampusAI Agent encountered an error. "
                    "Check that Ollama is running and that the required "
                    f"model is available. Details: {error}"
                )

# --------------------------------------------------
# 4. Student feedback summarization
# --------------------------------------------------

elif feature == "Summarize Student Feedback":
    st.header("Student Feedback Summary")

    st.write(
        "Identify common feedback themes, concerns, positive experiences, "
        "and comments that do not match the configured themes."
    )

    st.caption(
        "Source: data/student_feedback.csv | "
        "Method: rule-based keyword matching"
    )

    if st.button("Summarize Feedback", type="primary"):
        try:
            with st.spinner("Analyzing the student feedback dataset..."):
                summary = summarize_feedback()

            st.markdown(summary)

        except (FileNotFoundError, ValueError) as error:
            st.error(str(error))
        except Exception as error:
            st.error(f"Could not summarize feedback: {error}")

# --------------------------------------------------
# 5. Automated student feedback report
# --------------------------------------------------

elif feature == "Generate Student Report":
    st.header("Student Feedback Report")

    st.write(
        "Generate a structured report containing positive feedback, "
        "main concerns, suggested actions, and dataset limitations."
    )

    if st.button("Generate Report", type="primary"):
        try:
            with st.spinner("Generating the student feedback report..."):
                report = generate_feedback_report()

            st.markdown(report)

            st.download_button(
                label="Download Report",
                data=report,
                file_name="CampusAI_Student_Feedback_Report.md",
                mime="text/markdown",
            )

        except (FileNotFoundError, ValueError) as error:
            st.error(str(error))
        except Exception as error:
            st.error(f"Could not generate the report: {error}")


# --------------------------------------------------
# 6. Footer
# --------------------------------------------------

st.divider()
st.caption(
    "CampusAI | University information retrieval, "
    "agent-based tool selection, and student feedback reporting"
)

