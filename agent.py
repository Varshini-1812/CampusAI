
from langchain.agents import create_agent
from langchain_ollama import ChatOllama

from agent_tools import CAMPUSAI_TOOLS


# Connect LangChain to the locally running Qwen model.
llm = ChatOllama(
    model="qwen2.5:3b",
    temperature=0,
)


# Create the CampusAI agent.
campusai_agent = create_agent(
    model=llm,
    tools=CAMPUSAI_TOOLS,
    system_prompt="""
You are CampusAI, a university student support assistant.

You have access to three tools:

1. search_university_information:
   Searches university documents for regulations, academic rules,
   courses, examinations, calendars, and student responsibilities.

2. summarize_student_feedback:
   Analyzes the available student feedback and returns a summary.

3. generate_student_report:
   Generates a report based on the available student feedback.

Tool usage instructions:

- For questions about university policies, regulations, courses,
  examinations, academic calendars, or student responsibilities,
  use search_university_information.

- For requests to summarize or analyze student feedback,
  use summarize_student_feedback.

- For requests to generate or create a student feedback report,
  use generate_student_report.

- Do not answer university policy questions using assumptions
  or general knowledge. Use the retrieved university documents.

- Do not invent university rules, dates, statistics, feedback,
  or student comments.

- Preserve important details, factual counts, and original
  comments returned by the tools.

- If the tools cannot find relevant information, clearly explain
  that the information was not found in the available documents.

- If a tool returns an error, do not fabricate an answer.

- Keep responses clear, concise, and helpful.
""",
)


def ask_campusai(message: str) -> str:
    """Send a user message to the CampusAI agent and return its response."""

    if not message or not message.strip():
        return "Please enter a question."

    try:
        result = campusai_agent.invoke(
            {
                "messages": [
                    {"role": "user", "content": message.strip()}
                ]
            }
        )

        # Find the final assistant response.
        for item in reversed(result["messages"]):
            if getattr(item, "type", None) != "ai":
                continue

            content = item.content

            if isinstance(content, str) and content.strip():
                return content.strip()

            # Handle structured content returned by some models.
            if isinstance(content, list):
                text_parts = []

                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            text_parts.append(block.get("text", ""))

                response = "\n".join(
                    part for part in text_parts if part
                ).strip()

                if response:
                    return response

        return "CampusAI could not generate a response."

    except Exception as error:
        print(f"CampusAI agent error: {error}")
        return (
            "CampusAI encountered an error while processing your "
            "question. Please check that Ollama is running and try again."
        )
