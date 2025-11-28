import sys

from agent import answer_question

def run_cli():
    print("=== FAQ AI Agent (CLI Mode) ===")
    print("Ask anything about the company policies / FAQs.")
    print("Type 'exit' to quit.\n")

    while True:
        user_q = input("You: ")
        if user_q.lower().strip() in {"exit", "quit"}:
            print("Goodbye 👋")
            break

        try:
            ans = answer_question(user_q)
            print(f"Agent: {ans}\n")
        except Exception as e:
            print(f"[Error] {e}\n")


def run_streamlit():
    import streamlit as st

    st.set_page_config(page_title="FAQ AI Agent", page_icon="🤖")
    st.title("🤖 FAQ AI Agent")
    st.write("Ask questions based on the internal FAQ document (`faq.txt`).")

    user_q = st.text_input("Your question:")

    if st.button("Ask") and user_q.strip():
        with st.spinner("Thinking..."):
            try:
                ans = answer_question(user_q)
                st.markdown("### Answer")
                st.write(ans)
            except Exception as e:
                st.error(f"Error: {e}")


if __name__ == "__main__":
    # If user runs: python app.py --cli  → CLI mode
    # Else: Streamlit mode
    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        run_cli()
    else:
        # To launch streamlit: streamlit run app.py
        # (This block is not used directly when running as `streamlit run app.py`,
        # but we keep it for safety.)
        run_streamlit()
