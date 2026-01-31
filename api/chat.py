import sys
from rag_core import get_llm, query_documents, run_local_agent, finalize_answer

def main():
    print("Initializing RAG Chat System... (Loading Model)")
    try:
        llm = get_llm()
    except Exception as e:
        print(f"Error initializing model: {e}")
        return

    print("Ready! Type 'exit' to quit.")
    
    while True:
        query = input("\nUser: ")
        if query.lower() in ["exit", "quit"]:
            break
        
        if not query.strip():
            continue
            
        print("Retrieving context...")
        retrieved = query_documents(query)

        response_text, tool_used, tool_trace = run_local_agent(query, retrieved, llm)
        final_response = finalize_answer(
            response_text, retrieved, tool_used=tool_used, tool_trace=tool_trace
        )

        print("\nAssistant:", final_response)

if __name__ == "__main__":
    # To run: python chat.py
    main()
