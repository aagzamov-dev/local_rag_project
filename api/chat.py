import sys
from rag_core import get_llm, query_documents, format_prompt

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
        retrieved = query_documents(query, top_k=5)
        
        if not retrieved:
            print("Assistant: I found no relevant documents in the database.")
            continue
            
        prompt = format_prompt(query, retrieved)
        
        print("\nAssistant: ", end="", flush=True)
        stream = llm.create_completion(
            prompt,
            max_tokens=1024,
            stop=["<|im_end|>", "User:", "Question:"],
            stream=True,
            temperature=0.1,
            repeat_penalty=1.1
        )
        
        for output in stream:
            text = output['choices'][0]['text']
            print(text, end="", flush=True)
        print() # Newline

if __name__ == "__main__":
    # To run: python chat.py
    main()
