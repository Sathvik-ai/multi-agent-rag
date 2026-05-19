from src.agents.reasoning import ReasoningAgent

def test_full_pipeline():
    print("Initializing Reasoning Agent...")
    agent = ReasoningAgent()
    
    # We ingested the Attention Is All You Need paper and a medical trials CSV.
    # Let's ask a question related to the medical trials CSV!
    question = "Which drug is targeting Asthma and what is its efficacy score?"
    
    print(f"\n[?] User Question: {question}")
    print("\nRetrieving context and formulating answer...")
    
    # Run the full retrieval and reasoning pipeline
    response = agent.ask(question)
    
    print("\n" + "="*50)
    print("🤖 GEMINI AGENT RESPONSE")
    print("="*50)
    print(response["answer"])
    print("\n" + "="*50)
    
    print("\n📚 Sources Retrieved (Top 3):")
    for idx, source in enumerate(response["sources"][:3], 1):
        print(f"\nSource {idx} [Score: {source['score']:.4f}]:")
        print(f"Title: {source['title']}")
        print(f"Text Snippet: {source['text'][:150]}...")

if __name__ == "__main__":
    test_full_pipeline()
