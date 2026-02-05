import time
from memory_manager import AgentMemoryCore


def main():
    """
    Test script for the AgentMemoryCore system.
    Simulates a conversation and tests memory recall.
    """
    print("=" * 60)
    print("AgentMemoryCore Test Script")
    print("=" * 60)
    
    # Initialize the memory core
    print("\n[Test] Initializing AgentMemoryCore...")
    core = AgentMemoryCore()
    
    # Simulate a conversation
    print("\n[Test] Starting conversation simulation...")
    conversations = [
        ("user", "Hi"),
        ("assistant", "Hello! How can I help you today?"),
        ("user", "1998年 I hate coriander, remember that."),
        ("assistant", "Got it! I'll remember that you don't like coriander."),
        ("user", "2025年 Also, I prefer spicy food."),
    ]
    
    for role, content in conversations:
        core.add_memory(role, content)
        time.sleep(0.5)  # Small delay for readability
    
    # Wait for flush to complete
    print("\n[Test] Waiting for flush to complete...")
    time.sleep(2)
    
    # Test memory consolidation
    print("\n" + "=" * 60)
    print("MEMORY CONSOLIDATION TEST")
    print("=" * 60)
    
    print("\n[Test] Adding more similar memories for consolidation test...")
    more_conversations = [
        ("user", "I really dislike cilantro in my food"),
        ("user", "Spicy dishes are my favorite"),
        ("user", "Can't stand coriander, it tastes like soap"),
        ("user", "The spicier the better for me"),
        ("user", "Please no cilantro in any of my meals")
    ]
    
    for role, content in more_conversations:
        core.add_memory(role, content)
        time.sleep(0.3)
    
    print("\n[Test] Running consolidation...")
    final_count = core.consolidate_memories(similarity_threshold=0.7)
    print(f"\n[Test] Final memory count: {final_count}")
    
    # Perform a recall search
    print("\n" + "=" * 60)
    print("RECALL TEST")
    print("=" * 60)
    
    print("\n[Test] Performing recall query...")
    query = "2025年 我喜欢吃什么？"
    print(f"Query: {query}")
    
    result = core.recall(query, top_k=10)
    
    # Print results
    print("\n" + "=" * 60)
    print("RECALL RESULTS")
    print("=" * 60)
    
    print("\n--- Short-Term Memory (Recent Context) ---")
    if result["short_term"]:
        for i, item in enumerate(result["short_term"], 1):
            print(f"{i}. [{item['role']}] {item['content']}")
    else:
        print("(empty)")
    
    print("\n--- Long-Term Memory (Relevant Insights) ---")
    if result["long_term"]:
        for i, item in enumerate(result["long_term"], 1):
            # Display original content without tags suffix
            original_content = item['metadata'].get('original_content', item['document'])
            print(f"{i}. {original_content}")
            tags = item['metadata'].get('tags', 'N/A')
            print(f"   Tags: {tags}")
            if 'distance' in item:
                print(f"   Distance: {item['distance']:.4f}")
    else:
        print("(empty)")
    
    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
