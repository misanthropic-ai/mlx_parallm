#!/usr/bin/env python3
"""Demo script for Extended Mind Transformers."""

import mlx.core as mx
from mlx_parallm.utils import load, generate
import uuid


def main():
    """Run the Extended Mind Transformers demo."""
    
    print("Extended Mind Transformers Demo")
    print("=" * 50)
    print("\nThis demo shows how external memories can improve LLM factual accuracy.")
    print("\nLoading model...")
    
    # Load model - you can change this to your preferred model
    # model_path = "NousResearch/DeepHermes-3-Llama-3-8B-Preview"
    model_path = "mlx-community/Llama-3.2-3B-Instruct-4bit"  # Smaller model for demo
    
    model, tokenizer = load(model_path, use_extended_mind=True)
    model.set_model_id(f"demo_{uuid.uuid4().hex[:8]}")
    
    print(f"✓ Model loaded: {model_path}")
    
    # Demo 1: Without memories
    print("\n" + "-"*50)
    print("DEMO 1: Asking about Alexander Grothendieck WITHOUT memories")
    print("-"*50)
    
    query = "When did Alexander Grothendieck get his French citizenship?"
    print(f"\nQuery: {query}")
    print("\nGenerating response...")
    
    response = generate(model, tokenizer, query, max_tokens=100)
    print(f"\nResponse: {response}")
    
    # Demo 2: Add memories
    print("\n" + "-"*50)
    print("DEMO 2: Adding Wikipedia knowledge as external memory")
    print("-"*50)
    
    ag_wiki_entry = """Alexander Grothendieck (28 March 1928 – 13 November 2014) was a stateless mathematician who became a French citizen in 1971. He became the leading figure in the creation of modern algebraic geometry. His research extended the scope of the field and added elements of commutative algebra, homological algebra, sheaf theory, and category theory to its foundations. He is considered by many to be the greatest mathematician of the twentieth century.

Grothendieck began his productive and public career as a mathematician in 1949. In 1958, he was appointed a research professor at the Institut des hautes études scientifiques (IHÉS) and remained there until 1970, when, driven by personal and political convictions, he left following a dispute over military funding. He received the Fields Medal in 1966 for advances in algebraic geometry, homological algebra, and K-theory."""
    
    print("\nMemory content:")
    print(ag_wiki_entry[:200] + "...")
    
    # Add the memory
    memory_tokens = mx.array(tokenizer.encode(ag_wiki_entry))
    model.add_memories(memory_tokens)
    print("\n✓ Memory added to model")
    
    # Demo 3: With memories
    print("\n" + "-"*50)
    print("DEMO 3: Asking the SAME question WITH memories")
    print("-"*50)
    
    print(f"\nQuery: {query}")
    print("\nGenerating response with access to memories...")
    
    response_with_memory = generate(model, tokenizer, query, max_tokens=100)
    print(f"\nResponse: {response_with_memory}")
    
    # Demo 4: Different questions
    print("\n" + "-"*50)
    print("DEMO 4: Testing other questions about Grothendieck")
    print("-"*50)
    
    questions = [
        "What field did Grothendieck work in?",
        "When did Grothendieck receive the Fields Medal?",
        "Where did Grothendieck work as a research professor?"
    ]
    
    for q in questions:
        print(f"\nQ: {q}")
        response = generate(model, tokenizer, q, max_tokens=50)
        print(f"A: {response}")
    
    # Demo 5: Add more memories
    print("\n" + "-"*50)
    print("DEMO 5: Adding more diverse memories")
    print("-"*50)
    
    additional_memories = [
        "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It is 330 meters (1,083 ft) tall and was completed in 1889.",
        "The speed of light in vacuum is exactly 299,792,458 meters per second (approximately 300,000 km/s or 186,000 mi/s).",
        "Python is a high-level, interpreted programming language created by Guido van Rossum and first released in 1991."
    ]
    
    print("\nAdding additional memories about:")
    for i, mem in enumerate(additional_memories):
        print(f"{i+1}. {mem[:50]}...")
        tokens = mx.array(tokenizer.encode(mem))
        model.add_memories(tokens)
    
    print("\n✓ All memories added")
    
    # Test the new memories
    test_questions = [
        "How tall is the Eiffel Tower?",
        "What is the speed of light?",
        "Who created Python programming language?"
    ]
    
    print("\nTesting retrieval of new memories:")
    for q in test_questions:
        print(f"\nQ: {q}")
        response = generate(model, tokenizer, q, max_tokens=50)
        print(f"A: {response}")
    
    # Cleanup
    print("\n" + "-"*50)
    print("Demo completed!")
    print("\nKey takeaways:")
    print("1. Extended Mind Transformers can access external memories during generation")
    print("2. Memories improve factual accuracy without fine-tuning")
    print("3. Multiple memories can be added and queried dynamically")
    print("4. This enables RAG-like capabilities with better integration")


if __name__ == "__main__":
    main()