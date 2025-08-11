import weaviate
import openai
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MTGRAGPipeline:
    def __init__(self, weaviate_client, openai_api_key: str, model: str = "gpt-4"):
        """
        Initialize the MTG RAG Pipeline
        
        Args:
            weaviate_client: Weaviate client instance
            openai_api_key: OpenAI API key
            model: OpenAI model to use (default: gpt-4)
        """
        self.client = weaviate_client
        self.openai_client = openai.OpenAI(api_key=openai_api_key)
        self.model = model
        
        # Collection names
        self.collections = {
            "rules": "MTGOfficialRules",
            "cards": "MTGCards", 
            "rulings": "MTGRulings"
        }
    
    def search_all_collections(
        self, 
        query_text: str, 
        limits: Dict[str, int] = None,
        min_score: float = 0.7
    ) -> Dict[str, List[Any]]:
        """
        Search across all MTG collections and return combined results
        
        Args:
            query_text: The search query
            limits: Dictionary specifying limit for each collection
            min_score: Minimum similarity score to include results
            
        Returns:
            Dictionary with results from each collection
        """
        if limits is None:
            limits = {"rules": 3, "cards": 3, "rulings": 3}
        
        results = {
            "rules": [],
            "cards": [], 
            "rulings": [],
            "search_metadata": {
                "query": query_text,
                "timestamp": datetime.now().isoformat(),
                "limits": limits,
                "min_score": min_score
            }
        }
        
        logger.info(f"Searching all collections for: '{query_text}'")
        
        # Search MTG Official Rules
        try:
            rules_collection = self.client.collections.get(self.collections["rules"])
            rules_response = rules_collection.query.near_text(
                query=query_text,
                limit=limits["rules"],
                return_metadata=["score", "distance"]
            )
            
            # Filter by score
            filtered_rules = [
                obj for obj in rules_response.objects 
                if obj.metadata.score >= min_score
            ]
            results["rules"] = filtered_rules
            
            logger.info(f"Found {len(filtered_rules)} relevant rules (from {len(rules_response.objects)} total)")
            
        except Exception as e:
            logger.error(f"Error searching rules: {e}")
            results["rules"] = []
        
        # Search MTG Cards
        try:
            cards_collection = self.client.collections.get(self.collections["cards"])
            cards_response = cards_collection.query.near_text(
                query=query_text,
                limit=limits["cards"],
                return_metadata=["score", "distance"]
            )
            
            # Filter by score
            filtered_cards = [
                obj for obj in cards_response.objects 
                if obj.metadata.score >= min_score
            ]
            results["cards"] = filtered_cards
            
            logger.info(f"Found {len(filtered_cards)} relevant cards (from {len(cards_response.objects)} total)")
            
        except Exception as e:
            logger.error(f"Error searching cards: {e}")
            results["cards"] = []
        
        # Search MTG Rulings
        try:
            rulings_collection = self.client.collections.get(self.collections["rulings"])
            rulings_response = rulings_collection.query.near_text(
                query=query_text,
                limit=limits["rulings"],
                return_metadata=["score", "distance"]
            )
            
            # Filter by score
            filtered_rulings = [
                obj for obj in rulings_response.objects 
                if obj.metadata.score >= min_score
            ]
            results["rulings"] = filtered_rulings
            
            logger.info(f"Found {len(filtered_rulings)} relevant rulings (from {len(rulings_response.objects)} total)")
            
        except Exception as e:
            logger.error(f"Error searching rulings: {e}")
            results["rulings"] = []
        
        return results
    
    def format_context_for_llm(self, search_results: Dict[str, List[Any]]) -> str:
        """
        Format Weaviate search results into context for LLM
        
        Args:
            search_results: Results from search_all_collections
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        # Add search metadata
        metadata = search_results.get("search_metadata", {})
        context_parts.append(f"SEARCH QUERY: {metadata.get('query', 'Unknown')}")
        context_parts.append(f"SEARCH TIMESTAMP: {metadata.get('timestamp', 'Unknown')}")
        context_parts.append("")
        
        # Add rules context
        if search_results["rules"]:
            context_parts.append("=== OFFICIAL RULES ===")
            for i, obj in enumerate(search_results["rules"], 1):
                rule_text = obj.properties.get("rule", "")
                score = getattr(obj.metadata, 'score', 0)
                context_parts.append(f"{i}. [Score: {score:.3f}] {rule_text}")
            context_parts.append("")
        
        # Add cards context  
        if search_results["cards"]:
            context_parts.append("=== RELEVANT CARDS ===")
            for i, obj in enumerate(search_results["cards"], 1):
                name = obj.properties.get("name", "Unknown Card")
                text = obj.properties.get("text", "")
                card_type = obj.properties.get("type", "")
                manacost = obj.properties.get("manacost", "")
                power = obj.properties.get("power", "")
                toughness = obj.properties.get("toughness", "")
                score = getattr(obj.metadata, 'score', 0)
                
                card_info = f"{i}. [Score: {score:.3f}] {name}"
                if manacost:
                    card_info += f" {manacost}"
                if card_type:
                    card_info += f" - {card_type}"
                if power and toughness:
                    card_info += f" ({power}/{toughness})"
                if text:
                    card_info += f"\n   Text: {text}"
                
                context_parts.append(card_info)
            context_parts.append("")
        
        # Add rulings context
        if search_results["rulings"]:
            context_parts.append("=== OFFICIAL RULINGS ===")
            for i, obj in enumerate(search_results["rulings"], 1):
                name = obj.properties.get("name", "Unknown Card")
                ruling = obj.properties.get("rulings", "")
                ruling_date = obj.properties.get("ruling_date", "")
                source = obj.properties.get("source", "")
                score = getattr(obj.metadata, 'score', 0)
                
                ruling_info = f"{i}. [Score: {score:.3f}] {name}"
                if ruling_date:
                    ruling_info += f" ({ruling_date})"
                if source:
                    ruling_info += f" - Source: {source}"
                ruling_info += f"\n   Ruling: {ruling}"
                
                context_parts.append(ruling_info)
            context_parts.append("")
        
        return "\n".join(context_parts)
    
    def generate_answer(self, question: str, context: str, temperature: float = 0.1) -> str:
        """
        Generate answer using OpenAI with the provided context
        
        Args:
            question: User's question
            context: Formatted context from search results
            temperature: OpenAI temperature parameter
            
        Returns:
            Generated answer
        """
        system_prompt = """You are an expert Magic: The Gathering judge with comprehensive knowledge of the game rules, cards, and official rulings. 

Your responsibilities:
1. Provide accurate, rule-based answers using the official context provided
2. Cite specific rules, cards, or rulings when relevant
3. Explain complex interactions clearly and step-by-step
4. If the provided context doesn't fully answer the question, clearly state what information is missing
5. Always prioritize official rules over card text when there are conflicts
6. Use proper MTG terminology

Answer format guidelines:
- Start with a direct answer to the question
- Provide detailed explanation with rule citations
- Include relevant card interactions if applicable
- End with any important caveats or edge cases"""

        user_prompt = f"""Based on the following official Magic: The Gathering information, please answer this question:

QUESTION: {question}

OFFICIAL CONTEXT:
{context}

Please provide a comprehensive answer using the official information provided above."""

        try:
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                max_tokens=1000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return f"I encountered an error while generating the answer: {str(e)}"
    
    def answer_question(
        self, 
        question: str, 
        limits: Dict[str, int] = None,
        min_score: float = 0.7,
        temperature: float = 0.1,
        include_debug: bool = False
    ) -> Dict[str, Any]:
        """
        Complete RAG pipeline: search collections and generate answer
        
        Args:
            question: User's MTG question
            limits: Search limits for each collection
            min_score: Minimum similarity score for results
            temperature: OpenAI temperature
            include_debug: Whether to include debug information
            
        Returns:
            Dictionary containing answer and metadata
        """
        logger.info(f"Processing question: '{question}'")
        
        # Step 1: Search all collections
        search_results = self.search_all_collections(question, limits, min_score)
        
        # Step 2: Format context
        context = self.format_context_for_llm(search_results)
        
        # Step 3: Generate answer
        answer = self.generate_answer(question, context, temperature)
        
        # Step 4: Prepare response
        result = {
            "question": question,
            "answer": answer,
            "timestamp": datetime.now().isoformat(),
            "sources_found": {
                "rules": len(search_results["rules"]),
                "cards": len(search_results["cards"]),
                "rulings": len(search_results["rulings"])
            }
        }
        
        if include_debug:
            result["debug"] = {
                "search_results": search_results,
                "formatted_context": context,
                "search_params": {
                    "limits": limits,
                    "min_score": min_score,
                    "temperature": temperature
                }
            }
        
        logger.info(f"Answer generated successfully. Sources: {result['sources_found']}")
        return result
    
    def batch_answer_questions(
        self, 
        questions: List[str], 
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Process multiple questions in batch
        
        Args:
            questions: List of questions to process
            **kwargs: Arguments to pass to answer_question
            
        Returns:
            List of answer dictionaries
        """
        results = []
        
        for i, question in enumerate(questions, 1):
            logger.info(f"Processing question {i}/{len(questions)}")
            try:
                result = self.answer_question(question, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing question {i}: {e}")
                results.append({
                    "question": question,
                    "answer": f"Error processing question: {str(e)}",
                    "error": True
                })
        
        return results


# Example usage and testing
def main():

# Replace with your actual Weaviate Cloud URL and API key
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url="xhttsi9br7swv5ue58ctq.c0.us-east1.gcp.weaviate.cloud",
        auth_credentials=weaviate.auth.AuthApiKey(api_key="TENRZEc5ckFOazMvMTEwVF8yc2JDdXE1UkM5aDZVd1lOR0ZyMW5IWE85NTVMY3NJS00vbGlrb0lqQzFNPV92MjAw")
    )
    """Example usage of the MTG RAG Pipeline"""
    
    # Initialize Weaviate client (adjust URL as needed)
    client = weaviate.connect_to_local()  # or your Weaviate instance
    
    # Initialize pipeline (you need to provide your OpenAI API key)
    pipeline = MTGRAGPipeline(
        weaviate_client=client,
        openai_api_key="your-openai-api-key-here",
        model="gpt-4"
    )
    
    # Test questions
    test_questions = [
        "How does trample work when a creature is blocked?",
        "What happens when Lightning Bolt targets a creature with protection from red?",
        "Can I sacrifice a creature to pay for its own ability?",
        "How does the stack work in Magic: The Gathering?",
        "What is the difference between a triggered ability and an activated ability?"
    ]
    
    # Single question example
    print("=== Single Question Example ===")
    result = pipeline.answer_question(
        question=test_questions[0],
        limits={"rules": 5, "cards": 3, "rulings": 3},
        min_score=0.6,
        include_debug=False
    )
    
    print(f"Question: {result['question']}")
    print(f"Answer: {result['answer']}")
    print(f"Sources found: {result['sources_found']}")
    print()
    
    # Batch processing example
    print("=== Batch Processing Example ===")
    batch_results = pipeline.batch_answer_questions(
        questions=test_questions[:3],
        limits={"rules": 3, "cards": 2, "rulings": 2},
        min_score=0.7
    )
    
    for i, result in enumerate(batch_results, 1):
        print(f"\n--- Question {i} ---")
        print(f"Q: {result['question']}")
        print(f"A: {result['answer'][:200]}...")  # Truncated for brevity
        print(f"Sources: {result['sources_found']}")
    
    # Close client
    client.close()


if __name__ == "__main__":
    main()