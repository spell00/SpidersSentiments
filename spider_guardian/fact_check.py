"""Spider fact-checking database and myth-busting utilities."""

from typing import List, Dict, Optional
import re

# Evidence-based spider facts with sources
SPIDER_FACTS = {
    "pest_control": {
        "fact": "Spiders eat 400-800 million tons of prey annually, controlling insect populations naturally.",
        "source": "The Science of Nature (2017) - Study by Martin Nyffeler & Klaus Birkhofer",
        "keywords": ["pest", "control", "eat", "insect", "population"],
        "myth_counters": ["useless", "no purpose"],
    },
    "venom_safety": {
        "fact": "Of 50,000+ spider species, only ~12 are dangerous to humans. Most are harmless.",
        "source": "American Association of Poison Control Centers",
        "keywords": ["dangerous", "poisonous", "venomous", "bite", "deadly"],
        "myth_counters": ["all spiders are deadly", "spider bites kill"],
    },
    "home_helpers": {
        "fact": "House spiders reduce mosquitoes, flies, and disease-carrying pests in homes.",
        "source": "Journal of Medical Entomology (2019)",
        "keywords": ["home", "house", "indoor", "bedroom"],
        "myth_counters": ["get out", "kill it"],
    },
    "ecosystem_role": {
        "fact": "Spiders are keystone predators in most ecosystems, supporting biodiversity.",
        "source": "Annual Review of Entomology (2018)",
        "keywords": ["ecosystem", "environment", "nature", "biodiversity"],
        "myth_counters": [],
    },
    "web_engineering": {
        "fact": "Spider silk is 5x stronger than steel by weight and more elastic than nylon.",
        "source": "Nature Materials Journal",
        "keywords": ["web", "silk", "strong", "material"],
        "myth_counters": [],
    },
    "harmless_majority": {
        "fact": "99.9% of spider species cannot harm humans. They're more scared of you than you are of them.",
        "source": "Burke Museum of Natural History",
        "keywords": ["afraid", "scared", "fear", "phobia"],
        "myth_counters": ["spiders attack", "aggressive"],
    },
    "beneficial_gardeners": {
        "fact": "Garden spiders can reduce crop damage by up to 70% by eating agricultural pests.",
        "source": "Journal of Applied Ecology (2020)",
        "keywords": ["garden", "crop", "farm", "agriculture"],
        "myth_counters": [],
    },
}

COMMON_MYTHS = {
    "swallow_in_sleep": {
        "myth": "You swallow spiders in your sleep",
        "truth": "This is completely false. Spiders avoid sleeping humans and vibrations.",
        "source": "Scientific American debunking (1993)",
    },
    "eggs_under_skin": {
        "myth": "Spider bites lay eggs under your skin",
        "truth": "Anatomically impossible. Spiders don't lay eggs in hosts.",
        "source": "American Arachnological Society",
    },
    "daddy_longlegs_deadly": {
        "myth": "Daddy longlegs are the most venomous but fangs too small",
        "truth": "False. They're not even spiders (opiliones), and they're not venomous.",
        "source": "UC Riverside Entomology",
    },
    "all_aggressive": {
        "myth": "Spiders are aggressive and chase people",
        "truth": "Spiders are shy and flee from threats. 'Chasing' is running toward dark shelter.",
        "source": "American Museum of Natural History",
    },
}


class SpiderFactChecker:
    """Intelligent fact-checking for spider-related claims."""
    
    def __init__(self):
        self.facts = SPIDER_FACTS
        self.myths = COMMON_MYTHS
    
    def detect_myth(self, text: str) -> Optional[Dict]:
        """Detect if text contains a common spider myth."""
        text_lower = text.lower()
        
        for myth_key, myth_data in self.myths.items():
            # Check for myth keywords
            myth_indicators = myth_data["myth"].lower().split()
            if any(word in text_lower for word in myth_indicators if len(word) > 4):
                return {
                    "myth_detected": True,
                    "myth_type": myth_key,
                    "myth_claim": myth_data["myth"],
                    "truth": myth_data["truth"],
                    "source": myth_data["source"],
                }
        
        return None
    
    def find_relevant_facts(self, text: str, top_k: int = 2) -> List[Dict]:
        """Find most relevant facts for a given text."""
        text_lower = text.lower()
        scored_facts = []
        
        for fact_key, fact_data in self.facts.items():
            score = 0
            
            # Score based on keyword matches
            for keyword in fact_data["keywords"]:
                if keyword in text_lower:
                    score += 2
            
            # Boost score if myth counter detected
            for myth_counter in fact_data["myth_counters"]:
                if myth_counter in text_lower:
                    score += 3
            
            if score > 0:
                scored_facts.append({
                    "fact": fact_data["fact"],
                    "source": fact_data["source"],
                    "relevance_score": score,
                    "category": fact_key,
                })
        
        # Sort by relevance
        scored_facts.sort(key=lambda x: x["relevance_score"], reverse=True)
        return scored_facts[:top_k]
    
    def generate_fact_response(self, text: str, max_length: int = 280) -> Optional[str]:
        """
        Generate a fact-based response to counter misinformation.
        Optimized for Twitter's character limit.
        """
        
        # Check for myths first
        myth_result = self.detect_myth(text)
        if myth_result:
            response = f"💡 Actually: {myth_result['truth']} (Source: {myth_result['source']})"
            if len(response) <= max_length:
                return response
        
        # Otherwise find relevant facts
        facts = self.find_relevant_facts(text, top_k=1)
        if facts:
            fact = facts[0]
            response = f"🕷️ Fun fact: {fact['fact']}"
            if len(response) <= max_length:
                return response
        
        return None
    
    def enhance_reply_with_facts(self, original_reply: str, tweet_text: str) -> str:
        """
        Enhance a reply with relevant facts if space allows.
        """
        
        # Try to find a short, relevant fact
        facts = self.find_relevant_facts(tweet_text, top_k=1)
        if not facts:
            return original_reply
        
        fact_snippet = facts[0]["fact"].split('.')[0] + '.'  # First sentence only
        
        # Check if we have room (leaving space for fact)
        if len(original_reply) + len(fact_snippet) + 10 <= 280:
            return f"{original_reply} 📚 {fact_snippet}"
        
        return original_reply


# Instant fact lookup for common questions
QUICK_FACTS = {
    "how many eyes": "Most spiders have 8 eyes, but some have 6, 4, 2, or even 0!",
    "how long live": "Tarantulas can live 20-30 years, while house spiders typically live 1-2 years.",
    "biggest spider": "Goliath birdeater (up to 12 inch leg span), but huntsman spiders have longer legs.",
    "smallest spider": "Patu digua is ~0.37mm, smaller than a pinhead!",
    "fastest spider": "Giant house spiders can run 1.73 ft/second (about 1.2 mph).",
    "how many species": "Over 50,000 known species, with likely thousands more undiscovered.",
}


def get_quick_fact(query: str) -> Optional[str]:
    """Quick fact lookup for common questions."""
    query_lower = query.lower()
    for key, fact in QUICK_FACTS.items():
        if key in query_lower:
            return fact
    return None


if __name__ == "__main__":
    # Demo
    checker = SpiderFactChecker()
    
    test_texts = [
        "I'm so scared of spiders, they're all dangerous!",
        "Should I kill this spider in my house?",
        "Spiders are useless and gross",
        "I heard you swallow spiders in your sleep",
    ]
    
    print("🕷️ Spider Fact-Checker Demo\n")
    for text in test_texts:
        print(f"Input: {text}")
        
        myth = checker.detect_myth(text)
        if myth:
            print(f"  ⚠️ MYTH DETECTED: {myth['myth_claim']}")
            print(f"  ✅ TRUTH: {myth['truth']}")
        
        facts = checker.find_relevant_facts(text, top_k=1)
        if facts:
            print(f"  📚 Relevant fact: {facts[0]['fact']}")
        
        response = checker.generate_fact_response(text)
        if response:
            print(f"  💬 Suggested response: {response}")
        
        print()
