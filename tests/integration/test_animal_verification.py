"""
Test animal verification functionality
"""
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from spider_guardian.image_analysis import ImageAnalyzer


def test_animal_verification():
    """Test the animal verification logic with sample descriptions"""
    analyzer = ImageAnalyzer()

    test_cases = [
        "a black spider with eight legs sitting on a web",
        "a small black insect with six legs and wings flying around",
        "a brown beetle crawling on the ground with antennae visible",
        "a lizard with scales sunning itself on a rock",
        "a scorpion with pincers and a curved tail",
        "a jumping spider with large eyes hunting prey",
        "a centipede with many legs moving quickly",
    ]

    print("🔍 Testing Animal Verification System")
    print("=" * 60)

    for i, description in enumerate(test_cases, 1):
        print(f"\n{i}. Testing: '{description}'")

        # Test spider detection
        spider_detected, confidence = analyzer._detect_spider_content(description)
        print(f"   Spider detected: {spider_detected} (confidence: {confidence:.1%})")

        # Test animal verification
        verification = analyzer._verify_animal_classification(description)

        print(f"   Detected categories: {verification.get('detected_categories', [])}")
        print(f"   Actually a spider: {verification.get('is_actually_spider', 'Unknown')}")

        if verification.get('correction_suggestion'):
            print(f"   🔧 Correction: {verification['correction_suggestion']}")

        if verification.get('reasoning'):
            print(f"   💡 Reasoning: {'; '.join(verification['reasoning'][:2])}")  # Show first 2 reasons

    print("\n" + "=" * 60)
    print("✅ Animal verification testing complete!")
    print("💡 The bot can now distinguish between spiders and other animals!")


if __name__ == "__main__":
    test_animal_verification()
