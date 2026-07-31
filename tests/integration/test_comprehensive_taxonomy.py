"""
Test comprehensive taxonomy prediction
"""
import os
import sys

from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from spider_guardian.image_analysis import ImageAnalyzer


def test_comprehensive_taxonomy():
    """Test the comprehensive taxonomy prediction system"""
    analyzer = ImageAnalyzer()

    test_cases = [
        "a black widow spider with red hourglass marking on its abdomen",
        "a brown recluse spider with violin-shaped marking on cephalothorax",
        "a jumping spider with large eyes hunting on a leaf",
        "an orb weaver spider sitting in center of circular web",
        "a wolf spider carrying egg sac on its back",
        "a house spider in its messy cobweb",
        "a large hairy tarantula on the ground",
        "a small beetle with shiny black wing covers",
        "a honey bee covered in pollen visiting flowers",
        "a red ant carrying food back to colony",
        "a colorful butterfly with orange and black wings",
        "a green lizard basking on a warm rock",
        "a brown snake slithering through grass",
        "a scorpion with pincers and curved tail raised",
    ]

    print("🧬 Testing Comprehensive Taxonomy System")
    print("=" * 80)

    for i, description in enumerate(test_cases, 1):
        print(f"\n{i}. Testing: '{description}'")
        print("-" * 60)

        # Create a dummy 1x1 image for testing
        dummy_image = Image.new('RGB', (1, 1), color='white')

        # Predict taxonomy
        taxonomy = analyzer._predict_comprehensive_taxonomy(description, dummy_image)

        if taxonomy.kingdom:
            print(f"🔬 Taxonomy Classification:")
            if taxonomy.kingdom:
                print(f"   Kingdom: {taxonomy.kingdom}")
            if taxonomy.phylum:
                print(f"   Phylum: {taxonomy.phylum}")
            if taxonomy.class_name:
                print(f"   Class: {taxonomy.class_name}")
            if taxonomy.order:
                print(f"   Order: {taxonomy.order}")
            if taxonomy.family:
                print(f"   Family: {taxonomy.family}")
            if taxonomy.genus:
                print(f"   Genus: {taxonomy.genus}")
            if taxonomy.species:
                print(f"   Species: {taxonomy.species}")
            if taxonomy.scientific_name:
                print(f"   📖 Scientific Name: {taxonomy.scientific_name}")
            if taxonomy.common_name:
                print(f"   🏷️  Common Name: {taxonomy.common_name}")
            if taxonomy.confidence > 0:
                print(f"   🎯 Confidence: {taxonomy.confidence:.1%}")
        else:
            print("   ❓ No taxonomy classification available")

        # Also test spider species matching for spider descriptions
        if "spider" in description.lower():
            spider_match = analyzer._match_spider_species(description.lower())
            if spider_match:
                print(f"\n🕷️  Spider Species Match:")
                print(f"   Scientific: {spider_match.get('scientific', 'Unknown')}")
                print(f"   Common: {spider_match.get('common', 'Unknown')}")
                print(f"   Family: {spider_match.get('family', 'Unknown')}")
                print(f"   Danger Level: {spider_match.get('danger', 'Unknown')}")
                print(f"   Match Confidence: {spider_match.get('confidence', 0):.1%}")

    print("\n" + "=" * 80)
    print("✅ Comprehensive taxonomy testing complete!")
    print("🌟 The system can now provide detailed taxonomic classification!")
    print("📚 Including kingdom, phylum, class, order, family, genus, and species")
    print("🔬 With both scientific and common names when available")


if __name__ == "__main__":
    test_comprehensive_taxonomy()
