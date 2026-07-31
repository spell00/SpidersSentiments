"""
Test script for image analysis functionality
"""
import logging
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from spider_guardian.image_analysis import ImageAnalyzer, image_analyzer
from spider_guardian.twitter_client import SocialPost

# Set up logging
logging.basicConfig(level=logging.INFO)


def test_image_analysis():
    """Test image analysis with sample URLs"""
    print("🔍 Testing Image Analysis for Spider Guardian Bot")
    print("=" * 60)

    # Test URLs - you can replace these with actual spider image URLs
    test_urls = [
        # Example URLs (replace with real ones)
        "https://a-z-animals.com/media/animals/images/original/black_widow_spider1.jpg",
        "https://www.terro.com/media/wysiwyg/tr/cms/learning-center/Spider/ter-insects-black-widow-spider-article-4.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/f/fe/Scorpion_Photograph_By_Shantanu_Kuveskar.jpg",
        "",
    ]

    # Test with mock post containing images
    mock_post = SocialPost(
        id="test123",
        text="Found this spider in my garden! Is it dangerous?",
        conversation_id="test123",
        image_urls=test_urls,
    )

    print(f"📷 Testing with {len(mock_post.image_urls)} images...")

    for i, url in enumerate(mock_post.image_urls):
        print(f"\n🖼️  Analyzing image {i+1}: {url}")

        try:
            result = image_analyzer.analyze_image_from_url(url)
            if result:
                print(f"✅ Analysis successful!")
                print(f"   Description: {result.description}")
                print(f"   Spider detected: {result.spider_detected}")
                print(f"   Confidence: {result.confidence:.1%}")
                if result.species_suggestion:
                    print(f"   Species: {result.species_suggestion}")
                print(f"   Safety: {result.danger_level}")
                if result.objects_detected:
                    print(f"   Objects: {', '.join(result.objects_detected)}")

                # Show comprehensive taxonomy
                if result.taxonomy and result.taxonomy.kingdom:
                    taxonomy = result.taxonomy
                    print(f"\n   🧬 Comprehensive Taxonomy:")
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
                        print(f"   🔬 Scientific Name: {taxonomy.scientific_name}")
                    if taxonomy.common_name:
                        print(f"   📖 Common Name: {taxonomy.common_name}")
                    if taxonomy.confidence > 0:
                        print(f"   🎯 Taxonomy Confidence: {taxonomy.confidence:.1%}")

                # Show animal verification results
                if result.animal_verification:
                    verification = result.animal_verification
                    print(f"\n   🔍 Animal Verification:")
                    print(f"   Actually a spider: {verification.get('is_actually_spider', 'Unknown')}")
                    if verification.get('detected_categories'):
                        print(f"   Detected as: {', '.join(verification['detected_categories'])}")
                    if verification.get('correction_suggestion'):
                        print(f"   ⚠️  Correction: {verification['correction_suggestion']}")

            else:
                print("❌ Analysis failed")
        except Exception as e:
            print(f"❌ Error: {e}")

    print("\n" + "=" * 60)
    print("💡 Tips:")
    print("- Replace test URLs with real spider images")
    print("- The bot will now automatically analyze images in tweets")
    print("- Check logs for detailed analysis results")


def test_model_loading():
    """Test if image analysis models load correctly"""
    print("🤖 Testing Model Loading...")

    try:
        analyzer = ImageAnalyzer()
        if analyzer.blip_model and analyzer.blip_processor:
            print("✅ BLIP model loaded successfully")
            print(f"   Device: {analyzer.device}")
        else:
            print("❌ Failed to load BLIP model")
    except Exception as e:
        print(f"❌ Model loading error: {e}")


if __name__ == "__main__":
    # Test model loading first
    test_model_loading()
    print()

    # Test image analysis (only if you have internet connection)
    choice = input("Test image analysis with sample URLs? (y/N): ").strip().lower()
    if choice in ['y', 'yes']:
        test_image_analysis()
    else:
        print("Skipping image analysis test")
        print("🚀 Image analysis is ready! Run your bot to see it in action.")
