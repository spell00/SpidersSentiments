"""
Image Analysis Module for Spider Guardian Bot
Handles image recognition and analysis from tweets with comprehensive taxonomy
"""
import os
import logging
import io
import requests
from typing import Dict, List, Optional, Tuple
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoImageProcessor, AutoModelForImageClassification
from dataclasses import dataclass

try:
    from .inat_client import INaturalistClient, INatPrediction
except ImportError:  # pragma: no cover - support direct script execution
    from spider_guardian.inat_client import INaturalistClient, INatPrediction  # type: ignore

@dataclass
class TaxonomyInfo:
    """Detailed taxonomic information"""
    kingdom: Optional[str] = None
    phylum: Optional[str] = None
    class_name: Optional[str] = None  # 'class' is reserved keyword
    order: Optional[str] = None
    family: Optional[str] = None
    genus: Optional[str] = None
    species: Optional[str] = None
    common_name: Optional[str] = None
    scientific_name: Optional[str] = None
    confidence: float = 0.0
    source: str = "heuristic"

@dataclass
class ImageAnalysisResult:
    """Result of image analysis"""
    description: str
    spider_detected: bool
    confidence: float
    species_suggestion: Optional[str] = None
    danger_level: str = "unknown"  # safe, caution, dangerous, unknown
    objects_detected: List[str] = None
    animal_verification: Optional[Dict] = None
    taxonomy: Optional[TaxonomyInfo] = None  # New comprehensive taxonomy
    image_url: Optional[str] = None
    
    def __post_init__(self):
        if self.objects_detected is None:
            self.objects_detected = []
        if self.animal_verification is None:
            self.animal_verification = {}
        if self.taxonomy is None:
            self.taxonomy = TaxonomyInfo()

class ImageAnalyzer:
    """Handles image analysis for spider-related content with comprehensive taxonomy"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_models()
        self._load_taxonomy_data()

        try:
            self.inat_client = INaturalistClient.from_env()
        except Exception as exc:  # pragma: no cover - defensive guard
            logging.warning("iNaturalist client initialisation failed: %s", exc)
            self.inat_client = None

        self.spider_keywords = [
            "spider",
            "arachnid",
            "web",
            "cobweb",
            "eight legs",
            "spinneret",
            "silk",
            "tarantula",
            "black widow",
            "brown recluse",
            "orb weaver",
            "wolf spider",
        ]

        self.animal_categories = {
            "spiders": {
                "keywords": ["spider", "arachnid", "web", "orb", "fangs", "silk"],
                "body_parts": ["fangs", "spinneret", "cephalothorax", "abdomen"],
                "characteristics": ["eight legs", "venom", "web building"],
            },
            "insects": {
                "keywords": ["insect", "bug", "wasp", "bee", "ant", "beetle", "fly", "moth", "butterfly"],
                "body_parts": ["antennae", "thorax", "wings", "proboscis"],
                "characteristics": ["six legs", "compound eyes", "exoskeleton"],
            },
            "other_arthropods": {
                "keywords": ["scorpion", "tick", "mite", "centipede", "millipede", "crustacean"],
                "body_parts": ["pincers", "stinger", "segments", "claws"],
                "characteristics": ["many legs", "segmented body", "tail stinger"],
            },
            "vertebrates": {
                "keywords": ["lizard", "snake", "frog", "bird", "mammal", "gecko", "cat", "dog"],
                "body_parts": ["scales", "feathers", "fur", "beak", "claws"],
                "characteristics": ["backbone", "warm blooded", "cold blooded"],
            },
        }

        self.dangerous_indicators = {
            "black widow": "dangerous",
            "red hourglass": "dangerous",
            "brown recluse": "dangerous",
            "violin mark": "dangerous",
            "fiddleback": "dangerous",
            "widow": "dangerous",
            "recluse": "dangerous",
            "tarantula": "caution",
            "large hairy spider": "caution",
            "scorpion": "dangerous",
        }
        
        # Enhanced spider detection patterns
        self.spider_patterns = {
            "common_spiders": {
                "jumping spider": {
                    "scientific": "Salticidae",
                    "keywords": ["jumping", "salticid", "large eyes", "compact body"],
                    "danger": "safe"
                },
                "orb weaver": {
                    "scientific": "Araneidae", 
                    "keywords": ["orb web", "circular web", "garden spider"],
                    "danger": "safe"
                },
                "wolf spider": {
                    "scientific": "Lycosidae",
                    "keywords": ["wolf", "hunting", "ground", "lycosid"],
                    "danger": "safe"
                },
                "black widow": {
                    "scientific": "Latrodectus",
                    "keywords": ["black widow", "red hourglass", "shiny black"],
                    "danger": "dangerous"
                },
                "brown recluse": {
                    "scientific": "Loxosceles reclusa",
                    "keywords": ["brown recluse", "violin mark", "fiddleback"],
                    "danger": "dangerous"
                },
                "house spider": {
                    "scientific": "Parasteatoda tepidariorum",
                    "keywords": ["house spider", "common house", "cobweb"],
                    "danger": "safe"
                },
                "cellar spider": {
                    "scientific": "Pholcidae",
                    "keywords": ["cellar spider", "daddy long legs", "thin legs"],
                    "danger": "safe"
                },
                "tarantula": {
                    "scientific": "Theraphosidae",
                    "keywords": ["tarantula", "large", "hairy", "theraphosid"],
                    "danger": "caution"
                }
            }
        }
        
        # Expanded taxonomy database for common animals
        self.taxonomy_db = self._build_comprehensive_taxonomy()
        
    def _build_comprehensive_taxonomy(self) -> Dict:
        """Build a comprehensive taxonomy database"""
        return {
            "arachnida": {
                "kingdom": "Animalia",
                "phylum": "Arthropoda", 
                "class": "Arachnida",
                "families": {
                    "araneae": {  # Spiders
                        "order": "Araneae",
                        "families": {
                            "salticidae": {
                                "family": "Salticidae",
                                "common_name": "Jumping Spiders",
                                "species": {
                                    "phidippus_audax": {
                                        "scientific": "Phidippus audax",
                                        "common": "Bold Jumping Spider",
                                        "danger": "safe"
                                    },
                                    "platycryptus_undatus": {
                                        "scientific": "Platycryptus undatus", 
                                        "common": "Tan Jumping Spider",
                                        "danger": "safe"
                                    }
                                }
                            },
                            "araneidae": {
                                "family": "Araneidae",
                                "common_name": "Orb Weavers",
                                "species": {
                                    "araneus_diadematus": {
                                        "scientific": "Araneus diadematus",
                                        "common": "European Garden Spider",
                                        "danger": "safe"
                                    }
                                }
                            },
                            "lycosidae": {
                                "family": "Lycosidae", 
                                "common_name": "Wolf Spiders",
                                "species": {
                                    "tigrosa_helluo": {
                                        "scientific": "Tigrosa helluo",
                                        "common": "Wetland Giant Wolf Spider",
                                        "danger": "safe"
                                    }
                                }
                            },
                            "theridiidae": {
                                "family": "Theridiidae",
                                "common_name": "Cobweb Spiders",
                                "species": {
                                    "latrodectus_mactans": {
                                        "scientific": "Latrodectus mactans",
                                        "common": "Southern Black Widow",
                                        "danger": "dangerous"
                                    },
                                    "parasteatoda_tepidariorum": {
                                        "scientific": "Parasteatoda tepidariorum",
                                        "common": "Common House Spider", 
                                        "danger": "safe"
                                    }
                                }
                            },
                            "sicariidae": {
                                "family": "Sicariidae",
                                "common_name": "Recluse Spiders",
                                "species": {
                                    "loxosceles_reclusa": {
                                        "scientific": "Loxosceles reclusa",
                                        "common": "Brown Recluse",
                                        "danger": "dangerous"
                                    }
                                }
                            }
                        }
                    },
                    "scorpiones": {  # Scorpions
                        "order": "Scorpiones",
                        "families": {
                            "buthidae": {
                                "family": "Buthidae",
                                "common_name": "Bark Scorpions"
                            }
                        }
                    }
                }
            },
            "insecta": {
                "kingdom": "Animalia",
                "phylum": "Arthropoda",
                "class": "Insecta",
                "orders": {
                    "coleoptera": {
                        "order": "Coleoptera",
                        "common_name": "Beetles"
                    },
                    "lepidoptera": {
                        "order": "Lepidoptera", 
                        "common_name": "Butterflies and Moths"
                    },
                    "hymenoptera": {
                        "order": "Hymenoptera",
                        "common_name": "Ants, Bees, and Wasps"
                    },
                    "diptera": {
                        "order": "Diptera",
                        "common_name": "Flies"
                    }
                }
            },
            "reptilia": {
                "kingdom": "Animalia",
                "phylum": "Chordata",
                "class": "Reptilia",
                "orders": {
                    "squamata": {
                        "order": "Squamata",
                        "suborders": {
                            "lacertilia": {
                                "suborder": "Lacertilia",
                                "common_name": "Lizards"
                            },
                            "serpentes": {
                                "suborder": "Serpentes", 
                                "common_name": "Snakes"
                            }
                        }
                    }
                }
            }
        }
    
    def _load_taxonomy_data(self):
        """Load additional taxonomy data from external sources if available"""
        # This could be expanded to load from GBIF, iNaturalist API, etc.
        pass
        
    def _load_models(self):
        """Load image analysis models"""
        try:
            # Load BLIP model for image captioning
            self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            self.blip_model.to(self.device)
            logging.info("✅ BLIP image captioning model loaded")
            
            # Try to load a more specialized model for animal classification
            try:
                # This is a hypothetical model - you might want to use a different one
                self.animal_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
                self.animal_model = AutoModelForImageClassification.from_pretrained("microsoft/resnet-50")
                self.animal_model.to(self.device)
                logging.info("✅ Animal classification model loaded")
            except Exception as e:
                logging.warning(f"Could not load specialized animal model: {e}")
                self.animal_processor = None
                self.animal_model = None
            
        except Exception as e:
            logging.error(f"Failed to load image models: {e}")
            self.blip_processor = None
            self.blip_model = None
    
    def analyze_image_from_url(self, image_url: str) -> Optional[ImageAnalysisResult]:
        """Download and analyze an image from URL"""
        try:
            # Download image
            response = requests.get(image_url, headers={'User-Agent': 'SpiderGuardian/1.0'})
            response.raise_for_status()
            
            # Convert to PIL Image
            image = Image.open(io.BytesIO(response.content))
            result = self.analyze_image(image)
            if result:
                result.image_url = image_url
            return result
            
        except Exception as e:
            logging.error(f"Failed to download/analyze image from {image_url}: {e}")
            return None
    
    def analyze_image(self, image: Image.Image) -> Optional[ImageAnalysisResult]:
        """Analyze a PIL Image for spider content"""
        if not self.blip_model or not self.blip_processor:
            logging.warning("Image analysis models not available")
            return None
            
        try:
            # Ensure image is RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Generate description using BLIP
            description = self._generate_description(image)
            
            # Detect spider-related content
            spider_detected, confidence = self._detect_spider_content(description)
            
            # Verify animal classification
            animal_verification = self._verify_animal_classification(description)
            
            # Predict detailed taxonomy
            taxonomy = self._predict_comprehensive_taxonomy(description, image)
            
            # Determine danger level and species
            species, danger_level = self._analyze_species_and_danger(description)

            species_override = None
            if taxonomy and taxonomy.common_name and taxonomy.confidence >= 0.3:
                species_override = taxonomy.common_name
            elif taxonomy and taxonomy.scientific_name and taxonomy.confidence >= 0.3:
                species_override = taxonomy.scientific_name

            if species_override:
                species = species_override

            if taxonomy and taxonomy.source == "inaturalist" and taxonomy.scientific_name:
                animal_verification.setdefault(
                    "external_taxonomy",
                    {
                        "scientific_name": taxonomy.scientific_name,
                        "common_name": taxonomy.common_name,
                        "kingdom": taxonomy.kingdom,
                        "phylum": taxonomy.phylum,
                        "class": taxonomy.class_name,
                        "order": taxonomy.order,
                        "confidence": taxonomy.confidence,
                        "source": "iNaturalist",
                    },
                )
                if taxonomy.class_name:
                    if taxonomy.class_name.lower() != "arachnida":
                        animal_verification["is_actually_spider"] = False
                        if not animal_verification.get("correction_suggestion"):
                            label = taxonomy.common_name or taxonomy.class_name
                            animal_verification["correction_suggestion"] = (
                                f"iNaturalist suggests this is a {label}"
                            )
                    elif taxonomy.confidence >= 0.3:
                        animal_verification["is_actually_spider"] = True
            
            # Extract objects
            objects = self._extract_objects(description)
            
            result = ImageAnalysisResult(
                description=description,
                spider_detected=spider_detected,
                confidence=confidence,
                species_suggestion=species,
                danger_level=danger_level,
                objects_detected=objects,
                animal_verification=animal_verification,
                taxonomy=taxonomy
            )
            
            logging.info(f"Image analysis result: {result}")
            return result
            
        except Exception as e:
            logging.error(f"Failed to analyze image: {e}")
            return None
    
    def _generate_description(self, image: Image.Image) -> str:
        """Generate description of the image using BLIP"""
        try:
            # Process image
            inputs = self.blip_processor(image, return_tensors="pt").to(self.device)
            
            # Generate caption
            with torch.no_grad():
                out = self.blip_model.generate(**inputs, max_length=50)
            
            description = self.blip_processor.decode(out[0], skip_special_tokens=True)
            return description
            
        except Exception as e:
            logging.error(f"Failed to generate image description: {e}")
            return "Unable to analyze image"
    
    def _detect_spider_content(self, description: str) -> Tuple[bool, float]:
        """Detect if description contains spider-related content"""
        description_lower = description.lower()
        
        # Check for spider keywords
        matches = 0
        total_keywords = len(self.spider_keywords)
        
        for keyword in self.spider_keywords:
            if keyword in description_lower:
                matches += 1
        
        # Calculate confidence based on keyword matches
        confidence = matches / total_keywords if total_keywords > 0 else 0.0
        spider_detected = matches > 0
        
        # Boost confidence for exact spider mentions
        if any(word in description_lower for word in ["spider", "arachnid", "tarantula"]):
            confidence = max(confidence, 0.7)
            spider_detected = True
        
        return spider_detected, confidence
    
    def _verify_animal_classification(self, description: str) -> Dict:
        """Verify what type of animal this actually is"""
        description_lower = description.lower()
        verification_result = {
            "detected_categories": [],
            "confidence_scores": {},
            "is_actually_spider": False,
            "correction_suggestion": None,
            "reasoning": []
        }
        
        # Check each animal category
        for category, details in self.animal_categories.items():
            score = 0
            matches = []
            total_possible = len(details["keywords"]) + len(details["body_parts"]) + len(details["characteristics"])
            
            # Check keywords
            for keyword in details["keywords"]:
                if keyword in description_lower:
                    score += 2  # Keywords get higher weight
                    matches.append(f"keyword: {keyword}")
            
            # Check body parts
            for part in details["body_parts"]:
                if part in description_lower:
                    score += 1.5
                    matches.append(f"body part: {part}")
            
            # Check characteristics
            for char in details["characteristics"]:
                if char in description_lower:
                    score += 1
                    matches.append(f"characteristic: {char}")
            
            if score > 0:
                confidence = min(score / (total_possible * 1.5), 1.0)  # Normalize to 0-1
                verification_result["detected_categories"].append(category)
                verification_result["confidence_scores"][category] = confidence
                if matches:
                    verification_result["reasoning"].append(f"{category}: {', '.join(matches)}")
        
        # Determine the most likely category
        if verification_result["confidence_scores"]:
            best_category = max(verification_result["confidence_scores"], 
                              key=verification_result["confidence_scores"].get)
            best_confidence = verification_result["confidence_scores"][best_category]
            
            # Check if it's actually a spider
            if best_category == "spiders" and best_confidence > 0.3:
                verification_result["is_actually_spider"] = True
            elif best_category != "spiders" and best_confidence > 0.4:
                verification_result["is_actually_spider"] = False
                verification_result["correction_suggestion"] = f"This appears to be a {best_category.replace('_', ' ')} rather than a spider"
            
            # Add specific corrections for common misidentifications
            if best_category == "insects":
                verification_result["reasoning"].append("Note: Insects have 6 legs and 3 body segments, spiders have 8 legs and 2 body segments")
            elif best_category == "other_arthropods":
                verification_result["reasoning"].append("Note: This arthropod is not a spider - check leg count and body structure")
            elif best_category == "vertebrates":
                verification_result["reasoning"].append("Note: This is a vertebrate animal, not an arachnid")
        
        return verification_result
    
    def _predict_comprehensive_taxonomy(self, description: str, image: Image.Image) -> TaxonomyInfo:
        """Predict taxonomic classification using iNaturalist when possible."""

        taxonomy = self._predict_taxonomy_with_inat(image)
        if taxonomy and taxonomy.kingdom:
            return taxonomy

        return self._predict_taxonomy_from_description(description)

    def _predict_taxonomy_with_inat(self, image: Image.Image) -> Optional[TaxonomyInfo]:
        """Use iNaturalist computer vision to derive taxonomy, if enabled."""

        if not self.inat_client or not self.inat_client.is_available:
            return None

        try:
            prediction = self.inat_client.identify_taxon(image)
        except Exception as exc:  # pragma: no cover - network interaction
            logging.warning("iNaturalist classification error: %s", exc)
            return None

        if not prediction or not prediction.taxon:
            return None

        return self._taxon_from_inat(prediction)

    def _taxon_from_inat(self, prediction: INatPrediction) -> TaxonomyInfo:
        """Convert an iNaturalist prediction into our TaxonomyInfo structure."""

        taxonomy = TaxonomyInfo(source="inaturalist")
        taxon = prediction.taxon or {}
        ancestors = taxon.get("ancestors") or []
        lineage = list(ancestors) + [taxon]

        rank_map = {
            "kingdom": "kingdom",
            "phylum": "phylum",
            "class": "class_name",
            "classis": "class_name",
            "order": "order",
            "family": "family",
            "genus": "genus",
            "species": "species",
        }

        for node in lineage:
            rank = (node.get("rank") or "").lower()
            name = node.get("name")
            if not rank or not name:
                continue
            mapped = rank_map.get(rank)
            if not mapped:
                continue
            setattr(taxonomy, mapped, name)

        # Ensure genus/species fall back to the focal taxon if needed
        if not taxonomy.genus and (taxon.get("rank") == "genus"):
            taxonomy.genus = taxon.get("name")
        if not taxonomy.species and (taxon.get("rank") == "species"):
            taxonomy.species = taxon.get("name")

        taxonomy.scientific_name = taxon.get("name")
        taxonomy.common_name = taxon.get("preferred_common_name") or taxon.get("english_common_name")
        confidence = float(prediction.score)
        taxonomy.confidence = max(0.0, min(confidence, 1.0))

        return taxonomy

    def _predict_taxonomy_from_description(self, description: str) -> TaxonomyInfo:
        """Fallback heuristic taxonomy when external services are unavailable."""

        description_lower = description.lower()
        taxonomy = TaxonomyInfo(source="heuristic")

        if any(word in description_lower for word in ["spider", "arachnid", "web", "eight legs"]):
            taxonomy.kingdom = "Animalia"
            taxonomy.phylum = "Arthropoda"
            taxonomy.class_name = "Arachnida"
            taxonomy.order = "Araneae"

            best_match = self._match_spider_species(description_lower)
            if best_match:
                taxonomy.family = best_match.get("family")
                taxonomy.genus = best_match.get("genus")
                taxonomy.species = best_match.get("species")
                taxonomy.scientific_name = best_match.get("scientific")
                taxonomy.common_name = best_match.get("common")
                taxonomy.confidence = best_match.get("confidence", 0.0)
            else:
                taxonomy.confidence = 0.5

        elif any(word in description_lower for word in ["insect", "bug", "beetle", "fly", "ant", "bee", "wasp", "six legs"]):
            taxonomy.kingdom = "Animalia"
            taxonomy.phylum = "Arthropoda"
            taxonomy.class_name = "Insecta"

            insect_orders = {
                "beetle": ("Coleoptera", "Beetles"),
                "fly": ("Diptera", "Flies"),
                "ant": ("Hymenoptera", "Ants, Bees, and Wasps"),
                "bee": ("Hymenoptera", "Ants, Bees, and Wasps"),
                "wasp": ("Hymenoptera", "Ants, Bees, and Wasps"),
                "butterfly": ("Lepidoptera", "Butterflies and Moths"),
                "moth": ("Lepidoptera", "Butterflies and Moths"),
                "grasshopper": ("Orthoptera", "Grasshoppers and Crickets"),
                "cricket": ("Orthoptera", "Grasshoppers and Crickets"),
            }

            for keyword, (order, common) in insect_orders.items():
                if keyword in description_lower:
                    taxonomy.order = order
                    taxonomy.common_name = common
                    taxonomy.confidence = 0.7
                    break
            else:
                taxonomy.confidence = 0.4

        elif any(word in description_lower for word in ["tarantula", "large spider", "hairy spider"]):
            taxonomy.kingdom = "Animalia"
            taxonomy.phylum = "Arthropoda"
            taxonomy.class_name = "Arachnida"
            taxonomy.order = "Araneae"
            taxonomy.family = "Theraphosidae"
            taxonomy.common_name = "Tarantula"
            taxonomy.scientific_name = "Theraphosidae"
            taxonomy.confidence = 0.8

        elif any(word in description_lower for word in ["scorpion", "pincers", "tail"]):
            taxonomy.kingdom = "Animalia"
            taxonomy.phylum = "Arthropoda"
            taxonomy.class_name = "Arachnida"
            taxonomy.order = "Scorpiones"
            taxonomy.common_name = "Scorpion"
            taxonomy.confidence = 0.8

        elif any(word in description_lower for word in ["lizard", "gecko", "scales"]):
            taxonomy.kingdom = "Animalia"
            taxonomy.phylum = "Chordata"
            taxonomy.class_name = "Reptilia"
            taxonomy.order = "Squamata"
            taxonomy.common_name = "Lizard"
            taxonomy.confidence = 0.7

        elif any(word in description_lower for word in ["snake", "serpent"]):
            taxonomy.kingdom = "Animalia"
            taxonomy.phylum = "Chordata"
            taxonomy.class_name = "Reptilia"
            taxonomy.order = "Squamata"
            taxonomy.common_name = "Snake"
            taxonomy.confidence = 0.7

        elif any(word in description_lower for word in ["frog", "toad", "amphibian"]):
            taxonomy.kingdom = "Animalia"
            taxonomy.phylum = "Chordata"
            taxonomy.class_name = "Amphibia"
            taxonomy.common_name = "Amphibian"
            taxonomy.confidence = 0.7

        return taxonomy
    
    def _match_spider_species(self, description: str) -> Optional[Dict]:
        """Match description to specific spider species"""
        best_match = None
        best_score = 0.0
        
        for common_name, spider_data in self.spider_patterns["common_spiders"].items():
            score = 0.0
            matches = 0
            
            # Check if common name is mentioned
            if common_name in description:
                score += 0.8
                matches += 1
            
            # Check for scientific name components
            scientific = spider_data.get("scientific", "")
            if scientific.lower() in description:
                score += 0.9
                matches += 1
            
            # Check for keywords
            for keyword in spider_data.get("keywords", []):
                if keyword in description:
                    score += 0.3
                    matches += 1
            
            # Normalize score
            if matches > 0:
                final_score = min(score, 1.0)
                if final_score > best_score:
                    best_score = final_score
                    
                    # Parse scientific name
                    sci_parts = scientific.split()
                    genus = sci_parts[0] if sci_parts else ""
                    species = sci_parts[1] if len(sci_parts) > 1 else ""
                    
                    best_match = {
                        "family": self._get_family_from_scientific(scientific),
                        "genus": genus,
                        "species": species,
                        "scientific": scientific,
                        "common": common_name.title(),
                        "confidence": final_score,
                        "danger": spider_data.get("danger", "unknown")
                    }
        
        return best_match if best_score > 0.3 else None
    
    def _get_family_from_scientific(self, scientific_name: str) -> str:
        """Get family name from scientific name (simplified lookup)"""
        family_mapping = {
            "Salticidae": "Salticidae",
            "Araneidae": "Araneidae", 
            "Lycosidae": "Lycosidae",
            "Latrodectus": "Theridiidae",
            "Loxosceles": "Sicariidae",
            "Parasteatoda": "Theridiidae",
            "Theraphosidae": "Theraphosidae",
            "Pholcidae": "Pholcidae"
        }
        
        for key, family in family_mapping.items():
            if key in scientific_name:
                return family
        return "Unknown"

    def _analyze_species_and_danger(self, description: str) -> Tuple[Optional[str], str]:
        """Analyze potential species and danger level"""
        description_lower = description.lower()
        
        # Check for dangerous species
        for indicator, danger in self.dangerous_indicators.items():
            if indicator in description_lower:
                return indicator.title(), danger
        
        # Check for common harmless species
        harmless_species = {
            "jumping spider": "safe",
            "garden spider": "safe", 
            "house spider": "safe",
            "orb weaver": "safe",
            "cellar spider": "safe"
        }
        
        for species, danger in harmless_species.items():
            if species in description_lower:
                return species.title(), danger
        
        # Default for general spider detection
        if any(word in description_lower for word in ["spider", "arachnid"]):
            return "Spider (species unknown)", "unknown"
        
        return None, "unknown"
    
    def _extract_objects(self, description: str) -> List[str]:
        """Extract objects mentioned in the description"""
        # Simple object extraction based on common words
        common_objects = [
            "web", "wall", "ceiling", "corner", "garden", "flower", "leaf",
            "hand", "finger", "book", "table", "chair", "window", "door"
        ]
        
        description_lower = description.lower()
        detected_objects = []
        
        for obj in common_objects:
            if obj in description_lower:
                detected_objects.append(obj)
        
        return detected_objects

# Enhanced SocialPost to include image analysis
@dataclass
class EnhancedSocialPost:
    """Extended SocialPost with image analysis"""
    id: str
    text: str
    author_handle: Optional[str] = None
    conversation_id: Optional[str] = None
    lang: Optional[str] = None
    like_count: int = 0
    repost_count: int = 0
    reply_count: int = 0
    impression_count: int = 0
    image_urls: List[str] = None
    image_analysis: List[ImageAnalysisResult] = None
    
    def __post_init__(self):
        if self.image_urls is None:
            self.image_urls = []
        if self.image_analysis is None:
            self.image_analysis = []

def create_image_aware_prompt(post_text: str, image_analysis: List[ImageAnalysisResult], context_documents) -> str:
    """Create a prompt that incorporates image analysis results"""
    
    # Base prompt components
    image_context = ""
    
    if image_analysis:
        image_descriptions = []
        spider_info = []
        verification_info = []
        taxonomy_info = []
        
        for analysis in image_analysis:
            image_descriptions.append(f"Image shows: {analysis.description}")
            
            # Add comprehensive taxonomy information
            if analysis.taxonomy and analysis.taxonomy.kingdom:
                taxonomy = analysis.taxonomy
                tax_parts = []
                if taxonomy.scientific_name:
                    tax_parts.append(f"Scientific name: {taxonomy.scientific_name}")
                if taxonomy.common_name:
                    tax_parts.append(f"Common name: {taxonomy.common_name}")
                if taxonomy.family:
                    tax_parts.append(f"Family: {taxonomy.family}")
                if taxonomy.order:
                    tax_parts.append(f"Order: {taxonomy.order}")
                if taxonomy.class_name:
                    tax_parts.append(f"Class: {taxonomy.class_name}")
                if taxonomy.source:
                    source_label = "iNaturalist computer vision" if taxonomy.source == "inaturalist" else taxonomy.source
                    tax_parts.append(f"Source: {source_label}")
                if taxonomy.confidence > 0:
                    tax_parts.append(f"Classification confidence: {taxonomy.confidence:.0%}")
                
                if tax_parts:
                    taxonomy_info.append(f"Taxonomic classification: {'; '.join(tax_parts)}")
            
            # Add animal verification information
            if analysis.animal_verification:
                verification = analysis.animal_verification
                if verification.get('correction_suggestion'):
                    verification_info.append(f"⚠️ Animal ID correction: {verification['correction_suggestion']}")
                elif verification.get('is_actually_spider'):
                    verification_info.append("✅ Confirmed as spider")
                
                if verification.get('detected_categories'):
                    categories = verification['detected_categories']
                    if len(categories) > 1:
                        verification_info.append(f"Detected as: {', '.join(categories)}")
            
            if analysis.spider_detected:
                spider_info.append(
                    f"Spider detected (confidence: {analysis.confidence:.1%})"
                )
                if analysis.species_suggestion:
                    spider_info.append(f"Possible species: {analysis.species_suggestion}")
                if analysis.danger_level != "unknown":
                    spider_info.append(f"Safety level: {analysis.danger_level}")
        
        if image_descriptions:
            image_context = "\n".join([
                "Visual context from attached images:",
                *[f"- {desc}" for desc in image_descriptions]
            ])
        
        if taxonomy_info:
            image_context += "\n" + "\n".join([
                "Scientific classification:",
                *[f"- {info}" for info in taxonomy_info]
            ])
        
        if verification_info:
            image_context += "\n" + "\n".join([
                "Animal identification verification:",
                *[f"- {info}" for info in verification_info]
            ])
        
        if spider_info:
            image_context += "\n" + "\n".join([
                "Spider analysis:",
                *[f"- {info}" for info in spider_info]
            ])
    
    return image_context

# Global image analyzer instance
image_analyzer = ImageAnalyzer()

if __name__ == "__main__":
    # Test the image analyzer
    test_url = "https://example.com/spider-image.jpg"  # Replace with actual image URL
    result = image_analyzer.analyze_image_from_url(test_url)
    if result:
        print(f"Analysis: {result}")
    else:
        print("Failed to analyze image")