"""Generate a high-impact spider-themed post from historical Spider Guardian data."""

from __future__ import annotations

import argparse
import json
import logging
import random
import textwrap
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ..config import ChatProviderConfig, SpiderGuardianConfig
from ..storage import SQLDataStore


STYLE_CUES: Sequence[str] = (
    "Kick off with a punchline-worthy visual gag about spider antics.",
    "Spin a short absurd metaphor that turns a spider into an overqualified roommate.",
    "Set up a myth as the straight man and knock it down with a witty comeback.",
    "Drop a playful science fact then pivot to a comedic, empathetic aside.",
    "End with an invite for goofy spider sightings or puns from the audience.",
)

HUMOR_DIRECTIVE = (
    "Keep it genuinely funny—lean on wordplay, playful exaggeration,"
    " or charming self-awareness, never mean-spirited."
)

INTENT_DIRECTIVES: Dict[str, str] = {
    "myth-busting": "Dispel fear with kind, factual reassurance.",
    "celebration": "Cheer for spiders as ecological heroes.",
    "education": "Teach one memorable scientific detail without jargon.",
    "call-to-action": "Motivate the audience to protect spiders or share a positive story.",
}

EMOJI = "\U0001F577"  # spider emoji, left as escape to keep source ASCII


@dataclass
class Highlight:
    text: str
    score: float
    meta_summary: str
    metadata: Dict[str, Any]
    link: str
    created_at: datetime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a high-performing spider post")
    parser.add_argument("--topic", default="spider myth busting", help="Theme for the new post")
    parser.add_argument("--intent", choices=sorted(INTENT_DIRECTIVES.keys()), default="myth-busting",
                        help="High-level communication goal")
    parser.add_argument("--model", default=None, help="Optional Hugging Face model override")
    parser.add_argument("--provider", default="local", help="Provider name to initialise (default: local)")
    parser.add_argument("--temperature", type=float, default=0.4, help="Sampling temperature for the provider")
    parser.add_argument("--num-candidates", type=int, default=3, help="How many alternatives to generate")
    parser.add_argument("--max-words", type=int, default=60, help="Maximum word count for the post")
    parser.add_argument("--sample-limit", type=int, default=40, help="How many historical posts to inspect")
    parser.add_argument("--prompt-examples", type=int, default=6,
                        help="Number of historic highlights to expose to the model")
    parser.add_argument("--min-engagement", type=float, default=0.0,
                        help="Minimum engagement score a sample must reach to be used")
    parser.add_argument("--export-training-data", type=Path,
                        help="If provided, dump top samples to this JSONL file for future fine-tuning")
    parser.add_argument("--show-prompt", action="store_true", help="Print the final prompt for inspection")
    parser.add_argument("--seed", type=int, help="Deterministic seed for reproducible variation")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def flatten_metrics(data: Dict[str, Any]) -> Dict[str, Any]:
    flattened: Dict[str, Any] = {}
    stack: List[Any] = [data]
    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            for key, value in current.items():
                if not key:
                    continue
                if isinstance(value, dict):
                    stack.append(value)
                elif isinstance(value, list):
                    stack.extend(item for item in value if isinstance(item, dict))
                else:
                    flattened[str(key).lower()] = value
        elif isinstance(current, list):
            stack.extend(item for item in current if isinstance(item, dict))
    return flattened


def to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.replace(",", "").strip()
        if not cleaned:
            return None
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def compute_engagement_score(metadata: Dict[str, Any], created_at: datetime, text: str) -> float:
    metrics = flatten_metrics(metadata)

    def pick(keys: Iterable[str]) -> float:
        for key in keys:
            value = to_float(metrics.get(key))
            if value is not None:
                return value
        return 0.0

    likes = pick(("likes", "like_count", "favorite_count", "faves"))
    reshares = pick(("retweets", "retweet_count", "reposts", "repost_count", "quotes"))
    replies = pick(("reply_count", "replies", "comments"))
    impressions = pick(("impressions", "views", "view_count"))
    quality = pick(("quality_score", "score", "engagement_score"))

    base = likes + 1.35 * reshares + 1.15 * replies + 0.45 * impressions + 2.0 * quality
    if base <= 0:
        base = 0.25 * len(text.split())

    age_factor = 1.0
    try:
        now = datetime.now(timezone.utc)
        if created_at.tzinfo is None:
            created = created_at.replace(tzinfo=timezone.utc)
        else:
            created = created_at
        age_days = max(0.0, (now - created).total_seconds() / 86400.0)
        if age_days < 1.0:
            age_factor = 1.2
        elif age_days < 7.0:
            age_factor = 1.05
        elif age_days > 90.0:
            age_factor = 0.9
    except Exception:
        pass

    return base * age_factor


def describe_metrics(metadata: Dict[str, Any]) -> str:
    metrics = flatten_metrics(metadata)

    def format_metric(label: str, keys: Sequence[str]) -> Optional[str]:
        value = None
        for key in keys:
            value = to_float(metrics.get(key))
            if value is not None and value > 0:
                break
        if value is None or value <= 0:
            return None
        if value.is_integer():
            display = str(int(value))
        else:
            display = f"{value:.1f}"
        return f"{label} {display}"

    parts: List[str] = []
    mappings = [
        ("likes", ("likes", "like_count", "favorite_count", "faves")),
        ("reshares", ("retweets", "reposts", "retweet_count", "repost_count")),
        ("replies", ("reply_count", "replies", "comments")),
        ("views", ("impressions", "views", "view_count")),
    ]
    for label, keys in mappings:
        metric = format_metric(label, keys)
        if metric:
            parts.append(metric)
    return ", ".join(parts[:3])


def extract_text(article_content: Any, fallback_title: str) -> Optional[str]:
    candidates: List[str] = []
    if isinstance(article_content, str):
        stripped = article_content.strip()
        if stripped:
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError:
                candidates.append(stripped)
            else:
                if isinstance(payload, dict):
                    for key in ("generated_reply", "reply_text", "text", "body", "content"):
                        value = payload.get(key)
                        if isinstance(value, str):
                            candidates.append(value.strip())
                elif isinstance(payload, list):
                    for item in payload:
                        if isinstance(item, str):
                            candidates.append(item.strip())
    if fallback_title:
        candidates.append(fallback_title.strip())

    cleaned: List[str] = []
    for candidate in candidates:
        normalized = " ".join(candidate.split())
        if len(normalized) < 24:
            continue
        if normalized not in cleaned:
            cleaned.append(normalized)
    if not cleaned:
        return None
    cleaned.sort(key=len, reverse=True)
    longest = cleaned[0]
    if len(longest.split()) > 60:
        return " ".join(longest.split()[:60])
    return longest


def gather_highlights(store: SQLDataStore, limit: int, min_score: float) -> List[Highlight]:
    highlights: List[Highlight] = []
    for article in store.iter_scraped_articles():
        text = extract_text(article.content, article.title)
        if not text:
            continue
        score = compute_engagement_score(article.metadata, article.created_at, text)
        if score < min_score:
            continue
        meta_summary = describe_metrics(article.metadata)
        highlights.append(
            Highlight(
                text=text,
                score=score,
                meta_summary=meta_summary,
                metadata=article.metadata,
                link=article.link,
                created_at=article.created_at,
            )
        )
    highlights.sort(key=lambda h: h.score, reverse=True)
    return highlights[:limit]


def format_highlight(highlight: Highlight) -> str:
    snippet = textwrap.shorten(highlight.text, width=140, placeholder="...")
    if highlight.meta_summary:
        return f"{snippet} ({highlight.meta_summary})"
    return snippet


def build_generation_prompt(topic: str, intent: str, highlights: Sequence[str], max_words: int,
                             style_cue: str) -> str:
    inspiration_block = "\n".join(f"- {line}" for line in highlights) if highlights else (
        "- Emphasise calm, factual reassurance about spiders."
    )
    directive = INTENT_DIRECTIVES.get(intent, "Keep the tone warm and factual.")
    instructions = textwrap.dedent(
        f"""
        Craft a social media post about {topic}.
        Goal: {directive}
        Tone: upbeat arachnid expert who sounds human and approachable.
        Humor: {HUMOR_DIRECTIVE}
        Constraints:
        1. Maximum {max_words} words.
        2. Include exactly one emoji that fits the message.
        3. Keep it first-person or inclusive ("we", "let's") where natural.
        Inspiration patterns (do not copy verbatim):
        {inspiration_block}
        Extra creative cue: {style_cue}
        Return just the finished post text.
        """
    ).strip()
    return instructions


def generate_with_provider(provider: Any, topic: str, intent: str, highlights: Sequence[str],
                           max_words: int, num_candidates: int) -> List[str]:
    outputs: List[str] = []
    for idx in range(num_candidates):
        style_cue = STYLE_CUES[idx % len(STYLE_CUES)]
        prompt = build_generation_prompt(topic, intent, highlights, max_words, style_cue)
        logging.debug("Prompt %d:%s%s", idx + 1, "\n" if logging.getLogger().isEnabledFor(logging.DEBUG) else " ", prompt)
        try:
            text = provider.generate(prompt)
        except Exception as exc:
            logging.warning("Provider generation failed on attempt %d: %s", idx + 1, exc)
            break
        cleaned = " ".join((text or "").strip().split())
        if cleaned:
            outputs.append(cleaned)
    return outputs


def build_fallback_post(topic: str, highlights: Sequence[Highlight], max_words: int) -> str:
    joke_bank = [
        "My house spider keeps sending rent reminders in mosquito legs—best landlord ever",
        "Orb weavers are basically night-shift decorators tossing glitter nets for gnats",
        "Tarantulas are just hairy vacuum robots with eight legs and zero judgment",
        "Every spider in the garden is an unpaid security guard with a taste for mosquitoes",
        "Let’s normalize escorting spiders outside like VIP guests instead of screaming extras",
    ]
    punchline = random.choice(joke_bank)
    if highlights:
        punchline = textwrap.shorten(highlights[0].text, width=120, placeholder="...")
    base = (
        f"{EMOJI} Comedy PSA: {punchline}. "
        "Spread a spider fun-fact today so fewer eight-legged roomies get eviction notices!"
    )
    words = base.split()
    if len(words) > max_words:
        base = " ".join(words[:max_words])
    return base


def generate_candidates(
    config: SpiderGuardianConfig,
    *,
    topic: str,
    intent: str,
    provider: Optional[Any] = None,
    store: Optional[SQLDataStore] = None,
    max_words: int = 60,
    num_candidates: int = 3,
    sample_limit: int = 40,
    prompt_examples: int = 6,
    min_engagement: float = 0.0,
) -> Tuple[List[str], List[str], List[Highlight], Optional[Any]]:
    """Generate candidate posts and return them alongside supporting context."""

    active_store = store or SQLDataStore(config.sql_database_path)
    highlights = gather_highlights(active_store, sample_limit, min_engagement)
    formatted_highlights = [format_highlight(sample) for sample in highlights[:prompt_examples]]

    active_provider = provider
    if active_provider is None:
        from ..providers import build_chat_providers

        providers = build_chat_providers(config.providers)
        active_provider = next((p for p in providers if p.is_available()), None)

    generated: List[str] = []
    if active_provider is not None:
        generated = generate_with_provider(
            provider=active_provider,
            topic=topic,
            intent=intent,
            highlights=formatted_highlights,
            max_words=max_words,
            num_candidates=num_candidates,
        )

    if not generated:
        generated.append(build_fallback_post(topic, highlights, max_words))

    return generated, formatted_highlights, highlights, active_provider


def export_training_dataset(samples: Sequence[Highlight], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for highlight in samples:
            record = {
                "text": highlight.text,
                "score": highlight.score,
                "meta_summary": highlight.meta_summary,
                "metadata": highlight.metadata,
                "link": highlight.link,
                "created_at": highlight.created_at.isoformat(),
            }
            json.dump(record, handle, ensure_ascii=False)
            handle.write("\n")
    logging.info("Exported %d samples to %s", len(samples), destination)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    if args.seed is not None:
        random.seed(args.seed)

    config = SpiderGuardianConfig()
    existing_providers = tuple(config.providers)
    if args.model:
        provider_configs = (
            ChatProviderConfig(name=args.provider, model=args.model, temperature=args.temperature),
        )
    else:
        provider_configs = tuple(
            ChatProviderConfig(name=prov.name, model=prov.model, temperature=args.temperature)
            for prov in existing_providers
        )
        if args.provider and provider_configs:
            first = provider_configs[0]
            if args.provider != first.name:
                logging.info(
                    "Switching provider implementation to '%s' (model=%s)",
                    args.provider,
                    first.model,
                )
                provider_configs = (
                    ChatProviderConfig(name=args.provider, model=first.model, temperature=args.temperature),
                )
    config.providers = provider_configs

    store = SQLDataStore(config.sql_database_path)
    generated, formatted_highlights, highlights, _provider = generate_candidates(
        config,
        topic=args.topic,
        intent=args.intent,
        store=store,
        max_words=args.max_words,
        num_candidates=args.num_candidates,
        sample_limit=args.sample_limit,
        prompt_examples=args.prompt_examples,
        min_engagement=args.min_engagement,
    )

    if args.export_training_data:
        export_training_dataset(highlights, args.export_training_data)

    print("Historic inspiration (top patterns):")
    if formatted_highlights:
        for item in formatted_highlights[:5]:
            print(f"- {item}")
    else:
        print("- No high-engagement samples found; using heuristics only.")

    if args.show_prompt and formatted_highlights:
        preview_prompt = build_generation_prompt(
            args.topic,
            args.intent,
            formatted_highlights,
            args.max_words,
            STYLE_CUES[0],
        )
        print("\nPrompt preview:\n" + preview_prompt)

    print("\nGenerated candidates:")
    for idx, text in enumerate(generated, start=1):
        print(f"{idx}. {text}")


__all__ = [
    "Highlight",
    "INTENT_DIRECTIVES",
    "generate_candidates",
    "main",
]


if __name__ == "__main__":
    main()
