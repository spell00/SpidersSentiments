"""Chat provider implementations for Spider Guardian."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import logging

from transformers import pipeline, __version__ as transformers_version

# Optional torch for device selection
import torch  # type: ignore

# LangChain imports are deferred inside the provider to avoid top-level import failures

from .config import ChatProviderConfig


class ChatProvider:
    """Abstract base class for chat providers."""

    def __init__(self, config: ChatProviderConfig) -> None:
        self.config = config

    def is_available(self) -> bool:
        raise NotImplementedError

    def generate(self, prompt: str, conversation: Optional[List[Dict[str, str]]] = None) -> str:
        raise NotImplementedError


class LocalModelChatProvider(ChatProvider):
    """Provider that uses a local HuggingFace transformer pipeline."""

    def __init__(self, config: ChatProviderConfig) -> None:
        super().__init__(config)
        self.generator = None
        if pipeline is None:
            logging.warning("transformers pipeline unavailable; local provider disabled")
            return

        model_name = getattr(self.config, "model", None)
        if not model_name:
            logging.error("Provider config missing 'model' attribute")
            return

        try:
            logging.info("Loading model %s...", model_name)
            is_gptq = "gptq" in model_name.lower()

            if is_gptq:
                try:
                    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
                except Exception as import_exc:  # pragma: no cover - best effort warning
                    logging.warning("Quantized model support unavailable: %s", import_exc)
                    raise

                if torch is None:
                    raise RuntimeError("PyTorch not available; cannot load GPTQ model")

                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )

                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map="auto",
                    trust_remote_code=False,
                    # quantization_config=quantization_config,
                )
                tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
                self.generator = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    trust_remote_code=False,
                )
                logging.info("Successfully loaded quantized GPTQ model %s with transformers native support", model_name)
            else:
                # Prefer GPU if available
                kwargs = {"trust_remote_code": False}
                if torch is not None:
                    try:
                        if torch.cuda.is_available():
                            kwargs["device_map"] = "auto"
                    except Exception:
                        pass
                self.generator = pipeline("text-generation", model=model_name, **kwargs)
                logging.info("Successfully loaded standard model %s", model_name)
        except Exception as exc:
            logging.warning("Local model failed to init: %s", exc)
            try:
                logging.info("Trying simplified model loading...")
                self.generator = pipeline("text-generation", model=model_name)
                logging.info("Successfully loaded model %s with simplified loading", model_name)
            except Exception as simplified_exc:
                logging.error("All model loading attempts failed: %s", simplified_exc)
                if isinstance(simplified_exc, KeyError) and "mistral" in str(simplified_exc).lower():
                    logging.info(
                        "transformers %s does not include 'mistral' support. Upgrade to transformers>=4.33.0 to load %s.",
                        transformers_version,
                        model_name,
                    )
                self.generator = None

    def is_available(self) -> bool:
        return self.generator is not None

    def generate(self, prompt: str, conversation: Optional[List[Dict[str, str]]] = None) -> str:
        if self.generator is None:
            raise RuntimeError("Local generator unavailable")

        result = self.generator(prompt, max_new_tokens=64, num_return_sequences=1)
        text = result[0]["generated_text"].strip()

        if text.startswith(prompt):
            text = text[len(prompt):].strip()
        else:
            idx = text.rfind(prompt)
            if idx != -1:
                text = text[idx + len(prompt):].strip()

        lines = [line for line in text.splitlines() if line.strip()]
        reply = lines[-1].strip() if lines else ""
        for prefix in ("Answer:", "Reply:", "Assistant:"):
            if reply.lower().startswith(prefix.lower()):
                reply = reply[len(prefix):].strip()
        return " ".join(reply.split())


class LangChainChatProvider(ChatProvider):
    """Provider that uses LangChain to wrap a Hugging Face chat or text model.

    We keep our prompt as a single human message to leverage LC templating and parsing.
    """

    def __init__(self, config: ChatProviderConfig) -> None:
        super().__init__(config)
        self._chain = None
        model_name = getattr(self.config, "model", None)
        if not model_name:
            logging.error("LangChain provider config missing 'model'")
            return
        try:
            # Defer imports so environments without LangChain don't error at module import time
            from langchain_core.prompts import ChatPromptTemplate  # type: ignore
            from langchain_core.output_parsers import StrOutputParser  # type: ignore
            from langchain_community.chat_models import ChatHuggingFace  # type: ignore
            from langchain_community.llms import HuggingFacePipeline  # type: ignore
            # Try chat wrapper first; if it fails, fall back to plain pipeline LLM
            try:
                chat = ChatHuggingFace(model=model_name)
                llm_or_chat = chat
            except Exception:
                if pipeline is None:
                    logging.error("transformers pipeline unavailable; cannot create HuggingFacePipeline")
                    return
                hf_gen = pipeline("text-generation", model=model_name)
                llm_or_chat = HuggingFacePipeline(pipeline=hf_gen)
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a concise social commenter. Reply naturally and briefly."),
                ("human", "{prompt}")
            ])
            self._chain = prompt | llm_or_chat | StrOutputParser()
            logging.info("LangChain provider initialised for model %s", model_name)
        except Exception as exc:
            logging.warning("LangChain provider init failed: %s", exc)
            self._chain = None

    def is_available(self) -> bool:
        return self._chain is not None

    def generate(self, prompt: str, conversation: Optional[List[Dict[str, str]]] = None) -> str:
        if self._chain is None:
            raise RuntimeError("LangChain provider unavailable")
        try:
            text = self._chain.invoke({"prompt": prompt})
        except Exception as exc:
            logging.debug("LangChain generation error: %s", exc)
            return ""
        text = (text or "").strip()
        # Strip common prefixes
        for prefix in ("Answer:", "Reply:", "Assistant:"):
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix):].strip()
        return " ".join(text.split())


def build_chat_providers(configs: Sequence[ChatProviderConfig]) -> List[ChatProvider]:
    """Instantiate all configured chat providers."""

    providers: List[ChatProvider] = []
    for provider_config in configs:
        name = getattr(provider_config, "name", "").lower()
        if not name:
            continue
        if name == "local":
            try:
                provider = LocalModelChatProvider(provider_config)
                if provider.is_available():
                    providers.append(provider)
                else:
                    logging.info("Skipping local provider; not available: %s", getattr(provider_config, "model", None))
            except Exception as exc:
                logging.warning("Local provider init failed: %s", exc)
        elif name == "langchain":
            try:
                provider = LangChainChatProvider(provider_config)
                if provider.is_available():
                    providers.append(provider)
                else:
                    logging.info("Skipping LangChain provider; not available: %s", getattr(provider_config, "model", None))
            except Exception as exc:
                logging.warning("LangChain provider init failed: %s", exc)
        else:
            logging.info("Unknown provider '%s' — skipping.", provider_config.name)
    if not providers:
        logging.warning("No chat providers available. Replies will not be generated.")
    return providers
