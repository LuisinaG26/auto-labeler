"""Adaptador de Ollama (API nativa), con probabilidades a partir de los logprobs."""
from __future__ import annotations

import base64
import io
import json
import math
import time
from collections.abc import Sequence

import httpx
from PIL import Image

from autolabel.vlm.base import VlmError

_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def encode_image(img: Image.Image, max_side: int | None = None) -> str:
    """JPEG en base64. Solo reduce, nunca amplía."""
    img = img.convert("RGB")
    if max_side and max(img.size) > max_side:
        scale = max_side / max(img.size)
        img = img.resize((round(img.width * scale), round(img.height * scale)), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode()


class OllamaVlm:
    def __init__(
        self,
        model: str,
        base_url: str = "http://ollama:11434",
        *,
        timeout: float = 180.0,
        max_image_side: int = 1024,
        retries: int = 2,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.model = model
        self.max_image_side = max_image_side
        self._retries = retries
        self._http = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout, transport=transport)

    # ------------------------------------------------------------------ bajo nivel

    def _chat(self, images: Sequence[Image.Image], prompt: str, system: str | None, **extra) -> dict:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({
            "role": "user",
            "content": prompt,
            "images": [encode_image(i, self.max_image_side) for i in images],
        })
        body = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "think": False,                       # sin cadena de pensamiento: respuesta directa
            "options": {"temperature": 0, **extra.pop("options", {})},
            **extra,
        }
        last: Exception | None = None
        for attempt in range(self._retries + 1):
            try:
                r = self._http.post("/api/chat", json=body)
                r.raise_for_status()
                return r.json()
            except (httpx.TransportError, httpx.HTTPStatusError) as e:
                last = e
                time.sleep(1.5 * (attempt + 1))
        raise VlmError(f"Ollama no respondió: {last}")

    # ------------------------------------------------------------------ contrato

    def choose(
        self,
        images: Sequence[Image.Image],
        prompt: str,
        options: Sequence[str],
        *,
        system: str | None = None,
    ) -> dict[str, float]:
        if not 2 <= len(options) <= len(_LETTERS):
            raise ValueError("choose necesita entre 2 y 26 opciones")
        letters = _LETTERS[: len(options)]
        listing = "\n".join(f"{letter}. {opt}" for letter, opt in zip(letters, options, strict=True))
        full_prompt = f"{prompt}\n\n{listing}\n\nAnswer with only the letter of the correct option."

        data = self._chat(
            images, full_prompt, system,
            logprobs=True, top_logprobs=20, options={"num_predict": 2},
        )
        probs = self._letter_probabilities(data, letters)
        return {opt: probs[letter] for letter, opt in zip(letters, options, strict=True)}

    def ask_json(
        self,
        images: Sequence[Image.Image],
        prompt: str,
        schema: dict,
        *,
        system: str | None = None,
    ) -> dict:
        data = self._chat(images, prompt, system, format=schema, options={"num_predict": 256})
        text = data.get("message", {}).get("content", "")
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise VlmError(f"Respuesta no válida como JSON: {text!r}") from e

    # ------------------------------------------------------------------ interpretación

    @staticmethod
    def _letter_probabilities(data: dict, letters: str) -> dict[str, float]:
        """Probabilidad de cada letra a partir de los logprobs del token que el modelo eligió.

        Solo cuenta el primer token que es una de las letras. Los demás (un espacio, un
        marcador de pensamiento) se saltan, y si el modelo nunca llega a responder se informa
        el error en vez de inventar una probabilidad a partir de alternativas lejanas.
        """
        valid = set(letters)               # un conjunto: con una cadena, "" contaría como contenida
        for entry in data.get("logprobs") or []:
            if entry["token"].strip().upper() not in valid:
                continue
            alternatives: dict[str, float] = {}
            for alt in [entry, *entry.get("top_logprobs", [])]:
                token = alt["token"].strip().upper()
                if token in valid:
                    alternatives[token] = max(alt["logprob"], alternatives.get(token, -math.inf))
            total = sum(math.exp(lp) for lp in alternatives.values())
            return {ch: (math.exp(alternatives[ch]) / total if ch in alternatives else 0.0) for ch in letters}

        # Sin logprobs: se usa la letra del texto y se informa certeza total
        text = (data.get("message", {}).get("content") or "").strip().upper()
        if not data.get("logprobs") and text[:1] and text[0] in valid:
            return {c: 1.0 if c == text[0] else 0.0 for c in letters}
        raise VlmError(
            f"El modelo no respondió con una opción (texto: {text!r}). Si es un modelo de "
            "razonamiento, usa su variante de instrucción."
        )
