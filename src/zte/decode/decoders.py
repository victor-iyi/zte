"""EEG→language decoders: retrieval and prefix-LM generation.

:class:`RetrievalDecoder` finds nearest text strings in an aligned bank (the
practical zero-shot path after EEG-OT-CLIP). :class:`PrefixLanguageDecoder`
maps an EEG embedding to soft prompt prefixes for a frozen (or toy) LM.
:class:`LanguageDecoder` is a thin facade selecting path(s) from
:class:`~zte.decode.config.DecoderConfig.mode`.
"""

# pylint: disable=import-outside-toplevel
from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from zte.decode.alignment import OTCLIPAligner
from zte.decode.config import DecoderConfig, GenerativeConfig
from zte.device import resolve_device
from zte.logging_utils import get_logger

_LOG = get_logger('decode.decoders')

type GenerativeBackend = Literal['auto', 'transformers', 'toy']


def _l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """L2-normalises rows of ``x``."""
    x = np.asarray(x, dtype=np.float32)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)


def _transformers_available() -> bool:
    """Returns whether ``transformers`` can be imported."""
    try:
        import transformers  # noqa: F401

        return True
    except ImportError:
        return False


class RetrievalDecoder:
    """Decode by nearest-neighbour retrieval in the aligned text space.

    Given aligned EEG embeddings and a bank of ``(text_emb, text_string)``,
    returns the nearest text(s). Extends the spirit of
    :class:`~zte.inference.retrieval.NearestNeighborIndex` for cross-modal
    EEG→text.
    """

    def __init__(
        self,
        text_bank_emb: np.ndarray,
        texts: list[str],
        aligner: OTCLIPAligner | None = None,
        *,
        bank_already_aligned: bool = True,
    ) -> None:
        """Builds a retrieval index over a text bank.

        Args:
            text_bank_emb: Text embeddings ``(N, D)``. When ``aligner`` is set and
                ``bank_already_aligned`` is ``False``, these are passed through
                ``aligner.encode_text``.
            texts: Surface strings of length ``N``.
            aligner: Optional OT-CLIP aligner used to project query EEG (and
                optionally the bank) into the shared space.
            bank_already_aligned: If ``True``, ``text_bank_emb`` is already in
                the shared space; otherwise project with ``aligner``.

        Raises:
            ValueError: If bank and texts lengths disagree.
        """
        emb = np.asarray(text_bank_emb, dtype=np.float32)
        if len(emb) != len(texts):
            raise ValueError(f'text_bank_emb ({len(emb)}) and texts ({len(texts)}) must align.')
        self.aligner = aligner
        # When an aligner is attached, project the bank into the shared space unless
        # the caller explicitly marks it as already aligned (and dims already match).
        if aligner is not None:
            proj_dim = int(aligner.config.proj_dim)
            needs_proj = (not bank_already_aligned) or (emb.ndim == 2 and emb.shape[1] != proj_dim)
            if needs_proj:
                with torch.no_grad():
                    t = torch.from_numpy(emb)
                    emb = aligner.encode_text(t).cpu().numpy().astype(np.float32)
        self.bank = _l2_normalize(emb)
        self.texts = list(texts)

    def __len__(self) -> int:
        """Number of bank entries."""
        return len(self.texts)

    def _project_eeg(self, eeg_emb: np.ndarray) -> np.ndarray:
        """Projects EEG into the retrieval space if an aligner is attached."""
        eeg = np.asarray(eeg_emb, dtype=np.float32)
        if eeg.ndim == 1:
            eeg = eeg[None, :]
        if self.aligner is not None:
            with torch.no_grad():
                t = torch.from_numpy(eeg)
                eeg = self.aligner.encode_eeg(t).cpu().numpy().astype(np.float32)
        return _l2_normalize(eeg)

    def decode(self, eeg_emb: np.ndarray, k: int = 1) -> list[str]:
        """Returns the top-1 (or top-``k`` first) nearest text per query.

        Args:
            eeg_emb: Query EEG embeddings ``(N, D)`` or ``(D,)``.
            k: Unused for the returned string (kept for API symmetry); use
                :meth:`decode_topk` for multiple neighbours. Top-1 is always returned.

        Returns:
            List of nearest text strings, one per query row.
        """
        del k  # top-1 path
        top = self.decode_topk(eeg_emb, k=1)
        return [row[0][0] if row else '' for row in top]

    def decode_topk(
        self, eeg_emb: np.ndarray, k: int = 5
    ) -> list[list[tuple[str, float]]]:
        """Returns top-``k`` ``(text, cosine)`` neighbours per query.

        Args:
            eeg_emb: Query EEG embeddings ``(N, D)`` or ``(D,)``.
            k: Neighbour count.

        Returns:
            One list of ``(text, similarity)`` pairs per query, best first.
        """
        q = self._project_eeg(eeg_emb)
        if len(self.bank) == 0:
            return [[] for _ in range(q.shape[0])]
        k = min(k, len(self.bank))
        sims = q @ self.bank.T
        part = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
        order = np.argsort(-np.take_along_axis(sims, part, axis=1), axis=1)
        part = np.take_along_axis(part, order, axis=1)
        out: list[list[tuple[str, float]]] = []
        for i in range(q.shape[0]):
            row = [(self.texts[int(j)], float(sims[i, j])) for j in part[i]]
            out.append(row)
        return out

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        texts: list[str] | None = None,
        text_bank_emb: np.ndarray | None = None,
        map_location: str | torch.device = 'cpu',
    ) -> RetrievalDecoder:
        """Restores a retrieval decoder from an alignment / decode checkpoint.

        Args:
            path: Checkpoint path containing ``aligner`` weights and optionally
                ``text_bank_emb`` / ``texts``.
            texts: Optional override for the text bank strings.
            text_bank_emb: Optional override for bank embeddings.
            map_location: ``torch.load`` map location.

        Returns:
            A ready :class:`RetrievalDecoder`.

        Raises:
            ValueError: If the bank cannot be resolved from the checkpoint or args.
        """
        payload = torch.load(path, map_location=map_location, weights_only=False)
        aligner = None
        if 'aligner' in payload or 'config' in payload:
            try:
                aligner = OTCLIPAligner.from_checkpoint(str(path), map_location=map_location)
            except Exception as exc:  # noqa: BLE001
                _LOG.warning('Could not restore aligner from %s (%s)', path, exc)
        bank = text_bank_emb
        bank_texts = texts
        if bank is None and 'text_bank_emb' in payload:
            bank = np.asarray(payload['text_bank_emb'], dtype=np.float32)
        if bank_texts is None and 'texts' in payload:
            bank_texts = list(payload['texts'])
        if bank is None or bank_texts is None:
            raise ValueError(
                'from_checkpoint requires text_bank_emb and texts in the payload or as arguments.'
            )
        return cls(bank, bank_texts, aligner=aligner, bank_already_aligned=True)


class PrefixLanguageDecoder(nn.Module):
    """Maps aligned EEG embeddings to soft prompt prefixes for a language model.

    Architecture (transformers backend):
      ``eeg_emb (B, D) → Linear → (B, prefix_len, lm_hidden)`` prepended to LM
      token embeddings; teacher-forced CE on target token ids.

    For ``backend='toy'``: a tiny character-level GRU decoder that does **not**
    need ``transformers`` — maps EEG → sequence of char logits over a vocab
    built from training texts. Enables full CI testing.
    """

    def __init__(
        self,
        config: GenerativeConfig | None = None,
        texts: list[str] | None = None,
    ) -> None:
        """Builds the prefix mapper and LM backbone.

        Args:
            config: Generative configuration.
            texts: Optional corpus used to build the toy character vocabulary.
        """
        super().__init__()
        self.config = config or GenerativeConfig()
        self.backend = self._resolve_backend(self.config.backend)
        self._device_spec = resolve_device(self.config.device, self.config.precision)
        self.device = self._device_spec.device

        if self.backend == 'toy':
            self._init_toy(texts or [])
        else:
            self._init_transformers()

        self.to(self.device)
        _LOG.info(
            'PrefixLanguageDecoder ready | backend=%s prefix_len=%d',
            self.backend,
            self.config.prefix_len,
        )

    @staticmethod
    def _resolve_backend(backend: GenerativeBackend) -> Literal['transformers', 'toy']:
        if backend == 'toy':
            return 'toy'
        if backend == 'transformers':
            if not _transformers_available():
                raise ImportError(
                    "GenerativeConfig.backend='transformers' but transformers is not installed."
                )
            return 'transformers'
        # auto
        return 'transformers' if _transformers_available() else 'toy'

    def _init_toy(self, texts: list[str]) -> None:
        """Initialises the dependency-free character GRU decoder."""
        chars = sorted({c for t in texts for c in t}) or list('abcdefghijklmnopqrstuvwxyz ')
        # Reserve 0=pad, 1=bos, 2=eos
        self.pad_id, self.bos_id, self.eos_id = 0, 1, 2
        self.char_to_id = {c: i + 3 for i, c in enumerate(chars)}
        self.id_to_char = {i: c for c, i in self.char_to_id.items()}
        vocab_size = len(self.char_to_id) + 3
        self.vocab_size = vocab_size
        hidden = 128
        self.prefix_mapper = nn.Linear(
            self.config.prefix_dim, self.config.prefix_len * hidden
        )
        self.embed = nn.Embedding(vocab_size, hidden, padding_idx=self.pad_id)
        self.rnn = nn.GRU(hidden, hidden, batch_first=True)
        self.lm_head = nn.Linear(hidden, vocab_size)
        self.lm_hidden = hidden
        self.tokenizer = None
        self.lm = None

    def _init_transformers(self) -> None:
        """Initialises a HuggingFace causal LM + prefix mapper."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.lm_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.lm = AutoModelForCausalLM.from_pretrained(self.config.lm_name)
        cfg = self.lm.config
        self.lm_hidden = int(getattr(cfg, 'n_embd', None) or cfg.hidden_size)
        if self.config.freeze_lm:
            for p in self.lm.parameters():
                p.requires_grad = False
            self.lm.eval()
        self.prefix_mapper = nn.Linear(
            self.config.prefix_dim, self.config.prefix_len * self.lm_hidden
        )
        self.vocab_size = int(self.lm.config.vocab_size)
        self.pad_id = int(self.tokenizer.pad_token_id)
        self.bos_id = int(getattr(self.tokenizer, 'bos_token_id', None) or self.pad_id)
        self.eos_id = int(self.tokenizer.eos_token_id or self.pad_id)
        self.embed = None
        self.rnn = None
        self.lm_head = None
        self.char_to_id = {}
        self.id_to_char = {}

    def map_prefix(self, eeg_emb: Tensor) -> Tensor:
        """Maps EEG embeddings to prefix hidden states.

        Args:
            eeg_emb: ``(B, prefix_dim)``.

        Returns:
            ``(B, prefix_len, lm_hidden)``.
        """
        b = eeg_emb.shape[0]
        flat = self.prefix_mapper(eeg_emb)
        return flat.view(b, self.config.prefix_len, self.lm_hidden)

    def forward(
        self,
        eeg_emb: Tensor,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Teacher-forced next-token loss conditioned on an EEG prefix.

        Args:
            eeg_emb: ``(B, prefix_dim)`` EEG (or aligned) embeddings.
            input_ids: Target token ids ``(B, T)`` (toy: char ids; HF: tokenizer ids).
            attention_mask: Optional mask ``(B, T)`` (``1`` = keep).

        Returns:
            ``(loss, logits)`` where logits are ``(B, T, vocab)`` over target positions.
        """
        eeg_emb = eeg_emb.to(self.device)
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        if self.backend == 'toy':
            return self._forward_toy(eeg_emb, input_ids, attention_mask)
        return self._forward_hf(eeg_emb, input_ids, attention_mask)

    def _forward_toy(
        self,
        eeg_emb: Tensor,
        input_ids: Tensor,
        attention_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        """Toy GRU teacher-forcing path."""
        assert self.embed is not None and self.rnn is not None and self.lm_head is not None
        prefix = self.map_prefix(eeg_emb)  # (B, P, H)
        # Initialise RNN state from mean prefix.
        h0 = prefix.mean(dim=1).unsqueeze(0)  # (1, B, H)
        tok_emb = self.embed(input_ids)  # (B, T, H)
        # Prepend a BOS embedding derived from the first prefix slot.
        bos = prefix[:, :1, :]
        inputs = torch.cat([bos, tok_emb[:, :-1, :]], dim=1)
        out, _ = self.rnn(inputs, h0)
        logits = self.lm_head(out)
        if attention_mask is None:
            attention_mask = (input_ids != self.pad_id).to(logits.dtype)
        loss = F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            input_ids.reshape(-1),
            ignore_index=self.pad_id,
        )
        return loss, logits

    def _forward_hf(
        self,
        eeg_emb: Tensor,
        input_ids: Tensor,
        attention_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        """HuggingFace causal-LM path with soft prefixes."""
        assert self.lm is not None
        prefix = self.map_prefix(eeg_emb)  # (B, P, H)
        tok_emb = self.lm.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([prefix, tok_emb], dim=1)
        b, p, _ = prefix.shape
        t = input_ids.shape[1]
        if attention_mask is None:
            attention_mask = torch.ones(b, t, device=self.device, dtype=torch.long)
        prefix_mask = torch.ones(b, p, device=self.device, dtype=attention_mask.dtype)
        full_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        # Labels: ignore prefix positions (``-100``).
        ignore = torch.full((b, p), -100, device=self.device, dtype=input_ids.dtype)
        labels = torch.cat([ignore, input_ids], dim=1)
        outputs = self.lm(
            inputs_embeds=inputs_embeds,
            attention_mask=full_mask,
            labels=labels,
        )
        # Slice logits to the text region for a stable API.
        logits = outputs.logits[:, p : p + t, :]
        return outputs.loss, logits

    @torch.no_grad()
    def generate(self, eeg_emb: Tensor | np.ndarray, max_new_tokens: int | None = None) -> list[str]:
        """Autoregressively generates strings conditioned on EEG embeddings.

        Args:
            eeg_emb: ``(B, D)`` or ``(D,)`` embeddings.
            max_new_tokens: Override for ``config.max_new_tokens``.

        Returns:
            One decoded string per batch row.
        """
        if isinstance(eeg_emb, np.ndarray):
            eeg_emb = torch.from_numpy(np.asarray(eeg_emb, dtype=np.float32))
        if eeg_emb.ndim == 1:
            eeg_emb = eeg_emb.unsqueeze(0)
        eeg_emb = eeg_emb.to(self.device)
        n_new = max_new_tokens if max_new_tokens is not None else self.config.max_new_tokens
        if self.backend == 'toy':
            return self._generate_toy(eeg_emb, n_new)
        return self._generate_hf(eeg_emb, n_new)

    def _generate_toy(self, eeg_emb: Tensor, max_new_tokens: int) -> list[str]:
        """Greedy char-level generation for the toy backend."""
        assert self.embed is not None and self.rnn is not None and self.lm_head is not None
        prefix = self.map_prefix(eeg_emb)
        h = prefix.mean(dim=1).unsqueeze(0)
        b = eeg_emb.shape[0]
        inp = prefix[:, :1, :]
        outs: list[list[int]] = [[] for _ in range(b)]
        for _ in range(max_new_tokens):
            out, h = self.rnn(inp, h)
            logits = self.lm_head(out[:, -1, :]) / max(self.config.temperature, 1e-5)
            # Greedy (top-p sampling is optional; keep CI deterministic).
            nxt = torch.argmax(logits, dim=-1)
            for i, tok in enumerate(nxt.tolist()):
                if tok == self.eos_id:
                    continue
                if tok in self.id_to_char:
                    outs[i].append(tok)
            inp = self.embed(nxt).unsqueeze(1)
        return [''.join(self.id_to_char.get(t, '') for t in seq) for seq in outs]

    def _generate_hf(self, eeg_emb: Tensor, max_new_tokens: int) -> list[str]:
        """Prefix-conditioned generation with a HuggingFace LM."""
        assert self.lm is not None and self.tokenizer is not None
        prefix = self.map_prefix(eeg_emb)
        # Start from EOS/BOS as a single seed token after the soft prefix.
        seed = torch.full(
            (eeg_emb.shape[0], 1),
            self.eos_id,
            device=self.device,
            dtype=torch.long,
        )
        tok_emb = self.lm.get_input_embeddings()(seed)
        inputs_embeds = torch.cat([prefix, tok_emb], dim=1)
        attention_mask = torch.ones(
            inputs_embeds.shape[:2], device=self.device, dtype=torch.long
        )
        gen = self.lm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=self.config.temperature != 1.0 or self.config.top_p < 1.0,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            pad_token_id=self.pad_id,
            eos_token_id=self.eos_id,
        )
        return self.tokenizer.batch_decode(gen, skip_special_tokens=True)

    def encode_texts(self, texts: list[str], max_length: int = 64) -> tuple[Tensor, Tensor]:
        """Tokenises texts for teacher forcing (toy or HF).

        Args:
            texts: Target strings.
            max_length: Truncation / pad length.

        Returns:
            ``(input_ids, attention_mask)`` each ``(B, T)``.
        """
        if self.backend == 'toy':
            return self._encode_toy(texts, max_length)
        assert self.tokenizer is not None
        enc = self.tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='pt',
        )
        return enc['input_ids'], enc['attention_mask']

    def _encode_toy(self, texts: list[str], max_length: int) -> tuple[Tensor, Tensor]:
        """Char-level encode with BOS/EOS."""
        rows: list[list[int]] = []
        for text in texts:
            ids = [self.bos_id] + [self.char_to_id.get(c, self.pad_id) for c in text] + [self.eos_id]
            ids = ids[:max_length]
            ids = ids + [self.pad_id] * (max_length - len(ids))
            rows.append(ids)
        input_ids = torch.tensor(rows, dtype=torch.long)
        attention_mask = (input_ids != self.pad_id).long()
        return input_ids, attention_mask

    def train(self, mode: bool = True) -> PrefixLanguageDecoder:
        """Keeps a frozen HF LM in eval while training the prefix mapper."""
        super().train(mode)
        if self.backend == 'transformers' and self.config.freeze_lm and self.lm is not None:
            self.lm.eval()
        return self


class LanguageDecoder:
    """Facade selecting retrieval and/or prefix-LM decode from :class:`DecoderConfig`."""

    def __init__(
        self,
        config: DecoderConfig | None = None,
        retrieval: RetrievalDecoder | None = None,
        generative: PrefixLanguageDecoder | None = None,
    ) -> None:
        """Wires optional backends.

        Args:
            config: Top-level decode config (defaults to :class:`DecoderConfig`).
            retrieval: Optional retrieval decoder.
            generative: Optional prefix-LM decoder.
        """
        self.config = config or DecoderConfig()
        self.retrieval = retrieval
        self.generative = generative

    def decode_sentences(
        self,
        eeg_emb: np.ndarray | Tensor,
        *,
        k: int = 1,
        max_new_tokens: int | None = None,
    ) -> list[str]:
        """Decodes EEG embeddings to sentence strings.

        Uses retrieval when ``mode`` is ``'retrieval'`` or ``'both'`` (preferring
        retrieval for the returned list). Falls back to the generative path when
        retrieval is unavailable or ``mode='prefix_lm'``.

        Args:
            eeg_emb: Query embeddings ``(N, D)``.
            k: Top-k for retrieval (top-1 string is returned).
            max_new_tokens: Generation length for the prefix-LM path.

        Returns:
            One string per query row.

        Raises:
            RuntimeError: If the configured backend is not attached.
        """
        mode = self.config.mode
        if mode in {'retrieval', 'both'} and self.retrieval is not None:
            arr = eeg_emb.detach().cpu().numpy() if torch.is_tensor(eeg_emb) else eeg_emb
            return self.retrieval.decode(arr, k=k)
        if mode in {'prefix_lm', 'both'} and self.generative is not None:
            return self.generative.generate(eeg_emb, max_new_tokens=max_new_tokens)
        raise RuntimeError(
            f'LanguageDecoder has no backend for mode={mode!r}; '
            'attach retrieval and/or generative decoders.'
        )

    @classmethod
    def from_config(
        cls,
        config: DecoderConfig,
        text_bank_emb: np.ndarray | None = None,
        texts: list[str] | None = None,
        aligner: OTCLIPAligner | None = None,
    ) -> LanguageDecoder:
        """Builds a facade from config and an optional text bank.

        Args:
            config: Decode pipeline config.
            text_bank_emb: Bank embeddings for retrieval.
            texts: Bank strings for retrieval / toy vocab.
            aligner: Optional trained aligner.

        Returns:
            A :class:`LanguageDecoder` with the requested backends attached.
        """
        retrieval = None
        generative = None
        if config.mode in {'retrieval', 'both'} and text_bank_emb is not None and texts is not None:
            retrieval = RetrievalDecoder(text_bank_emb, texts, aligner=aligner)
        if config.mode in {'prefix_lm', 'both'}:
            generative = PrefixLanguageDecoder(config.generative, texts=texts)
        return cls(config, retrieval=retrieval, generative=generative)

    @staticmethod
    def _align_config_fields(raw: dict[str, Any], cls: type) -> dict[str, Any]:
        """Filters a dict to dataclass field names."""
        allowed = {f.name for f in fields(cls)}
        return {k: v for k, v in raw.items() if k in allowed}
