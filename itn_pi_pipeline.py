"""
ITN(Inverse Text Normalization) + PI(Punctuation Insertion) 통합 인퍼런스 파이프라인.

입력 텍스트 → ITN 추론 → PI 추론 → 최종 문장(들) 반환.
모델 경로는 config.ModelPaths 또는 파이프라인 생성자에서 지정합니다.
"""

from __future__ import annotations

from typing import List, Optional, Union

import torch
from transformers import (
    AutoTokenizer,
    BartForConditionalGeneration,
    BertForTokenClassification,
    PreTrainedTokenizerFast,
)

from config import ModelPaths, DEFAULT_PATHS


# ---------------------------------------------------------------------------
# ITN 인퍼런스
# ---------------------------------------------------------------------------

def _load_itn(model_path: str) -> tuple[BartForConditionalGeneration, AutoTokenizer]:
    model = BartForConditionalGeneration.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if torch.cuda.is_available():
        model = model.cuda()
    return model, tokenizer


def itn_infer(
    model: BartForConditionalGeneration,
    tokenizer: AutoTokenizer,
    text: str,
    *,
    device: Optional[torch.device] = None,
    max_length: int = 150,
    num_beams: int = 1,
) -> str:
    """단일 문장에 대해 ITN 추론을 수행합니다."""
    if device is None:
        device = next(model.parameters()).device
    inputs = tokenizer(text, padding=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        generated_ids = model.generate(
            inputs["input_ids"],
            num_beams=num_beams,
            do_sample=False,
            max_length=max_length,
        )
    return tokenizer.decode(
        generated_ids[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )


# ---------------------------------------------------------------------------
# PI 인퍼런스
# ---------------------------------------------------------------------------

PI_LABELS = ["NOPE", ".", ",", "!", "?"]


def _load_pi(model_path: str) -> tuple[BertForTokenClassification, PreTrainedTokenizerFast]:
    model = BertForTokenClassification.from_pretrained(model_path, num_labels=5)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)
    if torch.cuda.is_available():
        model = model.cuda()
    return model, tokenizer


def _split_into_chunks_by_tokens(
    text: str,
    tokenizer: PreTrainedTokenizerFast,
    chunk_size_tokens: int = 500,
    overlap_tokens: int = 0,
    max_length: int = 512,
) -> List[str]:
    """
    텍스트를 토큰 수 기준으로 청크 단위로 자릅니다.
    max_length(512)를 넘을 때만 분할하며, 청크당 chunk_size_tokens개 이하로 잘라
    [CLS]/[SEP] 등을 고려해 max_length 이내가 되도록 합니다.
    """
    encoded = tokenizer.encode(text, add_special_tokens=True)
    if len(encoded) <= max_length:
        return [text] if text else []
    # [CLS], [SEP] 제외한 본문 토큰만 분할
    content_ids = encoded[1:-1]
    chunks: List[str] = []
    start = 0
    while start < len(content_ids):
        end = min(start + chunk_size_tokens, len(content_ids))
        chunk_ids = content_ids[start:end]
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=False)
        if chunk_text.strip():
            chunks.append(chunk_text.strip())
        if end >= len(content_ids):
            break
        start = end - overlap_tokens if overlap_tokens else end
    return chunks


def _pi_infer_single_chunk(
    model: BertForTokenClassification,
    tokenizer: PreTrainedTokenizerFast,
    text: str,
    *,
    device: torch.device,
    max_length: int = 512,
) -> List[str]:
    """길이 제한 이내의 단일 텍스트에 대해 PI 추론만 수행합니다."""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    seq_len = inputs["attention_mask"].shape[1]
    inputs["token_type_ids"] = torch.zeros(
        (1, seq_len), dtype=torch.long, device=inputs["input_ids"].device
    )
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=2)
    predicted_labels = predictions[0].cpu().numpy().tolist()
    predicted_labels = [PI_LABELS[idx] for idx in predicted_labels]
    token_ids = inputs["input_ids"][0].cpu().numpy().tolist()
    res: List[int] = []
    for tid, label in zip(token_ids, predicted_labels):
        if tid == 3646:
            if label != "NOPE":
                res.append(tokenizer.convert_tokens_to_ids(label))
            res.append(3646)
        else:
            res.append(tid)
    decoded = tokenizer.decode(res)
    decoded = (
        decoded.replace("<s>", "")
        .replace("</s>", "")
        .replace("<pad>", "")
        .strip()
    )
    return decoded.split("╏")


def pi_infer(
    model: BertForTokenClassification,
    tokenizer: PreTrainedTokenizerFast,
    text: str,
    *,
    device: Optional[torch.device] = None,
    max_length: int = 512,
    chunk_size: int = 500,
    chunk_overlap: int = 0,
) -> List[str]:
    """
    텍스트에 대해 PI(구두점 삽입) 추론을 수행합니다.
    입력에 '╏'가 있으면 세그먼트로 나누어 처리하고, 리스트로 반환합니다.
    토큰 수가 max_length(기본 512)를 넘으면 chunk_size(기본 500 토큰) 단위로 잘라
    각각 추론한 뒤 결과를 이어 붙여 하나의 리스트로 반환합니다.
    """
    if device is None:
        device = next(model.parameters()).device
    encoded = tokenizer.encode(text, add_special_tokens=True)
    if len(encoded) <= max_length:
        return _pi_infer_single_chunk(
            model, tokenizer, text, device=device, max_length=max_length
        )
    chunks = _split_into_chunks_by_tokens(
        text,
        tokenizer,
        chunk_size_tokens=chunk_size,
        overlap_tokens=chunk_overlap,
        max_length=max_length,
    )
    merged: List[str] = []
    for chunk in chunks:
        segs = _pi_infer_single_chunk(
            model, tokenizer, chunk, device=device, max_length=max_length
        )
        merged.extend(segs)
    return merged


# ---------------------------------------------------------------------------
# 통합 파이프라인
# ---------------------------------------------------------------------------


class ITNPIPipeline:
    """ITN → PI 순서로 실행하는 통합 파이프라인."""

    def __init__(
        self,
        paths: Optional[ModelPaths] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        self.paths = paths or DEFAULT_PATHS
        self._device = torch.device(device) if isinstance(device, str) else device
        self._itn_model: Optional[BartForConditionalGeneration] = None
        self._itn_tokenizer: Optional[AutoTokenizer] = None
        self._pi_model: Optional[BertForTokenClassification] = None
        self._pi_tokenizer: Optional[PreTrainedTokenizerFast] = None

    def _ensure_itn_loaded(self) -> None:
        if self._itn_model is not None:
            return
        if self.paths.itn is None:
            raise ValueError(
                "ITN 모델 경로가 설정되지 않았습니다. config.ModelPaths(itn='...') 로 설정하세요."
            )
        self._itn_model, self._itn_tokenizer = _load_itn(self.paths.itn)
        if self._device is not None:
            self._itn_model = self._itn_model.to(self._device)

    def _ensure_pi_loaded(self) -> None:
        if self._pi_model is not None:
            return
        if self.paths.pi is None:
            raise ValueError(
                "PI 모델 경로가 설정되지 않았습니다. config.ModelPaths(pi='...') 로 설정하세요."
            )
        self._pi_model, self._pi_tokenizer = _load_pi(self.paths.pi)
        if self._device is not None:
            self._pi_model = self._pi_model.to(self._device)

    def run(self, text: str) -> str:
        """
        단일 문장에 대해 ITN → PI 파이프라인을 실행합니다.
        반환: 구두점이 삽입된 최종 문장 하나.
        """
        self._ensure_itn_loaded()
        self._ensure_pi_loaded()
        assert self._itn_model is not None and self._itn_tokenizer is not None
        assert self._pi_model is not None and self._pi_tokenizer is not None
        device = self._device or next(self._itn_model.parameters()).device
        text = text +" < es >"
        itn_out = itn_infer(self._itn_model, self._itn_tokenizer, text, device=device)
        itn_out = itn_out.replace("< es >", "").strip()
        segments = pi_infer(self._pi_model, self._pi_tokenizer, itn_out, device=device)
        return " ".join(s.strip() for s in segments if s.strip())

    def run_dialogue(self, utterances: List[str]) -> List[str]:
        """
        여러 발화에 대해 ITN → PI 파이프라인을 실행합니다.
        각 발화를 ITN한 뒤 '╏'로 이어 붙여 PI를 수행하고, 구두점이 붙은 발화 리스트를 반환합니다.
        """
        self._ensure_itn_loaded()
        self._ensure_pi_loaded()
        assert self._itn_model is not None and self._itn_tokenizer is not None
        assert self._pi_model is not None and self._pi_tokenizer is not None
        device = self._device or next(self._itn_model.parameters()).device
        itn_results = [
            itn_infer(self._itn_model, self._itn_tokenizer, u, device=device)
            for u in utterances
        ]
        combined = "╏".join(itn_results)
        return pi_infer(self._pi_model, self._pi_tokenizer, combined, device=device)


def create_pipeline(
    itn_path: Optional[str] = None,
    pi_path: Optional[str] = None,
    device: Optional[str] = None,
) -> ITNPIPipeline:
    """모델 경로를 직접 지정해 파이프라인을 생성합니다."""
    paths = ModelPaths(itn=itn_path, pi=pi_path)
    return ITNPIPipeline(paths=paths, device=device)


# ---------------------------------------------------------------------------
# CLI / 스크립트 진입점
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    
    import sys

    if len(sys.argv) < 2:
        print("사용법: python -m pipeline '입력 문장'")
        print("모델 경로는 config.ModelPaths 또는 create_pipeline(itn_path=..., pi_path=...) 로 설정하세요.")
        sys.exit(0)
    text = sys.argv[1]
    try:
        
        
        # 추후 모델 경로 설정 후 실행 예시:
        paths = ModelPaths(
            itn="/data/public/hklee/workspace/ITN_master/korean_ITN/ITN_Model/v3_checkpoints/checkpoint-base",
            pi="/data/public/hklee/workspace/ITN_master/korean_PI/ixi-PI/PI_Model/checkpoint-65700",
        )

        pipeline = ITNPIPipeline(paths=paths)
        print(pipeline.run(text))
    except ValueError as e:
        print(f"오류: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"오류: {e}", file=sys.stderr)
        sys.exit(1)