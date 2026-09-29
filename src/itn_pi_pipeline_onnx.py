"""
ITN(Inverse Text Normalization) + PI(Punctuation Insertion) ONNX 통합 인퍼런스 파이프라인.

ONNX quantized 모델을 로드하여 ITN → PI 순서로 추론합니다.
- ITN: 문장 단위 (string).
- PI: 버퍼 단위 (여러 발화를 '╏'로 이은 문자열 → list).

발화가 하나씩 들어올 때: process_utterance(utterance) 사용.
  버퍼를 일정 크기(K)만 유지하고, PI는 해당 버퍼로 수행한 뒤 마지막 발화 결과만 반환.
  새 대화 시작 시 reset_buffer() 호출.
"""

from __future__ import annotations

import re
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional

import torch
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from optimum.onnxruntime import ORTModelForSeq2SeqLM, ORTModelForTokenClassification

# decoder_model_merged.onnx 생성 시 사용 (Optimum 신규 버전 대응)
try:
    from optimum.onnx.graph_transformations import merge_decoders as _merge_decoders
except ImportError:
    _merge_decoders = None


def _ensure_decoder_merged(dir_path: Path) -> None:
    """dir_path에 decoder_model.onnx와 decoder_with_past_model.onnx만 있고
    decoder_model_merged.onnx가 없으면 merge하여 생성합니다.
    (Optimum 일부 버전은 decoder_model_merged.onnx를 요구함)
    """
    merged = dir_path / "decoder_model_merged.onnx"
    if merged.exists():
        return
    dec = dir_path / "decoder_model.onnx"
    dec_past = dir_path / "decoder_with_past_model.onnx"
    if not dec.exists() or not dec_past.exists() or _merge_decoders is None:
        return
    try:
        _merge_decoders(
            str(dec),
            str(dec_past),
            save_path=str(merged),
            strict=False,
        )
    except Exception:
        pass


def _load_itn_onnx(model_path: str) -> tuple[ORTModelForSeq2SeqLM, AutoTokenizer]:
    """ONNX quantized ITN 모델과 토크나이저를 로드합니다.
    양자화 디렉터리(encoder_model_quantized.onnx 등)인 경우 표준 파일명으로 복사한 뒤 로드합니다.
    decoder_model_merged.onnx를 요구하는 Optimum 버전 대응으로, 필요 시 decoder 두 파일을 merge합니다.
    """
    path = Path(model_path)
    encoder_std = path / "encoder_model.onnx"
    encoder_quant = path / "encoder_model_quantized.onnx"
    decoder_merged = path / "decoder_model_merged.onnx"
    decoder_std = path / "decoder_model.onnx"
    decoder_past = path / "decoder_with_past_model.onnx"

    def _load_from_dir(dir_path: str):
        return ORTModelForSeq2SeqLM.from_pretrained(dir_path)

    if encoder_std.exists():
        if decoder_merged.exists():
            model = _load_from_dir(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            return model, tokenizer
        if decoder_std.exists() and decoder_past.exists():
            with tempfile.TemporaryDirectory(prefix="itn_onnx_") as tmp_dir:
                tmp = Path(tmp_dir)
                for f in path.iterdir():
                    if f.suffix in (".onnx", ".json") or f.name.startswith("tokenizer"):
                        shutil.copy2(f, tmp / f.name)
                _ensure_decoder_merged(tmp)
                model = _load_from_dir(str(tmp))
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            return model, tokenizer
        model = _load_from_dir(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        return model, tokenizer

    if encoder_quant.exists():
        renames = [
            ("encoder_model_quantized.onnx", "encoder_model.onnx"),
            ("decoder_model_quantized.onnx", "decoder_model.onnx"),
            ("decoder_with_past_model_quantized.onnx", "decoder_with_past_model.onnx"),
        ]
        with tempfile.TemporaryDirectory(prefix="itn_onnx_") as tmp_dir:
            tmp = Path(tmp_dir)
            for src_name, dst_name in renames:
                src = path / src_name
                if src.exists():
                    shutil.copy2(src, tmp / dst_name)
            for f in path.iterdir():
                if f.suffix == ".json" or f.name.startswith("tokenizer"):
                    shutil.copy2(f, tmp / f.name)
            _ensure_decoder_merged(tmp)
            model = _load_from_dir(str(tmp))
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        return model, tokenizer

    raise FileNotFoundError(
        f"ITN ONNX 모델을 찾을 수 없습니다: {model_path} "
        "(encoder_model.onnx 또는 encoder_model_quantized.onnx 필요)"
    )


def itn_infer_onnx(
    model: ORTModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    text: str,
    *,
    max_length: int = 150,
    num_beams: int = 1,
) -> str:
    """단일 문장에 대해 ITN ONNX 추론을 수행합니다."""
    inputs = tokenizer(text, padding=True, return_tensors="pt")

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
# PI ONNX 인퍼런스
# ---------------------------------------------------------------------------

PI_LABELS = ["NOPE", ".", ",", "!", "?"]


def _load_pi_onnx(model_path: str) -> tuple[ORTModelForTokenClassification, PreTrainedTokenizerFast]:
    """ONNX quantized PI 모델과 토크나이저를 로드합니다.
    model_quantized.onnx만 있는 경우 file_name으로 지정해 로드합니다.
    """
    path = Path(model_path)
    if (path / "model_quantized.onnx").exists() and not (path / "model.onnx").exists():
        model = ORTModelForTokenClassification.from_pretrained(
            model_path, file_name="model_quantized.onnx"
        )
    else:
        model = ORTModelForTokenClassification.from_pretrained(model_path)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)
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


def _pi_infer_single_chunk_onnx(
    model: ORTModelForTokenClassification,
    tokenizer: PreTrainedTokenizerFast,
    text: str,
    *,
    max_length: int = 512,
) -> List[str]:
    """길이 제한 이내의 단일 텍스트에 대해 PI ONNX 추론을 수행합니다."""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )

    seq_len = inputs["attention_mask"].shape[1]
    inputs["token_type_ids"] = torch.zeros((1, seq_len), dtype=torch.long)

    with torch.no_grad():
        outputs = model(**inputs)
        # ONNX 모델의 출력은 logits를 포함합니다
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
        predictions = torch.argmax(logits, dim=2)

    predicted_labels = predictions[0].cpu().numpy().tolist()
    predicted_labels = [PI_LABELS[idx] for idx in predicted_labels]

    token_ids = inputs["input_ids"][0].cpu().numpy().tolist()
    res: List[int] = []

    for tid, label in zip(token_ids, predicted_labels):
        if tid == 3646:  # 특수 토큰 ID (╏)
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


def _ensure_trailing_space_per_segment(text: str) -> str:
    """
    PI 모델 입력: ╏로 구분된 각 문장 끝에 공백 하나가 있어야 정상 추론됩니다.
    파인튜닝 시 해당 방식으로 학습되었으므로 추론 시에도 동일하게 적용합니다.
    """
    if not text.strip():
        return text
    return "╏".join((s.rstrip() + " ") if s.strip() else s for s in text.split("╏"))


def pi_infer_onnx(
    model: ORTModelForTokenClassification,
    tokenizer: PreTrainedTokenizerFast,
    text: str,
    *,
    max_length: int = 512,
    chunk_size: int = 500,
    chunk_overlap: int = 0,
) -> List[str]:
    """
    텍스트에 대해 PI(구두점 삽입) ONNX 추론을 수행합니다.
    입력에 '╏'가 있으면 세그먼트로 나누어 처리하고, 리스트로 반환합니다.
    토큰 수가 max_length(기본 512)를 넘으면 chunk_size(기본 500 토큰) 단위로 잘라
    각각 추론한 뒤 결과를 이어 붙여 하나의 리스트로 반환합니다.
    문장 끝 공백은 파인튜닝 시와 동일하게 내부에서 보정합니다.
    """
    text = _ensure_trailing_space_per_segment(text)
    encoded = tokenizer.encode(text, add_special_tokens=True)

    if len(encoded) <= max_length:
        return _pi_infer_single_chunk_onnx(
            model, tokenizer, text, max_length=max_length
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
        segs = _pi_infer_single_chunk_onnx(
            model, tokenizer, chunk, max_length=max_length
        )
        merged.extend(segs)

    return merged


# ---------------------------------------------------------------------------
# 통합 파이프라인
# ---------------------------------------------------------------------------


def _strip_role_prefix(text: str) -> str:
    """RX:, TX: 접두사를 제거하여 ITN 분석 시 역할 기호가 결과에 영향을 주지 않도록 합니다."""
    t = text.strip()
    if t.upper().startswith("RX:"):
        return t[3:].strip()
    if t.upper().startswith("TX:"):
        return t[3:].strip()
    return t


class ITNPIPipelineONNX:
    """
    ONNX quantized 모델을 사용하는 ITN → PI 통합 파이프라인.

    - ITN: 문장 단위 (string → string).
    - PI: 버퍼 단위 (여러 발화를 '╏'로 이은 string → list).
    발화가 하나씩 들어올 때는 process_utterance()를 사용하고, 버퍼는 최근 K개만 유지하며
    마지막 발화에 대한 구두점 결과만 반환합니다.
    """

    def __init__(
        self,
        itn_path: Optional[str] = None,
        pi_path: Optional[str] = None,
        buffer_size: int = 10,
        pi_max_length: int = 512,
    ) -> None:
        """
        Args:
            itn_path: ITN ONNX(-quantized) 모델 디렉터리 경로.
            pi_path: PI ONNX(-quantized) 모델 디렉터리 경로.
            buffer_size: process_utterance() 시 유지할 최대 발화 개수(K).
                         토큰 수 제한 내에서만 유지됩니다.
            pi_max_length: PI 입력 최대 토큰 수. 버퍼를 이어 붙인 길이가 넘으면
                           오래된 발화부터 제거합니다 (기본 512).
        """
        self.itn_path = itn_path
        self.pi_path = pi_path
        self.buffer_size = max(1, buffer_size)
        self.pi_max_length = pi_max_length
        self._itn_model: Optional[ORTModelForSeq2SeqLM] = None
        self._itn_tokenizer: Optional[AutoTokenizer] = None
        self._pi_model: Optional[ORTModelForTokenClassification] = None
        self._pi_tokenizer: Optional[PreTrainedTokenizerFast] = None
        self._buffer: List[str] = []

    def _ensure_itn_loaded(self) -> None:
        """ITN ONNX 모델이 로드되지 않았다면 로드합니다."""
        if self._itn_model is not None:
            return
        if self.itn_path is None:
            raise ValueError(
                "ITN 모델 경로가 설정되지 않았습니다. "
                "ITNPIPipelineONNX(itn_path='...') 로 설정하세요."
            )
        self._itn_model, self._itn_tokenizer = _load_itn_onnx(self.itn_path)

    def _ensure_pi_loaded(self) -> None:
        """PI ONNX 모델이 로드되지 않았다면 로드합니다."""
        if self._pi_model is not None:
            return
        if self.pi_path is None:
            raise ValueError(
                "PI 모델 경로가 설정되지 않았습니다. "
                "ITNPIPipelineONNX(pi_path='...') 로 설정하세요."
            )
        self._pi_model, self._pi_tokenizer = _load_pi_onnx(self.pi_path)

    def run(self, text: str) -> str:
        """
        단일 문장에 대해 ITN → PI 파이프라인을 실행합니다.

        Args:
            text: ITN 입력 텍스트 (예: "이번에 서울시 티오가 몇 명 남았지?")

        Returns:
            구두점이 삽입된 최종 문장 하나.
        """
        self._ensure_itn_loaded()
        self._ensure_pi_loaded()

        assert self._itn_model is not None and self._itn_tokenizer is not None
        assert self._pi_model is not None and self._pi_tokenizer is not None

        # ITN 추론: RX:/TX: 제거 후 < es > 토큰 추가
        text_clean = _strip_role_prefix(text)
        text_with_eos = text_clean + " < es >"
        itn_out = itn_infer_onnx(self._itn_model, self._itn_tokenizer, text_with_eos)

        # < es > 제거
        itn_out = itn_out.replace("< es >", "").strip()

        # PI 추론 (문장 끝 공백은 pi_infer_onnx 내부에서 보정)
        segments = pi_infer_onnx(self._pi_model, self._pi_tokenizer, itn_out)

        return " ".join(s.strip() for s in segments if s.strip())

    def run_dialogue(self, utterances: List[str]) -> List[str]:
        """
        여러 발화에 대해 ITN → PI 파이프라인을 실행합니다.
        각 발화를 ITN한 뒤 '╏'로 이어 붙여 PI를 수행하고, 구두점이 붙은 발화 리스트를 반환합니다.
        반환 리스트 길이는 항상 입력 발화 수와 동일합니다 (PI가 더 많은 세그먼트를 만들면 병합).

        Args:
            utterances: 발화 리스트 (예: ["RX:첫 번째 발화", "TX:두 번째 발화"])

        Returns:
            구두점이 삽입된 발화 리스트 (len(utterances)개).
        """
        self._ensure_itn_loaded()
        self._ensure_pi_loaded()

        assert self._itn_model is not None and self._itn_tokenizer is not None
        assert self._pi_model is not None and self._pi_tokenizer is not None

        # 각 발화에 대해 ITN 수행 (RX:/TX: 제거 후 < es > 종결 토큰 추가하여 분석)
        itn_results = [
            itn_infer_onnx(
                self._itn_model,
                self._itn_tokenizer,
                _strip_role_prefix(u) + " < es >",
            )
            .replace("< es >", "")
            .strip()
            for u in utterances
        ]

        # '╏'로 결합하여 PI 수행
        combined = "╏".join(itn_results)
        segments = pi_infer_onnx(self._pi_model, self._pi_tokenizer, combined)

        n = len(utterances)
        if len(segments) > n:
            # PI가 구두점 등으로 세그먼트를 더 나눈 경우: 초과분을 마지막 발화로 병합
            segments = segments[: n - 1] + [" ".join(s.strip() for s in segments[n - 1 :] if s.strip())]
        elif len(segments) < n:
            # 부족한 경우 빈 문자열로 패딩
            segments = segments + [""] * (n - len(segments))
        return segments

    def run_dialogue_debug(
        self, utterances: List[str]
    ) -> dict:
        """
        run_dialogue와 동일한 처리하되, ITN / PI 단계별 결과를 반환합니다.
        """
        self._ensure_itn_loaded()
        self._ensure_pi_loaded()

        assert self._itn_model is not None and self._itn_tokenizer is not None
        assert self._pi_model is not None and self._pi_tokenizer is not None

        itn_results = [
            itn_infer_onnx(
                self._itn_model,
                self._itn_tokenizer,
                _strip_role_prefix(u) + " < es >",
            )
            .replace("< es >", "")
            .strip()
            for u in utterances
        ]
        combined = "╏".join(itn_results)
        pi_segments_raw = pi_infer_onnx(
            self._pi_model, self._pi_tokenizer, combined
        )

        n = len(utterances)
        segments = list(pi_segments_raw)
        if len(segments) > n:
            segments = segments[: n - 1] + [
                " ".join(s.strip() for s in segments[n - 1 :] if s.strip())
            ]
        elif len(segments) < n:
            segments = segments + [""] * (n - len(segments))
        return {
            "itn_results": itn_results,
            "combined": combined,
            "pi_segments_raw": pi_segments_raw,
            "final": segments,
        }

    def process_utterance(self, utterance: str) -> str:
        """
        발화가 하나씩 들어올 때 호출합니다.
        ITN(문장 단위) → 버퍼에 추가 → 토큰 수·발화 개수 제한으로 trim → PI(버퍼) 수행 후
        마지막 발화에 대한 구두점 결과만 반환합니다.
        """
        self._ensure_itn_loaded()
        self._ensure_pi_loaded()

        assert self._itn_model is not None and self._itn_tokenizer is not None
        assert self._pi_model is not None and self._pi_tokenizer is not None

        # ITN: RX:/TX: 제거 후 문장 단위 (필요 시 < es > 추가)
        text_clean = _strip_role_prefix(utterance)
        text_with_eos = text_clean.strip()
        if not text_with_eos.endswith("< es >"):
            text_with_eos = text_with_eos + " < es >"
        itn_out = itn_infer_onnx(
            self._itn_model, self._itn_tokenizer, text_with_eos
        )
        itn_out = itn_out.replace("< es >", "").strip()

        # 버퍼에 추가 (PI 학습 시 문장 끝 공백 사용 → 저장 시 끝에 공백 하나)
        self._buffer.append((itn_out.rstrip() + " ") if itn_out.strip() else itn_out)

        # 1) 토큰 수 제한: PI max_length를 넘으면 오래된 발화부터 제거
        while len(self._buffer) > 1:
            combined = "╏".join(self._buffer)
            encoded = self._pi_tokenizer.encode(
                combined, add_special_tokens=True
            )
            if len(encoded) <= self.pi_max_length:
                break
            self._buffer = self._buffer[1:]

        # 2) 발화 개수 제한: buffer_size 초과 시 오래된 발화부터 제거
        if len(self._buffer) > self.buffer_size:
            self._buffer = self._buffer[-self.buffer_size :]

        # PI: 버퍼 전체를 '╏'로 이어서 추론
        combined = "╏".join(self._buffer)
        segments = pi_infer_onnx(
            self._pi_model, self._pi_tokenizer, combined
        )

        # 마지막 발화에 대한 결과만 반환
        if not segments:
            return itn_out
        return segments[-1].strip()

    def get_buffer(self) -> List[str]:
        """현재 버퍼 내용을 복사본으로 반환합니다 (디버깅·로그용)."""
        return list(self._buffer)

    def reset_buffer(self) -> None:
        """버퍼를 비웁니다. 새 대화 세션을 시작할 때 호출하세요."""
        self._buffer = []


def create_pipeline_onnx(
    itn_path: Optional[str] = None,
    pi_path: Optional[str] = None,
    buffer_size: int = 10,
    pi_max_length: int = 512,
) -> ITNPIPipelineONNX:
    """
    모델 경로를 직접 지정해 ONNX 파이프라인을 생성합니다.

    Args:
        itn_path: ITN ONNX quantized 모델 경로
        pi_path: PI ONNX quantized 모델 경로
        buffer_size: process_utterance() 시 유지할 최대 발화 개수 (기본 10)
        pi_max_length: PI 입력 최대 토큰 수, 버퍼 trim 기준 (기본 512)

    Returns:
        ITNPIPipelineONNX 인스턴스
    """
    return ITNPIPipelineONNX(
        itn_path=itn_path,
        pi_path=pi_path,
        buffer_size=buffer_size,
        pi_max_length=pi_max_length,
    )


# ---------------------------------------------------------------------------
# CLI / 스크립트 진입점
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="ITN+PI ONNX 파이프라인 단발 실행")
    parser.add_argument("text", help="ITN 입력 문장")
    parser.add_argument("--itn-path", required=True, help="ITN ONNX(-quantized) 모델 디렉터리")
    parser.add_argument("--pi-path", required=True, help="PI ONNX(-quantized) 모델 디렉터리")
    args = parser.parse_args()

    try:
        pipeline = ITNPIPipelineONNX(itn_path=args.itn_path, pi_path=args.pi_path)
        result = pipeline.run(args.text)
        print(result)
    except Exception as e:
        print(f"오류: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
