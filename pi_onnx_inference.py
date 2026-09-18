#%%
"""
PI 양자화 ONNX 모델 인퍼런스.

pi_quantization.py로 생성한 {ckpt}-onnx-quantized 디렉터리를 사용합니다.
경로 설정 후 실행하세요.
"""
from typing import List

from transformers import PreTrainedTokenizerFast
from optimum.onnxruntime import ORTModelForTokenClassification

# ---------------------------------------------------------------------------
# 설정: PI 양자화 모델 경로 (pi_quantization 실행 후 생성된 디렉터리)
# ---------------------------------------------------------------------------
QUANTIZED_DIR = "/data/public/hklee/workspace/ITN_master/korean_PI/ixi-PI/PI_Model/checkpoint-65700-onnx-quantized"  # 예: "/path/to/pi/checkpoint-onnx-quantized"

if not QUANTIZED_DIR:
    raise ValueError(
        "PI 양자화 모델 경로를 설정하세요: QUANTIZED_DIR = '/path/to/...-onnx-quantized'"
    )

# 양자화된 ONNX 모델(model_quantized.onnx)·토크나이저 로드
model = ORTModelForTokenClassification.from_pretrained(
    QUANTIZED_DIR, file_name="model_quantized.onnx"
)
tokenizer = PreTrainedTokenizerFast.from_pretrained(QUANTIZED_DIR)

PI_LABELS = ["NOPE", ".", ",", "!", "?"]


#%%
def get_punc(text: str, max_length: int = 512, truncate_tail_chars: int = 384) -> List[str]:
    """
    텍스트에 구두점을 삽입한 뒤 '╏' 기준으로 분할된 문장 리스트를 반환합니다.
    pi_inference.get_punc와 동일한 후처리 규칙을 사용합니다.
    """
    if len(text) >= max_length:
        text = text[-truncate_tail_chars:]
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    seq_len = inputs["attention_mask"].shape[1]
    inputs["token_type_ids"] = inputs["attention_mask"].new_zeros(1, seq_len).long()

    outputs = model(**inputs)
    predictions = outputs.logits.argmax(dim=2)
    predicted_labels = [PI_LABELS[i] for i in predictions[0].tolist()]
    token_ids = inputs["input_ids"][0].tolist()

    res: List[int] = []
    for tid, label in zip(token_ids, predicted_labels):
        if tid == 3646:
            if label != "NOPE":
                res.append(tokenizer.convert_tokens_to_ids(label))
            res.append(3646)
        else:
            res.append(tid)

    decoded = tokenizer.decode(res)
    print("tokenizer decode result : " + decoded)
    decoded = (
        decoded.replace("<s>", "")
        .replace("</s>", "")
        .replace("<pad>", "")
        .strip()
    )
    return decoded.split("╏")

#%%
if __name__ == "__main__":
    # sample = ["RX: 이번에 서울시 TO가 몇 명 남았지 ", "TX: 제 번호는 01012345678이에요 ", "RX: 2025년 3월 15일까지 해주세요 "]
    sample = ["TX: 여보세요 안녕하세요 비주얼살롱 성수점입니다 ", "RX: 예약 좀 하려고 전화드렸는데요 ", "TX: 네 어떤 시술을 원하실까요 "]
    # sample = ["TX: 여보세유 나유 지금 출장 중인디 통화 가능하겄어유 ", "RX: 어 당연허지유 출장 갔다는 거 까먹고 있었구먼유 어디유 지금 ", "TX: 부산이유 여기서 협력사 미팅 있는디 곧 끝날 것 같어유 근디 팝업 행사 때문에 전화했어유 팝업 준비허는 거 잘 되고 있어유 "]
    print("PI 입력: ", sample)
    result = get_punc("╏".join(sample))
    print("PI 출력:", " ".join(result))

# %%
