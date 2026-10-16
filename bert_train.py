import numpy as np
import pickle
import random
import torch
from torch.utils.data import DataLoader

import torch
import datasets 
from datasets import load_dataset, load_metric, Dataset, DatasetDict
from transformers import BertForTokenClassification, BertTokenizerFast, Trainer, TrainingArguments, BertConfig
from transformers import AutoModel,  AutoTokenizer, BertModel, BertConfig, BertForSequenceClassification, Trainer, TrainingArguments

from transformers import PreTrainedTokenizerFast

# Load the BART tokenizer
tokenizer = PreTrainedTokenizerFast.from_pretrained('checkpoint_tokenizer')

from datasets import load_from_disk

# 디스크에서 데이터셋 로드
tokenized_datasets = load_from_disk('tokenized_datasets')

# 로드한 데이터셋 확인
print(tokenized_datasets)

# 모델 로드
from transformers import DataCollatorForTokenClassification

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

# +
config = BertConfig(
    vocab_size=tokenizer.vocab_size,  # 어휘 사전 크기
    hidden_size=512,   # Transformer 모델의 hidden state 크기
    num_hidden_layers=2,  # Transformer 모델의 층 개수
    num_attention_heads=2,  # 어텐션 헤드 개수
    intermediate_size=512,  # Transformer 모델 내부의 히든 레이어 크기
    max_position_embeddings=512,  # 최대 위치 임베딩 개수
    type_vocab_size=1,   # 타입 어휘 사전 크기 (e.g., segment A/B)
    initializer_range=0.02,  # 가중치 초기화 범위
    layer_norm_eps=1e-12,   # 레이어 정규화 epsilon 값
    pad_token_id=tokenizer.pad_token_id,   # 패딩 토큰 ID
    gradient_checkpointing=False,  # 그래디언트 체크포인팅 사용 여부
    position_embedding_type="absolute",  # 위치 임베딩 타입 (absolute 또는 relative)
    use_cache=True,   # 캐시 사용 여부
    num_labels=5
)
#model = BertForTokenClassification(config)
model =  BertForTokenClassification.from_pretrained("model_checkpoints/checkpoint-98600-100epc/", num_labels=5)

# TrainingArguments 설정
training_args = TrainingArguments(
    output_dir="./model_checkpoints",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=384,
    per_device_eval_batch_size=128,
    num_train_epochs=100,
    weight_decay=0.01,
)
# -

metric = load_metric("seqeval")

def compute_metrics(p):
    label_list = ["▁",".",",","!","?"]
    
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(pred, label) if l != -100]
        for pred, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for l in label if l != -100]
        for label in labels
    ]
    
    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


# 클래스 가중치 설정
class_weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])  # Tensor로 변환

# 손실 함수 정의 (CrossEntropyLoss 사용 예시)
class WeightedLoss(torch.nn.CrossEntropyLoss):
    def __init__(self, weight=None, **kwargs):
        super().__init__(weight=weight, **kwargs)

# Trainer에서 사용할 손실 함수 정의
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # 손실 계산 시 클래스 가중치 적용
        loss_fct = WeightedLoss(weight=class_weights.to(logits.device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss

# CustomTrainer 사용
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# 모델 학습
trainer.train()

# 평가
trainer.evaluate()
