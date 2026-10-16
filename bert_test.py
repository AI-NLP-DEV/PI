# %%
import torch
from transformers import BertForTokenClassification, BertTokenizerFast, PreTrainedTokenizerFast

# 모델과 토크나이저 로드
model_path = "./model_checkpoints/checkpoint-65700/"  # 체크포인트 경로 수정
model = BertForTokenClassification.from_pretrained(model_path, num_labels=5)
# Load the BART tokenizer
tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)

# %%
from datasets import load_from_disk

# 디스크에서 데이터셋 로드
tokenized_datasets = load_from_disk('tokenized_datasets')

# 로드한 데이터셋 확인
print(tokenized_datasets)

# %%
sentences = tokenized_datasets["test"]["sentence"]
input_ids = tokenized_datasets["test"]["input_ids"]
attention_mask = tokenized_datasets["test"]["attention_mask"]
token_type_ids = tokenized_datasets["test"]["token_type_ids"]

# 단건

# %%
text = ['TX:네 네 네 네 ','RX:웅 ','TX:이 가입 인증을 위해 화면에 보이는 인증번호 2자리를 입력해주세요 ','RX:인증번호가 입력되지 않았습니다 ','TX:처음부터 다시 입력하여 주십시오 ','RX:어 이게 이런 문제가 있네 ','TX:입력되지 않았습니다 ']
text = "╏".join(text)

# %%
def get_punc(text):
    # 입력 문장 토큰화
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs["token_type_ids"] = torch.Tensor([0]*len(inputs["attention_mask"][0])).long()
    # 예측 수행
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=2)
    
    # 예측 결과를 정리
    predicted_labels = predictions[0].numpy().tolist()
    label_list = ["NOPE", ".", ",", "!", "?"]  # 원래 레이블 리스트
    
    # 레이블을 실제 값으로 변환
    predicted_labels = [label_list[label] for label in predicted_labels if label != -100]
    
    # 결과 출력
    #print("입력 문장:", text)
    #print("예측된 레이블:", predicted_labels)
    
    res = []
    for i in zip(list(inputs["input_ids"][0].numpy()), predicted_labels):
        if i[0] ==3646:
            if i[1] == "NOPE":
                pass
            else:
                res.append(tokenizer.convert_tokens_to_ids(i[1]))
            res.append(3646)
        else:
            res.append(i[0])        
    
    #print("출력 문장:", tokenizer.decode(res))
    return tokenizer.decode(res)

# %% [markdown]
# ## 여러건

# %%
from tqdm import tqdm
results = []
for k in tqdm(zip(input_ids, attention_mask, token_type_ids)):
    inputs = {"input_ids":torch.Tensor([k[0]]).long(),"attention_mask":torch.Tensor([k[1]]).long(),"token_type_ids":torch.Tensor([k[2]]).long()}
    # 예측 수행
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=2)
    
    # 예측 결과를 정리
    predicted_labels = predictions[0].numpy().tolist()
    label_list = ["NOPE", ".", ",", "!", "?"]  # 원래 레이블 리스트
    
    # 레이블을 실제 값으로 변환
    predicted_labels = [label_list[label] for label in predicted_labels if label != -100]
    
    # 결과 출력
    #print("입력 문장:", text)
    #print("예측된 레이블:", predicted_labels)
    
    res = []
    for i in zip(list(inputs["input_ids"][0].numpy()), predicted_labels):
        if i[0] ==3646:
            if i[1] == "NOPE":
                pass
            else:
                res.append(tokenizer.convert_tokens_to_ids(i[1]))
            res.append(3646)
        else:
            res.append(i[0])        
    
    #print("출력 문장:", tokenizer.decode(res))
    results.append((tokenizer.decode(k[0]),tokenizer.decode(res)))

# %%


# %%
for pair results:
    inputs = pair[0]
    outputs = pair[1]
    "\n".join(inputs.replace("<s>","").replace("</s>","").replace("<pad>","").strip().split("╏"))
    "\n".join(outputs.replace("<s>","").replace("</s>","").replace("<pad>","").strip().split("╏"))

# %%
print("#########STT+ITN 처리결과###########")
for i in results[500][0].replace("<s>","").replace("</s>","").replace("<pad>","").strip().split("╏"):
    print(i)
print()
print("#####STT+ITN+문장부호 처리결과######")
for i in results[500][1].replace("<s>","").replace("</s>","").replace("<pad>","").strip().split("╏"):
    print(i)

# %%
def get_punc(text):
    # 입력 문장 토큰화
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs["token_type_ids"] = torch.Tensor([0]*len(inputs["attention_mask"][0])).long()
    # 예측 수행
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=2)
    
    # 예측 결과를 정리
    predicted_labels = predictions[0].numpy().tolist()
    label_list = ["NOPE", ".", ",", "!", "?"]  # 원래 레이블 리스트
    
    # 레이블을 실제 값으로 변환
    predicted_labels = [label_list[label] for label in predicted_labels if label != -100]
    
    # 결과 출력
    #print("입력 문장:", text)
    #print("예측된 레이블:", predicted_labels)
    
    res = []
    for i in zip(list(inputs["input_ids"][0].numpy()), predicted_labels):
        if i[0] ==3646:
            if i[1] == "NOPE":
                pass
            else:
                res.append(tokenizer.convert_tokens_to_ids(i[1]))
            res.append(3646)
        else:
            res.append(i[0])        
    
    #print("출력 문장:", tokenizer.decode(res))
    res = tokenizer.decode(res)
    return res.replace("<s>","").replace("</s>","").replace("<pad>","").strip().split("╏")[-1]

# %%
import random
random.shuffle(results)

# %%
def mk_input_format(concat_sent):
    sent_list = concat_sent.replace("<s>","").replace("</s>","").replace("<pad>","").strip().split("╏")
    tmp = []
    tmp2 = []
    for i in sent_list:
        tmp2.append(i)
        tmp.append(tmp2.copy())
    return tmp

# %%
from tqdm import tqdm
inputs = []
for i in tqdm(range(len(results))):
    tmp = []
    if i==600:
        break
    tmp2 = []
    inputs.append(mk_input_format(results[i][0]))

# %%
outputs = []
for i in tqdm(inputs):
    tmp2 = []
    for j in i:
        tmp2.append(get_punc("╏".join(j)))
    outputs.append(tmp2)

# %%
import pandas as pd

# 딕셔너리 데이터
data = {
    'PI 처리 전': ["\n".join(i[-1]) for i in inputs],
    'PI 처리 후': ["\n".join(i) for i in outputs]
}

# DataFrame으로 변환
df = pd.DataFrame(data)
df.to_csv("PI.csv",encoding="cp949")

# %%



