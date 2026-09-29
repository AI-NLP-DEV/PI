#%%
import torch
from transformers import BertForTokenClassification, BertTokenizerFast
from transformers import PreTrainedTokenizerFast

#%%
# 모델과 토크나이저 로드
model_path = "/data/public/hklee/workspace/ITN_master/korean_PI/ixi-PI/PI_Model/checkpoint-65700"  # 체크포인트 경로 수정
model = BertForTokenClassification.from_pretrained(model_path, num_labels=5)
tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)
#%%
def get_punc(model, tokenizer, text):

    print(len(text))
    # 입력 문장 토큰화
    if len(text)>=512:
        text = text[-384:] 
    print("입력 문장 : " +  text)
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
    print(res)

    return res.replace("<s>","").replace("</s>","").replace("<pad>","").strip().split("╏")


#%%

if __name__ == "__main__":
    # sample_dialogue = ['RX:일상에 즐거운 변화를 만드는 LG 유플러스 김도훈입니다 ', 'TX:네 핸드폰을 분실해 가지고 ', 'RX:휴대폰 잃어버리셨어요 안타깝습니다 바로 분실 신고 진행해 드릴게요 분실 한번 호가 5870-977 아 죄송합니 이 2번 분실했기 때문에 이 번호가 아니시겠네요 죄송합니다 번호가 어떻게 되세요 ', 'TX:010 ', 'RX:네 ', 'TX:7184에 ', 'RX:네 ', 'TX:4238이요 ', 'RX:잠시만요 ', 'RX:명의자분 성함하고 생년월일 어떻게 되세요 ', 'TX:잃어버린 핸드폰 주인이요 ', 'RX:네 7184-4238번 말씀해 주셨잖아요 ', 'TX:네 네 장선미구요 ', 'RX:네 ', 'TX:901030이요 ', 'RX:어 혹시 본인이 아니신가요 ', 'TX:저 전 이 명의는 핸드폰 남편 거예요 ', 'RX:아니요 아니요 그니까 지금 전화 주신 고객님이 장선미 고객님 맞으세요 ', 'TX:네네 맞아요 ', 'RX:아 네 네 그 휴대폰 요금 납부하는 방법은 은행지로 카드 중 어떻게 되십니까 ', 'TX:카드인 것 같은데요 ', 'RX:확인되었습니다 장선미 고객님 정보 확인 감사드립니다 휴대폰 발신 정지 처리 진행해 드릴 거구요 휴대폰 습득하신 분 저희 고객 센터 연락 왔을 때 연락드릴 수 있는 번호 하나 여쭤보겠습니다 ', 'TX:네 01058702 ', 'RX:네 ', 'TX:777 하나요 ', 'RX:네 메모 남겨놓겠구요 말씀하셨던 번호로 분실 관련된 정보 제공 및 도움 드릴 수 있는 방법 있으면 연락드릴 거구요 ', 'TX:예 ', 'RX:네 정지 시에는 월 4400원의 정지 기본료가 날짜 계산되어 기존 요금제 대신해서 청구가 됩니다 정지 해제를 원하실 때는 명의자분과 지금처럼 통화가 돼야지만 해지 가능할 수 있는 부분 참고해 주시길 바라겠구요 ', 'TX:예 ', 'RX:그리고 정지 시에는 멤버십 등급은 하락될 수 있는 부분 참고해 주시구요 그리고 휴대폰을 못 찾을 경우에는 분실 파손 전담팀으로 연락 주시면 임대폰이나 기기 변경 관련 상담도 받아볼 수가 있기 때문에요 정상 업무 시간에 상담 받아볼 수 있도록 연락처도 같이 문자 메시지 발송해드릴 겁니다 이렇게 주요 사항 따로 문자 메시지 발송해드릴 거구요 네 정지 모두 완료되었습니다 ', 'TX:그러면은 전화만 뭐 문자는 기다리기만 하면 되는 건가요 ']
    # sample_dialogue = ['RX:일상에 즐거운 변화를 만드는 LG 유플러스 김도훈입니다 ', 'TX:네 핸드폰을 분실해 가지고 ', 'RX:휴대폰 잃어버리셨어요 안타깝습니다 바로 분실 신고 진행해 드릴게요 분실 한번 호가 5870-977 아 죄송합니 이 2번 분실했기 때문에 이 번호가 아니시겠네요 죄송합니다 번호가 어떻게 되세요 ', 'TX:010 ', 'RX:네 ', 'TX:7184에 ', 'RX:네 ', 'TX:4238이요 ', 'RX:잠시만요 ', 'RX:명의자분 성함하고 생년월일 어떻게 되세요 ', 'TX:잃어버린 핸드폰 주인이요 ', 'RX:네 7184-4238번 말씀해 주셨잖아요 ', 'TX:네 네 장선미구요 ', 'RX:네 ', 'TX:901030이요 ', 'RX:어 혹시 본인이 아니신가요 ', 'TX:저 전 이 명의는 핸드폰 남편 거예요 ']
    # sample_dialogue = ["안녕하세요 저는 조정제입니다 반갑습니다 이천이십팔년 사월구일이네요 "]
    # sample_dialogue = ['일상에 즐거운 변화를 만드는 LG 유플러스 김도훈입니다 ']
    sample_dialogue = ["RX: 이번에 서울시 TO가 몇 명 남았지 ", "TX: 제 번호는 01012345678이에요 " , "RX: 2025년 3월 15일까지 해주세요 "]
    print("#####sample_dialogue#####")
    # print(sample_dialogue)

    print("#####PI sample_dialogue#####")
    result = get_punc(model, tokenizer, "╏".join(sample_dialogue))
    print(' '.join(map(str, result)))
    # result = get_punc(model, tokenizer, sample_dialogue)

# %%
