import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
QA_input =[{'question': 'why is conversion important?',
            'context': 'The option to convert models between FARM and transformers gives freedom to the user and let people easily switch between frameworks'},
            {'question': 'How many programming languages does BLOOM supports?',
             'context' : "BLOOM has 176 billion parameter and can generate text in 46 natural languages and 13 programming languages"},]

model_name = 'deepset/roberta-base-squad2'

model = AutoModelForQuestionAnswering . from_pretrained(model_name)
tokenizer = AutoTokenizer . from_pretrained(model_name)

inputs0 = tokenizer(QA_input[0]['question'], QA_input[0]['context'], return_tensors="pt")
output0 = model(**inputs0)

inputs1 = tokenizer(QA_input[1]['question'], QA_input[1]['context'], return_tensors="pt")
output1 = model(**inputs1)

answer_start_idx = torch.argmax(output0.start_logits)
answer_end_idx = torch.argmax(output0.end_logits)

answer_tokens = inputs0.input_ids[0, answer_start_idx: answer_end_idx + 1]
answer = tokenizer.decode(answer_tokens)
print("ques:{}\nanswer: {}".format(QA_input[0]['question'],answer))

answer_start_idx = torch.argmax(output1.start_logits)
answer_end_idx = torch.argmax(output1.end_logits)

answer_tokens = inputs1.input_ids[0, answer_start_idx: answer_end_idx + 1]
answer = tokenizer.decode(answer_tokens)
print("ques:{}\nanswer: {}".format(QA_input[1]['question'],answer))