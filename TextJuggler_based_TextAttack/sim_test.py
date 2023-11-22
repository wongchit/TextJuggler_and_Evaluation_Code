from transformers import AutoTokenizer, AutoModelForMaskedLM
model =AutoModelForMaskedLM.from_pretrained('IDEA-CCNL/Erlangshen-SimCSE-110M-Chinese')
tokenizer = AutoTokenizer.from_pretrained('IDEA-CCNL/Erlangshen-SimCSE-110M-Chinese')

import torch
from sklearn.metrics.pairwise import cosine_similarity

texta = '今天天气真不错，我们去散步吧！'
textb = '今天天气真不错，我们出去走走吧！'
inputs_a = tokenizer(texta, return_tensors="pt")
inputs_b = tokenizer(textb, return_tensors="pt")

outputs_a = model(**inputs_a, output_hidden_states=True)
texta_embedding = outputs_a.hidden_states[-1][:, 0, :].squeeze()

outputs_b = model(**inputs_b, output_hidden_states=True)
textb_embedding = outputs_b.hidden_states[-1][:, 0, :].squeeze()

# if you use cuda, the text_embedding should be textb_embedding.cpu().numpy()
# 或者用torch.no_grad():
with torch.no_grad():
    silimarity_soce = cosine_similarity(texta_embedding.reshape(1, -1), textb_embedding .reshape(1, -1))[0][0]
print(silimarity_soce)


