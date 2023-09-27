import torch
model_path = 'path of the quantized model'
lora_path = 'path of the saved LoRA adapters'
merged_path = 'target path of the merged model'
scale = 16 /64
group_size = 32

model = torch.load(model_path, map_location='cpu')
lora = torch.load(lora_path, map_location='cpu')
tmp_keys = [key[17:-14] for key in lora.keys() if 'lora_A' in key]
for tmp_key in tmp_keys:
    model[tmp_key+'.qzeros'] -= (lora['base_model.model.'+tmp_key+'.lora_B.weight'] @ lora['base_model.model.'+tmp_key+'.lora_A.weight']).t() * scale / group_size /model[tmp_key+'.scales']

torch.save(model, merged_path)

