from transformers import Gemma3ForCausalLM, Gemma3TextConfig, AutoTokenizer
# load gemma3 model
local_path = "/fast/nchandak/models/gemma-3-4b-it" 
config = Gemma3TextConfig.from_pretrained(local_path)
tokenizer = AutoTokenizer.from_pretrained(local_path)
config.architectures = ["Gemma3ForCausalLM"]
model = Gemma3ForCausalLM.from_pretrained(local_path, config= config, trust_remote_code=True)
# Now save
save_path = "/fast/nchandak/models/gemma-3-4b-it-text"
model.save_pretrained(save_path)
config.save_pretrained(save_path) 
tokenizer.save_pretrained(save_path)