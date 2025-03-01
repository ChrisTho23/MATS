import torch as t
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

logger = logging.getLogger(__name__)

class DeepSeekModel:
    def __init__(self, model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"):
        """
        Initialize the DeepSeek model and tokenizer
        
        Args:
            model_name: HuggingFace model identifier
        """
        logger.info(f"Loading model and tokenizer from {model_name}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=True
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=t.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        logger.info("Model loaded successfully")
    
    def generate(self, prompt, temperature=0.6, top_p=0.95, do_sample=True, max_length=2048):
        """
        Generate text from the model given a prompt
        
        Args:
            prompt: Input text prompt
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            do_sample: Whether to use sampling
            
        Returns:
            The raw generated text
        """
        # Encode the prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # Generate text
        with t.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample
            )
        
        # Decode the generated text
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Return the raw generated text without the prompt
        return generated_text[len(prompt):]
