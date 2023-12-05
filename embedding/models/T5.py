from transformers import CLIPImageProcessor, T5EncoderModel, T5Tokenizer
import torch


class T5Embedder:
    def __init__(self, device):
        self.tokenizer = T5Tokenizer()
        self.encoder = T5EncoderModel()
        self.max_length = 77
        self.device = device
        
        
    def _text_preprocessing(self, text, clean_caption=False):
        if not isinstance(text, (tuple, list)):
            text = [text]

        def process(text: str):
            text = text.lower().strip()
            return text

        return [process(t) for t in text]
    
    def __call__(self, prompt):
        prompt = self._text_preprocessing(prompt)
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(
            prompt, padding="longest", return_tensors="pt"
        ).input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
            text_input_ids, untruncated_ids
        ):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.max_length - 1 : -1])
            # logger.warning(
            #     "The following part of your input was truncated because CLIP can only handle sequences up to"
            #     f" {self.max_length} tokens: {removed_text}"
            # )

        attention_mask = text_inputs.attention_mask.to(self.device)

        prompt_embeds = self.text_encoder(
            text_input_ids.to(self.device),
            attention_mask=attention_mask,
        )
        prompt_embeds = prompt_embeds[0]
        return prompt_embeds