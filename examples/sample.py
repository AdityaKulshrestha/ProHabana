import os 
import time
import torch
from transformers import AutoModelForSeq2SeqLM, BitsAndBytesConfig
from IndicTransTokenizer import IndicProcessor, IndicTransTokenizer
from transformers import AutoModelForSeq2SeqLM
from prohabana.main import log_details

os.environ['PT_HPU_ENABLE_GENERIC_STREAM'] = '1'
os.environ['PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES'] = '0'

import habana_frameworks.torch as ht
import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.hpu.graphs as htgraphs

class IndicTranslator:
    def __init__(self, direction, checkpoint_dir):
        self.direction = direction
        self.checkpoint_dir = checkpoint_dir
        self.tokenizer = IndicTransTokenizer(direction=self.direction)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.checkpoint_dir, trust_remote_code=True, output_attentions=True)
        self.model = ht.hpu.wrap_in_hpu_graph(self.model)
        self.ip = IndicProcessor(inference=True)
        self.BATCH_SIZE = 1
        self.DEVICE = torch.device('hpu')
        self.model.to(self.DEVICE)
        
    def pre_print(self, print_str: str):
        print("=================================================")
        print(print_str)
        print("=================================================")

    def preprocess_input(self, sentences, src_lang, tgt_lang):
        #self.pre_print("Pre-proxessing input")
        preprocessed = self.ip.preprocess_batch(sentences, src_lang=src_lang, tgt_lang=tgt_lang)
        inputs = self.tokenizer(preprocessed, src=True, truncation=True, padding="max_length", max_length=128, return_tensors="pt", return_attention_mask=True).to(self.DEVICE)
        return inputs

    def translate(self, inputs, tgt_lang):
        #self.pre_print("Starting translation")
        with torch.no_grad(), torch.autocast(device_type="hpu", dtype=torch.bfloat16, enabled=True):
                generated_tokens = self.model.generate(**inputs, use_cache=True, min_length=0, max_length=128, num_beams=5, num_return_sequences=1)
        decoded_tokens = self.tokenizer.batch_decode(generated_tokens.detach().cpu().tolist(), src=False)
        #self.pre_print("Post-processing")
        postprocessed = self.ip.postprocess_batch(decoded_tokens, lang=tgt_lang)
        return postprocessed

    @log_details
    def batch_translate(self, input_sentences, src_lang, tgt_lang):
        translations = []
        for i in range(0, len(input_sentences), self.BATCH_SIZE):
            batch = input_sentences[i:i+self.BATCH_SIZE]
            inputs = self.preprocess_input(batch, src_lang, tgt_lang)
            translated_batch = self.translate(inputs, tgt_lang)
            translations.extend(translated_batch)
            del inputs
        
        return translations

    def single_translate(self, sentence, src_lang, tgt_lang):
        translation = self.batch_translate([sentence], src_lang, tgt_lang)
        #self.pre_print(f"Translation: {translation}")
        return translation

# Example usage
if __name__ == "__main__":
    translator = IndicTranslator("indic-indic", "ai4bharat/indictrans2-indic-indic-1B")

    hi_sents = [
    "जब मैं छोटा था, तो मैं हर दिन पार्क में जाता था।",
    "उसके पास कई पुरानी किताबें हैं, जो उसने अपने पूर्वजों से विरासत में पाई हैं।",
    "मैं समझ नहीं पा रहा हूँ कि अपनी समस्या कैसे हल करूँ।",
    "वह बहुत मेहनती और बुद्धिमान है, इसी कारण उसे सभी अच्छे अंक मिले हैं।",
    "हमने पिछले हफ्ते एक नई फिल्म देखी, जो बहुत प्रेरणादायक थी।",
    "अगर तुम उस समय मुझसे मिले होते, तो हम बाहर खाने चले जाते।",
    "वह अपनी बहन के साथ एक नई साड़ी खरीदने बाजार गई।",
    "राज ने मुझे बताया कि वह अगले महीने अपनी दादी के घर जा रहा है।",
    "सभी बच्चे पार्टी में मज़े कर रहे थे और बहुत सारी मिठाइयाँ खा रहे थे।",
    "मेरे दोस्त ने मुझे अपने जन्मदिन की पार्टी में आमंत्रित किया है, और मैं उसे एक उपहार दूँगा।"
]
    src_lang, tgt_lang = "hin_Deva", "ben_Beng"

    # Warmup 
    st = time.time()
    hi_translations = translator.batch_translate(hi_sents[0], src_lang, tgt_lang)
    print("Time taken for warmup: ", time.time() - st)

    st = time.time()
    hi_translations = translator.batch_translate(hi_sents[1:], src_lang, tgt_lang)
    print("Time taken for translation: ", time.time() - st)


    print(f"\n{src_lang} - {tgt_lang}")
    for input_sentence, translation in zip(hi_sents, hi_translations):
        print(f"{src_lang}: {input_sentence}")
        print(f"{tgt_lang}: {translation}")
        
    
    sample_sent = "मैं आज कार्यालय जा रहा हूं और मुझे विश्वास है कि यह एक अच्छा दिन होगा।"
    
    st = time.time()
    hi_translations = translator.single_translate(sample_sent, src_lang, tgt_lang)
    print(hi_translations)
    print("Time taken for translation: ", time.time() - st)
        
# Time taken for warmup:  234.33291220664978
# Time taken for warmup:  1.790390968322754
