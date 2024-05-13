from constants import category_to_id
import torch
from transformers import (
    BertTokenizer,
    BertModel,
    BertForMaskedLM,
    RobertaTokenizer,
    RobertaForMaskedLM,
    RobertaModel,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    GPT2Model,
    GPTNeoForCausalLM,
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
    GPTJForCausalLM,
)

from utils.llm_helper import LanguageModelService


class L3MVN_LanguageModelService_ZeroShot(LanguageModelService):
    def __init__(self, model_name, device):
        super().__init__(model_name, device)
        self.scoring_fxn = self._configure_lm(model_name)

    def _configure_lm(self, lm):
        """
        Configure the language model, tokenizer, and embedding generator function.

        Sets self.lm, self.lm_model, self.tokenizer, and self.embedder based on the
        selected language model inputted to this function.

        Args:
            lm: str representing name of LM to use

        Returns:
            None
        """

        if lm == "BERT":
            raise NotImplemented
        elif lm == "BERT-large":
            raise NotImplemented
        elif lm == "RoBERTa":
            raise NotImplemented
        elif lm == "RoBERTa-large":
            tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
            lm_model = RobertaForMaskedLM.from_pretrained("roberta-large")
            start = "<s>"
            end = "</s>"
            mask_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("<mask>"))[0]
        elif lm == "GPT2-large":
            lm_model = GPT2LMHeadModel.from_pretrained("gpt2-large")
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")
        else:
            print("Model option " + lm + " not implemented yet")
            raise NotImplemented

        lm_model.eval()
        lm_model = lm_model.to(self.device)

        def scoring_fxn(text):
            tokens_tensor = tokenizer.encode(
                text, add_special_tokens=False, return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                output = lm_model(tokens_tensor, labels=tokens_tensor)
                # print(output)
                loss = output[0]

                return -loss

        return scoring_fxn

    def construct_dist(self, objs):
        query_str = "A room containing "
        for ob in objs:
            query_str += ob + ", "
        query_str += "and"

        TEMP = []
        for label in category_to_id:
            TEMP_STR = query_str + " "
            TEMP_STR += label + "."

            # print(TEMP_STR)
            score = self.scoring_fxn(TEMP_STR)
            TEMP.append(score)
        dist = torch.tensor(TEMP)

        # print(dist)
        return dist
