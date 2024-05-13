from transformers import (
    BertTokenizer,
    BertModel,
    RobertaTokenizer,
    RobertaModel,
    GPT2Tokenizer,
    GPT2Model,
    AutoTokenizer,
    AutoModel,
)

from model import FeedforwardNet
from utils.llm_helper import LanguageModelService
import torch


class L3MVN_LanguageModelService_FFN(LanguageModelService):
    def __init__(self, model_name, device):
        super().__init__(model_name, device)
        self.model, self.tokenizer, self.start_token, self.end_token = (
            self._configure_lm(model_name)
        )
        self.model.eval().to(self.device)

    def _configure_lm(self, lm):
        if lm.startswith("BERT"):
            tokenizer = BertTokenizer.from_pretrained(
                "bert-base-uncased" if lm == "BERT" else "bert-large-uncased"
            )
            model = BertModel.from_pretrained(
                "bert-base-uncased" if lm == "BERT" else "bert-large-uncased"
            )
            start, end = "[CLS]", "[SEP]"
        elif lm.startswith("RoBERTa"):
            tokenizer = RobertaTokenizer.from_pretrained(
                "roberta-base" if lm == "RoBERTa" else "roberta-large"
            )
            model = RobertaModel.from_pretrained(
                "roberta-base" if lm == "RoBERTa" else "roberta-large"
            )
            start, end = "<s>", "</s>"
        elif lm == "GPT2-large":
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")
            model = GPT2Model.from_pretrained("gpt2-large")
            start, end = (
                None,
                None,
            )  # GPT-2 does not use special start or end tokens
        elif lm == "GPT-Neo":
            tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
            model = AutoModel.from_pretrained("EleutherAI/gpt-neo-1.3B")
            start, end = None, None
        elif lm == "GPT-J":
            tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
            model = AutoModel.from_pretrained(
                "EleutherAI/gpt-j-6B",
                revision="float16",
                torch_dtype=torch.float16,
            )
            start, end = None, None
        else:
            raise ValueError(f"Model {lm} not supported.")

        return model, tokenizer, start, end

    def embed_sentence(self, sentence):
        """
        Embeds a given sentence using the configured language model.

        Args:
            sentence (str): The sentence to embed.

        Returns:
            torch.Tensor: The embedding of the sentence.
        """
        # Optional: Add start and end tokens if defined for the model
        if self.start_token and self.end_token:
            sentence = f"{self.start_token} {sentence} {self.end_token}"

        # Tokenize the sentence and convert to tensor format
        tokens_tensor = self.tokenizer.encode(sentence, return_tensors="pt").to(
            self.device
        )

        # Generate embeddings using the language model
        with torch.no_grad():
            outputs = self.model(tokens_tensor)
            # hidden state is a tuple
            hidden_state = outputs.last_hidden_state

        # Return the embedding of the first token
        # (CLS token for BERT-like models)
        return hidden_state[:, -1]

    '''
    Original Version
            def embedder(query_str):
            query_str = start + " " + query_str + " " + end
            tokenized_text = tokenizer.tokenize(query_str)
            tokens_tensor = torch.tensor(
                [tokenizer.convert_tokens_to_ids(tokenized_text)])
            """ tokens_tensor = torch.tensor([indexed_tokens.to(self.device)])
                """
            tokens_tensor = tokens_tensor.to(
                device)  # if you have gpu

            with torch.no_grad():
                outputs = lm_model(tokens_tensor)
                # hidden state is a tuple
                hidden_state = outputs.last_hidden_state

            # Shape (batch size=1, num_tokens, hidden state size)
            # Return just the start token's embeddinge
            return hidden_state[:, -1]

        return embedder
    '''

    def object_query_constructor(self, objects):
        """
        Construct a query string based on a list of objects

        Args:
            objects: torch.tensor of object indices contained in a room

        Returns:
            str query describing the room, eg "This is a room containing
                toilets and sinks."
        """
        assert len(objects) > 0
        query_str = "This room contains "
        names = []
        for ob in objects:
            names.append(ob)
        if len(names) == 1:
            query_str += names[0]
        elif len(names) == 2:
            query_str += names[0] + " and " + names[1]
        else:
            for name in names[:-1]:
                query_str += name + ", "
            query_str += "and " + names[-1]
        query_str += "."
        return query_str


def setup_language_model_and_ff_net(args, device, category_ids):
    """
    Sets up the language model service and initializes the feedforward network.

    Args:
        args: Configuration arguments including model names and other settings.
        device: The device (CPU or GPU) to run the models on.
        category_ids: A list or array of category IDs for output size determination.

    Returns:
        lm_service: An instance of the LanguageModelService class for embedding sentences.
        ff_net: An initialized FeedforwardNet model.
    """
    # Initialize the language model service with the desired model
    lm_service = L3MVN_LanguageModelService_FFN("RoBERTa-large", device)
    # lm_service = LanguageModelService(args.language_model, device)

    # Initialize the feedforward network based on the output size needed
    output_size = len(category_ids)
    ff_net = FeedforwardNet(1024, output_size).to(device)

    # Load pre-trained weights into the feedforward network if specified
    if args.load != "0":
        print("Loading LLM model {}".format(args.load))
        """NOTE: version original
        state_dict = torch.load(args.load,
                                map_location=lambda storage, loc: storage)
        """
        ff_net_state_dict = torch.load(args.load, map_location=device)
        ff_net.load_state_dict(ff_net_state_dict)

    ff_net.eval()  # Set the network to evaluation mode

    return lm_service, ff_net
