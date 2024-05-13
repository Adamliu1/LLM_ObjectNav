class LanguageModelService:
    def __init__(self, model_name, device):
        self.model_name = model_name
        self.device = device

    def _configure_lm(self, lm):
        raise NotImplementedError
