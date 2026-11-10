DEFAULT_NLI_MODEL = 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'

LABEL_TO_STATUS = {
    'entailment': 'supported',
    'contradiction': 'contradicted',
    'neutral': 'insufficient',
}


def load_nli_model(model_name: str = DEFAULT_NLI_MODEL):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).eval()

    labels = {i: label.lower() for i, label in model.config.id2label.items()}
    if set(labels.values()) != set(LABEL_TO_STATUS):
        raise ValueError(f'Unexpected NLI labels: {sorted(labels.values())}')
    statuses = [LABEL_TO_STATUS[labels[i]] for i in range(len(labels))]

    def predict(premise: str, hypothesis: str) -> dict:
        inputs = tokenizer(
            premise, hypothesis, truncation='only_first', max_length=512, return_tensors='pt'
        )
        with torch.no_grad():
            logits = model(**inputs).logits[0]
        probabilities = torch.softmax(logits, dim=-1).tolist()
        return dict(zip(statuses, probabilities))

    return predict