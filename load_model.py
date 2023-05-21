from transformers import ViTForImageClassification, ViTImageProcessor

# Load the model and tokenizer from the local disk
def ModelLoader(path):
    model = ViTForImageClassification.from_pretrained(path)
    tokenizer = ViTImageProcessor.from_pretrained(path)
    return model, tokenizer
