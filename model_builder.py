from models.simple_cnn import SimpleCNN

def build_model(cfg):
    model = SimpleCNN(num_classes=cfg["num_classes"])
    return model
