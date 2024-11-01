import detoxify

class DetoxifyEvaluator:
    def __init__(self):
        self.model = detoxify.Detoxify("original")
    
    def evaluate(self, text):
        return self.model.predict(text)