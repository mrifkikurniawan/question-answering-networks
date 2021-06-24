
class Question(object):
    def __init__(self, text: str):
        self._text = text
        self._len = len(self._text)
        
    @property
    def len(self):
        return self._len
    
    @property
    def data(self):
        return self._text

    def tokenize(self, tokenizer: callable, postprocess: callable=None):
        text = self._text.copy()
        if postprocess:
            text = postprocess(text)
        self.token = tokenizer(text)



class Answer(object):
    def __init__(self, data: dict):
        self._text = data['text']
        self._start = data['answer_start']
        self._len = len(self._text)
        self._end = self._start + self._len
        
    @property
    def len(self):
        return self._len
    
    @property
    def data(self):
        return self._text
    
    @property
    def start(self):
        return self._start

    @property
    def end(self):
        return self._end

    def tokenize(self, tokenizer: callable, postprocess: callable=None):
        text = self._text.copy()
        if postprocess:
            text = postprocess(text)
        self.token = tokenizer(text)



class Context(object):
    def __init__(self, text: str):
        self._text = text
        self._len = len(self._text)
        
    @property
    def len(self):
        return self._len
    
    @property
    def data(self):
        return self._text
    
    def tokenize(self, tokenizer: callable, postprocess: callable=None):
        text = self._text.copy()
        if postprocess:
            text = postprocess(text)
        self.token = tokenizer(text)