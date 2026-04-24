from flask import Flask, render_template, request, jsonify
import re
from collections import Counter

app = Flask(__name__)

class AutoComplete:
    def __init__(self):
        self.words = Counter()
        self.next_words = {}
    
    def train(self, texts):
        for text in texts:
            words = re.findall(r'[a-z]+', text.lower())
            for i, w in enumerate(words):
                self.words[w] += 1
                if i < len(words)-1:
                    if w not in self.next_words:
                        self.next_words[w] = Counter()
                    self.next_words[w][words[i+1]] += 1
    
    def predict(self, text, k=5):
        words = re.findall(r'[a-z]+', text.lower())
        if words and words[-1] in self.next_words:
            return [w for w, _ in self.next_words[words[-1]].most_common(k)]
        return [w for w, _ in self.words.most_common(k)]

training_data = [
    "hello how are you", "how are you doing", "i am fine", "good morning",
    "what is your name", "my name is ai", "nice to meet you", "thank you",
    "please help me", "can you help", "i need assistance", "of course",
    "i love you", "i love programming", "i love python"
]

model = AutoComplete()
model.train(training_data)

print("Model ready!")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text', '')
    suggestions = model.predict(text, 5)
    return jsonify({'suggestions': suggestions})

if __name__ == '__main__':
    print("Server on http://127.0.0.1:5000")
    app.run(debug=True, port=5000)