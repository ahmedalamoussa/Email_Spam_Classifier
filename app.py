from flask import Flask, request, render_template, jsonify
import pickle

# Charger le modèle et le vecteur
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Créer l'application Flask
app = Flask(__name__)

# Page principale (facultative)
@app.route('/')
def home():
    return render_template('index.html')  # si tu as un formulaire HTML

# Endpoint API pour prédiction
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # on attend {"text": "ton email ici"}
    text = [data['text']]
    
    # Transformer et prédire
    X = vectorizer.transform(text)
    prediction = model.predict(X)[0]  # 0 = ham, 1 = spam
    return jsonify({'prediction': int(prediction)})

# Lancer le serveur
if __name__ == '__main__':
    app.run(debug=True)
