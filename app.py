from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Model aur PCA load karein
kmeans_model = pickle.load(open("smartcart_kmeans_model.pkl", "rb"))
pca_model = pickle.load(open("pca_transformer.pkl", "rb"))

@app.route("/")
def home():
    # Agar aapne templates folder mein index.html banaya hai, toh render_template use karein
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Form se saare inputs lein (apne dataset ke columns ke anusar yahan add karein)
        income = float(request.form["Income"])
        recency = float(request.form["Recency"])
        total_spending = float(request.form["total_spending"])
        
        # NOTE: Agar aapke model ko aur bhi columns chahiye, toh unhe yahan add karein.
        # Filhal hum array bana rahe hain (agar model ko exact wahi features chahiye jo training mein the)
        user_input = np.array([[income, recency, total_spending]])
        
        # 1. Pehle PCA transform karein
        user_pca = pca_model.transform(user_input)
        
        # 2. KMeans se cluster predict karein
        cluster_id = int(kmeans_model.predict(user_pca)[0])

        return render_template("index.html", prediction_text=f"Customer belongs to KMeans Cluster: {cluster_id}")
    
    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)