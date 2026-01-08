from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model
model = pickle.load(open("Hasil_Model_Random_Forest.pkl", "rb"))

# Daftar provinsi
provinsi_list = [
    "Aceh",
    "Sumatera Utara",
    "Sumatera Barat",
    "Riau",
    "Jambi",
    "Sumatera Selatan",
    "Bengkulu",
    "Lampung"
]


@app.route("/", methods=["GET", "POST"])
def index():
    hasil = None

    if request.method == "POST":
        luas_panen = float(request.form["luas_panen"])
        curah_hujan = float(request.form["curah_hujan"])
        kelembapan = float(request.form["kelembapan"])
        suhu = float(request.form["suhu"])
        provinsi = request.form["provinsi"]

        # fitur numerik
        fitur = [luas_panen, curah_hujan, kelembapan, suhu]

        # one-hot encoding provinsi 
        for p in provinsi_list:
            fitur.append(1 if provinsi == p else 0)

        data = np.array([fitur])
        hasil = model.predict(data)[0]

    return render_template("form.html", hasil=hasil)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
