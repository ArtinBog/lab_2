from flask import Flask, render_template
from sklearn.externals import joblib

app = Flask(__name__)

@app.route("/")
def index():
    model = joblib.load('regr_tree_1.pkl')
    prediction = str(model.predict([[239000.0, 12, 4.0, 4772, 118, 8350.0, 0, 0]]).round(1)[0])
    return render_template("index.html", prediction=prediction)

if __name__ == ("__main__"):
    app.run(debug=True)
