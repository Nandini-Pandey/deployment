from flask import Flask, render_template, jsonify, request
from chatbot import get_response
app=Flask(__name__)
@app.get('/')
def index_get():
    return render_template("index.html")


@app.post("/predict")
def predict():
    text=request.get_json().get("message")
    response=get_response(text)
    message={"answer": response}
    return jsonify(message)


@app.route("/music")
def music():
    return render_template('indexm.html')


@app.route("/game")
def game():
    return render_template("indexg.html")

@app.route("/resource")
def resource():
    return render_template("indexr.html")

if __name__=="__main__":
    app.run(debug=True)
