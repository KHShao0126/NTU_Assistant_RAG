from flask import Flask, render_template, request
from bm25_version1 import search_bm25
import os

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    result = ""
    if request.method == 'POST':
        query = request.form['query']
        result = search_bm25(query)
    return render_template('index.html', result=result)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)