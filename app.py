import flask 
from keras.models import load_model
from nltk.corpus import stopwords
import nltk
import re
import gensim

#loading model
model = load_model('final_lstm.h5')

#f1
def wordlist(essay, remove_stopwords):
    
    essay = re.sub("[^a-zA-Z]", " ", essay)
    words = essay.lower().split()
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    return (words)

#f2
def Make_sentences(essay, remove_stopwords):
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    raw_sentences = tokenizer.tokenize(essay.strip())
    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            sentences.append(wordlist(raw_sentence, remove_stopwords))
    return sentences

#f3
def makeFeatureVec(words, model, num_features):
    
    featureVec = np.zeros((num_features),dtype="float32")
    num_words = 0.
    index2word_set = set(model.wv.index_to_key)
    for word in words:
        if word in index2word_set:
            num_words += 1
            featureVec = np.add(featureVec,model.wv[word])        
    featureVec = np.divide(featureVec,num_words)
    return featureVec

#f4
def getAvgFeatureVecs(essays, model, num_features):

    counter = 0
    essayFeatureVecs = np.zeros((len(essays),num_features),dtype="float32")
    for essay in essays:
        essayFeatureVecs[counter] = makeFeatureVec(essay, model, num_features)
        counter = counter + 1
    return essayFeatureVecs

#cleaning the data
def preProcess(text): 
    model = gensim.models.Word2Vec.load("word2vecmodel.bin")
    clean_test_essays = []
    clean_test_essays.append(wordlist( text, remove_stopwords=True ))
    testDataVecs = getAvgFeatureVecs( clean_test_essays, model, 300 )
    testDataVecs = np.array(testDataVecs)
    testDataVecs = np.reshape(testDataVecs, (testDataVecs.shape[0], 1, testDataVecs.shape[1]))
    return testDataVecs

#defining app
app = flask.Flask(__name__, template_folder = 'templates')


#main route
@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return(flask.render_template('main.html'))

    if flask.request.method == 'POST':

        essay = flask.request.form['e']

        essay_pro = preProcess(essay)

        pred = model.predict(essay_pro)

        return flask.render_template('main.html', result = pred)


