import os
import nltk
import json
import random
import openai
import requests
import re
from googlesearch import search
from flask import Flask, render_template, request, jsonify
from nltk.corpus import wordnet as wn
from pywsd.similarity import max_similarity
from pywsd.lesk import adapted_lesk
import string
from summarizer import Summarizer
from flashtext import KeywordProcessor
import pke
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
openai.api_key = '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'
current_question_id = 0
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    # Get the text from the form
    full_text = request.form['text']

    # Perform summarization using Summarizer
    model = Summarizer()
    result = model(full_text, min_length=60, max_length=500, ratio=0.4)
    summarized_text = ''.join(result)

    # Perform keyword extraction and processing
    keywords = get_nouns_multipartite(full_text)
    filtered_keys = []
    for keyword in keywords:
        if keyword.lower() in summarized_text.lower():
            filtered_keys.append(keyword)

    # Tokenize sentences and create keyword-sentence mapping
    sentences = tokenize_sentences(summarized_text)
    keyword_sentence_mapping = get_sentences_for_keyword(filtered_keys, sentences)

    # Generate key distractor list
    key_distractor_list = {}
    for keyword in keyword_sentence_mapping:
        wordsense = get_wordsense(keyword_sentence_mapping[keyword][0], keyword)
        if wordsense:
            distractors = get_distractors_wordnet(wordsense, keyword)
            if len(distractors) == 0:
                distractors = get_distractors_conceptnet(keyword)
            if len(distractors) != 0:
                key_distractor_list[keyword] = distractors
        else:
            distractors = get_distractors_conceptnet(keyword)
            if len(distractors) != 0:
                key_distractor_list[keyword] = distractors

    # Prepare response data
    response_data = []
    index = 1
    for each in key_distractor_list:
        sentence = keyword_sentence_mapping[each][0]
        pattern = re.compile(each, re.IGNORECASE)
        output = pattern.sub(" _______ ", sentence)
        choices = [each.capitalize()] + key_distractor_list[each]
        correct_answer = each.capitalize()
        top4choices = choices[:4]
        random.shuffle(top4choices)
        optionchoices = ['a', 'b', 'c', 'd']
        options = []
        for idx, choice in enumerate(top4choices):
            options.append({"letter": optionchoices[idx], "choice": choice})
        response_data.append({"index": index, "output": output, "correct_answer": correct_answer, "options": options})
        index = index + 1

    return jsonify(response_data)


# Helper functions

def get_nouns_multipartite(text):
    out = []

    extractor = pke.unsupervised.MultipartiteRank()
    extractor.load_document(input=text)
    #    not contain punctuation marks or stopwords as candidates.
    pos = {'PROPN'}
    # pos = {'VERB', 'ADJ', 'NOUN'}
    stoplist = list(string.punctuation)
    stoplist = ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']
    stoplist = stopwords.words('english')
    # extractor.load_document(input='/content/egypt.txt',
    #                            language='en',
    #                           stoplist=stoplist,
    #                          normalization=None)
    extractor.candidate_selection()

    # 4. build the Multipartite graph and rank candidates using random walk,
    #    alpha controls the weight adjustment mechanism, see TopicRank for
    #    threshold/method parameters.
    extractor.candidate_weighting(alpha=1.1,
                                  threshold=0.75,
                                  method='average')
    keyphrases = extractor.get_n_best(n=20)

    for key in keyphrases:
        out.append(key[0])

    return out

def tokenize_sentences(text):
    sentences = [sent_tokenize(text)]
    sentences = [y for x in sentences for y in x]
    # Remove any short sentences less than 20 letters.
    sentences = [sentence.strip() for sentence in sentences if len(sentence) > 20]
    return sentences

def get_sentences_for_keyword(keywords, sentences):
    keyword_processor = KeywordProcessor()
    keyword_sentences = {}
    for word in keywords:
        keyword_sentences[word] = []
        keyword_processor.add_keyword(word)
    for sentence in sentences:
        keywords_found = keyword_processor.extract_keywords(sentence)
        for key in keywords_found:
            keyword_sentences[key].append(sentence)

    for key in keyword_sentences.keys():
        values = keyword_sentences[key]
        values = sorted(values, key=len, reverse=True)
        keyword_sentences[key] = values
    return keyword_sentences

def get_distractors_wordnet(syn, word):
    distractors = []
    word = word.lower()
    orig_word = word
    if len(word.split()) > 0:
        word = word.replace(" ", "_")
    hypernym = syn.hypernyms()
    if len(hypernym) == 0:
        return distractors
    for item in hypernym[0].hyponyms():
        name = item.lemmas()[0].name()
        # print ("name ",name, " word",orig_word)
        if name == orig_word:
            continue
        name = name.replace("_", " ")
        name = " ".join(w.capitalize() for w in name.split())
        if name is not None and name not in distractors:
            distractors.append(name)
    return distractors

def get_wordsense(sent, word):
    word = word.lower()

    if len(word.split()) > 0:
        word = word.replace(" ", "_")

    synsets = wn.synsets(word, 'n')
    if synsets:
        wup = max_similarity(sent, word, 'wup', pos='n')
        adapted_lesk_output = adapted_lesk(sent, word, pos='n')
        synsets = [synset for synset in (wup, adapted_lesk_output) if synset]
        if synsets:
            chosen_synset = min(synsets, key=lambda syn: syn.offset())
            return chosen_synset
    return None

# ... Rest of your code ...


# ... Rest of your code ...


def get_distractors_conceptnet(word):
    word = word.lower()
    original_word = word
    if (len(word.split()) > 0):
        word = word.replace(" ", "_")
    distractor_list = []
    url = "http://api.conceptnet.io/query?node=/c/en/%s/n&rel=/r/PartOf&start=/c/en/%s&limit=5" % (word, word)
    obj = requests.get(url).json()

    for edge in obj['edges']:
        link = edge['end']['term']

        url2 = "http://api.conceptnet.io/query?node=%s&rel=/r/PartOf&end=%s&limit=10" % (link, link)
        obj2 = requests.get(url2).json()
        for edge in obj2['edges']:
            word2 = edge['start']['label']
            if word2 not in distractor_list and original_word.lower() not in word2.lower():
                distractor_list.append(word2)

    return distractor_list


def load_quiz_data(category):
    try:
        with open(os.path.join("OpenTriviaQA_JSON", f"{category}.json"), "r") as file:
            quiz_data = json.load(file)
        return quiz_data
    except FileNotFoundError:
        return []


def process_user_input(user_input):
    tokens = nltk.word_tokenize(user_input.lower())
    return tokens[0]


@app.route('/select_category', methods=['POST'])
def select_category():
    request_data = request.get_json()
    category = request_data.get('category', None)

    if not category:
        return jsonify({"error": "Invalid input data."}), 400

    categories = [file_name[:-5] for file_name in os.listdir("OpenTriviaQA_JSON") if file_name.endswith(".json")]
    if category not in categories:
        return jsonify({"error": "Invalid category name."}), 400

    return jsonify({"message": f"Category '{category}' selected."})


@app.route('/get_questions', methods=['POST'])
def get_questions():
    request_data = request.get_json()
    category = request_data.get('category', None)

    if not category:
        return jsonify({"error": "Invalid input data."}), 400

    quiz_data = load_quiz_data(category)
    if not quiz_data:
        return jsonify({"error": "No quiz data found for the selected category."}), 400

    # Assign unique IDs to each question
    for idx, question in enumerate(quiz_data):
        question['id'] = idx

    return jsonify({"questions": quiz_data})


@app.route('/answer_question', methods=['POST'])
def answer_question():
    global current_question_id

    request_data = request.get_json()
    category = request_data.get('category', None)
    user_answers = request_data.get('user_answers', None)

    if not category or not user_answers:
        return jsonify({"error": "Invalid input data."}), 400

    quiz_data = load_quiz_data(category)
    if not quiz_data:
        return jsonify({"error": "No quiz data found for the selected category."}), 400
    return jsonify(quiz_data)
    import sys
    sys.exit()
    num_questions = min(10, len(quiz_data))
    score = 0


    for idx in range(current_question_id, current_question_id + num_questions):

        if idx >= len(quiz_data):
            break
        user_answer = user_answers.get(str(quiz_data[idx]['id']), '').lower()

        if user_answer == quiz_data[idx]['answer'].lower():
            score += 1





def generate_suggestions(incorrect_answers):
    prompt = "For the questions you answered wrong:\n\n"
    for idx, answer in enumerate(incorrect_answers):
        prompt += f"{idx + 1}. Q: {answer['question']}\nYour answer: {answer['user_answer']}\n\n"

    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=150
    )

    return response.choices[0].text.strip()


# Function to get relevant websites for topics
def get_relevant_websites(topics):
    websites_data = []
    for topic in topics:
        search_query = f"Best resources for learning {topic}"
        search_results = search(search_query, num_results=5)

        websites_list = list(search_results)  # Convert the generator to a list

        websites_data.append({"topic": topic, "websites": websites_list})

    return websites_data


@app.route('/quiz_completed', methods=['POST'])
def quiz_completed():
    data = request.json
    incorrect_answers = data.get('incorrect_answers', [])
    topics = data.get('topics', [])

    suggestions = generate_suggestions(incorrect_answers)
    # suggestions = []
    relevant_websites = get_relevant_websites(topics)

    response = {
        "suggestions": suggestions,
        "relevant_websites": relevant_websites
    }

    return jsonify(response)


@app.route('/generate_quiz', methods=['POST'])
def generate_quiz():
    try:
        topic = request.json['topic']

        # Modify the prompt as needed
        prompt = """
        tell me about sql.
        """

        response = openai.Completion.create(
            engine="davinci",
            prompt=prompt,
            max_tokens=600
        )

        quiz = response.choices[0].text.strip()

        return jsonify({'quiz': quiz})
    except Exception as e:
        return jsonify({'error': str(e)}), 500




if __name__ == '__main__':
    app.run(debug=True)
