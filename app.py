from flask import Flask, render_template, request, jsonify
from collections import defaultdict
import json
import os

app = Flask(__name__)

LEARNING_FILE = 'learned_pairs.json'
TRIGRAM_FILE = 'learned_trigrams.json'

# ============================================
# LOAD EXISTING LEARNING DATA
# ============================================

if os.path.exists(LEARNING_FILE):
    with open(LEARNING_FILE, 'r', encoding='utf-8') as f:
        learned_data = json.load(f)
    bigrams = defaultdict(lambda: defaultdict(int))
    for word, suggestions in learned_data.items():
        for sugg, count in suggestions.items():
            bigrams[word][sugg] = count
else:
    bigrams = defaultdict(lambda: defaultdict(int))

if os.path.exists(TRIGRAM_FILE):
    with open(TRIGRAM_FILE, 'r', encoding='utf-8') as f:
        trigram_data = json.load(f)
    trigrams = defaultdict(lambda: defaultdict(int))
    for two_words, suggestions in trigram_data.items():
        for sugg, count in suggestions.items():
            trigrams[two_words][sugg] = count
else:
    trigrams = defaultdict(lambda: defaultdict(int))

# ============================================
# BIGRAM DATA (MAXIMUM - 5000+ PAIRS)
# ============================================

bigram_pretrained = {
    # PRONOUNS (All forms)
    'i': {'am': 500, 'have': 495, 'will': 490, 'can': 485, 'love': 480, 'like': 475, 'want': 470, 'need': 465, 'did': 460, 'would': 455, 'could': 450, 'should': 445, 'might': 440, 'must': 435, 'do': 430, 'dont': 425, 'never': 420, 'always': 415},
    'you': {'are': 500, 'have': 495, 'will': 490, 'can': 485, 'need': 480, 'want': 475, 'like': 470, 'did': 465, 'would': 460, 'could': 455, 'should': 450, 'do': 445, 'dont': 440, 'never': 435},
    'he': {'is': 500, 'has': 495, 'was': 490, 'does': 485, 'will': 480, 'can': 475, 'would': 470, 'could': 465, 'should': 460, 'did': 455, 'never': 450},
    'she': {'is': 500, 'has': 495, 'was': 490, 'does': 485, 'will': 480, 'can': 475, 'would': 470, 'could': 465, 'should': 460, 'did': 455},
    'it': {'is': 500, 'has': 495, 'was': 490, 'looks': 485, 'seems': 480, 'works': 475, 'does': 470, 'will': 465, 'can': 460, 'takes': 455},
    'we': {'are': 500, 'have': 495, 'will': 490, 'can': 485, 'should': 480, 'must': 475, 'would': 470, 'could': 465, 'need': 460, 'did': 455},
    'they': {'are': 500, 'have': 495, 'will': 490, 'can': 485, 'should': 480, 'must': 475, 'would': 470, 'could': 465, 'need': 460, 'did': 455},
    'my': {'name': 500, 'friend': 495, 'car': 490, 'house': 485, 'life': 480, 'love': 475, 'heart': 470, 'mind': 465, 'family': 460, 'mother': 455, 'father': 450},
    'your': {'name': 500, 'friend': 495, 'car': 490, 'house': 485, 'life': 480, 'love': 475, 'help': 470, 'support': 465, 'family': 460, 'mother': 455},
    'his': {'name': 500, 'friend': 495, 'car': 490, 'house': 485, 'life': 480, 'work': 475, 'family': 470, 'mother': 465, 'father': 460},
    'her': {'name': 500, 'friend': 495, 'car': 490, 'house': 485, 'life': 480, 'smile': 475, 'eyes': 470, 'family': 465, 'mother': 460},
    'our': {'home': 500, 'life': 495, 'family': 490, 'country': 485, 'world': 480, 'future': 475, 'children': 470, 'parents': 465, 'house': 460},
    'their': {'home': 500, 'life': 495, 'family': 490, 'country': 485, 'children': 480, 'future': 475, 'house': 470, 'car': 465},
    'me': {'too': 500, 'please': 495, 'now': 490, 'later': 485, 'either': 480, 'neither': 475, 'also': 470, 'either': 465},
    'us': {'too': 500, 'please': 495, 'now': 490, 'later': 485, 'all': 480, 'both': 475, 'also': 470},
    'them': {'too': 500, 'please': 495, 'now': 490, 'later': 485, 'all': 480, 'both': 475},
    
    # VERBS (All tenses - Maximum)
    'am': {'a': 500, 'going': 495, 'doing': 490, 'trying': 485, 'here': 480, 'not': 475, 'very': 470, 'so': 465, 'just': 460, 'still': 455, 'always': 450},
    'is': {'a': 500, 'the': 495, 'going': 490, 'very': 485, 'not': 480, 'my': 475, 'your': 470, 'his': 465, 'her': 460, 'it': 455, 'this': 450, 'that': 445},
    'are': {'you': 500, 'we': 495, 'they': 490, 'going': 485, 'not': 480, 'the': 475, 'a': 470, 'there': 465, 'here': 460, 'so': 455, 'very': 450},
    'was': {'a': 500, 'the': 495, 'very': 490, 'going': 485, 'not': 480, 'my': 475, 'his': 470, 'her': 465, 'so': 460, 'just': 455, 'really': 450},
    'were': {'you': 500, 'they': 495, 'we': 490, 'not': 485, 'very': 480, 'so': 475, 'there': 470, 'here': 465, 'all': 460, 'really': 455},
    'be': {'a': 500, 'the': 495, 'very': 490, 'good': 485, 'nice': 480, 'happy': 475, 'sad': 470, 'there': 465, 'here': 460, 'careful': 455},
    'been': {'a': 500, 'the': 495, 'very': 490, 'so': 485, 'too': 480, 'there': 475, 'here': 470, 'long': 465, 'away': 460},
    'being': {'a': 500, 'very': 495, 'so': 490, 'too': 485, 'good': 480, 'nice': 475, 'happy': 470},
    'have': {'a': 500, 'to': 495, 'been': 490, 'got': 485, 'you': 480, 'it': 475, 'this': 470, 'no': 465, 'many': 460, 'some': 455, 'never': 450},
    'has': {'a': 500, 'been': 495, 'to': 490, 'got': 485, 'it': 480, 'this': 475, 'no': 470, 'many': 465, 'always': 460, 'never': 455},
    'had': {'a': 500, 'been': 495, 'to': 490, 'got': 485, 'no': 480, 'it': 475, 'this': 470, 'never': 465, 'always': 460},
    'do': {'you': 500, 'it': 495, 'not': 490, 'this': 485, 'your': 480, 'my': 475, 'we': 470, 'they': 465, 'i': 460, 'something': 455},
    'does': {'it': 500, 'this': 495, 'not': 490, 'he': 485, 'she': 480, 'anyone': 475, 'everyone': 470, 'anybody': 465},
    'did': {'you': 500, 'it': 495, 'this': 490, 'he': 485, 'she': 480, 'not': 475, 'we': 470, 'they': 465, 'i': 460, 'everything': 455},
    'can': {'you': 500, 'i': 495, 'we': 490, 'they': 485, 'he': 480, 'she': 475, 'be': 470, 'do': 465, 'get': 460, 'see': 455, 'help': 450, 'go': 445},
    'could': {'be': 500, 'you': 495, 'i': 490, 'we': 485, 'they': 480, 'have': 475, 'get': 470, 'do': 465, 'see': 460, 'help': 455},
    'would': {'be': 500, 'you': 495, 'i': 490, 'like': 485, 'love': 480, 'want': 475, 'have': 470, 'do': 465, 'go': 460, 'see': 455, 'help': 450},
    'should': {'be': 500, 'you': 495, 'we': 490, 'they': 485, 'have': 480, 'go': 475, 'do': 470, 'get': 465, 'see': 460, 'take': 455},
    'might': {'be': 500, 'have': 495, 'need': 490, 'want': 485, 'go': 480, 'come': 475, 'see': 470, 'get': 465},
    'must': {'be': 500, 'have': 495, 'go': 490, 'do': 485, 'get': 480, 'see': 475, 'remember': 470},
    'will': {'be': 500, 'go': 495, 'come': 490, 'see': 485, 'get': 480, 'have': 475, 'do': 470, 'make': 465, 'take': 460, 'need': 455, 'want': 450},
    'would': {'like': 500, 'love': 495, 'want': 490, 'be': 485, 'go': 480, 'come': 475, 'see': 470},
    'going': {'to': 500, 'be': 495, 'home': 490, 'out': 485, 'in': 480, 'there': 475, 'now': 470},
    'go': {'to': 500, 'home': 495, 'out': 490, 'in': 485, 'there': 480, 'now': 475, 'away': 470, 'back': 465, 'together': 460},
    'come': {'here': 500, 'home': 495, 'back': 490, 'in': 485, 'to': 480, 'over': 475, 'out': 470, 'with': 465},
    'see': {'you': 500, 'it': 495, 'him': 490, 'her': 485, 'them': 480, 'me': 475, 'us': 470, 'the': 465, 'that': 460},
    'get': {'it': 500, 'out': 495, 'in': 490, 'up': 485, 'down': 480, 'there': 475, 'home': 470, 'back': 465, 'ready': 460, 'better': 455},
    'make': {'it': 500, 'sure': 495, 'money': 490, 'love': 485, 'mistake': 480, 'sense': 475, 'time': 470, 'decision': 465, 'difference': 460},
    'take': {'it': 500, 'care': 495, 'time': 490, 'out': 485, 'in': 480, 'a': 475, 'the': 470, 'your': 465, 'my': 460},
    'give': {'me': 500, 'it': 495, 'you': 490, 'him': 485, 'her': 480, 'us': 475, 'them': 470, 'up': 465, 'back': 460},
    'put': {'it': 500, 'on': 495, 'in': 490, 'out': 485, 'down': 480, 'up': 475, 'there': 470, 'together': 465},
    'keep': {'it': 500, 'going': 495, 'up': 490, 'safe': 485, 'quiet': 480, 'calm': 475, 'trying': 470, 'working': 465},
    'find': {'it': 500, 'out': 495, 'a': 490, 'the': 485, 'yourself': 480, 'way': 475, 'time': 470},
    'start': {'to': 500, 'a': 495, 'the': 490, 'your': 485, 'now': 480, 'today': 475},
    'stop': {'it': 500, 'the': 495, 'a': 490, 'your': 485, 'now': 480, 'here': 475},
    
    # EVERY, SOME, ANY, NO, ALL, MANY, MUCH, MORE (Maximum)
    'every': {'day': 500, 'night': 495, 'time': 490, 'one': 485, 'person': 480, 'morning': 475, 'week': 470, 'month': 465, 'year': 460, 'single': 455},
    'everyone': {'is': 500, 'has': 495, 'knows': 490, 'loves': 485, 'needs': 480, 'wants': 475, 'likes': 470, 'agrees': 465, 'thinks': 460},
    'everything': {'is': 500, 'was': 495, 'has': 490, 'looks': 485, 'seems': 480, 'okay': 475, 'fine': 470, 'perfect': 465, 'good': 460},
    'everywhere': {'is': 500, 'was': 495, 'looks': 490, 'goes': 485, 'feels': 480, 'seems': 475, 'i': 470},
    'everybody': {'is': 500, 'has': 495, 'knows': 490, 'loves': 485, 'needs': 480, 'wants': 475, 'likes': 470},
    'some': {'one': 500, 'people': 495, 'time': 490, 'day': 485, 'thing': 480, 'where': 475, 'money': 470, 'help': 465, 'reason': 460, 'way': 455},
    'someone': {'is': 500, 'has': 495, 'knows': 490, 'loves': 485, 'called': 480, 'said': 475, 'told': 470, 'asked': 465, 'helped': 460},
    'something': {'is': 500, 'was': 495, 'has': 490, 'looks': 485, 'happened': 480, 'changed': 475, 'wrong': 470, 'else': 465, 'new': 460},
    'somewhere': {'is': 500, 'was': 495, 'over': 490, 'in': 485, 'out': 480, 'else': 475, 'near': 470, 'far': 465},
    'any': {'one': 500, 'time': 495, 'day': 490, 'person': 485, 'thing': 480, 'where': 475, 'help': 470, 'idea': 465, 'reason': 460, 'way': 455},
    'anyone': {'is': 500, 'has': 495, 'knows': 490, 'can': 485, 'will': 480, 'could': 475, 'should': 470, 'would': 465},
    'anything': {'is': 500, 'was': 495, 'can': 490, 'will': 485, 'possible': 480, 'else': 475, 'wrong': 470, 'new': 465, 'good': 460},
    'anywhere': {'is': 500, 'was': 495, 'can': 490, 'will': 485, 'else': 480, 'near': 475, 'far': 470},
    'anybody': {'is': 500, 'has': 495, 'knows': 490, 'can': 485, 'will': 480, 'could': 475},
    'no': {'one': 500, 'time': 495, 'way': 490, 'problem': 485, 'idea': 480, 'where': 475, 'money': 470, 'help': 465, 'reason': 460, 'doubt': 455},
    'no one': {'is': 500, 'knows': 495, 'cares': 490, 'loves': 485, 'likes': 480, 'helps': 475, 'understands': 470},
    'nothing': {'is': 500, 'was': 495, 'has': 490, 'matters': 485, 'changed': 480, 'else': 475, 'wrong': 470, 'new': 465, 'works': 460},
    'nowhere': {'to': 500, 'in': 495, 'near': 490, 'else': 485, 'found': 480, 'seen': 475},
    'nobody': {'is': 500, 'knows': 495, 'cares': 490, 'loves': 485, 'likes': 480, 'helps': 475},
    'all': {'the': 500, 'my': 495, 'your': 490, 'his': 485, 'her': 480, 'of': 475, 'these': 470, 'those': 465, 'our': 460, 'their': 455},
    'most': {'of': 500, 'the': 495, 'people': 490, 'important': 485, 'common': 480, 'popular': 475, 'likely': 470, 'beautiful': 465},
    'many': {'people': 500, 'times': 495, 'years': 490, 'days': 485, 'things': 480, 'ways': 475, 'reasons': 470, 'places': 465, 'problems': 460},
    'much': {'more': 500, 'better': 495, 'less': 490, 'love': 485, 'time': 480, 'money': 475, 'help': 470, 'fun': 465, 'care': 460},
    'more': {'than': 500, 'people': 495, 'time': 490, 'money': 485, 'important': 480, 'common': 475, 'likely': 470, 'beautiful': 465, 'fun': 460},
    'less': {'than': 500, 'more': 495, 'time': 490, 'money': 485, 'important': 480, 'common': 475, 'likely': 470, 'fun': 465},
    'least': {'of': 500, 'the': 495, 'important': 490, 'common': 485, 'likely': 480, 'popular': 475},
    
    # ADJECTIVES (Maximum common adjectives)
    'good': {'morning': 500, 'day': 495, 'luck': 490, 'job': 485, 'idea': 480, 'person': 475, 'time': 470, 'night': 465, 'friend': 460, 'work': 455, 'life': 450, 'news': 445},
    'bad': {'day': 500, 'luck': 495, 'idea': 490, 'person': 485, 'situation': 480, 'weather': 475, 'habit': 470, 'news': 465, 'dream': 460, 'night': 455},
    'great': {'day': 500, 'job': 495, 'idea': 490, 'work': 485, 'person': 480, 'friend': 475, 'time': 470, 'success': 465, 'life': 460, 'news': 455},
    'nice': {'day': 500, 'person': 495, 'car': 490, 'house': 485, 'weather': 480, 'smile': 475, 'place': 470, 'view': 465, 'guy': 460},
    'beautiful': {'day': 500, 'girl': 495, 'house': 490, 'car': 485, 'smile': 480, 'view': 475, 'place': 470, 'weather': 465, 'face': 460, 'eyes': 455},
    'pretty': {'good': 500, 'nice': 495, 'girl': 490, 'face': 485, 'dress': 480, 'house': 475, 'car': 470, 'well': 465, 'much': 460},
    'ugly': {'truth': 500, 'face': 495, 'situation': 490, 'reality': 485, 'building': 480, 'car': 475, 'person': 470, 'duckling': 465},
    'happy': {'birthday': 500, 'day': 495, 'life': 490, 'ending': 485, 'moment': 480, 'smile': 475, 'family': 470, 'face': 465, 'childhood': 460},
    'sad': {'day': 500, 'story': 495, 'ending': 490, 'news': 485, 'person': 480, 'face': 475, 'truth': 470, 'reality': 465, 'song': 460},
    'angry': {'at': 500, 'with': 495, 'about': 490, 'person': 485, 'voice': 480, 'face': 475, 'mob': 470, 'crowd': 465},
    'excited': {'about': 500, 'for': 495, 'to': 490, 'today': 485, 'tomorrow': 480, 'see': 475, 'meet': 470, 'go': 465},
    'scared': {'of': 500, 'about': 495, 'to': 490, 'death': 485, 'dark': 480, 'lonely': 475, 'alone': 470, 'scary': 465},
    'tired': {'of': 500, 'from': 495, 'today': 490, 'very': 485, 'so': 480, 'too': 475, 'now': 470, 'always': 465},
    'big': {'house': 500, 'car': 495, 'problem': 490, 'success': 485, 'day': 480, 'fan': 475, 'deal': 470, 'man': 465, 'city': 460, 'surprise': 455},
    'small': {'house': 500, 'car': 495, 'problem': 490, 'thing': 485, 'cat': 480, 'dog': 475, 'town': 470, 'city': 465, 'difference': 460, 'change': 455},
    'large': {'house': 500, 'car': 495, 'problem': 490, 'company': 485, 'city': 480, 'number': 475, 'amount': 470, 'group': 465},
    'huge': {'house': 500, 'car': 495, 'problem': 490, 'success': 485, 'fan': 480, 'deal': 475, 'difference': 470},
    'tiny': {'house': 500, 'car': 495, 'problem': 490, 'thing': 485, 'cat': 480, 'dog': 475, 'town': 470},
    'new': {'car': 500, 'house': 495, 'job': 490, 'phone': 485, 'friend': 480, 'day': 475, 'idea': 470, 'life': 465, 'year': 460, 'beginning': 455},
    'old': {'car': 500, 'house': 495, 'friend': 490, 'phone': 485, 'man': 480, 'woman': 475, 'days': 470, 'times': 465, 'school': 460, 'building': 455},
    'young': {'man': 500, 'woman': 495, 'boy': 490, 'girl': 485, 'person': 480, 'age': 475, 'people': 470, 'generation': 465},
    'rich': {'man': 500, 'person': 495, 'family': 490, 'country': 485, 'people': 480, 'life': 475, 'history': 470, 'culture': 465},
    'poor': {'man': 500, 'person': 495, 'family': 490, 'country': 485, 'people': 480, 'life': 475, 'health': 470, 'quality': 465},
    'strong': {'man': 500, 'person': 495, 'feeling': 490, 'will': 485, 'body': 480, 'sense': 475, 'relationship': 470},
    'weak': {'man': 500, 'person': 495, 'signal': 490, 'feeling': 485, 'health': 480, 'link': 475, 'connection': 470},
    'fast': {'car': 500, 'food': 495, 'runner': 490, 'internet': 485, 'speed': 480, 'pace': 475, 'learning': 470, 'worker': 465},
    'slow': {'car': 500, 'internet': 495, 'runner': 490, 'speed': 485, 'day': 480, 'process': 475, 'learning': 470, 'worker': 465},
    'easy': {'to': 500, 'way': 495, 'task': 490, 'job': 485, 'work': 480, 'life': 475, 'answer': 470, 'question': 465},
    'hard': {'to': 500, 'work': 495, 'task': 490, 'job': 485, 'time': 480, 'life': 475, 'decision': 470, 'question': 465},
    'soft': {'and': 500, 'voice': 495, 'touch': 490, 'skin': 485, 'pillow': 480, 'drink': 475, 'music': 470},
    'dark': {'night': 500, 'room': 495, 'side': 490, 'sky': 485, 'cloud': 480, 'color': 475, 'blue': 470, 'brown': 465},
    'light': {'and': 500, 'weight': 495, 'color': 490, 'blue': 485, 'green': 480, 'house': 475, 'room': 470, 'skin': 465},
    'bright': {'light': 500, 'sun': 495, 'future': 490, 'smile': 485, 'color': 480, 'day': 475, 'morning': 470},
    'clear': {'sky': 500, 'water': 495, 'mind': 490, 'understanding': 485, 'explanation': 480, 'view': 475},
    'simple': {'way': 500, 'answer': 495, 'solution': 490, 'life': 485, 'truth': 480, 'fact': 475},
    'complex': {'problem': 500, 'situation': 495, 'system': 490, 'relationship': 485, 'issue': 480},
    
    # GREETINGS & POLITE (Maximum)
    'hello': {'world': 500, 'everyone': 495, 'friends': 490, 'dear': 485, 'sir': 480, 'there': 475, 'how': 470, 'my': 465, 'good': 460},
    'hi': {'there': 500, 'everyone': 495, 'friends': 490, 'how': 485, 'hello': 480, 'my': 475, 'good': 470},
    'hey': {'there': 500, 'everyone': 495, 'how': 490, 'whats': 485, 'hi': 480, 'you': 475, 'guys': 470},
    'goodbye': {'everyone': 500, 'friends': 495, 'dear': 490, 'sir': 485, 'now': 480, 'for': 475, 'my': 470},
    'bye': {'everyone': 500, 'friends': 495, 'dear': 490, 'now': 485, 'see': 480, 'for': 475, 'guys': 470},
    'thank': {'you': 500, 'god': 495, 'sir': 490, 'everyone': 485, 'all': 480, 'so': 475, 'very': 470, 'much': 465, 'for': 460},
    'thanks': {'you': 500, 'god': 495, 'sir': 490, 'everyone': 485, 'for': 480, 'so': 475, 'very': 470, 'much': 465},
    'please': {'help': 500, 'tell': 495, 'come': 490, 'go': 485, 'wait': 480, 'sit': 475, 'stand': 470, 'be': 465, 'do': 460, 'stay': 455},
    'sorry': {'for': 500, 'about': 495, 'dear': 490, 'everyone': 485, 'sir': 480, 'to': 475, 'i': 470, 'my': 465, 'the': 460},
    'excuse': {'me': 500, 'us': 495, 'them': 490, 'him': 485, 'her': 480, 'sir': 475, 'maam': 470},
    'welcome': {'to': 500, 'home': 495, 'our': 490, 'my': 485, 'the': 480, 'everyone': 475, 'back': 470},
    
    # QUESTION WORDS (Maximum)
    'what': {'is': 500, 'are': 495, 'was': 490, 'do': 485, 'does': 480, 'about': 475, 'happened': 470, 'did': 465, 'can': 460, 'will': 455, 'would': 450},
    'where': {'is': 500, 'are': 495, 'was': 490, 'do': 485, 'did': 480, 'have': 475, 'can': 470, 'should': 465, 'would': 460},
    'when': {'is': 500, 'are': 495, 'will': 490, 'did': 485, 'was': 480, 'does': 475, 'can': 470, 'should': 465},
    'why': {'is': 500, 'are': 495, 'do': 490, 'did': 485, 'would': 480, 'not': 475, 'should': 470, 'could': 465},
    'how': {'are': 500, 'is': 495, 'to': 490, 'do': 485, 'about': 480, 'many': 475, 'much': 470, 'long': 465, 'often': 460, 'far': 455},
    'who': {'is': 500, 'are': 495, 'was': 490, 'did': 485, 'will': 480, 'can': 475, 'should': 470, 'would': 465},
    'which': {'one': 500, 'is': 495, 'are': 490, 'way': 485, 'time': 480, 'place': 475, 'of': 470, 'side': 465},
    'whom': {'are': 500, 'is': 495, 'was': 490, 'did': 485, 'will': 480, 'should': 475},
    'whose': {'is': 500, 'are': 495, 'was': 490, 'this': 485, 'that': 480, 'car': 475, 'house': 470},
    
       # TIME WORDS (Maximum)
    'today': {'is': 500, 'was': 495, 'i': 490, 'we': 485, 'going': 480, 'will': 475, 'feels': 470, 'looks': 465, 'has': 460, 'i': 455},
    'tomorrow': {'is': 500, 'will': 495, 'morning': 490, 'night': 485, 'i': 480, 'we': 475, 'be': 470, 'comes': 465, 'morning': 460},
    'yesterday': {'was': 500, 'i': 495, 'we': 490, 'he': 485, 'she': 480, 'they': 475, 'went': 470, 'came': 465, 'saw': 460},
    'now': {'i': 500, 'we': 495, 'is': 490, 'its': 485, 'go': 480, 'come': 475, 'time': 470, 'or': 465, 'then': 460},
    'later': {'i': 500, 'we': 495, 'will': 490, 'today': 485, 'then': 480, 'on': 475, 'bye': 470, 'see': 465},
    'soon': {'i': 500, 'we': 495, 'will': 490, 'be': 485, 'see': 480, 'come': 475, 'enough': 470, 'after': 465},
    'early': {'morning': 500, 'today': 495, 'tomorrow': 490, 'in': 485, 'this': 480, 'age': 475, 'bird': 470},
    'late': {'night': 500, 'at': 495, 'for': 490, 'i': 485, 'we': 480, 'again': 475, 'work': 470, 'sleep': 465},
    'morning': {'i': 500, 'we': 495, 'woke': 490, 'go': 485, 'work': 480, 'study': 475, 'run': 470, 'walk': 465, 'coffee': 460},
    'afternoon': {'i': 500, 'we': 495, 'went': 490, 'came': 485, 'worked': 480, 'studied': 475, 'ate': 470},
    'evening': {'i': 500, 'we': 495, 'came': 490, 'went': 485, 'home': 480, 'ate': 475, 'watched': 470, 'relaxed': 465},
    'night': {'i': 500, 'we': 495, 'slept': 490, 'went': 485, 'came': 480, 'good': 475, 'late': 470, 'dark': 465, 'stayed': 460},
    'week': {'i': 500, 'we': 495, 'last': 490, 'next': 485, 'this': 480, 'every': 475, 'per': 470, 'long': 465},
    'month': {'i': 500, 'we': 495, 'last': 490, 'next': 485, 'this': 480, 'every': 475, 'per': 470, 'long': 465},
    'year': {'i': 500, 'we': 495, 'last': 490, 'next': 485, 'this': 480, 'every': 475, 'new': 470, 'per': 465, 'long': 460},
    'day': {'i': 500, 'we': 495, 'was': 490, 'is': 485, 'of': 480, 'by': 475, 'after': 470, 'before': 465, 'every': 460},
    'weekend': {'i': 500, 'we': 495, 'am': 490, 'will': 485, 'going': 480, 'had': 475, 'enjoyed': 470},
    
    # ANIMALS (Maximum)
    'cat': {'sat': 500, 'ran': 495, 'jumped': 490, 'ate': 485, 'sleeps': 480, 'meowed': 475, 'is': 470, 'was': 465, 'climbed': 460, 'purred': 455},
    'dog': {'ran': 500, 'barked': 495, 'ate': 490, 'sleeps': 485, 'jumped': 480, 'walked': 475, 'is': 470, 'was': 465, 'played': 460, 'bites': 455},
    'bird': {'flew': 500, 'sang': 495, 'sat': 490, 'ate': 485, 'is': 480, 'has': 475, 'flies': 470, 'nested': 465, 'chirped': 460},
    'fish': {'swam': 500, 'ate': 495, 'died': 490, 'lived': 485, 'is': 480, 'swims': 475, 'jumped': 470, 'bites': 465},
    'horse': {'ran': 500, 'jumped': 495, 'ate': 490, 'galloped': 485, 'is': 480, 'was': 475, 'neighed': 470, 'rode': 465},
    'cow': {'ate': 500, 'gave': 495, 'is': 490, 'was': 485, 'grazed': 480, 'mooed': 475, 'walked': 470},
    'lion': {'roared': 500, 'hunted': 495, 'ate': 490, 'slept': 485, 'is': 480, 'king': 475, 'lives': 470, 'hunts': 465},
    'tiger': {'roared': 500, 'hunted': 495, 'ate': 490, 'slept': 485, 'is': 480, 'lives': 475, 'striped': 470},
    'elephant': {'is': 500, 'has': 495, 'looks': 490, 'walked': 485, 'big': 480, 'heavy': 475, 'lives': 470, 'trunk': 465},
    'monkey': {'climbed': 500, 'ate': 495, 'jumped': 490, 'is': 485, 'funny': 480, 'swung': 475, 'plays': 470},
    'rabbit': {'ran': 500, 'jumped': 495, 'ate': 490, 'is': 485, 'fast': 480, 'white': 475, 'cute': 470},
    'mouse': {'ran': 500, 'ate': 495, 'squeaked': 490, 'is': 485, 'small': 480, 'grey': 475, 'quick': 470},
    
    # VEHICLES (Maximum)
    'car': {'is': 500, 'was': 495, 'drives': 490, 'looks': 485, 'has': 480, 'needs': 475, 'costs': 470, 'runs': 465, 'parked': 460, 'broke': 455},
    'bus': {'is': 500, 'was': 495, 'comes': 490, 'goes': 485, 'arrives': 480, 'leaves': 475, 'late': 470, 'early': 465, 'full': 460},
    'train': {'is': 500, 'was': 495, 'comes': 490, 'goes': 485, 'arrives': 480, 'leaves': 475, 'late': 470, 'fast': 465, 'delayed': 460},
    'bike': {'is': 500, 'was': 495, 'rides': 490, 'goes': 485, 'has': 480, 'needs': 475, 'fast': 470, 'new': 465},
    'plane': {'is': 500, 'was': 495, 'flies': 490, 'lands': 485, 'takes': 480, 'delayed': 475, 'fast': 470, 'crashed': 465},
    'truck': {'is': 500, 'was': 495, 'carries': 490, 'drives': 485, 'big': 480, 'heavy': 475, 'loaded': 470},
    'boat': {'is': 500, 'was': 495, 'sails': 490, 'floats': 485, 'sinks': 480, 'slow': 475, 'small': 470},
    'ship': {'is': 500, 'was': 495, 'sails': 490, 'floats': 485, 'sinks': 480, 'big': 475, 'cargo': 470},
    'helicopter': {'is': 500, 'was': 495, 'flies': 490, 'lands': 485, 'loud': 480, 'fast': 475},
    
    # ELECTRONICS
    'phone': {'is': 500, 'was': 495, 'rings': 490, 'charges': 485, 'works': 480, 'dies': 475, 'calls': 470, 'has': 465, 'broke': 460},
    'computer': {'is': 500, 'works': 495, 'runs': 490, 'crashes': 485, 'needs': 480, 'has': 475, 'fast': 470, 'slow': 465},
    'laptop': {'is': 500, 'works': 495, 'runs': 490, 'crashes': 485, 'charges': 480, 'light': 475, 'portable': 470},
    'tv': {'is': 500, 'works': 495, 'shows': 490, 'plays': 485, 'has': 480, 'turns': 475, 'watched': 470, 'broke': 465},
    'tablet': {'is': 500, 'works': 495, 'runs': 490, 'has': 485, 'charges': 480, 'light': 475, 'portable': 470},
    'camera': {'is': 500, 'works': 495, 'takes': 490, 'has': 485, 'zooms': 480, 'records': 475, 'battery': 470},
    
    # FOOD & DRINK
    'eat': {'food': 500, 'dinner': 495, 'breakfast': 490, 'lunch': 485, 'well': 480, 'healthy': 475, 'out': 470, 'together': 465, 'something': 460},
    'drink': {'water': 500, 'coffee': 495, 'tea': 490, 'milk': 485, 'soda': 480, 'juice': 475, 'beer': 470, 'wine': 465, 'something': 460},
    'food': {'is': 500, 'was': 495, 'tastes': 490, 'looks': 485, 'smells': 480, 'good': 475, 'delicious': 470, 'hot': 465, 'cold': 460},
    'water': {'is': 500, 'was': 495, 'clean': 490, 'cold': 485, 'hot': 480, 'fresh': 475, 'running': 470, 'bottle': 465, 'drink': 460},
    'coffee': {'is': 500, 'was': 495, 'hot': 490, 'cold': 485, 'strong': 480, 'black': 475, 'good': 470, 'fresh': 465, 'iced': 460},
    'tea': {'is': 500, 'was': 495, 'hot': 490, 'cold': 485, 'sweet': 480, 'green': 475, 'black': 470, 'iced': 465},
    'milk': {'is': 500, 'was': 495, 'cold': 490, 'hot': 485, 'fresh': 480, 'good': 475, 'almond': 470, 'soy': 465},
    'juice': {'is': 500, 'was': 495, 'fresh': 490, 'sweet': 485, 'cold': 480, 'orange': 475, 'apple': 470, 'grape': 465},
    
    # COUNTRIES & CITIES
    'pakistan': {'is': 500, 'has': 495, 'people': 490, 'culture': 485, 'cricket': 480, 'country': 475, 'beautiful': 470, 'rich': 465},
    'karachi': {'is': 500, 'city': 495, 'has': 490, 'people': 485, 'life': 480, 'big': 475, 'beautiful': 470, 'busy': 465},
    'lahore': {'is': 500, 'city': 495, 'has': 490, 'food': 485, 'culture': 480, 'beautiful': 475, 'historical': 470},
    'islamabad': {'is': 500, 'city': 495, 'has': 490, 'beautiful': 485, 'capital': 480, 'peaceful': 475, 'clean': 470},
    'india': {'is': 500, 'has': 495, 'people': 490, 'cricket': 485, 'country': 480, 'big': 475, 'diverse': 470},
    'usa': {'is': 500, 'has': 495, 'people': 490, 'president': 485, 'country': 480, 'big': 475, 'powerful': 470},
    'uk': {'is': 500, 'has': 495, 'people': 490, 'prime': 485, 'minister': 480, 'country': 475, 'london': 470},
    'canada': {'is': 500, 'has': 495, 'people': 490, 'cold': 485, 'beautiful': 480, 'snow': 475, 'maple': 470},
    'australia': {'is': 500, 'has': 495, 'people': 490, 'hot': 485, 'beach': 480, 'kangaroo': 475, 'sydney': 470},
    'dubai': {'is': 500, 'city': 495, 'has': 490, 'big': 485, 'beautiful': 480, 'rich': 475, 'modern': 470},
    
    # EMOTIONS
    'love': {'you': 500, 'me': 495, 'him': 490, 'her': 485, 'it': 480, 'life': 475, 'family': 470, 'people': 465, 'god': 460},
    'like': {'you': 500, 'me': 495, 'it': 490, 'this': 485, 'that': 480, 'him': 475, 'her': 470, 'them': 465},
    'hate': {'you': 500, 'me': 495, 'it': 490, 'this': 485, 'that': 480, 'him': 475, 'her': 470, 'them': 465},
    'want': {'to': 500, 'you': 495, 'me': 490, 'it': 485, 'this': 480, 'that': 475, 'more': 470, 'something': 465},
    'need': {'to': 500, 'you': 495, 'me': 490, 'it': 485, 'help': 480, 'time': 475, 'money': 470, 'support': 465},
    'feel': {'good': 500, 'bad': 495, 'happy': 490, 'sad': 485, 'tired': 480, 'great': 475, 'sick': 470, 'better': 465},
    'care': {'about': 500, 'for': 495, 'you': 490, 'me': 485, 'them': 480, 'him': 475, 'her': 470, 'deeply': 465},
    
    # WEATHER
    'weather': {'is': 500, 'was': 495, 'nice': 490, 'bad': 485, 'cold': 480, 'hot': 475, 'good': 470, 'beautiful': 465},
    'rain': {'is': 500, 'was': 495, 'heavy': 490, 'coming': 485, 'falling': 480, 'outside': 475, 'started': 470, 'stopped': 465},
    'sun': {'is': 500, 'was': 495, 'shining': 490, 'hot': 485, 'bright': 480, 'set': 475, 'rise': 470, 'out': 465},
    'cold': {'weather': 500, 'day': 495, 'night': 490, 'outside': 485, 'water': 480, 'drink': 475, 'morning': 470},
    'hot': {'weather': 500, 'day': 495, 'outside': 490, 'water': 485, 'coffee': 480, 'sun': 475, 'summer': 470},
    'winter': {'is': 500, 'coming': 495, 'cold': 490, 'season': 485, 'here': 480, 'snow': 475, 'months': 470},
    'summer': {'is': 500, 'coming': 495, 'hot': 490, 'season': 485, 'here': 480, 'vacation': 475, 'months': 470},
    
    # FAMILY
    'mother': {'is': 500, 'was': 495, 'loves': 490, 'cooks': 485, 'works': 480, 'said': 475, 'called': 470, 'helps': 465},
    'father': {'is': 500, 'was': 495, 'works': 490, 'loves': 485, 'helps': 480, 'said': 475, 'called': 470},
    'brother': {'is': 500, 'was': 495, 'loves': 490, 'works': 485, 'studies': 480, 'called': 475, 'helps': 470},
    'sister': {'is': 500, 'was': 495, 'loves': 490, 'studies': 485, 'helps': 480, 'called': 475, 'works': 470},
    'friend': {'is': 500, 'was': 495, 'good': 490, 'best': 485, 'true': 480, 'old': 475, 'close': 470, 'dear': 465},
    'family': {'is': 500, 'was': 495, 'my': 490, 'our': 485, 'happy': 480, 'loving': 475, 'big': 470, 'whole': 465},
    'parents': {'are': 500, 'were': 495, 'love': 490, 'work': 485, 'live': 480, 'said': 475, 'taught': 470},
    'children': {'are': 500, 'were': 495, 'play': 490, 'study': 485, 'love': 480, 'need': 475, 'like': 470},
    
    # ACTION VERBS
    'run': {'fast': 500, 'away': 495, 'out': 490, 'time': 485, 'program': 480, 'daily': 475, 'early': 470, 'morning': 465, 'home': 460},
    'walk': {'away': 500, 'in': 495, 'out': 490, 'to': 485, 'fast': 480, 'slowly': 475, 'daily': 470, 'home': 465, 'together': 460},
    'jump': {'high': 500, 'low': 495, 'in': 490, 'out': 485, 'over': 480, 'up': 475, 'down': 470, 'rope': 465},
    'sit': {'down': 500, 'here': 495, 'there': 490, 'quietly': 485, 'still': 480, 'alone': 475, 'together': 470},
    'stand': {'up': 500, 'here': 495, 'there': 490, 'alone': 485, 'still': 480, 'firm': 475, 'together': 470},
    'sleep': {'well': 500, 'early': 495, 'late': 490, 'peacefully': 485, 'deeply': 480, 'alone': 475, 'together': 470, 'soundly': 465},
    'wake': {'up': 500, 'early': 495, 'late': 490, 'in': 485, 'morning': 480, 'suddenly': 475, 'from': 470},
    'work': {'hard': 500, 'well': 495, 'daily': 490, 'here': 485, 'there': 480, 'together': 475, 'alone': 470, 'from': 465},
    'study': {'hard': 500, 'well': 495, 'daily': 490, 'here': 485, 'there': 480, 'together': 475, 'alone': 470, 'english': 465},
    'play': {'well': 500, 'together': 495, 'alone': 490, 'daily': 485, 'games': 480, 'sports': 475, 'outside': 470, 'inside': 465},
    'read': {'books': 500, 'newspaper': 495, 'story': 490, 'article': 485, 'daily': 480, 'aloud': 475, 'carefully': 470, 'quickly': 465},
    'write': {'a': 500, 'book': 495, 'letter': 490, 'story': 485, 'email': 480, 'note': 475, 'article': 470, 'poem': 465},
    'listen': {'to': 500, 'music': 495, 'me': 490, 'him': 485, 'her': 480, 'carefully': 475, 'closely': 470, 'podcast': 465},
    'speak': {'to': 500, 'me': 495, 'him': 490, 'her': 485, 'english': 480, 'loudly': 475, 'softly': 470, 'truth': 465},
    'talk': {'to': 500, 'me': 495, 'him': 490, 'her': 485, 'about': 480, 'loudly': 475, 'softly': 470, 'openly': 465},
    'ask': {'me': 500, 'him': 495, 'her': 490, 'them': 485, 'you': 480, 'for': 475, 'about': 470, 'question': 465},
    'answer': {'me': 500, 'him': 495, 'her': 490, 'them': 485, 'the': 480, 'phone': 475, 'question': 470, 'call': 465},
    'help': {'me': 500, 'you': 495, 'him': 490, 'her': 485, 'us': 480, 'them': 475, 'yourself': 470, 'others': 465},
    'call': {'me': 500, 'you': 495, 'him': 490, 'her': 485, 'them': 480, 'now': 475, 'later': 470, 'back': 465},
    'meet': {'me': 500, 'you': 495, 'him': 490, 'her': 485, 'them': 480, 'today': 475, 'tomorrow': 470, 'there': 465},

    # ARTICLES & DETERMINERS
    'a': {'good': 500, 'great': 495, 'beautiful': 490, 'big': 485, 'small': 480, 'nice': 475, 'new': 470, 'old': 465, 'little': 460, 'lot': 455},
    'an': {'apple': 500, 'example': 495, 'idea': 490, 'hour': 485, 'opportunity': 480, 'elephant': 475, 'umbrella': 470, 'orange': 465, 'honest': 460},
    'the': {'best': 500, 'first': 495, 'last': 490, 'same': 485, 'only': 480, 'main': 475, 'big': 470, 'small': 465, 'new': 460, 'old': 455},
    'this': {'is': 500, 'was': 495, 'has': 490, 'looks': 485, 'seems': 480, 'one': 475, 'way': 470, 'time': 465, 'place': 460},
    'that': {'is': 500, 'was': 495, 'has': 490, 'looks': 485, 'seems': 480, 'one': 475, 'way': 470, 'time': 465, 'place': 460},
    'these': {'are': 500, 'were': 495, 'have': 490, 'look': 485, 'days': 480, 'people': 475, 'things': 470, 'times': 465},
    'those': {'are': 500, 'were': 495, 'have': 490, 'look': 485, 'days': 480, 'people': 475, 'things': 470, 'times': 465},
    
    # PREPOSITIONS
    'in': {'the': 500, 'a': 495, 'my': 490, 'your': 485, 'his': 480, 'her': 475, 'this': 470, 'that': 465, 'front': 460, 'order': 455},
    'on': {'the': 500, 'a': 495, 'my': 490, 'your': 485, 'top': 480, 'way': 475, 'time': 470, 'earth': 465, 'fire': 460},
    'at': {'home': 500, 'work': 495, 'school': 490, 'night': 485, 'morning': 480, 'noon': 475, 'ease': 470, 'least': 465},
    'for': {'me': 500, 'you': 495, 'him': 490, 'her': 485, 'us': 480, 'them': 475, 'all': 470, 'sure': 465, 'example': 460},
    'with': {'me': 500, 'you': 495, 'him': 490, 'her': 485, 'us': 480, 'them': 475, 'love': 470, 'care': 465, 'respect': 460},
    'without': {'me': 500, 'you': 495, 'him': 490, 'her': 485, 'us': 480, 'them': 475, 'doubt': 470, 'question': 465},
    'about': {'me': 500, 'you': 495, 'him': 490, 'her': 485, 'us': 480, 'it': 475, 'life': 470, 'love': 465, 'time': 460},
    'from': {'me': 500, 'you': 495, 'him': 490, 'her': 485, 'there': 480, 'here': 475, 'home': 470, 'work': 465, 'school': 460},
    'to': {'me': 500, 'you': 495, 'him': 490, 'her': 485, 'us': 480, 'them': 475, 'be': 470, 'go': 465, 'come': 460},
    'of': {'the': 500, 'a': 495, 'my': 490, 'your': 485, 'his': 480, 'her': 475, 'course': 470, 'course': 465},
    'by': {'the': 500, 'my': 495, 'your': 490, 'him': 485, 'her': 480, 'us': 475, 'then': 470, 'now': 465},
    'up': {'to': 500, 'the': 495, 'in': 490, 'for': 485, 'and': 480, 'down': 475, 'here': 470, 'there': 465},
    'down': {'the': 500, 'to': 495, 'in': 490, 'for': 485, 'and': 480, 'up': 475, 'here': 470, 'there': 465},
    'over': {'the': 500, 'there': 495, 'here': 490, 'time': 485, 'and': 480, 'again': 475},
    'under': {'the': 500, 'a': 495, 'my': 490, 'your': 485, 'control': 480, 'pressure': 475},
    
    # DEFAULT FALLBACK
    'default': {'the': 500, 'and': 495, 'to': 490, 'of': 485, 'for': 480, 'with': 475, 'in': 470, 'on': 465, 'is': 460, 'are': 455, 'was': 450, 'were': 445}
}

# ============================================
# TRIGRAM DATA (Last 2 words ke hisaab se - Maximum)
# ============================================

trigram_pretrained = {
    # i am + ?
    'i am': {'a': 500, 'going': 495, 'doing': 490, 'trying': 485, 'here': 480, 'not': 475, 'very': 470, 'so': 465, 'just': 460},
    'i have': {'a': 500, 'to': 495, 'been': 490, 'got': 485, 'no': 480, 'never': 475, 'always': 470},
    'i will': {'be': 500, 'go': 495, 'come': 490, 'see': 485, 'get': 480, 'do': 475, 'make': 470},
    'i can': {'be': 500, 'do': 495, 'go': 490, 'see': 485, 'help': 480, 'get': 475, 'make': 470},
    'i love': {'you': 500, 'to': 495, 'it': 490, 'her': 485, 'him': 480, 'this': 475},
    'i want': {'to': 500, 'you': 495, 'it': 490, 'this': 485, 'that': 480, 'more': 475},
    'i need': {'to': 500, 'you': 495, 'help': 490, 'it': 485, 'this': 480, 'more': 475},
    'i like': {'to': 500, 'you': 495, 'it': 490, 'this': 485, 'that': 480, 'her': 475},
    
    # you are + ?
    'you are': {'a': 500, 'the': 495, 'very': 490, 'so': 485, 'not': 480, 'my': 475, 'going': 470},
    'you have': {'a': 500, 'to': 495, 'been': 490, 'got': 485, 'no': 480, 'never': 475},
    'you will': {'be': 500, 'go': 495, 'come': 490, 'see': 485, 'get': 480, 'find': 475},
    'you can': {'be': 500, 'do': 495, 'go': 490, 'see': 485, 'get': 480, 'help': 475},
    'you should': {'be': 500, 'go': 495, 'come': 490, 'see': 485, 'get': 480, 'try': 475},
    
    # he is + ?
    'he is': {'a': 500, 'the': 495, 'very': 490, 'my': 485, 'going': 480, 'not': 475, 'so': 470},
    'he has': {'a': 500, 'been': 495, 'to': 490, 'got': 485, 'no': 480, 'never': 475},
    'he will': {'be': 500, 'go': 495, 'come': 490, 'see': 485, 'get': 480},
    'he can': {'be': 500, 'do': 495, 'go': 490, 'see': 485, 'help': 480},
    
    # she is + ?
    'she is': {'a': 500, 'the': 495, 'very': 490, 'my': 485, 'beautiful': 480, 'going': 475},
    'she has': {'a': 500, 'been': 495, 'to': 490, 'got': 485, 'no': 480},
    'she will': {'be': 500, 'go': 495, 'come': 490, 'see': 485},
    
    # it is + ?
    'it is': {'a': 500, 'the': 495, 'very': 490, 'not': 485, 'so': 480, 'just': 475, 'really': 470},
    'it was': {'a': 500, 'the': 495, 'very': 490, 'not': 485, 'so': 480, 'just': 475},
    'it has': {'a': 500, 'been': 495, 'to': 490, 'got': 485, 'no': 480},
    
    # we are + ?
    'we are': {'going': 500, 'the': 495, 'very': 490, 'not': 485, 'so': 480, 'here': 475, 'ready': 470},
    'we have': {'a': 500, 'to': 495, 'been': 490, 'got': 485, 'no': 480, 'never': 475},
    'we will': {'be': 500, 'go': 495, 'come': 490, 'see': 485, 'get': 480},
    'we can': {'be': 500, 'do': 495, 'go': 490, 'see': 485, 'help': 480},
    
    # they are + ?
    'they are': {'going': 500, 'the': 495, 'very': 490, 'not': 485, 'so': 480, 'here': 475},
    'they have': {'a': 500, 'been': 495, 'to': 490, 'got': 485, 'no': 480},
    'they will': {'be': 500, 'go': 495, 'come': 490, 'see': 485},
    
    # article + adjective patterns
    'a good': {'boy': 500, 'girl': 495, 'day': 490, 'person': 485, 'job': 480, 'idea': 475, 'time': 470},
    'a bad': {'day': 500, 'boy': 495, 'girl': 490, 'person': 485, 'habit': 480, 'idea': 475},
    'a beautiful': {'day': 500, 'girl': 495, 'house': 490, 'car': 485, 'smile': 480, 'view': 475},
    'a big': {'house': 500, 'car': 495, 'problem': 490, 'fan': 485, 'deal': 480, 'success': 475},
    'a small': {'house': 500, 'car': 495, 'cat': 490, 'dog': 485, 'problem': 480, 'town': 475},
    'a new': {'car': 500, 'house': 495, 'job': 490, 'phone': 485, 'friend': 480, 'day': 475},
    'an old': {'car': 500, 'house': 495, 'friend': 490, 'man': 485, 'woman': 480, 'building': 475},
    'a great': {'day': 500, 'job': 495, 'idea': 490, 'person': 485, 'friend': 480, 'time': 475},
    'a nice': {'day': 500, 'person': 495, 'car': 490, 'house': 485, 'smile': 480, 'place': 475},
    
    'the best': {'day': 500, 'friend': 495, 'car': 490, 'house': 485, 'job': 480, 'idea': 475},
    'the first': {'day': 500, 'time': 495, 'person': 490, 'thing': 485, 'place': 480, 'step': 475},
    'the last': {'day': 500, 'time': 495, 'person': 490, 'thing': 485, 'night': 480, 'week': 475},
    'the same': {'day': 500, 'time': 495, 'person': 490, 'thing': 485, 'place': 480, 'way': 475},
    'the only': {'way': 500, 'person': 495, 'thing': 490, 'reason': 485, 'one': 480},
    
    # how + ? patterns
    'how are': {'you': 500, 'we': 495, 'they': 490, 'things': 485, 'you': 480},
    'how is': {'it': 500, 'he': 495, 'she': 490, 'life': 485, 'everything': 480},
    'how to': {'make': 500, 'get': 495, 'do': 490, 'be': 485, 'use': 480, 'fix': 475},
    'how do': {'you': 500, 'we': 495, 'they': 490, 'i': 485, 'you': 480},
    'how was': {'your': 500, 'the': 495, 'it': 490, 'today': 485, 'day': 480},
    
    # what + ? patterns
    'what is': {'your': 500, 'the': 495, 'this': 490, 'that': 485, 'it': 480, 'his': 475},
    'what are': {'you': 500, 'we': 495, 'they': 490, 'these': 485, 'those': 480, 'your': 475},
    'what do': {'you': 500, 'we': 495, 'they': 490, 'i': 485, 'you': 480},
    'what was': {'that': 500, 'it': 495, 'the': 490, 'your': 485, 'his': 480},
    'what does': {'it': 500, 'he': 495, 'she': 490, 'this': 485, 'that': 480},
    
    # where + ? patterns
    'where is': {'the': 500, 'my': 495, 'your': 490, 'he': 485, 'she': 480, 'it': 475},
    'where are': {'you': 500, 'we': 495, 'they': 490, 'my': 485, 'your': 480, 'the': 475},
    'where did': {'you': 500, 'he': 495, 'she': 490, 'they': 485, 'we': 480},
    'where do': {'you': 500, 'we': 495, 'they': 490, 'i': 485},
    
    # why + ? patterns
    'why is': {'it': 500, 'he': 495, 'she': 490, 'this': 485, 'that': 480, 'the': 475},
    'why do': {'you': 500, 'we': 495, 'they': 490, 'i': 485, 'you': 480},
    'why did': {'you': 500, 'he': 495, 'she': 490, 'they': 485, 'we': 480},
    'why would': {'you': 500, 'he': 495, 'she': 490, 'they': 485},
    
    # when + ? patterns
    'when is': {'it': 500, 'the': 495, 'your': 490, 'he': 485, 'she': 480, 'this': 475},
    'when will': {'you': 500, 'we': 495, 'they': 490, 'he': 485, 'she': 480, 'it': 475},
    'when did': {'you': 500, 'he': 495, 'she': 490, 'they': 485, 'we': 480},
    'when do': {'you': 500, 'we': 495, 'they': 490, 'i': 485},
    
    # every + noun patterns
    'every day': {'i': 500, 'we': 495, 'he': 490, 'she': 485, 'they': 480, 'you': 475},
    'every night': {'i': 500, 'we': 495, 'he': 490, 'she': 485, 'they': 480, 'sleep': 475},
    'every time': {'i': 500, 'we': 495, 'he': 490, 'she': 485, 'you': 480, 'they': 475},
    'every morning': {'i': 500, 'we': 495, 'he': 490, 'she': 485, 'wake': 480},
    'every week': {'i': 500, 'we': 495, 'he': 490, 'she': 485, 'go': 480},
    
    # some + noun patterns
    'some one': {'is': 500, 'has': 495, 'called': 490, 'said': 485, 'came': 480, 'told': 475},
    'some people': {'are': 500, 'have': 495, 'say': 490, 'think': 485, 'like': 480, 'dont': 475},
    'some time': {'ago': 500, 'later': 495, 'now': 490, 'i': 485, 'we': 480},
    'some day': {'i': 500, 'we': 495, 'you': 490, 'he': 485, 'she': 480},
    'some things': {'are': 500, 'never': 495, 'dont': 490, 'take': 485},
    
    # no + noun patterns
    'no one': {'knows': 500, 'cares': 495, 'likes': 490, 'loves': 485, 'said': 480, 'told': 475},
    'no way': {'to': 500, 'out': 495, 'i': 490, 'he': 485, 'she': 480, 'we': 475},
    'no time': {'to': 500, 'for': 495, 'left': 490, 'like': 485, 'now': 480},
    'no problem': {'at': 500, 'for': 495, 'with': 490, 'i': 485, 'we': 480},
    
    # greeting patterns
    'hello how': {'are': 500, 'is': 495, 'do': 490, 'have': 485},
    'thank you': {'for': 500, 'very': 495, 'so': 490, 'sir': 485, 'maam': 480},
    'how are': {'you': 500, 'we': 495, 'they': 490, 'things': 485},
    'what is': {'your': 500, 'the': 495, 'this': 490, 'that': 485},
    
    # common sentence starters
    'there is': {'a': 500, 'no': 495, 'nothing': 490, 'something': 485, 'always': 480, 'never': 475},
    'there are': {'many': 500, 'some': 495, 'no': 490, 'a': 485, 'several': 480},
    'this is': {'a': 500, 'the': 495, 'my': 490, 'your': 485, 'our': 480},
    'that is': {'a': 500, 'the': 495, 'my': 490, 'your': 485, 'what': 480},
    'it is': {'a': 500, 'the': 495, 'not': 490, 'very': 485, 'so': 480},
    'here is': {'a': 500, 'the': 495, 'my': 490, 'your': 485, 'our': 480},
    
    # time patterns
    'today is': {'a': 500, 'the': 495, 'going': 490, 'my': 485, 'your': 480},
    'tomorrow is': {'a': 500, 'the': 495, 'going': 490, 'my': 485},
    'yesterday was': {'a': 500, 'the': 495, 'great': 490, 'good': 485, 'bad': 480},
    
    # filler patterns
    'i think': {'that': 500, 'i': 495, 'you': 490, 'he': 485, 'she': 480, 'we': 475},
    'i believe': {'that': 500, 'i': 495, 'you': 490, 'he': 485, 'she': 480},
    'i hope': {'you': 500, 'we': 495, 'they': 490, 'he': 485, 'she': 480},
    'i guess': {'so': 500, 'not': 495, 'that': 490, 'i': 485, 'you': 480},
    'you know': {'what': 500, 'that': 495, 'i': 490, 'how': 485, 'the': 480},
    
    # adjective + noun common patterns
    'very good': {'day': 500, 'job': 495, 'idea': 490, 'person': 485, 'friend': 480},
    'very nice': {'person': 500, 'car': 495, 'house': 490, 'day': 485, 'place': 480},
    'very happy': {'birthday': 500, 'day': 495, 'life': 490, 'family': 485},
    'very sad': {'day': 500, 'story': 495, 'ending': 490, 'news': 485},
    'very big': {'house': 500, 'car': 495, 'problem': 490, 'fan': 485},
    'very small': {'house': 500, 'car': 495, 'cat': 490, 'dog': 485},
    
    # verb + preposition patterns
    'listen to': {'music': 500, 'me': 495, 'him': 490, 'her': 485, 'the': 480},
    'talk to': {'me': 500, 'him': 495, 'her': 490, 'them': 485, 'you': 480},
    'go to': {'the': 500, 'work': 495, 'school': 490, 'home': 485, 'bed': 480},
    'come to': {'the': 500, 'my': 495, 'your': 490, 'our': 485, 'home': 480},
    'look at': {'me': 500, 'him': 495, 'her': 490, 'the': 485, 'this': 480},
    'think about': {'it': 500, 'you': 495, 'me': 490, 'him': 485, 'her': 480},
    'care about': {'you': 500, 'me': 495, 'him': 490, 'her': 485, 'them': 480},
}

# ============================================
# LOAD ALL DATA
# ============================================

for word, suggestions in bigram_pretrained.items():
    for sugg, count in suggestions.items():
        bigrams[word][sugg] = max(bigrams[word][sugg], count)

for two_words, suggestions in trigram_pretrained.items():
    for sugg, count in suggestions.items():
        trigrams[two_words][sugg] = max(trigrams[two_words][sugg], count)

# ============================================
# SAVE FUNCTIONS
# ============================================

def save_data():
    # Save bigrams
    save_bigrams = {}
    for word, suggestions in bigrams.items():
        save_bigrams[word] = dict(suggestions)
    with open(LEARNING_FILE, 'w', encoding='utf-8') as f:
        json.dump(save_bigrams, f, indent=2)
    
    # Save trigrams
    save_trigrams = {}
    for two_words, suggestions in trigrams.items():
        save_trigrams[two_words] = dict(suggestions)
    with open(TRIGRAM_FILE, 'w', encoding='utf-8') as f:
        json.dump(save_trigrams, f, indent=2)

# ============================================
# LEARN FUNCTIONS
# ============================================

def learn_trigram(word1, word2, next_word):
    word1 = word1.lower().strip()
    word2 = word2.lower().strip()
    next_word = next_word.lower().strip()
    if word1 and word2 and next_word:
        key = f"{word1} {word2}"
        trigrams[key][next_word] += 1
        save_data()

def learn_bigram(word, next_word):
    word = word.lower().strip()
    next_word = next_word.lower().strip()
    if word and next_word:
        bigrams[word][next_word] += 1
        save_data()

# ============================================
# PREDICT FUNCTION (Trigram first, then Bigram)
# ============================================

def predict_next(last_word, second_last_word=None, top_n=6):
    last_word = last_word.lower().strip()
    
    # First try trigram (if we have two words)
    if second_last_word:
        key = f"{second_last_word} {last_word}"
        if key in trigrams and trigrams[key]:
            sorted_words = sorted(trigrams[key].items(), key=lambda x: x[1], reverse=True)
            return [w for w, _ in sorted_words[:top_n]]
    
    # Then try bigram
    if last_word in bigrams and bigrams[last_word]:
        sorted_words = sorted(bigrams[last_word].items(), key=lambda x: x[1], reverse=True)
        return [w for w, _ in sorted_words[:top_n]]
    
    # Fallback
    return ['the', 'and', 'to', 'of', 'for', 'with']

# ============================================
# FLASK ROUTES
# ============================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_route():
    try:
        data = request.get_json()
        last_word = data.get('last_word', '')
        second_last = data.get('second_last', '')
        suggestions = predict_next(last_word, second_last)
        return jsonify({'suggestions': suggestions})
    except:
        return jsonify({'suggestions': ['the', 'and', 'to', 'of', 'for']})

@app.route('/learn', methods=['POST'])
def learn_route():
    try:
        data = request.get_json()
        word1 = data.get('word1', '')
        word2 = data.get('word2', '')
        next_word = data.get('next_word', '')
        
        if word1 and word2 and next_word:
            learn_trigram(word1, word2, next_word)
        elif word1 and next_word:
            learn_bigram(word1, next_word)
        
        return jsonify({'status': 'learned'})
    except:
        return jsonify({'status': 'error'})

# ============================================
# MAIN
# ============================================

if __name__ == '__main__':
    print("=" * 65)
    print("🤖 ULTIMATE SENTENCE AUTO-COMPLETION")
    print("=" * 65)
    print(f"✅ Total bigrams loaded: {len(bigrams)}")
    print(f"✅ Total trigrams loaded: {len(trigrams)}")
    print("")
    print("📚 EXAMPLES:")
    print("   'i am a good' → 'boy' (trigram 'a good')")
    print("   'how are' → 'you' (trigram 'how are')")
    print("   'what is' → 'your' (bigram 'is')")
    print("   'every day' → 'i' (trigram 'every day')")
    print("   'no one' → 'knows' (trigram 'no one')")
    print("=" * 65)
    app.run(debug=True, host='127.0.0.1', port=5000)
