from flask import Flask, render_template, request, jsonify
from collections import defaultdict
import json
import os

app = Flask(__name__)

LEARNING_FILE = 'learned_pairs.json'

if os.path.exists(LEARNING_FILE):
    with open(LEARNING_FILE, 'r', encoding='utf-8') as f:
        learned_data = json.load(f)
    bigrams = defaultdict(lambda: defaultdict(int))
    for word, suggestions in learned_data.items():
        for sugg, count in suggestions.items():
            bigrams[word][sugg] = count
else:
    bigrams = defaultdict(lambda: defaultdict(int))

# ============================================
# MEGA DATASET (2000+ COMMON WORDS)
# ============================================

pretrained = {
    # ===== PRONOUNS (All forms) =====
    'i': {'am': 500, 'have': 490, 'will': 480, 'can': 470, 'love': 460, 'like': 450, 'want': 440, 'need': 430, 'did': 420, 'would': 410, 'could': 400, 'should': 390, 'might': 380, 'must': 370, 'do': 360, 'dont': 350},
    'you': {'are': 500, 'have': 490, 'will': 480, 'can': 470, 'need': 460, 'want': 450, 'like': 440, 'did': 430, 'would': 420, 'could': 410, 'should': 400, 'do': 390, 'dont': 380},
    'he': {'is': 500, 'has': 490, 'was': 480, 'does': 470, 'will': 460, 'can': 450, 'would': 440, 'could': 430, 'should': 420, 'did': 410},
    'she': {'is': 500, 'has': 490, 'was': 480, 'does': 470, 'will': 460, 'can': 450, 'would': 440, 'could': 430, 'should': 420},
    'it': {'is': 500, 'has': 490, 'was': 480, 'looks': 470, 'seems': 460, 'works': 450, 'does': 440, 'will': 430, 'can': 420},
    'we': {'are': 500, 'have': 490, 'will': 480, 'can': 470, 'should': 460, 'must': 450, 'would': 440, 'could': 430, 'need': 420},
    'they': {'are': 500, 'have': 490, 'will': 480, 'can': 470, 'should': 460, 'must': 450, 'would': 440, 'could': 430, 'need': 420},
    'my': {'name': 500, 'friend': 490, 'car': 480, 'house': 470, 'life': 460, 'love': 450, 'heart': 440, 'mind': 430, 'family': 420},
    'your': {'name': 500, 'friend': 490, 'car': 480, 'house': 470, 'life': 460, 'love': 450, 'help': 440, 'support': 430},
    'his': {'name': 500, 'friend': 490, 'car': 480, 'house': 470, 'life': 460, 'work': 450, 'family': 440},
    'her': {'name': 500, 'friend': 490, 'car': 480, 'house': 470, 'life': 460, 'smile': 450, 'eyes': 440},
    'our': {'home': 500, 'life': 490, 'family': 480, 'country': 470, 'world': 460, 'future': 450, 'children': 440},
    'their': {'home': 500, 'life': 490, 'family': 480, 'country': 470, 'children': 460, 'future': 450},
    'me': {'too': 500, 'please': 490, 'now': 480, 'later': 470, 'either': 460, 'neither': 450, 'also': 440},
    'him': {'too': 500, 'please': 490, 'now': 480, 'later': 470, 'either': 460, 'also': 450},
    'her': {'too': 500, 'please': 490, 'now': 480, 'later': 470, 'either': 460, 'also': 450},
    'us': {'too': 500, 'please': 490, 'now': 480, 'later': 470, 'all': 460, 'both': 450},
    'them': {'too': 500, 'please': 490, 'now': 480, 'later': 470, 'all': 460, 'both': 450},
    
    # ===== VERBS (All tenses) =====
    'am': {'going': 500, 'doing': 490, 'trying': 480, 'here': 470, 'not': 460, 'very': 450, 'so': 440, 'just': 430, 'still': 420, 'always': 410},
    'is': {'a': 500, 'the': 490, 'going': 480, 'very': 470, 'not': 460, 'my': 450, 'your': 440, 'his': 430, 'her': 420, 'it': 410, 'this': 400},
    'are': {'you': 500, 'we': 490, 'they': 480, 'going': 470, 'not': 460, 'the': 450, 'a': 440, 'there': 430, 'here': 420, 'so': 410},
    'was': {'a': 500, 'the': 490, 'very': 480, 'going': 470, 'not': 460, 'my': 450, 'his': 440, 'her': 430, 'so': 420, 'just': 410},
    'were': {'you': 500, 'they': 490, 'we': 480, 'not': 470, 'very': 460, 'so': 450, 'there': 440, 'here': 430, 'all': 420},
    'be': {'a': 500, 'the': 490, 'very': 480, 'good': 470, 'nice': 460, 'happy': 450, 'sad': 440, 'there': 430},
    'been': {'a': 500, 'the': 490, 'very': 480, 'so': 470, 'too': 460, 'there': 450, 'here': 440},
    'have': {'a': 500, 'to': 490, 'been': 480, 'got': 470, 'you': 460, 'it': 450, 'this': 440, 'no': 430, 'many': 420},
    'has': {'a': 500, 'been': 490, 'to': 480, 'got': 470, 'it': 460, 'this': 450, 'no': 440, 'many': 430},
    'had': {'a': 500, 'been': 490, 'to': 480, 'got': 470, 'no': 460, 'it': 450, 'this': 440},
    'do': {'you': 500, 'it': 490, 'not': 480, 'this': 470, 'your': 460, 'my': 450, 'we': 440, 'they': 430, 'i': 420},
    'does': {'it': 500, 'this': 490, 'not': 480, 'he': 470, 'she': 460, 'anyone': 450, 'everyone': 440},
    'did': {'you': 500, 'it': 490, 'this': 480, 'he': 470, 'she': 460, 'not': 450, 'we': 440, 'they': 430, 'i': 420},
    'can': {'you': 500, 'i': 490, 'we': 480, 'they': 470, 'he': 460, 'she': 450, 'be': 440, 'do': 430, 'get': 420, 'see': 410},
    'could': {'be': 500, 'you': 490, 'i': 480, 'we': 470, 'they': 460, 'have': 450, 'get': 440, 'do': 430},
    'would': {'be': 500, 'you': 490, 'i': 480, 'like': 470, 'love': 460, 'want': 450, 'have': 440, 'do': 430},
    'should': {'be': 500, 'you': 490, 'we': 480, 'they': 470, 'have': 460, 'go': 450, 'do': 440, 'get': 430},
    'might': {'be': 500, 'have': 490, 'need': 480, 'want': 470, 'go': 460, 'come': 450},
    'must': {'be': 500, 'have': 490, 'go': 480, 'do': 470, 'get': 460, 'see': 450},
    'will': {'be': 500, 'go': 490, 'come': 480, 'see': 470, 'get': 460, 'have': 450, 'do': 440, 'make': 430, 'take': 420},
    'go': {'to': 500, 'home': 490, 'out': 480, 'in': 470, 'there': 460, 'now': 450, 'away': 440, 'back': 430},
    'come': {'here': 500, 'home': 490, 'back': 480, 'in': 470, 'to': 460, 'over': 450, 'out': 440},
    'see': {'you': 500, 'it': 490, 'him': 480, 'her': 470, 'them': 460, 'me': 450, 'us': 440, 'the': 430},
    'get': {'it': 500, 'out': 490, 'in': 480, 'up': 470, 'down': 460, 'there': 450, 'home': 440, 'back': 430},
    'make': {'it': 500, 'sure': 490, 'money': 480, 'love': 470, 'mistake': 460, 'sense': 450, 'time': 440},
    'take': {'it': 500, 'care': 490, 'time': 480, 'out': 470, 'in': 460, 'a': 450, 'the': 440},
    'give': {'me': 500, 'it': 490, 'you': 480, 'him': 470, 'her': 460, 'us': 450, 'them': 440, 'up': 430},
    'put': {'it': 500, 'on': 490, 'in': 480, 'out': 470, 'down': 460, 'up': 450, 'there': 440},
    'keep': {'it': 500, 'going': 490, 'up': 480, 'safe': 470, 'quiet': 460, 'calm': 450, 'trying': 440},
    
    # ===== EVERY, SOME, ANY, NO, ALL, MANY, MUCH, MORE, MOST, LEAST =====
    'every': {'day': 500, 'night': 490, 'time': 480, 'one': 470, 'person': 460, 'morning': 450, 'week': 440, 'month': 430, 'year': 420},
    'everyone': {'is': 500, 'has': 490, 'knows': 480, 'loves': 470, 'needs': 460, 'wants': 450, 'likes': 440, 'agrees': 430},
    'everything': {'is': 500, 'was': 490, 'has': 480, 'looks': 470, 'seems': 460, 'okay': 450, 'fine': 440, 'perfect': 430},
    'everywhere': {'is': 500, 'was': 490, 'looks': 480, 'goes': 470, 'feels': 460, 'seems': 450},
    'everybody': {'is': 500, 'has': 490, 'knows': 480, 'loves': 470, 'needs': 460, 'wants': 450},
    'some': {'one': 500, 'people': 490, 'time': 480, 'day': 470, 'thing': 460, 'where': 450, 'money': 440, 'help': 430},
    'someone': {'is': 500, 'has': 490, 'knows': 480, 'loves': 470, 'called': 460, 'said': 450, 'told': 440},
    'something': {'is': 500, 'was': 490, 'has': 480, 'looks': 470, 'happened': 460, 'changed': 450, 'wrong': 440},
    'somewhere': {'is': 500, 'was': 490, 'over': 480, 'in': 470, 'out': 460, 'else': 450, 'near': 440},
    'any': {'one': 500, 'time': 490, 'day': 480, 'person': 470, 'thing': 460, 'where': 450, 'help': 440, 'idea': 430},
    'anyone': {'is': 500, 'has': 490, 'knows': 480, 'can': 470, 'will': 460, 'could': 450, 'should': 440},
    'anything': {'is': 500, 'was': 490, 'can': 480, 'will': 470, 'possible': 460, 'else': 450, 'wrong': 440},
    'anywhere': {'is': 500, 'was': 490, 'can': 480, 'will': 470, 'else': 460, 'near': 450},
    'anybody': {'is': 500, 'has': 490, 'knows': 480, 'can': 470, 'will': 460},
    'no': {'one': 500, 'time': 490, 'way': 480, 'problem': 470, 'idea': 460, 'where': 450, 'money': 440, 'help': 430},
    'no one': {'is': 500, 'knows': 490, 'cares': 480, 'loves': 470, 'likes': 460, 'helps': 450},
    'nothing': {'is': 500, 'was': 490, 'has': 480, 'matters': 470, 'changed': 460, 'else': 450, 'wrong': 440},
    'nowhere': {'to': 500, 'in': 490, 'near': 480, 'else': 470, 'found': 460},
    'nobody': {'is': 500, 'knows': 490, 'cares': 480, 'loves': 470, 'likes': 460},
    'all': {'the': 500, 'my': 490, 'your': 480, 'his': 470, 'her': 460, 'of': 450, 'these': 440, 'those': 430},
    'most': {'of': 500, 'the': 490, 'people': 480, 'important': 470, 'common': 460, 'popular': 450, 'likely': 440},
    'many': {'people': 500, 'times': 490, 'years': 480, 'days': 470, 'things': 460, 'ways': 450, 'reasons': 440},
    'much': {'more': 500, 'better': 490, 'less': 480, 'love': 470, 'time': 460, 'money': 450, 'help': 440},
    'more': {'than': 500, 'people': 490, 'time': 480, 'money': 470, 'important': 460, 'common': 450, 'likely': 440},
    'less': {'than': 500, 'more': 490, 'time': 480, 'money': 470, 'important': 460, 'common': 450},
    'least': {'of': 500, 'the': 490, 'important': 480, 'common': 470, 'likely': 460},
    
    # ===== ADJECTIVES (Common) =====
    'good': {'morning': 500, 'day': 490, 'luck': 480, 'job': 470, 'idea': 460, 'person': 450, 'time': 440, 'night': 430, 'friend': 420, 'work': 410},
    'bad': {'day': 500, 'luck': 490, 'idea': 480, 'person': 470, 'situation': 460, 'weather': 450, 'habit': 440, 'news': 430, 'dream': 420},
    'great': {'day': 500, 'job': 490, 'idea': 480, 'work': 470, 'person': 460, 'friend': 450, 'time': 440, 'success': 430, 'life': 420},
    'nice': {'day': 500, 'person': 490, 'car': 480, 'house': 470, 'weather': 460, 'smile': 450, 'place': 440, 'view': 430},
    'beautiful': {'day': 500, 'girl': 490, 'house': 480, 'car': 470, 'smile': 460, 'view': 450, 'place': 440, 'weather': 430, 'face': 420},
    'pretty': {'good': 500, 'nice': 490, 'girl': 480, 'face': 470, 'dress': 460, 'house': 450, 'car': 440, 'well': 430},
    'ugly': {'truth': 500, 'face': 490, 'situation': 480, 'reality': 470, 'building': 460, 'car': 450, 'person': 440},
    'happy': {'birthday': 500, 'day': 490, 'life': 480, 'ending': 470, 'moment': 460, 'smile': 450, 'family': 440, 'face': 430},
    'sad': {'day': 500, 'story': 490, 'ending': 480, 'news': 470, 'person': 460, 'face': 450, 'truth': 440, 'reality': 430},
    'angry': {'at': 500, 'with': 490, 'about': 480, 'person': 470, 'voice': 460, 'face': 450, 'mob': 440},
    'excited': {'about': 500, 'for': 490, 'to': 480, 'today': 470, 'tomorrow': 460, 'see': 450, 'meet': 440},
    'scared': {'of': 500, 'about': 490, 'to': 480, 'death': 470, 'dark': 460, 'lonely': 450, 'alone': 440},
    'tired': {'of': 500, 'from': 490, 'today': 480, 'very': 470, 'so': 460, 'too': 450, 'now': 440},
    'big': {'house': 500, 'car': 490, 'problem': 480, 'success': 470, 'day': 460, 'fan': 450, 'deal': 440, 'man': 430, 'city': 420},
    'small': {'house': 500, 'car': 490, 'problem': 480, 'thing': 470, 'cat': 460, 'dog': 450, 'town': 440, 'city': 430},
    'large': {'house': 500, 'car': 490, 'problem': 480, 'company': 470, 'city': 460, 'number': 450, 'amount': 440},
    'new': {'car': 500, 'house': 490, 'job': 480, 'phone': 470, 'friend': 460, 'day': 450, 'idea': 440, 'life': 430, 'year': 420},
    'old': {'car': 500, 'house': 490, 'friend': 480, 'phone': 470, 'man': 460, 'woman': 450, 'days': 440, 'times': 430},
    'young': {'man': 500, 'woman': 490, 'boy': 480, 'girl': 470, 'person': 460, 'age': 450, 'people': 440},
    'rich': {'man': 500, 'person': 490, 'family': 480, 'country': 470, 'people': 460, 'life': 450},
    'poor': {'man': 500, 'person': 490, 'family': 480, 'country': 470, 'people': 460, 'life': 450},
    'strong': {'man': 500, 'person': 490, 'feeling': 480, 'will': 470, 'body': 460, 'sense': 450},
    'weak': {'man': 500, 'person': 490, 'signal': 480, 'feeling': 470, 'health': 460, 'link': 450},
    'fast': {'car': 500, 'food': 490, 'runner': 480, 'internet': 470, 'speed': 460, 'pace': 450, 'learning': 440},
    'slow': {'car': 500, 'internet': 490, 'runner': 480, 'speed': 470, 'day': 460, 'process': 450, 'learning': 440},
    'easy': {'to': 500, 'way': 490, 'task': 480, 'job': 470, 'work': 460, 'life': 450, 'answer': 440},
    'hard': {'to': 500, 'work': 490, 'task': 480, 'job': 470, 'time': 460, 'life': 450, 'decision': 440},
    'soft': {'and': 500, 'voice': 490, 'touch': 480, 'skin': 470, 'pillow': 460, 'drink': 450},
    'dark': {'night': 500, 'room': 490, 'side': 480, 'sky': 470, 'cloud': 460, 'color': 450},
    'light': {'and': 500, 'weight': 490, 'color': 480, 'blue': 470, 'green': 460, 'house': 450},
    
    # ===== GREETINGS & POLITE =====
    'hello': {'world': 500, 'everyone': 490, 'friends': 480, 'dear': 470, 'sir': 460, 'there': 450, 'how': 440, 'my': 430},
    'hi': {'there': 500, 'everyone': 490, 'friends': 480, 'how': 470, 'hello': 460, 'my': 450},
    'hey': {'there': 500, 'everyone': 490, 'how': 480, 'whats': 470, 'hi': 460, 'you': 450},
    'goodbye': {'everyone': 500, 'friends': 490, 'dear': 480, 'sir': 470, 'now': 460, 'for': 450},
    'bye': {'everyone': 500, 'friends': 490, 'dear': 480, 'now': 470, 'see': 460, 'for': 450},
    'thank': {'you': 500, 'god': 490, 'sir': 480, 'everyone': 470, 'all': 460, 'so': 450, 'very': 440},
    'thanks': {'you': 500, 'god': 490, 'sir': 480, 'everyone': 470, 'for': 460, 'so': 450},
    'please': {'help': 500, 'tell': 490, 'come': 480, 'go': 470, 'wait': 460, 'sit': 450, 'stand': 440, 'be': 430},
    'sorry': {'for': 500, 'about': 490, 'dear': 480, 'everyone': 470, 'sir': 460, 'to': 450, 'i': 440},
    'excuse': {'me': 500, 'us': 490, 'them': 480, 'him': 470, 'her': 460, 'sir': 450},
    'welcome': {'to': 500, 'home': 490, 'our': 480, 'my': 470, 'the': 460, 'everyone': 450},
    
    # ===== QUESTION WORDS =====
    'what': {'is': 500, 'are': 490, 'was': 480, 'do': 470, 'does': 460, 'about': 450, 'happened': 440, 'did': 430, 'can': 420},
    'where': {'is': 500, 'are': 490, 'was': 480, 'do': 470, 'did': 460, 'have': 450, 'can': 440, 'should': 430},
    'when': {'is': 500, 'are': 490, 'will': 480, 'did': 470, 'was': 460, 'does': 450, 'can': 440},
    'why': {'is': 500, 'are': 490, 'do': 480, 'did': 470, 'would': 460, 'not': 450, 'should': 440},
    'how': {'are': 500, 'is': 490, 'to': 480, 'do': 470, 'about': 460, 'many': 450, 'much': 440, 'long': 430, 'often': 420},
    'who': {'is': 500, 'are': 490, 'was': 480, 'did': 470, 'will': 460, 'can': 450, 'should': 440},
    'which': {'one': 500, 'is': 490, 'are': 480, 'way': 470, 'time': 460, 'place': 450, 'of': 440},
    'whom': {'are': 500, 'is': 490, 'was': 480, 'did': 470, 'will': 460},
    'whose': {'is': 500, 'are': 490, 'was': 480, 'this': 470, 'that': 460},
    
    # ===== TIME WORDS =====
    'today': {'is': 500, 'was': 490, 'i': 480, 'we': 470, 'going': 460, 'will': 450, 'feels': 440, 'looks': 430},
    'tomorrow': {'is': 500, 'will': 490, 'morning': 480, 'night': 470, 'i': 460, 'we': 450, 'be': 440, 'comes': 430},
    'yesterday': {'was': 500, 'i': 490, 'we': 480, 'he': 470, 'she': 460, 'they': 450, 'went': 440, 'came': 430},
    'now': {'i': 500, 'we': 490, 'is': 480, 'its': 470, 'go': 460, 'come': 450, 'time': 440, 'or': 430},
    'later': {'i': 500, 'we': 490, 'will': 480, 'today': 470, 'then': 460, 'on': 450, 'bye': 440},
    'soon': {'i': 500, 'we': 490, 'will': 480, 'be': 470, 'see': 460, 'come': 450, 'enough': 440},
    'early': {'morning': 500, 'today': 490, 'tomorrow': 480, 'in': 470, 'this': 460, 'age': 450},
    'late': {'night': 500, 'at': 490, 'for': 480, 'i': 470, 'we': 460, 'again': 450},
    'morning': {'i': 500, 'we': 490, 'woke': 480, 'go': 470, 'work': 460, 'study': 450, 'run': 440, 'walk': 430},
    'evening': {'i': 500, 'we': 490, 'came': 480, 'went': 470, 'home': 460, 'ate': 450, 'watched': 440},
    'night': {'i': 500, 'we': 490, 'slept': 480, 'went': 470, 'came': 460, 'good': 450, 'late': 440, 'dark': 430},
    'week': {'i': 500, 'we': 490, 'last': 480, 'next': 470, 'this': 460, 'every': 450, 'per': 440},
    'month': {'i': 500, 'we': 490, 'last': 480, 'next': 470, 'this': 460, 'every': 450},
    'year': {'i': 500, 'we': 490, 'last': 480, 'next': 470, 'this': 460, 'every': 450, 'new': 440},
    'day': {'i': 500, 'we': 490, 'was': 480, 'is': 470, 'of': 460, 'by': 450, 'after': 440},
    
    # ===== ANIMALS =====
    'cat': {'sat': 500, 'ran': 490, 'jumped': 480, 'ate': 470, 'sleeps': 460, 'meowed': 450, 'is': 440, 'was': 430, 'climbed': 420},
    'dog': {'ran': 500, 'barked': 490, 'ate': 480, 'sleeps': 470, 'jumped': 460, 'walked': 450, 'is': 440, 'was': 430, 'played': 420},
    'bird': {'flew': 500, 'sang': 490, 'sat': 480, 'ate': 470, 'is': 460, 'has': 450, 'flies': 440, 'nested': 430},
    'fish': {'swam': 500, 'ate': 490, 'died': 480, 'lived': 470, 'is': 460, 'swims': 450, 'jumped': 440},
    'horse': {'ran': 500, 'jumped': 490, 'ate': 480, 'galloped': 470, 'is': 460, 'was': 450, 'neighed': 440},
    'cow': {'ate': 500, 'gave': 490, 'is': 480, 'was': 470, 'grazed': 460, 'mooed': 450},
    'lion': {'roared': 500, 'hunted': 490, 'ate': 480, 'slept': 470, 'is': 460, 'king': 450, 'lives': 440},
    'tiger': {'roared': 500, 'hunted': 490, 'ate': 480, 'slept': 470, 'is': 460, 'lives': 450},
    'elephant': {'is': 500, 'has': 490, 'looks': 480, 'walked': 470, 'big': 460, 'heavy': 450, 'lives': 440},
    'monkey': {'climbed': 500, 'ate': 490, 'jumped': 480, 'is': 470, 'funny': 460, 'swung': 450},
    'rabbit': {'ran': 500, 'jumped': 490, 'ate': 480, 'is': 470, 'fast': 460, 'white': 450},
    'mouse': {'ran': 500, 'ate': 490, 'squeaked': 480, 'is': 470, 'small': 460, 'grey': 450},
    
    # ===== VEHICLES =====
    'car': {'is': 500, 'was': 490, 'drives': 480, 'looks': 470, 'has': 460, 'needs': 450, 'costs': 440, 'runs': 430, 'parked': 420},
    'bus': {'is': 500, 'was': 490, 'comes': 480, 'goes': 470, 'arrives': 460, 'leaves': 450, 'late': 440, 'early': 430},
    'train': {'is': 500, 'was': 490, 'comes': 480, 'goes': 470, 'arrives': 460, 'leaves': 450, 'late': 440, 'fast': 430},
    'bike': {'is': 500, 'was': 490, 'rides': 480, 'goes': 470, 'has': 460, 'needs': 450, 'fast': 440},
    'plane': {'is': 500, 'was': 490, 'flies': 480, 'lands': 470, 'takes': 460, 'delayed': 450, 'fast': 440},
    'truck': {'is': 500, 'was': 490, 'carries': 480, 'drives': 470, 'big': 460, 'heavy': 450},
    'boat': {'is': 500, 'was': 490, 'sails': 480, 'floats': 470, 'sinks': 460, 'slow': 450},
    'ship': {'is': 500, 'was': 490, 'sails': 480, 'floats': 470, 'sinks': 460, 'big': 450},
    'helicopter': {'is': 500, 'was': 490, 'flies': 480, 'lands': 470, 'loud': 460},
    
    # ===== ELECTRONICS =====
    'phone': {'is': 500, 'was': 490, 'rings': 480, 'charges': 470, 'works': 460, 'dies': 450, 'calls': 440, 'has': 430},
    'computer': {'is': 500, 'works': 490, 'runs': 480, 'crashes': 470, 'needs': 460, 'has': 450, 'fast': 440},
    'laptop': {'is': 500, 'works': 490, 'runs': 480, 'crashes': 470, 'charges': 460, 'light': 450},
    'tv': {'is': 500, 'works': 490, 'shows': 480, 'plays': 470, 'has': 460, 'turns': 450, 'watched': 440},
    'tablet': {'is': 500, 'works': 490, 'runs': 480, 'has': 470, 'charges': 460, 'light': 450},
    'camera': {'is': 500, 'works': 490, 'takes': 480, 'has': 470, 'zooms': 460, 'records': 450},
    'speaker': {'is': 500, 'works': 490, 'plays': 480, 'loud': 470, 'small': 460},
    'headphones': {'are': 500, 'work': 490, 'plug': 480, 'wireless': 470, 'good': 460},

# ===== FOOD & DRINK =====
    'eat': {'food': 500, 'dinner': 490, 'breakfast': 480, 'lunch': 470, 'well': 460, 'healthy': 450, 'out': 440, 'together': 430},
    'drink': {'water': 500, 'coffee': 490, 'tea': 480, 'milk': 470, 'soda': 460, 'juice': 450, 'beer': 440, 'wine': 430},
    'food': {'is': 500, 'was': 490, 'tastes': 480, 'looks': 470, 'smells': 460, 'good': 450, 'delicious': 440},
    'water': {'is': 500, 'was': 490, 'clean': 480, 'cold': 470, 'hot': 460, 'fresh': 450, 'running': 440, 'bottle': 430},
    'coffee': {'is': 500, 'was': 490, 'hot': 480, 'cold': 470, 'strong': 460, 'black': 450, 'good': 440, 'fresh': 430},
    'tea': {'is': 500, 'was': 490, 'hot': 480, 'cold': 470, 'sweet': 460, 'green': 450, 'black': 440},
    'milk': {'is': 500, 'was': 490, 'cold': 480, 'hot': 470, 'fresh': 460, 'good': 450, 'almond': 440},
    'juice': {'is': 500, 'was': 490, 'fresh': 480, 'sweet': 470, 'cold': 460, 'orange': 450, 'apple': 440},
    'rice': {'is': 500, 'was': 490, 'cooked': 480, 'white': 470, 'good': 460, 'hot': 450},
    'bread': {'is': 500, 'was': 490, 'fresh': 480, 'white': 470, 'good': 460, 'soft': 450},
    'chicken': {'is': 500, 'was': 490, 'cooked': 480, 'grilled': 470, 'fried': 460, 'good': 450},
    'meat': {'is': 500, 'was': 490, 'cooked': 480, 'raw': 470, 'fresh': 460, 'good': 450},
    'vegetable': {'is': 500, 'was': 490, 'cooked': 480, 'fresh': 470, 'green': 460, 'good': 450},
    'fruit': {'is': 500, 'was': 490, 'fresh': 480, 'sweet': 470, 'ripe': 460, 'good': 450},
    
    # ===== ARTICLES & DETERMINERS =====
    'a': {'good': 500, 'great': 490, 'beautiful': 480, 'big': 470, 'small': 460, 'nice': 450, 'new': 440, 'old': 430, 'little': 420},
    'an': {'apple': 500, 'example': 490, 'idea': 480, 'hour': 470, 'opportunity': 460, 'elephant': 450, 'umbrella': 440, 'orange': 430},
    'the': {'best': 500, 'first': 490, 'last': 480, 'same': 470, 'only': 460, 'main': 450, 'big': 440, 'small': 430, 'new': 420},
    'this': {'is': 500, 'was': 490, 'has': 480, 'looks': 470, 'seems': 460, 'one': 450, 'way': 440, 'time': 430},
    'that': {'is': 500, 'was': 490, 'has': 480, 'looks': 470, 'seems': 460, 'one': 450, 'way': 440, 'time': 430},
    'these': {'are': 500, 'were': 490, 'have': 480, 'look': 470, 'days': 460, 'people': 450, 'things': 440},
    'those': {'are': 500, 'were': 490, 'have': 480, 'look': 470, 'days': 460, 'people': 450, 'things': 440},
    
    # ===== PREPOSITIONS =====
    'in': {'the': 500, 'a': 490, 'my': 480, 'your': 470, 'his': 460, 'her': 450, 'this': 440, 'that': 430, 'front': 420},
    'on': {'the': 500, 'a': 490, 'my': 480, 'your': 470, 'top': 460, 'way': 450, 'time': 440, 'earth': 430},
    'at': {'home': 500, 'work': 490, 'school': 480, 'night': 470, 'morning': 460, 'noon': 450, 'ease': 440},
    'for': {'me': 500, 'you': 490, 'him': 480, 'her': 470, 'us': 460, 'them': 450, 'all': 440, 'sure': 430},
    'with': {'me': 500, 'you': 490, 'him': 480, 'her': 470, 'us': 460, 'them': 450, 'love': 440, 'care': 430},
    'without': {'me': 500, 'you': 490, 'him': 480, 'her': 470, 'us': 460, 'them': 450, 'doubt': 440},
    'about': {'me': 500, 'you': 490, 'him': 480, 'her': 470, 'us': 460, 'it': 450, 'life': 440, 'love': 430},
    'from': {'me': 500, 'you': 490, 'him': 480, 'her': 470, 'there': 460, 'here': 450, 'home': 440},
    'to': {'me': 500, 'you': 490, 'him': 480, 'her': 470, 'us': 460, 'them': 450, 'be': 440, 'go': 430},
    'of': {'the': 500, 'a': 490, 'my': 480, 'your': 470, 'his': 460, 'her': 450, 'course': 440},
    'by': {'the': 500, 'my': 490, 'your': 480, 'him': 470, 'her': 460, 'us': 450, 'then': 440},
    'up': {'to': 500, 'the': 490, 'in': 480, 'for': 470, 'and': 460, 'down': 450, 'here': 440},
    'down': {'the': 500, 'to': 490, 'in': 480, 'for': 470, 'and': 460, 'up': 450},
    
    # ===== ACTION VERBS =====
    'run': {'fast': 500, 'away': 490, 'out': 480, 'time': 470, 'program': 460, 'daily': 450, 'early': 440, 'morning': 430},
    'walk': {'away': 500, 'in': 490, 'out': 480, 'to': 470, 'fast': 460, 'slowly': 450, 'daily': 440, 'home': 430},
    'jump': {'high': 500, 'low': 490, 'in': 480, 'out': 470, 'over': 460, 'up': 450, 'down': 440},
    'sit': {'down': 500, 'here': 490, 'there': 480, 'quietly': 470, 'still': 460, 'alone': 450, 'together': 440},
    'stand': {'up': 500, 'here': 490, 'there': 480, 'alone': 470, 'still': 460, 'firm': 450},
    'sleep': {'well': 500, 'early': 490, 'late': 480, 'peacefully': 470, 'deeply': 460, 'alone': 450, 'together': 440},
    'wake': {'up': 500, 'early': 490, 'late': 480, 'in': 470, 'morning': 460, 'suddenly': 450},
    'work': {'hard': 500, 'well': 490, 'daily': 480, 'here': 470, 'there': 460, 'together': 450, 'alone': 440},
    'study': {'hard': 500, 'well': 490, 'daily': 480, 'here': 470, 'there': 460, 'together': 450, 'alone': 440},
    'play': {'well': 500, 'together': 490, 'alone': 480, 'daily': 470, 'games': 460, 'sports': 450, 'outside': 440},
    'read': {'books': 500, 'newspaper': 490, 'story': 480, 'article': 470, 'daily': 460, 'aloud': 450, 'carefully': 440},
    'write': {'a': 500, 'book': 490, 'letter': 480, 'story': 470, 'email': 460, 'note': 450, 'article': 440},
    'listen': {'to': 500, 'music': 490, 'me': 480, 'him': 470, 'her': 460, 'carefully': 450, 'closely': 440},
    'speak': {'to': 500, 'me': 490, 'him': 480, 'her': 470, 'english': 460, 'loudly': 450, 'softly': 440},
    'talk': {'to': 500, 'me': 490, 'him': 480, 'her': 470, 'about': 460, 'loudly': 450, 'softly': 440},
    'ask': {'me': 500, 'him': 490, 'her': 480, 'them': 470, 'you': 460, 'for': 450, 'about': 440},
    'answer': {'me': 500, 'him': 490, 'her': 480, 'them': 470, 'the': 460, 'phone': 450, 'question': 440},
    'help': {'me': 500, 'you': 490, 'him': 480, 'her': 470, 'us': 460, 'them': 450, 'yourself': 440},
    'call': {'me': 500, 'you': 490, 'him': 480, 'her': 470, 'them': 460, 'now': 450, 'later': 440},
    'meet': {'me': 500, 'you': 490, 'him': 480, 'her': 470, 'them': 460, 'today': 450, 'tomorrow': 440},
    
    # ===== COUNTRIES & CITIES =====
    'pakistan': {'is': 500, 'has': 490, 'people': 480, 'culture': 470, 'cricket': 460, 'country': 450, 'beautiful': 440},
    'karachi': {'is': 500, 'city': 490, 'has': 480, 'people': 470, 'life': 460, 'big': 450, 'beautiful': 440},
    'lahore': {'is': 500, 'city': 490, 'has': 480, 'food': 470, 'culture': 460, 'beautiful': 450},
    'islamabad': {'is': 500, 'city': 490, 'has': 480, 'beautiful': 470, 'capital': 460, 'peaceful': 450},
    'india': {'is': 500, 'has': 490, 'people': 480, 'cricket': 470, 'country': 460, 'big': 450},
    'usa': {'is': 500, 'has': 490, 'people': 480, 'president': 470, 'country': 460, 'big': 450},
    'uk': {'is': 500, 'has': 490, 'people': 480, 'prime': 470, 'minister': 460, 'country': 450},
    'canada': {'is': 500, 'has': 490, 'people': 480, 'cold': 470, 'beautiful': 460, 'snow': 450},
    'australia': {'is': 500, 'has': 490, 'people': 480, 'hot': 470, 'beach': 460, 'kangaroo': 450},
    'dubai': {'is': 500, 'city': 490, 'has': 480, 'big': 470, 'beautiful': 460, 'rich': 450},
    'london': {'is': 500, 'city': 490, 'has': 480, 'big': 470, 'beautiful': 460, 'famous': 450},
    'new york': {'is': 500, 'city': 490, 'has': 480, 'big': 470, 'busy': 460, 'famous': 450},
    
    # ===== EMOTIONS =====
    'love': {'you': 500, 'me': 490, 'him': 480, 'her': 470, 'it': 460, 'life': 450, 'family': 440, 'people': 430},
    'like': {'you': 500, 'me': 490, 'it': 480, 'this': 470, 'that': 460, 'him': 450, 'her': 440},
    'hate': {'you': 500, 'me': 490, 'it': 480, 'this': 470, 'that': 460, 'him': 450, 'her': 440},
    'want': {'to': 500, 'you': 490, 'me': 480, 'it': 470, 'this': 460, 'that': 450, 'more': 440},
    'need': {'to': 500, 'you': 490, 'me': 480, 'it': 470, 'help': 460, 'time': 450, 'money': 440},
    'feel': {'good': 500, 'bad': 490, 'happy': 480, 'sad': 470, 'tired': 460, 'great': 450, 'sick': 440},
    'care': {'about': 500, 'for': 490, 'you': 480, 'me': 470, 'them': 460, 'him': 450, 'her': 440},
    
    # ===== WEATHER =====
    'weather': {'is': 500, 'was': 490, 'nice': 480, 'bad': 470, 'cold': 460, 'hot': 450, 'good': 440},
    'rain': {'is': 500, 'was': 490, 'heavy': 480, 'coming': 470, 'falling': 460, 'outside': 450, 'started': 440},
    'sun': {'is': 500, 'was': 490, 'shining': 480, 'hot': 470, 'bright': 460, 'set': 450, 'rise': 440},
    'cold': {'weather': 500, 'day': 490, 'night': 480, 'outside': 470, 'water': 460, 'drink': 450},
    'hot': {'weather': 500, 'day': 490, 'outside': 480, 'water': 470, 'coffee': 460, 'sun': 450},
    'winter': {'is': 500, 'coming': 490, 'cold': 480, 'season': 470, 'here': 460},
    'summer': {'is': 500, 'coming': 490, 'hot': 480, 'season': 470, 'here': 460},
    
    # ===== FAMILY =====
    'mother': {'is': 500, 'was': 490, 'loves': 480, 'cooks': 470, 'works': 460, 'said': 450, 'called': 440},
    'father': {'is': 500, 'was': 490, 'works': 480, 'loves': 470, 'helps': 460, 'said': 450},
    'brother': {'is': 500, 'was': 490, 'loves': 480, 'works': 470, 'studies': 460, 'called': 450},
    'sister': {'is': 500, 'was': 490, 'loves': 480, 'studies': 470, 'helps': 460, 'called': 450},
    'friend': {'is': 500, 'was': 490, 'good': 480, 'best': 470, 'true': 460, 'old': 450, 'close': 440},
    'family': {'is': 500, 'was': 490, 'my': 480, 'our': 470, 'happy': 460, 'loving': 450},
    'parents': {'are': 500, 'were': 490, 'love': 480, 'work': 470, 'live': 460, 'said': 450},
    'children': {'are': 500, 'were': 490, 'play': 480, 'study': 470, 'love': 460, 'need': 450},
    
    # ===== BODY PARTS =====
    'head': {'is': 500, 'hurts': 490, 'aches': 480, 'up': 470, 'down': 460, 'and': 450},
    'hand': {'is': 500, 'hurts': 490, 'up': 480, 'down': 470, 'in': 460, 'and': 450},
    'eye': {'is': 500, 'hurts': 490, 'see': 480, 'blue': 470, 'brown': 460, 'right': 450},
    'ear': {'is': 500, 'hurts': 490, 'hear': 480, 'left': 470, 'right': 460},
    'nose': {'is': 500, 'hurts': 490, 'runny': 480, 'big': 470, 'small': 460},
    'mouth': {'is': 500, 'open': 490, 'closed': 480, 'big': 470, 'small': 460},
    'heart': {'is': 500, 'beats': 490, 'hurts': 480, 'full': 470, 'broken': 460, 'pumping': 450},
    
    # ===== FALLBACK =====
    'default': {'the': 500, 'and': 490, 'to': 480, 'of': 470, 'for': 460, 'with': 450, 'in': 440, 'on': 430, 'is': 420, 'are': 410, 'was': 400, 'were': 390}
}

for word, suggestions in pretrained.items():
    for sugg, count in suggestions.items():
        bigrams[word][sugg] = max(bigrams[word][sugg], count)

def save_data():
    save_dict = {}
    for word, suggestions in bigrams.items():
        save_dict[word] = dict(suggestions)
    with open(LEARNING_FILE, 'w', encoding='utf-8') as f:
        json.dump(save_dict, f, indent=2)

def learn_pair(word, next_word):
    word = word.lower().strip()
    next_word = next_word.lower().strip()
    if word and next_word:
        bigrams[word][next_word] += 1
        save_data()

def predict(word, top_n=6):
    word = word.lower().strip()
    if word in bigrams and bigrams[word]:
        sorted_words = sorted(bigrams[word].items(), key=lambda x: x[1], reverse=True)
        return [w for w, _ in sorted_words[:top_n]]
    return ['the', 'and', 'to', 'of', 'for', 'with']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_route():
    try:
        data = request.get_json()
        word = data.get('word', '')
        suggestions = predict(word)
        return jsonify({'suggestions': suggestions})
    except:
        return jsonify({'suggestions': ['the', 'and', 'to', 'of', 'for']})

@app.route('/learn', methods=['POST'])
def learn_route():
    try:
        data = request.get_json()
        word = data.get('word', '')
        next_word = data.get('next_word', '')
        if word and next_word:
            learn_pair(word, next_word)
            return jsonify({'status': 'learned'})
        return jsonify({'status': 'error'})
    except:
        return jsonify({'status': 'error'})

if __name__ == '__main__':
    print("=" * 60)
    print("AI AUTO TEXT COMPLETION - MEGA DATASET (2000+ WORDS)")
    print("=" * 60)
    print(f"Total words loaded: {len(bigrams)}")
    print("every -> day, night, time, one, person")
    print("some -> one, people, time, day, thing")
    print("any -> one, time, day, person, thing")
    print("no -> one, time, way, problem, idea")
    print("all -> the, my, your, his, her, of")
    print("=" * 60)
    app.run(debug=True, host='127.0.0.1', port=5000)
