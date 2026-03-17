"""
SolasGPT inference server.
Loads trained model and serves chat completions over HTTP.
Connects to TurboWarp Node backend via /chat-plain replacement.

Usage:
  python inference_server.py                  # port 8788 by default
  python inference_server.py --port 8788
"""

import argparse
import json
import math
import os
import re
from datetime import datetime
from collections import Counter

import torch
from flask import Flask, jsonify, request

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

from model import SolasGPT

app = Flask(__name__)

# ─────────────── globals ────────────────────────
model = None
stoi: dict = {}
itos: dict = {}
device = 'cuda' if torch.cuda.is_available() else 'cpu'
CHECKPOINT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_checkpoint.pt')
DATASET = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'conversations.txt')
FEEDBACK_LOG = os.getenv(
    'FEEDBACK_LOG_PATH',
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'feedback_log.jsonl')
)
sessions: dict[str, list[str]] = {}   # short-term history per session
HISTORY_TURNS = 8
knowledge_pairs: list[tuple[str, str]] = []
embedding_model = None
embedding_model_name = os.getenv('EMBEDDING_MODEL_NAME', 'all-MiniLM-L6-v2')
embedding_enabled = os.getenv('USE_EMBEDDINGS', 'true').lower() == 'true'
embedding_error = ''
knowledge_embeddings = None
knowledge_question_tokens: list[list[str]] = []
knowledge_idf: dict[str, float] = {}
knowledge_vectors: list[dict[str, float]] = []
knowledge_norms: list[float] = []
STOPWORDS = {
    'a', 'an', 'the', 'is', 'are', 'in', 'on', 'at', 'of', 'for', 'to', 'and',
    'or', 'do', 'does', 'did', 'can', 'could', 'would', 'should', 'how', 'what',
    'when', 'where', 'why', 'which', 'who', 'whom', 'this', 'that', 'these', 'those',
    'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'my', 'your', 'our', 'their',
    'there', 'many', 'much', 'be', 'as', 'with', 'about', 'into', 'from', 'by', 'one'
}
# ────────────────────────────────────────────────


def encode(text: str) -> list[int]:
    return [stoi[c] for c in text if c in stoi]


def decode(ids: list[int]) -> str:
    return ''.join(itos[i] for i in ids)


def normalize_message(text: str) -> str:
    return re.sub(r'\s+', ' ', text.lower()).strip()


def load_knowledge_pairs() -> list[tuple[str, str]]:
    if not os.path.exists(DATASET):
        return []

    pairs: list[tuple[str, str]] = []
    pending_user = None
    with open(DATASET, 'r', encoding='utf-8') as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line.startswith('User: '):
                pending_user = line[6:].strip()
            elif line.startswith('Assistant: ') and pending_user:
                pairs.append((pending_user, line[11:].strip()))
                pending_user = None
    return pairs


def tokenize_for_vector(text: str) -> list[str]:
    normalized = normalize_message(text)
    tokens = re.findall(r"[a-z0-9']+", normalized)
    filtered = [token for token in tokens if token not in STOPWORDS]
    return filtered or tokens


def build_vector_index(pairs: list[tuple[str, str]]):
    global knowledge_question_tokens, knowledge_idf, knowledge_vectors, knowledge_norms

    questions = [question for question, _ in pairs]
    knowledge_question_tokens = [tokenize_for_vector(question) for question in questions]

    doc_count = max(1, len(knowledge_question_tokens))
    document_frequency: Counter[str] = Counter()
    for tokens in knowledge_question_tokens:
        document_frequency.update(set(tokens))

    knowledge_idf = {
        token: math.log((1 + doc_count) / (1 + freq)) + 1.0
        for token, freq in document_frequency.items()
    }

    knowledge_vectors = []
    knowledge_norms = []
    for tokens in knowledge_question_tokens:
        counts = Counter(tokens)
        total = max(1, sum(counts.values()))
        vector: dict[str, float] = {}
        for token, count in counts.items():
            tf = count / total
            vector[token] = tf * knowledge_idf.get(token, 1.0)
        norm = math.sqrt(sum(value * value for value in vector.values()))
        knowledge_vectors.append(vector)
        knowledge_norms.append(norm)


def load_embedding_model():
    global embedding_model, embedding_error
    if not embedding_enabled:
        embedding_error = 'disabled'
        return None
    if embedding_model is not None:
        return embedding_model
    if SentenceTransformer is None:
        embedding_error = 'sentence-transformers not installed'
        return None

    try:
        embedding_model = SentenceTransformer(embedding_model_name)
        embedding_error = ''
        return embedding_model
    except Exception as exc:
        embedding_error = str(exc)
        embedding_model = None
        return None


def build_embedding_index(pairs: list[tuple[str, str]]):
    global knowledge_embeddings
    model_instance = load_embedding_model()
    if model_instance is None or not pairs:
        knowledge_embeddings = None
        return

    questions = [question for question, _ in pairs]
    knowledge_embeddings = model_instance.encode(
        questions,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )


def embedding_similarity(query_text: str, context_text: str = '') -> tuple[int | None, float]:
    if embedding_model is None or knowledge_embeddings is None or len(knowledge_pairs) == 0:
        return None, 0.0

    query_embedding = embedding_model.encode(
        [query_text],
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0]

    scores = torch.matmul(knowledge_embeddings, query_embedding)
    if context_text:
        context_embedding = embedding_model.encode(
            [context_text],
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        scores = scores + 0.08 * torch.matmul(knowledge_embeddings, context_embedding)

    best_index = int(torch.argmax(scores).item())
    best_score = float(scores[best_index].item())
    return best_index, best_score


def vectorize_query(text: str) -> tuple[dict[str, float], float]:
    tokens = tokenize_for_vector(text)
    counts = Counter(tokens)
    total = max(1, sum(counts.values()))
    vector: dict[str, float] = {}
    for token, count in counts.items():
        if token not in knowledge_idf:
            continue
        tf = count / total
        vector[token] = tf * knowledge_idf[token]
    norm = math.sqrt(sum(value * value for value in vector.values()))
    return vector, norm


def cosine_similarity_sparse(a: dict[str, float], a_norm: float,
                             b: dict[str, float], b_norm: float) -> float:
    if a_norm <= 0 or b_norm <= 0:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    dot = 0.0
    for token, value in a.items():
        dot += value * b.get(token, 0.0)
    return dot / (a_norm * b_norm)


def overlap_score(a: str, b: str) -> float:
    a_tokens = set(re.findall(r"[a-z0-9']+", normalize_message(a)))
    b_tokens = set(re.findall(r"[a-z0-9']+", normalize_message(b)))
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / max(1, len(a_tokens | b_tokens))


def last_assistant_message(history: list[str]) -> str:
    for line in reversed(history):
        if line.startswith('Assistant: '):
            return line[11:].strip()
    return ''


def is_safe_math_expression(text: str) -> bool:
    return bool(re.fullmatch(r"[0-9\s\+\-\*\/\(\)\.]+", text))


def evaluate_math(text: str) -> str | None:
    raw = normalize_message(text)
    if not raw:
        return None

    patterns = [
        r"what is (.+)",
        r"calculate (.+)",
        r"solve (.+)",
    ]
    expr = None
    for pattern in patterns:
        match = re.fullmatch(pattern, raw)
        if match:
            expr = match.group(1)
            break

    if expr is None:
        expr = raw

    expr = expr.replace('x', '*').replace('times', '*').replace('plus', '+')
    expr = expr.replace('minus', '-').replace('divided by', '/').strip()
    if not is_safe_math_expression(expr):
        return None

    try:
        value = eval(expr, {'__builtins__': {}}, {})
        if isinstance(value, (int, float)) and math.isfinite(value):
            if abs(value - int(value)) < 1e-10:
                return f"{int(value)}"
            return f"{value:.6g}"
    except Exception:
        return None
    return None


def factual_reply(user_message: str) -> str | None:
    text = normalize_message(user_message)
    compact = re.sub(r"[^a-z0-9\s']+", '', text).strip()
    if not compact:
        return None

    def normalize_unit(raw_unit: str) -> str | None:
        unit = raw_unit.lower()
        if unit in {'second', 'seconds', 'sec', 'secs'}:
            return 'second'
        if unit in {'minute', 'minutes', 'min', 'mins'}:
            return 'minute'
        if unit in {'hour', 'hours', 'hr', 'hrs'}:
            return 'hour'
        if unit in {'day', 'days'}:
            return 'day'
        if unit in {'week', 'weeks'}:
            return 'week'
        if unit in {'year', 'years'}:
            return 'year'
        return None

    unit_match = re.fullmatch(r"how many ([a-z]+) (are )?in (a|an|one) ([a-z]+)", compact)
    if unit_match:
        asked_unit = normalize_unit(unit_match.group(1))
        container_unit = normalize_unit(unit_match.group(4))
        unit_seconds = {
            'second': 1,
            'minute': 60,
            'hour': 60 * 60,
            'day': 24 * 60 * 60,
            'week': 7 * 24 * 60 * 60,
            'year': 365 * 24 * 60 * 60,
        }
        if asked_unit and container_unit:
            ratio = unit_seconds[container_unit] / unit_seconds[asked_unit]
            rounded = int(round(ratio)) if abs(ratio - round(ratio)) < 1e-9 else round(ratio, 2)
            formatted = f"{rounded:,}" if isinstance(rounded, int) else f"{rounded:,.2f}".rstrip('0').rstrip('.')
            asked_label = asked_unit if rounded == 1 else f"{asked_unit}s"
            return f"There are {formatted} {asked_label} in a {container_unit}."

    quantity_unit_match = re.fullmatch(r"how many ([a-z]+) (are )?in (\d+(?:\.\d+)?) ([a-z]+)", compact)
    if quantity_unit_match:
        asked_unit = normalize_unit(quantity_unit_match.group(1))
        quantity = float(quantity_unit_match.group(3))
        container_unit = normalize_unit(quantity_unit_match.group(4))
        unit_seconds = {
            'second': 1,
            'minute': 60,
            'hour': 60 * 60,
            'day': 24 * 60 * 60,
            'week': 7 * 24 * 60 * 60,
            'year': 365 * 24 * 60 * 60,
        }
        if asked_unit and container_unit and quantity > 0:
            ratio = (quantity * unit_seconds[container_unit]) / unit_seconds[asked_unit]
            rounded = int(round(ratio)) if abs(ratio - round(ratio)) < 1e-9 else round(ratio, 2)
            formatted = f"{rounded:,}" if isinstance(rounded, int) else f"{rounded:,.2f}".rstrip('0').rstrip('.')
            asked_label = asked_unit if rounded == 1 else f"{asked_unit}s"
            quantity_label = f"{int(quantity)}" if abs(quantity - int(quantity)) < 1e-9 else f"{quantity:g}"
            container_label = container_unit if abs(quantity - 1) < 1e-9 else f"{container_unit}s"
            return f"There are {formatted} {asked_label} in {quantity_label} {container_label}."

    year_match = re.fullmatch(r"how many years until (\d{4})", compact)
    if year_match:
        target_year = int(year_match.group(1))
        current_year = int(os.getenv('CURRENT_YEAR_OVERRIDE', '0')) or int(datetime.now().year)
        diff = target_year - current_year
        if diff > 0:
            return f"There are {diff} years until {target_year}."
        if diff == 0:
            return f"{target_year} is this year."
        return f"{target_year} was {abs(diff)} years ago."

    facts: list[tuple[str, str]] = [
        (r"^how many days (are )?in (a|an|one) week$", 'There are 7 days in a week.'),
        (r"^how many hours (are )?in (a|an|one) day$", 'There are 24 hours in a day.'),
        (r"^how many minutes (are )?in (a|an|one) hour$", 'There are 60 minutes in an hour.'),
        (r"^how many seconds (are )?in (a|an|one) minute$", 'There are 60 seconds in a minute.'),
        (r"^how many seconds (are )?in (a|an|one) hour$", 'There are 3,600 seconds in an hour.'),
        (r"^how many minutes (are )?in (a|an|one) day$", 'There are 1,440 minutes in a day.'),
        (r"^how many seconds (are )?in (a|an|one) day$", 'There are 86,400 seconds in a day.'),
        (r"^how many months (are )?in (a|an|one) year$", 'There are 12 months in a year.'),
        (r"^how many weeks (are )?in (a|an|one) year$", 'There are 52 weeks in a year (about 52.14).'),
        (r"^what is the capital of france$", 'The capital of France is Paris.'),
        (r"^what is the capital of japan$", 'The capital of Japan is Tokyo.'),
        (r"^which planet is known as the red planet$", 'Mars is known as the Red Planet.'),
        (r"^what is h2o$", 'H2O is water.'),
        (r"^how many letters are in the english alphabet$", 'There are 26 letters in the English alphabet.'),
    ]

    for pattern, answer in facts:
        if re.fullmatch(pattern, compact):
            return answer
    return None


def intent_reply(user_message: str, history: list[str]) -> str | None:
    text = normalize_message(user_message)
    if not text:
        return None

    fact = factual_reply(user_message)
    if fact is not None:
        return fact

    if text in {'hi', 'hello', 'hey', 'yo'}:
        return 'Hello! I am SolasAI. What would you like to ask?'
    if text in {'thanks', 'thank you', 'thx'}:
        return "You're welcome."
    if text in {'bye', 'goodbye', 'see you'}:
        return 'Goodbye!'

    if text in {'who are you', 'what is your name'}:
        return 'I am SolasAI, a custom local chatbot model.'

    if text in {'explain more', 'more details', 'go deeper', 'elaborate'}:
        previous = last_assistant_message(history)
        if previous:
            return f"Sure. In simple terms: {previous}"
        return 'Sure. Ask a specific topic and I will explain it step by step.'

    if 'think in vertices' in text:
        return 'No. AI models do not think in vertices; they process text as tokens and vectors with probabilities.'

    math_result = evaluate_math(user_message)
    if math_result is not None:
        return f"The answer is {math_result}."

    return None


def retrieval_reply(user_message: str, history: list[str]) -> tuple[str | None, float]:
    query = normalize_message(user_message)
    if not query or not knowledge_pairs:
        return None, 0.0

    context_text = normalize_message(' '.join(history[-4:]))

    embed_index, embed_score = embedding_similarity(user_message, context_text)
    if embed_index is not None:
        question, answer = knowledge_pairs[embed_index]
        return answer, max(0.0, min(embed_score, 1.0))

    if not knowledge_vectors:
        return None, 0.0

    query_vector, query_norm = vectorize_query(user_message)
    context_vector, context_norm = vectorize_query(context_text) if context_text else ({}, 0.0)
    if query_norm <= 0:
        return None, 0.0

    best_answer = None
    best_score = 0.0
    for index, (question, answer) in enumerate(knowledge_pairs):
        question_vector = knowledge_vectors[index]
        question_norm = knowledge_norms[index]
        score = cosine_similarity_sparse(query_vector, query_norm, question_vector, question_norm)
        if context_text and context_norm > 0:
            score += 0.08 * cosine_similarity_sparse(context_vector, context_norm, question_vector, question_norm)
        score = min(score, 1.0)
        if score > best_score:
            best_score = score
            best_answer = answer

    return best_answer, min(best_score, 1.0)


def clean_reply(text: str) -> str:
    reply = text.replace('\r', ' ').strip()
    reply = reply.split('\nUser:')[0].split('\nAssistant:')[0].strip()
    reply = re.sub(r'\s+', ' ', reply)
    reply = re.sub(r'(.)\1{4,}', r'\1\1', reply)
    reply = reply.lstrip(':;- ').strip()
    if reply and reply[0].islower():
        reply = reply[0].upper() + reply[1:]
    if len(reply) > 420:
        reply = reply[:420].rsplit(' ', 1)[0].strip()

    if reply and reply[-1] not in '.!?':
        for mark in '.!?':
            cut = reply.rfind(mark)
            if cut >= max(20, int(len(reply) * 0.5)):
                reply = reply[: cut + 1]
                break
        if reply and reply[-1] not in '.!?':
            reply = f"{reply}."
    return reply


def is_low_quality_reply(reply: str) -> tuple[bool, str]:
    text = normalize_message(reply)
    if not text:
        return True, 'empty'

    generic_markers = [
        'i am still learning',
        'ask a specific question',
        'i could not understand that'
    ]
    for marker in generic_markers:
        if marker in text:
            return True, f'generic:{marker}'

    if len(text.split()) < 3:
        return True, 'too_short'
    return False, ''


def log_feedback_event(session_id: str, user_message: str, assistant_reply: str):
    needs_improvement, reason = is_low_quality_reply(assistant_reply)
    event = {
        'ts': datetime.utcnow().isoformat(timespec='seconds') + 'Z',
        'sessionId': session_id,
        'user': user_message,
        'assistant': assistant_reply,
        'needsImprovement': needs_improvement,
        'reason': reason,
    }

    try:
        os.makedirs(os.path.dirname(FEEDBACK_LOG), exist_ok=True)
        with open(FEEDBACK_LOG, 'a', encoding='utf-8') as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + '\n')
    except Exception:
        pass


def looks_bad(text: str) -> bool:
    reply = clean_reply(text)
    if len(reply) < 2:
        return True
    if 'User:' in reply or 'Assistant:' in reply:
        return True
    letters = sum(ch.isalpha() for ch in reply)
    spaces = sum(ch.isspace() for ch in reply)
    good_chars = sum(ch.isalnum() or ch in " .,?!'\"-" for ch in reply)
    if good_chars / max(1, len(reply)) < 0.85:
        return True
    if letters / max(1, len(reply) - spaces) < 0.45:
        return True
    words = reply.split()
    if len(words) >= 3 and len(set(words[: min(8, len(words))])) <= 2:
        return True
    if any(token in reply.lower() for token in ['asisistat', 'llllist', 'feryere']):
        return True
    return False


def assume_high_probability_reply(user_message: str, best_answer: str | None = None) -> str:
    if best_answer:
        return clean_reply(best_answer)

    text = normalize_message(user_message)
    compact = re.sub(r'\s+', ' ', text).strip()
    if not compact:
        return 'I will assume you want a clear answer. Here is the best approach: define your goal, provide key details, and I will give a direct solution.'

    topic = re.sub(r'^(what|who|when|where|which|why|how|can|could|would|should|is|are|do|does|did)\b\s*', '', compact, flags=re.IGNORECASE).strip(' ?.!')
    if not topic:
        topic = compact[:60]

    return clean_reply(
        f'I will assume the most likely intent is practical help with {topic}. '
        'Start by defining the goal, listing the important inputs, doing the first sensible step, and checking the result before continuing. '
        'If you want a more exact answer, you can give me a little more context and I can make the explanation more specific.'
    )


def generate_reply(prompt: str, max_new_tokens: int = 150, temperature: float = 0.8,
                   top_k: int = 40) -> str:
    assert model is not None
    context = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
    if context.shape[1] == 0:
        return "I could not understand that."

    out = model.generate(context, max_new_tokens=max_new_tokens,
                          temperature=temperature, top_k=top_k)
    generated = decode(out[0].tolist()[len(encode(prompt)):])
    reply = clean_reply(generated)
    return reply or "..."


def answer_message(user_message: str, history: list[str]) -> str:
    rule = intent_reply(user_message, history)
    if rule:
        return clean_reply(rule)

    best_answer, score = retrieval_reply(user_message, history)
    if best_answer and score >= 0.74:
        return clean_reply(best_answer)

    prompt = build_prompt(history, user_message)
    reply = generate_reply(prompt, max_new_tokens=160, temperature=0.7, top_k=24)
    if looks_bad(reply) and best_answer and score >= 0.42:
        return clean_reply(best_answer)
    if looks_bad(reply):
        return assume_high_probability_reply(user_message, best_answer)
    return clean_reply(reply)


def build_prompt(history: list[str], user_message: str) -> str:
    lines = history[-(HISTORY_TURNS * 2):]
    lines.append('System: Give a clear answer with helpful detail. When useful, include a short explanation or a few steps.')
    lines.append(f"User: {user_message}")
    lines.append("Assistant:")
    return '\n'.join(lines) + ''


def load_model():
    global model, stoi, itos, knowledge_pairs
    if not os.path.exists(CHECKPOINT):
        raise FileNotFoundError(
            f"No checkpoint found at {CHECKPOINT}. Run train.py first."
        )
    print(f"Loading checkpoint from {CHECKPOINT} ...")
    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    config = ckpt['config']
    stoi = ckpt['stoi']
    itos = ckpt['itos']

    model = SolasGPT(
        vocab_size=config['vocab_size'],
        embed_dim=config['embed_dim'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        block_size=config['block_size'],
        dropout=0.0,   # inference mode
    ).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    knowledge_pairs = load_knowledge_pairs()
    build_vector_index(knowledge_pairs)
    build_embedding_index(knowledge_pairs)
    print(f"Model loaded. Parameters: {model.param_count():,}  Device: {device}")
    print(f"Loaded knowledge pairs: {len(knowledge_pairs)}")
    retrieval_mode = 'embeddings' if knowledge_embeddings is not None else 'tf-idf-fallback'
    print(f"Vector index size: {len(knowledge_vectors)}")
    print(f"Retrieval mode: {retrieval_mode}")
    if embedding_error:
        print(f"Embedding status: {embedding_error}")


@app.get('/health')
def health():
    feedback_events = 0
    if os.path.exists(FEEDBACK_LOG):
        try:
            with open(FEEDBACK_LOG, 'r', encoding='utf-8') as handle:
                feedback_events = sum(1 for _ in handle)
        except Exception:
            feedback_events = 0

    return jsonify({
        'ok': model is not None,
        'checkpoint': CHECKPOINT,
        'retrievalMode': 'embeddings' if knowledge_embeddings is not None else 'tf-idf-fallback',
        'embeddingModel': embedding_model_name,
        'embeddingReady': knowledge_embeddings is not None,
        'embeddingError': embedding_error,
        'feedbackLogPath': FEEDBACK_LOG,
        'feedbackEvents': feedback_events,
    })


@app.post('/chat')
def chat():
    data = request.get_json(silent=True) or {}
    session_id = str(data.get('sessionId', 'default')).strip() or 'default'
    user_message = str(data.get('message', '')).strip()

    if not user_message:
        return jsonify({'ok': False, 'error': 'message is required'}), 400

    if model is None:
        return jsonify({'ok': False, 'error': 'Model not loaded. Run train.py first.'}), 503

    history = sessions.get(session_id, [])
    reply = answer_message(user_message, history)
    log_feedback_event(session_id, user_message, reply)

    # Update session history
    new_history = history + [f"User: {user_message}", f"Assistant: {reply}"]
    sessions[session_id] = new_history[-(HISTORY_TURNS * 2):]

    return jsonify({'ok': True, 'reply': reply, 'sessionId': session_id})


@app.post('/chat-plain')
def chat_plain():
    data = request.get_json(silent=True) or {}
    session_id = str(data.get('sessionId', 'default')).strip() or 'default'
    user_message = str(data.get('message', '')).strip()

    if not user_message:
        return 'ERROR: message is required', 400

    if model is None:
        return 'ERROR: Model not loaded. Run train.py first.', 503

    history = sessions.get(session_id, [])
    reply = answer_message(user_message, history)
    log_feedback_event(session_id, user_message, reply)

    new_history = history + [f"User: {user_message}", f"Assistant: {reply}"]
    sessions[session_id] = new_history[-(HISTORY_TURNS * 2):]

    return reply, 200, {'Content-Type': 'text/plain; charset=utf-8'}


@app.post('/reset')
def reset():
    data = request.get_json(silent=True) or {}
    session_id = str(data.get('sessionId', 'default')).strip() or 'default'
    sessions.pop(session_id, None)
    return jsonify({'ok': True, 'sessionId': session_id})


@app.post('/feedback')
def feedback():
    data = request.get_json(silent=True) or {}
    session_id = str(data.get('sessionId', 'default')).strip() or 'default'
    user_message = str(data.get('message', '')).strip()
    assistant_reply = str(data.get('reply', '')).strip()
    rating = normalize_message(str(data.get('rating', '')).strip())
    improvement = str(data.get('improvement', '')).strip()

    if not rating:
        return jsonify({'ok': False, 'error': 'rating is required'}), 400

    is_positive = rating in {'✓', '✅', 'tick', 'good', '1', 'yes', 'y'}
    is_negative = rating in {'✗', '❌', 'x', 'bad', '0', 'no', 'n'}
    needs_improvement = is_negative or (not is_positive)

    event = {
        'ts': datetime.utcnow().isoformat(timespec='seconds') + 'Z',
        'sessionId': session_id,
        'user': user_message,
        'assistant': assistant_reply,
        'rating': rating,
        'improvement': improvement,
        'needsImprovement': needs_improvement,
        'reason': 'user-negative-feedback' if is_negative else ('user-positive-feedback' if is_positive else 'user-feedback')
    }

    try:
        os.makedirs(os.path.dirname(FEEDBACK_LOG), exist_ok=True)
        with open(FEEDBACK_LOG, 'a', encoding='utf-8') as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + '\n')
    except Exception:
        return jsonify({'ok': False, 'error': 'failed to write feedback'}), 500

    return jsonify({'ok': True, 'sessionId': session_id, 'rating': rating, 'needsImprovement': needs_improvement})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=int(os.getenv('PORT', '8788')))
    parser.add_argument('--host', type=str, default=os.getenv('HOST', '0.0.0.0'))
    args = parser.parse_args()

    load_model()
    print(f"SolasGPT inference server running on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port)
