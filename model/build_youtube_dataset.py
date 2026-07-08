#!/usr/bin/env python3
"""
Build Minecraft learning pairs from YouTube transcripts.

This script does not process raw video frames. It learns from captions/transcripts
for user-provided YouTube videos and converts them into training pairs:

  User: how do i <goal> in minecraft
  Assistant: Step-by-step guidance inferred from tutorial transcript.

Usage:
  /mnt/data/SolasAI/.venv/bin/python build_youtube_dataset.py \
    --url https://www.youtube.com/watch?v=<id>

  /mnt/data/SolasAI/.venv/bin/python build_youtube_dataset.py \
    --urls-file data/youtube_urls.txt --merge
"""

from __future__ import annotations

import argparse
import json
import os
import re
import xml.etree.ElementTree as ET
import urllib.parse
import urllib.request
from collections import Counter

try:
    from youtube_transcript_api import YouTubeTranscriptApi
except Exception:
    YouTubeTranscriptApi = None

try:
    from youtube_transcript_api.proxies import GenericProxyConfig
except Exception:
    GenericProxyConfig = None


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DATA = os.path.join(BASE_DIR, 'data', 'youtube_minecraft_lessons.txt')
MAIN_DATA = os.path.join(BASE_DIR, 'data', 'conversations.txt')

ACTION_KEYWORDS = {
    'mine', 'mining', 'craft', 'crafting', 'smelt', 'build', 'building',
    'place', 'break', 'farm', 'farming', 'breed', 'trade', 'enchant',
    'loot', 'bridge', 'dig', 'collect', 'gather', 'survive', 'spawn',
    'portal', 'nether', 'end', 'redstone', 'iron', 'diamond', 'food',
    'armor', 'weapon', 'villager', 'house', 'base', 'chest', 'torch',
    'wood', 'stone', 'cobblestone'
}

GOAL_KEYWORDS = [
    'survival', 'starter house', 'base', 'mining', 'iron farm', 'food farm',
    'villager trading', 'diamond mining', 'nether portal', 'enchanting',
    'redstone', 'speedrun', 'combat', 'pvp', 'building'
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', action='append', default=[], help='YouTube video URL (repeatable)')
    parser.add_argument('--urls-file', type=str, default='', help='Path to newline-separated YouTube URLs')
    parser.add_argument('--languages', type=str, default='en,en-US', help='Transcript language priority')
    parser.add_argument('--max-videos', type=int, default=20)
    parser.add_argument('--proxy', type=str, default='', help='HTTP proxy URL for YouTube requests')
    parser.add_argument('--proxy-https', type=str, default='', help='HTTPS proxy URL for YouTube requests')
    parser.add_argument('--output', type=str, default=OUTPUT_DATA)
    parser.add_argument('--merge', action='store_true')
    parser.add_argument('--merge-target', type=str, default=MAIN_DATA)
    return parser.parse_args()


def resolve_proxy_urls(args: argparse.Namespace) -> tuple[str, str]:
    http_proxy = str(args.proxy or os.getenv('SOLASAI_YOUTUBE_PROXY') or os.getenv('HTTP_PROXY') or '').strip()
    https_proxy = str(args.proxy_https or os.getenv('SOLASAI_YOUTUBE_PROXY_HTTPS') or os.getenv('HTTPS_PROXY') or '').strip()
    if http_proxy and not https_proxy:
        https_proxy = http_proxy
    return http_proxy, https_proxy


def configure_urllib_proxy(http_proxy: str, https_proxy: str):
    proxies: dict[str, str] = {}
    if http_proxy:
        proxies['http'] = http_proxy
    if https_proxy:
        proxies['https'] = https_proxy
    if proxies:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler(proxies))
        urllib.request.install_opener(opener)


def create_transcript_api(http_proxy: str, https_proxy: str):
    if YouTubeTranscriptApi is None:
        return None
    if GenericProxyConfig is not None and (http_proxy or https_proxy):
        try:
            return YouTubeTranscriptApi(
                proxy_config=GenericProxyConfig(
                    http_url=http_proxy or None,
                    https_url=https_proxy or None,
                )
            )
        except Exception:
            pass
    try:
        return YouTubeTranscriptApi()
    except Exception:
        return None


def extract_video_id(url_or_id: str) -> str | None:
    text = str(url_or_id or '').strip()
    if not text:
        return None
    if re.fullmatch(r'[A-Za-z0-9_-]{11}', text):
        return text

    parsed = urllib.parse.urlparse(text)
    host = parsed.netloc.lower()
    if 'youtu.be' in host:
        candidate = parsed.path.strip('/').split('/')[0]
        return candidate if re.fullmatch(r'[A-Za-z0-9_-]{11}', candidate or '') else None

    if 'youtube.com' in host:
        query = urllib.parse.parse_qs(parsed.query)
        candidate = (query.get('v') or [''])[0]
        if re.fullmatch(r'[A-Za-z0-9_-]{11}', candidate or ''):
            return candidate
        parts = [p for p in parsed.path.split('/') if p]
        if len(parts) >= 2 and parts[0] in {'shorts', 'live'}:
            candidate = parts[1]
            if re.fullmatch(r'[A-Za-z0-9_-]{11}', candidate or ''):
                return candidate

    return None


def load_urls(args: argparse.Namespace) -> list[str]:
    urls = [u.strip() for u in args.url if str(u).strip()]
    if args.urls_file:
        with open(args.urls_file, 'r', encoding='utf-8') as handle:
            for line in handle:
                line = line.strip()
                if line and not line.startswith('#'):
                    urls.append(line)
    deduped = list(dict.fromkeys(urls))
    return deduped[: max(1, int(args.max_videos))]


def fetch_video_title(video_url: str) -> str:
    oembed_url = (
        'https://www.youtube.com/oembed?format=json&url=' +
        urllib.parse.quote(video_url, safe='')
    )
    try:
        with urllib.request.urlopen(oembed_url, timeout=8) as resp:
            payload = json.loads(resp.read().decode('utf-8', errors='ignore'))
            return str(payload.get('title') or '').strip()
    except Exception:
        return ''


def fetch_transcript(video_id: str, languages: list[str], transcript_api) -> tuple[list[dict], str]:
    last_reason = ''
    if transcript_api is not None:
        try:
            fetched = transcript_api.fetch(video_id, languages=languages)
            rows = [{'text': seg.text} for seg in fetched if getattr(seg, 'text', '')]
            if rows:
                return rows, ''
        except Exception as exc:
            # fall through to HTTP fallback below
            last_reason = f'youtube_transcript_api: {exc.__class__.__name__}'

    # Fallback path when youtube_transcript_api is unavailable in the runtime.
    tracks = fetch_caption_tracks(video_id)
    if not tracks:
        if not last_reason:
            last_reason = 'no caption tracks returned by YouTube'
        return [], last_reason

    preferred = [lang.lower() for lang in languages]
    sorted_tracks = sorted(
        tracks,
        key=lambda t: language_priority(t.get('lang_code', ''), preferred)
    )

    for track in sorted_tracks:
        row_data = fetch_transcript_from_track(video_id, track)
        if row_data:
            return row_data, ''

    if not last_reason:
        last_reason = 'caption tracks found, but transcript payload was empty or blocked'
    return [], last_reason


def language_priority(lang_code: str, preferred_languages: list[str]) -> int:
    lang = (lang_code or '').lower()
    for idx, preferred in enumerate(preferred_languages):
        if lang == preferred or lang.startswith(preferred + '-'):
            return idx
    return len(preferred_languages) + 1


_INNERTUBE_HEADERS = {
    'Content-Type': 'application/json',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                  'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36',
    'Accept-Language': 'en-US,en;q=0.9',
    'X-YouTube-Client-Name': '1',
    'X-YouTube-Client-Version': '2.20240415.01.00',
    'Origin': 'https://www.youtube.com',
}
_INNERTUBE_API_KEY = 'AIzaSyAO_FJ2SlqU8Q4STtR6HosjMFnEiP3cksk'
_INNERTUBE_PLAYER_URL = (
    'https://www.youtube.com/youtubei/v1/player?key=' + _INNERTUBE_API_KEY
)


def _innertube_player(video_id: str) -> dict:
    body = json.dumps({
        'videoId': video_id,
        'context': {
            'client': {
                'clientName': 'WEB',
                'clientVersion': '2.20240415.01.00',
                'hl': 'en',
                'gl': 'US',
            }
        },
    }).encode('utf-8')
    req = urllib.request.Request(
        _INNERTUBE_PLAYER_URL,
        data=body,
        headers=_INNERTUBE_HEADERS,
        method='POST',
    )
    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            return json.loads(resp.read().decode('utf-8', errors='ignore'))
    except Exception:
        return {}


def fetch_caption_tracks(video_id: str) -> list[dict]:
    player = _innertube_player(video_id)
    captions = player.get('captions') or {}
    renderer = captions.get('playerCaptionsTracklistRenderer') or {}
    raw_tracks = renderer.get('captionTracks') or []
    tracks: list[dict] = []
    for t in raw_tracks:
        if not isinstance(t, dict):
            continue
        tracks.append({
            'lang_code': t.get('languageCode', ''),
            'name': (t.get('name') or {}).get('simpleText', ''),
            'kind': t.get('kind', ''),
            'base_url': t.get('baseUrl', ''),
        })
    return tracks


def fetch_transcript_from_track(video_id: str, track: dict) -> list[dict]:
    base_url = str(track.get('base_url') or '').strip()
    if not base_url:
        return []

    # Request JSON3 format which includes timestamped segments
    if 'fmt=' in base_url:
        url = re.sub(r'fmt=[^&]*', 'fmt=json3', base_url)
    else:
        url = base_url + ('&' if '?' in base_url else '?') + 'fmt=json3'

    req = urllib.request.Request(url, headers={
        'User-Agent': _INNERTUBE_HEADERS['User-Agent'],
        'Accept-Language': _INNERTUBE_HEADERS['Accept-Language'],
    })
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            payload = json.loads(resp.read().decode('utf-8', errors='ignore'))
    except Exception:
        return []

    events = payload.get('events') if isinstance(payload, dict) else None
    if isinstance(events, list):
        rows: list[dict] = []
        for event in events:
            if not isinstance(event, dict):
                continue
            segs = event.get('segs')
            if not isinstance(segs, list):
                continue
            text = ''.join(str(seg.get('utf8', '')) for seg in segs if isinstance(seg, dict)).strip()
            text = clean_text(text)
            if text:
                rows.append({'text': text})
        return rows

    # Fallback: try XML (fmt=srv3 / default)
    url_xml = re.sub(r'fmt=[^&]*', '', base_url).rstrip('&?')
    req_xml = urllib.request.Request(url_xml, headers={
        'User-Agent': _INNERTUBE_HEADERS['User-Agent'],
    })
    try:
        with urllib.request.urlopen(req_xml, timeout=10) as resp:
            xml_text = resp.read().decode('utf-8', errors='ignore')
        root = ET.fromstring(xml_text)
        rows = []
        for text_el in root.iter('text'):
            txt = clean_text(text_el.text or '')
            if txt:
                rows.append({'text': txt})
        return rows
    except Exception:
        return []


def clean_text(text: str) -> str:
    text = re.sub(r'\[[^\]]*\]', ' ', str(text or ''))
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def split_sentences(text: str) -> list[str]:
    if not text:
        return []
    raw = re.split(r'(?<=[.!?])\s+|\n+', text)
    out: list[str] = []
    for sentence in raw:
        sentence = clean_text(sentence)
        if len(sentence) < 20:
            continue
        if len(sentence.split()) < 5:
            continue
        out.append(sentence)
    return out


def looks_actionable(sentence: str) -> bool:
    words = {w.strip('.,!?').lower() for w in sentence.split()}
    if len(words & ACTION_KEYWORDS) > 0:
        return True
    lower = sentence.lower()
    return any(token in lower for token in ['first', 'then', 'next', 'after that', 'finally'])


def infer_goal(title: str, action_sentences: list[str]) -> str:
    title_l = title.lower()
    for goal in GOAL_KEYWORDS:
        if goal in title_l:
            return goal

    token_counter: Counter[str] = Counter()
    for sentence in action_sentences:
        for token in re.findall(r"[a-zA-Z']+", sentence.lower()):
            if token in ACTION_KEYWORDS and len(token) > 2:
                token_counter[token] += 1

    if token_counter:
        common = [token for token, _ in token_counter.most_common(2)]
        return ' and '.join(common)

    return 'play survival effectively'


def build_pair(video_url: str, title: str, transcript_rows: list[dict]) -> tuple[str, str] | None:
    merged = ' '.join(clean_text(row.get('text', '')) for row in transcript_rows)
    sentences = split_sentences(merged)
    if not sentences:
        return None

    action_sentences = [s for s in sentences if looks_actionable(s)]
    if len(action_sentences) < 3:
        return None

    selected_steps: list[str] = []
    for sentence in action_sentences:
        if sentence in selected_steps:
            continue
        selected_steps.append(sentence)
        if len(selected_steps) >= 6:
            break

    goal = infer_goal(title, selected_steps)
    question = f'how do i {goal} in minecraft'
    steps_text = ' '.join(f'{idx + 1}) {step}' for idx, step in enumerate(selected_steps))
    answer = (
        f'From observed Minecraft tutorial behavior ({video_url}): {steps_text} '
        'Adapt to your world seed, tools, and biome.'
    )
    return question, clean_text(answer)


def parse_existing_pairs(path: str) -> set[tuple[str, str]]:
    if not os.path.exists(path):
        return set()
    pairs: set[tuple[str, str]] = set()
    pending_user = None
    with open(path, 'r', encoding='utf-8') as handle:
        for raw in handle:
            line = raw.strip()
            if line.startswith('User: '):
                pending_user = line[6:].strip()
            elif line.startswith('Assistant: ') and pending_user:
                pairs.add((pending_user, line[11:].strip()))
                pending_user = None
    return pairs


def write_pairs(path: str, pairs: list[tuple[str, str]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as handle:
        for question, answer in pairs:
            handle.write(f'User: {question}\n')
            handle.write(f'Assistant: {answer}\n')


def append_unique_pairs(path: str, pairs: list[tuple[str, str]]) -> int:
    existing = parse_existing_pairs(path)
    to_add = [(q, a) for q, a in pairs if (q, a) not in existing]
    if not to_add:
        return 0
    with open(path, 'a', encoding='utf-8') as handle:
        for question, answer in to_add:
            handle.write(f'\nUser: {question}\n')
            handle.write(f'Assistant: {answer}\n')
    return len(to_add)


def main() -> int:
    args = parse_args()
    http_proxy, https_proxy = resolve_proxy_urls(args)
    configure_urllib_proxy(http_proxy, https_proxy)
    transcript_api = create_transcript_api(http_proxy, https_proxy)

    urls = load_urls(args)
    if not urls:
        print('No URLs provided. Use --url or --urls-file.')
        return 1

    languages = [p.strip() for p in str(args.languages).split(',') if p.strip()]
    pairs: list[tuple[str, str]] = []

    for url in urls:
        vid = extract_video_id(url)
        if not vid:
            print(f'Skipping invalid YouTube URL: {url}')
            continue

        canonical_url = f'https://www.youtube.com/watch?v={vid}'
        title = fetch_video_title(canonical_url)
        transcript, reason = fetch_transcript(vid, languages, transcript_api)

        if not transcript:
            detail = f' [{reason}]' if reason else ''
            print(f'Skipping (no transcript): {canonical_url}{detail}')
            continue

        pair = build_pair(canonical_url, title, transcript)
        if not pair:
            print(f'Skipping (not enough actionable tutorial text): {canonical_url}')
            continue

        pairs.append(pair)
        print(f'Learned from: {canonical_url}')

    pairs = list(dict.fromkeys(pairs))
    write_pairs(args.output, pairs)
    print(f'YouTube training pairs written: {len(pairs)} -> {args.output}')

    if args.merge:
        added = append_unique_pairs(args.merge_target, pairs)
        print(f'Merged unique pairs: {added} -> {args.merge_target}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())