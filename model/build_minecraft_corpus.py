#!/usr/bin/env python3
"""
Expand conversations.txt with Minecraft-focused training pairs.

This generator intentionally creates a large corpus of structured, high-signal
 Q/A pairs rather than generic filler. It covers survival progression, combat,
 farms, exploration, building, redstone, and SolasAI mod usage patterns.
"""

from __future__ import annotations

import argparse
import os
import re


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DATA = os.path.join(BASE_DIR, 'data', 'minecraft_expanded_corpus.txt')
MAIN_DATA = os.path.join(BASE_DIR, 'data', 'conversations.txt')


TOPICS = [
    {
        'goal': 'survive the first day',
        'aliases': ['survive my first day', 'survive the first night', 'start a new survival world'],
        'steps': ['punch logs immediately', 'craft a crafting table and basic wooden tools', 'upgrade to stone tools fast', 'collect food before sunset', 'build a small shelter and place torches'],
        'tip': 'Do not wander too far from spawn until you have food, stone, and a safe place to stay.',
        'pitfall': 'A common mistake is exploring too long with no bed, no food, and no shelter ready.'
    },
    {
        'goal': 'get wood fast',
        'aliases': ['gather logs quickly', 'collect wood early game', 'farm wood at the start'],
        'steps': ['cut nearby trees first', 'turn logs into planks and sticks', 'craft a stone axe quickly', 'replant saplings near your base', 'keep extra logs for tools, torches, and chests'],
        'tip': 'A stone axe gives a big speed boost over punching or using your hand.',
        'pitfall': 'Do not burn through all your wood on decoration before making core tools and storage.'
    },
    {
        'goal': 'find iron early',
        'aliases': ['get iron quickly', 'mine iron early game', 'reach iron tools fast'],
        'steps': ['craft stone tools first', 'search surface caves or stony hills', 'light your path so mobs do not surround you', 'smelt raw iron as soon as you can', 'prioritize a shield, bucket, pickaxe, and armor pieces'],
        'tip': 'Your first iron should usually go into a shield and pickaxe because they increase survival and progression speed.',
        'pitfall': 'Do not waste your first iron on low-impact items when you still lack defense and utility.'
    },
    {
        'goal': 'find diamonds',
        'aliases': ['mine diamonds efficiently', 'get diamonds in survival', 'diamond mine the smart way'],
        'steps': ['bring an iron pickaxe, food, torches, and a water bucket', 'mine at strong diamond levels', 'use branch mining or explore deep caves carefully', 'block lava flows before mining exposed ore', 'save diamonds for key tools first'],
        'tip': 'A water bucket makes deep mining much safer because it controls lava and prevents fall damage.',
        'pitfall': 'Do not stand directly on top of a diamond vein and break the block under yourself into lava.'
    },
    {
        'goal': 'make a starter house',
        'aliases': ['build an early base', 'make a simple survival house', 'build a safe starter home'],
        'steps': ['pick flat land near wood, water, and animals', 'start with a compact layout', 'add chests, furnaces, a bed, and lighting', 'use slabs, stairs, and windows to make it look better', 'leave space to expand farms and storage later'],
        'tip': 'A small functional base is better than a huge unfinished shell on day one.',
        'pitfall': 'Many players overbuild too early and delay tools, food, and mining progress.'
    },
    {
        'goal': 'get food fast',
        'aliases': ['make food early game', 'stop starving in minecraft', 'build up a food supply'],
        'steps': ['kill a few animals or gather easy crops immediately', 'cook food instead of eating it raw', 'plant seeds near water', 'keep extra food in your hotbar before exploring', 'expand into a steady crop or animal farm'],
        'tip': 'Cooked food and a simple wheat or potato farm stabilize survival much faster than constant hunting.',
        'pitfall': 'Do not sprint everywhere if you are already low on food.'
    },
    {
        'goal': 'use a shield well',
        'aliases': ['fight with a shield', 'survive combat with a shield', 'block attacks better'],
        'steps': ['craft a shield as soon as you get iron', 'raise it before skeleton shots or creeper blasts', 'lower it only when you are ready to hit', 'keep distance from axe-wielding enemies', 'use terrain and corners to limit incoming attacks'],
        'tip': 'A shield dramatically improves early survival and is one of the best first iron crafts.',
        'pitfall': 'Holding a shield too late is almost the same as not having one at all.'
    },
    {
        'goal': 'fight skeletons safely',
        'aliases': ['beat skeletons', 'survive skeletons in caves', 'deal with skeleton arrows'],
        'steps': ['use a shield or move behind cover', 'close distance between arrow volleys', 'attack after they shoot', 'avoid flat open ground', 'heal before chain-fighting multiple ranged mobs'],
        'tip': 'Corners and uneven terrain make skeletons much easier to control.',
        'pitfall': 'Running straight at a skeleton in the open usually costs a lot of health.'
    },
    {
        'goal': 'avoid creeper deaths',
        'aliases': ['survive creepers', 'fight creepers safely', 'stop creepers from blowing me up'],
        'steps': ['light your base area well', 'listen for footsteps and hissing', 'hit creepers once and back away', 'use a shield or bow when possible', 'never corner yourself while fighting one'],
        'tip': 'Backing up after each hit is the simplest and safest creeper pattern.',
        'pitfall': 'Trying to tank a creeper blast in weak armor usually ends badly.'
    },
    {
        'goal': 'explore caves safely',
        'aliases': ['go caving without dying', 'survive cave exploration', 'loot caves better'],
        'steps': ['bring blocks, torches, food, a shield, and water', 'light branches as you enter them', 'block dangerous openings when needed', 'retreat to heal instead of forcing every fight', 'mark your exit path clearly'],
        'tip': 'Controlled cave exploration is faster than reckless rushing because you lose less gear and time.',
        'pitfall': 'Do not drop into dark vertical spaces without checking for mobs or lava first.'
    },
    {
        'goal': 'make a wheat farm',
        'aliases': ['start farming wheat', 'grow wheat efficiently', 'build a basic crop farm'],
        'steps': ['place the farm near water', 'hoe enough farmland for your seeds', 'light the area so growth continues safely', 'harvest mature crops and replant immediately', 'expand once you have spare seeds'],
        'tip': 'Even a tiny early crop farm saves time because it becomes a renewable food source.',
        'pitfall': 'Do not jump on farmland and break your own planting rows.'
    },
    {
        'goal': 'breed animals',
        'aliases': ['make an animal farm', 'breed cows and sheep', 'get renewable meat and leather'],
        'steps': ['fence a small pen', 'bring two animals inside with the right food', 'breed them and wait for cooldowns', 'expand the herd slowly', 'avoid overkilling your breeders'],
        'tip': 'Cows are especially valuable because they give both food and leather for books.',
        'pitfall': 'Killing all adults too early resets your farm progress.'
    },
    {
        'goal': 'trade with villagers',
        'aliases': ['start villager trading', 'use villagers for gear', 'build a trading setup'],
        'steps': ['secure a village or cure zombie villagers', 'protect them from mobs', 'assign workstations to get useful professions', 'level trades you actually need', 'lock in strong trades before changing job blocks'],
        'tip': 'Librarians and toolsmith-related trades can speed progression dramatically.',
        'pitfall': 'Do not let zombies or raids wipe out your trading setup.'
    },
    {
        'goal': 'get mending',
        'aliases': ['find a mending book', 'obtain mending enchantment', 'set up mending from villagers'],
        'steps': ['secure a villager', 'use a lectern to reroll librarian trades', 'check the first book trade repeatedly', 'lock the trade when mending appears', 'pair it with an XP source later'],
        'tip': 'Villager rerolling is usually the most reliable way to get mending in survival.',
        'pitfall': 'Do not break the workstation after trading if you want to keep the mending offer.'
    },
    {
        'goal': 'make an enchanting setup',
        'aliases': ['build an enchant table room', 'start enchanting gear', 'get level 30 enchants'],
        'steps': ['mine obsidian and diamonds for the table', 'collect leather and sugar cane for books', 'place bookshelves with the correct gap', 'gain XP from mining, smelting, or farming', 'enchant core tools before luxury items'],
        'tip': 'A full bookshelf setup is worth rushing because it improves gear quality a lot.',
        'pitfall': 'If bookshelves are placed wrong, you lose the level 30 setup without noticing.'
    },
    {
        'goal': 'make a nether portal',
        'aliases': ['go to the nether', 'build a portal quickly', 'enter the nether safely'],
        'steps': ['collect obsidian or cast the frame with lava and water', 'bring armor, food, blocks, and a shield', 'spawn-proof the area around the portal', 'write down portal coordinates', 'avoid overcommitting on the first trip'],
        'tip': 'Your first Nether trip should be short and goal-focused, not a full expedition.',
        'pitfall': 'Many runs die because players forget spare blocks and safe portal positioning.'
    },
    {
        'goal': 'survive the nether',
        'aliases': ['travel the nether safely', 'stop dying in the nether', 'explore the nether better'],
        'steps': ['wear at least one gold piece around piglins', 'carry fire resistance if possible', 'bridge carefully with guard rails when needed', 'mark paths back to your portal', 'fight only when the terrain favors you'],
        'tip': 'Navigation discipline matters more in the Nether because getting lost is expensive.',
        'pitfall': 'Do not sprint into open ledges or attack piglins casually near a crowd.'
    },
    {
        'goal': 'get blaze rods',
        'aliases': ['farm blazes', 'beat a blaze spawner', 'collect blaze rods for the end'],
        'steps': ['locate a Nether fortress', 'secure escape routes around the spawner', 'use cover to block fireballs', 'fight blazes one at a time when possible', 'leave with enough rods for brewing and eyes'],
        'tip': 'A few placed blocks around the spawner make blaze fights much easier to control.',
        'pitfall': 'Charging a blaze spawner room without cover usually causes chain damage and panic.'
    },
    {
        'goal': 'barter with piglins',
        'aliases': ['trade with piglins', 'get ender pearls from piglins', 'use gold bartering'],
        'steps': ['wear gold armor first', 'trap or isolate piglins in a safe area', 'drop gold ingots one at a time or use a bartering setup', 'sort the drops you need', 'store pearls and utility items for progression'],
        'tip': 'Bartering is a reliable supplement for ender pearls and fire resistance materials.',
        'pitfall': 'Do not anger piglins in the same area where you plan to barter.'
    },
    {
        'goal': 'find a stronghold',
        'aliases': ['locate the stronghold', 'use eyes of ender correctly', 'reach the end portal'],
        'steps': ['craft enough eyes of ender', 'throw an eye and follow the direction', 'move a long distance and throw again to triangulate', 'dig carefully when the eye changes direction sharply', 'search stone brick structures underground'],
        'tip': 'Triangulating from two distant throws is faster than following every eye step-by-step.',
        'pitfall': 'Do not dig straight down when you think you are above the stronghold.'
    },
    {
        'goal': 'beat the ender dragon',
        'aliases': ['kill the dragon', 'win the end fight', 'finish the dragon fight safely'],
        'steps': ['bring strong armor, bow, blocks, water, and food', 'destroy end crystals first', 'avoid looking at endermen unless ready to fight', 'use water or careful movement to survive knockback', 'hit the dragon during perches only when it is safe'],
        'tip': 'The fight gets much easier once all crystals are gone and you keep a calm rhythm.',
        'pitfall': 'Greeding too many hits during a perch often leads to unnecessary deaths.'
    },
    {
        'goal': 'get elytra',
        'aliases': ['find an end city ship', 'unlock elytra travel', 'loot end cities safely'],
        'steps': ['beat the dragon first', 'enter the outer End through a gateway', 'bridge carefully between islands', 'loot cities methodically and watch for shulkers', 'take the elytra from the ship and bring rockets later'],
        'tip': 'Slow, safe End city looting is better than rushing and falling into the void.',
        'pitfall': 'Do not carry irreplaceable gear if you are not confident crossing End gaps yet.'
    },
    {
        'goal': 'get ancient debris',
        'aliases': ['mine netherite', 'find ancient debris efficiently', 'upgrade to netherite'],
        'steps': ['bring fire resistance, beds or TNT if using them, and strong picks', 'mine in good Nether depth ranges', 'clear lava safely before exposing debris', 'collect gold and upgrade templates as needed', 'upgrade your best diamond gear first'],
        'tip': 'Netherite is strongest when used on the items you keep on you most often.',
        'pitfall': 'Do not use explosive mining carelessly in cramped lava-heavy terrain.'
    },
    {
        'goal': 'brew potions',
        'aliases': ['start brewing', 'make useful potions', 'set up a brewing stand'],
        'steps': ['get blaze rods for powder', 'collect nether wart and bottles', 'brew awkward potions first', 'add the ingredient for the effect you want', 'store utility potions before risky fights'],
        'tip': 'Potion brewing becomes much easier once you memorize the awkward potion base step.',
        'pitfall': 'Forgetting blaze powder as brewing fuel is a very common setup mistake.'
    },
    {
        'goal': 'make an iron farm',
        'aliases': ['build a basic iron farm', 'farm iron automatically', 'set up villager iron production'],
        'steps': ['bring villagers to a safe controlled area', 'set up beds and work access correctly', 'add the zombie or scare mechanic the design requires', 'spawn-proof nearby surfaces', 'collect the iron safely with lava or water channels'],
        'tip': 'Use a proven design for your version because villager mechanics are strict.',
        'pitfall': 'Small placement errors often break iron farms completely.'
    },
    {
        'goal': 'make an xp farm',
        'aliases': ['get experience fast', 'farm xp efficiently', 'build a simple xp setup'],
        'steps': ['pick a source like furnaces, mobs, villagers, or quartz mining', 'design around reliability first', 'store drops automatically when possible', 'use the farm regularly to repair mending gear', 'upgrade throughput only after the farm works consistently'],
        'tip': 'A simple dependable XP farm beats a complicated broken one.',
        'pitfall': 'Do not chase a huge build before you understand the core spawning or storage logic.'
    },
    {
        'goal': 'build a mob farm',
        'aliases': ['farm hostile mobs', 'make a dark room mob grinder', 'collect mob drops automatically'],
        'steps': ['build high enough or spawn-proof the surrounding area', 'create dark spawning platforms', 'move mobs into a kill chamber', 'collect drops safely', 'light caves nearby if rates are poor'],
        'tip': 'Spawn-proofing outside the farm often matters as much as the farm itself.',
        'pitfall': 'If nearby caves are unlit, your farm can feel much slower than expected.'
    },
    {
        'goal': 'organize storage',
        'aliases': ['make storage better', 'sort my items', 'build a useful storage room'],
        'steps': ['group items by type and frequency of use', 'keep crafting materials near workstations', 'label important chests or sections', 'store backup gear separately from daily tools', 'leave room for expansion as your world grows'],
        'tip': 'Good storage saves more time than most players expect.',
        'pitfall': 'One giant unsorted chest room causes constant friction later.'
    },
    {
        'goal': 'branch mine efficiently',
        'aliases': ['strip mine better', 'mine with tunnels', 'get ores from branch mining'],
        'steps': ['choose a productive depth', 'make a main hallway with side branches', 'keep branch spacing efficient', 'light and mark your route clearly', 'return to smelt and sort instead of hoarding everything on one trip'],
        'tip': 'Structured branch mining is boring but very reliable for steady ore income.',
        'pitfall': 'If you tunnel with poor spacing, you waste time rechecking the same blocks.'
    },
    {
        'goal': 'fight in pvp better',
        'aliases': ['improve minecraft pvp', 'win more melee fights', 'get better at combat'],
        'steps': ['manage spacing and movement first', 'time hits instead of spam clicking blindly', 'use a shield or utility item when the matchup calls for it', 'keep healing ready on your hotbar', 'disengage when the trade is bad'],
        'tip': 'Positioning and rhythm usually matter more than panic aggression.',
        'pitfall': 'Taking bad fights while low on healing or trapped in bad terrain loses games quickly.'
    },
    {
        'goal': 'aim with a bow',
        'aliases': ['shoot arrows better', 'use bows in combat', 'land more bow shots'],
        'steps': ['fully charge the bow when possible', 'lead moving targets a little', 'shoot from stable footing', 'use cover between shots', 'carry enough arrows for long fights'],
        'tip': 'Bow accuracy improves a lot when you slow down and pick cleaner shots.',
        'pitfall': 'Spamming rushed low-charge shots wastes arrows and pressure.'
    },
    {
        'goal': 'clutch with a water bucket',
        'aliases': ['learn water bucket clutches', 'survive falls with water', 'practice bucket saving'],
        'steps': ['keep the bucket on a comfortable hotbar slot', 'look down early as you fall', 'place water just before impact', 'pick it back up fast if needed', 'practice from safe heights before using it in real runs'],
        'tip': 'Consistent muscle memory matters more than one flashy successful clutch.',
        'pitfall': 'Waiting too long to swap to the bucket is the usual reason clutches fail.'
    },
    {
        'goal': 'bridge safely',
        'aliases': ['speed bridge safer', 'cross gaps without falling', 'build bridges in risky areas'],
        'steps': ['carry plenty of blocks', 'watch your angle and rhythm', 'add side rails in dangerous spots', 'pause if you lose timing', 'bridge from positions where knockback is less likely'],
        'tip': 'Safe bridging is about consistency, not just raw speed.',
        'pitfall': 'Overcommitting to fast bridging under pressure leads to avoidable falls.'
    },
    {
        'goal': 'play bedwars better',
        'aliases': ['improve at bedwars', 'win more bedwars games', 'rush beds smarter'],
        'steps': ['gather early resources efficiently', 'coordinate a quick first rush or defense plan', 'protect your bed with simple useful layers', 'watch mid control and enemy rotations', 'push advantages after a successful break'],
        'tip': 'Bedwars is easier when your early game has a clear plan instead of random rushing.',
        'pitfall': 'Ignoring defense and map awareness can throw good starts away.'
    },
    {
        'goal': 'escape mobs at night',
        'aliases': ['survive the night outside', 'get away from hostile mobs', 'reset a bad night fight'],
        'steps': ['sprint only when necessary', 'use terrain to break line of sight', 'block off small choke points', 'eat before you are one hit from death', 'return to a lit safe zone quickly'],
        'tip': 'Resetting a fight is often smarter than forcing one when multiple mobs are stacked on you.',
        'pitfall': 'Panic running into darker terrain usually adds even more enemies.'
    },
    {
        'goal': 'breathe underwater longer',
        'aliases': ['survive underwater', 'explore ocean safely', 'loot underwater without drowning'],
        'steps': ['carry doors, magma understanding, or water-breathing potions depending on version and goal', 'watch your oxygen before committing deeper', 'clear hostile drowned carefully', 'loot fast and reset your air often', 'upgrade gear if you plan to stay underwater a lot'],
        'tip': 'Underwater exploration is much easier once you prepare breathing and mobility tools first.',
        'pitfall': 'Drowning because you stayed one loot chest too long is a common avoidable death.'
    },
    {
        'goal': 'build with better detail',
        'aliases': ['make builds look nicer', 'improve building depth', 'decorate a minecraft house better'],
        'steps': ['vary block textures slightly', 'use stairs, slabs, trapdoors, and fences for depth', 'break up flat walls with supports or windows', 'add a clear roof shape', 'finish the area around the build with paths and landscaping'],
        'tip': 'Exterior depth and surrounding terrain work usually improve a build more than size alone.',
        'pitfall': 'Large flat walls with no texture or shape make even expensive builds look unfinished.'
    },
    {
        'goal': 'learn redstone basics',
        'aliases': ['understand simple redstone', 'start using redstone', 'make beginner redstone builds'],
        'steps': ['learn how power travels through dust', 'practice with levers, buttons, pistons, and repeaters', 'build one small mechanism at a time', 'test each section before combining them', 'label or separate parts if the circuit gets messy'],
        'tip': 'Redstone gets easier once you stop trying to build everything as one giant first attempt.',
        'pitfall': 'Debugging is much harder when you skip small tests and wire the whole build at once.'
    },
    {
        'goal': 'avoid getting lost',
        'aliases': ['navigate better in survival', 'find my base again', 'travel without losing my way'],
        'steps': ['write down coordinates of important places', 'use landmarks and paths', 'carry materials for markers', 'sleep or set spawn strategically', 'avoid long aimless trips when your inventory is full'],
        'tip': 'Simple navigation habits save more gear than any fancy recovery plan.',
        'pitfall': 'Exploring with no coordinates or markers turns small detours into full recovery problems.'
    },
    {
        'goal': 'manage inventory better',
        'aliases': ['keep my inventory clean', 'organize my hotbar', 'carry the right items'],
        'steps': ['assign fixed hotbar slots for tools, food, blocks, and utility', 'drop junk regularly', 'carry backup torches and blocks', 'store valuables before risky trips', 'restock before leaving base again'],
        'tip': 'A clean hotbar improves reaction speed in combat and exploration.',
        'pitfall': 'A cluttered inventory makes emergencies much harder to handle.'
    },
]


MOD_USAGE_PAIRS = [
    ('how do i ask solasai something in minecraft', 'Use /solasai ask <prompt> or /askai <prompt>. The mod sends your prompt to the backend and posts the reply back into chat.'),
    ('how do i change the solasai backend', 'Use /solasai backend <url>. The client stores the endpoint and sends future chat and mc-agent requests there.'),
    ('how do i check the solasai backend url', 'Use /solasai backend with no extra argument. It prints the current configured backend endpoint.'),
    ('how do i see the current ai task', 'Use /solasai task or /solasai tas. The client shows the current task text from the controller.'),
    ('how do i check the stronghold estimate', 'Use /solasai stronghold. If the mod has triangulated enough eye throws, it will print the estimated stronghold coordinates.'),
    ('can solasai read chat', 'Yes. The Fabric client registers a chat receive event and can react to incoming chat lines when auto-reply is enabled.'),
    ('can solasai send chat messages', 'Yes. The client can send normal chat through the player network handler and can also issue slash commands when needed.'),
    ('how do i toggle the ai overlay', 'Press F3 plus Tab. The mod toggles the debug overlay and prints whether the overlay is now on or off.'),
    ('how do i start ai control', 'Press Ctrl plus Tab while in game with no screen open. That opens the SolasAI prompt screen so you can start the controller with an objective.'),
    ('how do i stop ai control', 'Press Escape or Escape plus Tab while the AI is active. The controller stops and releases movement inputs.'),
    ('what does ai join do', 'AI Join connects to a multiplayer server with an objective and starts the controller once the world loads.'),
    ('what does solasai bases do', 'The /solasai bases command asks the bot service for saved base candidates on the current server and prints them in chat.'),
    ('can the mod run swarms', 'Yes. The client can send a swarm start request to the bot service with count, username mode, jobs, and auto-think settings.'),
    ('does solasai use chat history', 'For chat replies, the backend keeps short session context so responses can stay consistent across recent messages.'),
    ('can solasai read minecraft state', 'Yes. The mc-agent endpoint receives a game state snapshot including health, food, enemies, and other context for decision making.'),
    ('how do i make chat replies trigger', 'Mention solasai in chat when auto-reply is enabled or use the ask commands directly if you want a forced backend reply.'),
    ('can solasai send slash commands', 'Yes. The controller already uses sendChatCommand when possible and falls back to a slash-prefixed chat message if needed.'),
]


QUESTION_TEMPLATES = [
    'how do i {goal} in minecraft',
    'best way to {goal} in minecraft',
    'help me {goal} in minecraft survival',
    'step by step how do i {goal} in minecraft',
    'what is the safest way to {goal} in minecraft',
    'tips to {goal} in minecraft',
    'how should i {goal} in minecraft',
    'what should i do to {goal} in minecraft',
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default=OUTPUT_DATA)
    parser.add_argument('--merge', action='store_true')
    parser.add_argument('--merge-target', type=str, default=MAIN_DATA)
    return parser.parse_args()


def clean_text(text: str) -> str:
    return re.sub(r'\s+', ' ', str(text or '')).strip()


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


def build_topic_answers(topic: dict) -> list[str]:
    steps = topic['steps']
    numbered = ' '.join(f'{idx + 1}) {step}.' for idx, step in enumerate(steps))
    short_steps = ' Then '.join(step[0].upper() + step[1:] for step in steps[:3]) + '.'
    tip = topic['tip']
    pitfall = topic['pitfall']
    goal = topic['goal']
    return [
        clean_text(f'To {goal}, follow this order: {numbered} {tip} {pitfall}'),
        clean_text(f'A solid survival plan is to {short_steps} After that, finish the rest of the setup instead of rushing. {tip} {pitfall}'),
        clean_text(f'The reliable way to {goal} is to prepare first, execute in a safe order, and avoid greed. {numbered} {tip}'),
    ]


def generate_pairs() -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []

    for topic in TOPICS:
        question_goals = [topic['goal'], *topic.get('aliases', [])]
        answers = build_topic_answers(topic)
        for goal_variant in question_goals:
            for template in QUESTION_TEMPLATES:
                question = clean_text(template.format(goal=goal_variant))
                for answer in answers:
                    pairs.append((question, answer))

    for question, answer in MOD_USAGE_PAIRS:
        pairs.append((clean_text(question), clean_text(answer)))

    slang_inputs = ['hello', 'hi', 'hey', 'yoo', 'yo', 'sup', 'dawg', 'what are you doing']
    slang_outputs = [
        'I am ready. Give me a Minecraft goal, a question, or a task and I will answer clearly.',
        'Tell me what you want to do in Minecraft and I will break it down step by step.',
        'I can help with Minecraft survival, progression, combat, building, and SolasAI mod commands.'
    ]
    for user_text in slang_inputs:
        for answer in slang_outputs:
            pairs.append((user_text, answer))

    deduped = list(dict.fromkeys((clean_text(q), clean_text(a)) for q, a in pairs if clean_text(q) and clean_text(a)))
    return deduped


def main() -> int:
    args = parse_args()
    pairs = generate_pairs()
    write_pairs(args.output, pairs)
    print(f'Minecraft corpus pairs written: {len(pairs)} -> {args.output}')

    if args.merge:
        added = append_unique_pairs(args.merge_target, pairs)
        print(f'Merged unique pairs: {added} -> {args.merge_target}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())