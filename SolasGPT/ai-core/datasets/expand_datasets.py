#!/usr/bin/env python3
"""
Expand training datasets by generating synthetic conversations,
code examples, and Minecraft knowledge.
"""

import random
import json

# ============================================
# CONVERSATION DATA GENERATION
# ============================================

CONVERSATION_TEMPLATES = [
    # Greetings
    ("[CHAT] User: Hello\nAssistant: Hi! How can I help you today?\n\n",
     "[CHAT] User: Hey there\nAssistant: Hello! What can I do for you?\n\n",
     "[CHAT] User: Hi\nAssistant: Hey! How are you doing?\n\n",
     "[CHAT] User: Good morning\nAssistant: Good morning! What brings you here today?\n\n"),
    
    # Questions about capabilities
    ("[CHAT] User: What can you do?\nAssistant: I can help with coding, answer questions, and assist with Minecraft gameplay. What would you like help with?\n\n",
     "[CHAT] User: How can you help me?\nAssistant: I can assist with programming tasks, general questions, and Minecraft strategies. What do you need?\n\n",
     "[CHAT] User: What are your abilities?\nAssistant: I can code, chat, and help with Minecraft. How may I assist you today?\n\n"),
    
    # Coding questions
    ("[CODE] User: How do I write a for loop in Python?\nAssistant: Here's a basic for loop in Python:\n```python\nfor i in range(10):\n    print(i)\n```\nThis will print numbers 0 through 9.\n\n",
     "[CODE] User: Explain Python lists\nAssistant: Lists in Python are ordered collections that can hold different types:\n```python\nmy_list = [1, 2, 'hello', True]\nmy_list.append('new item')\nprint(my_list[0])  # prints 1\n```\n\n",
     "[CODE] User: What is a function?\nAssistant: A function is a reusable block of code:\n```python\ndef greet(name):\n    return f'Hello, {name}!'\n\nprint(greet('Alice'))\n```\n\n"),
    
    # Minecraft questions
    ("[MINECRAFT] User: How do I find diamonds in Minecraft?\nAssistant: To find diamonds: mine at Y-level -59 to -64 in version 1.18+, bring iron pickaxe or better, and watch for lava. Diamond ore spawns in small veins.\n\n",
     "[MINECRAFT] User: How do I make a farm?\nAssistant: Basic farm steps: find water source, till dirt with hoe in 4-block radius of water, plant seeds, add light sources. Crops grow faster with water nearby.\n\n",
     "[MINECRAFT] User: What's the best way to fight the Ender Dragon?\nAssistant: Ender Dragon tips: bring bow with arrows, destroy end crystals first, wear diamond armor, bring ender pearls, and aim for the head when it perches.\n\n"),
    
    # General knowledge
    ("[CHAT] User: What is Python?\nAssistant: Python is a high-level programming language known for its simplicity and readability. It's great for beginners and used in web development, AI, data science, and more.\n\n",
     "[CHAT] User: Explain variables\nAssistant: Variables store data values. In Python: `x = 5` creates a variable named 'x' storing the number 5. Variables can change: `x = 10` updates it to 10.\n\n",
     "[CHAT] User: What are data types?\nAssistant: Data types define what kind of data is stored: integers (1, 2, 3), floats (3.14), strings ('hello'), booleans (True/False), and more complex types like lists and dictionaries.\n\n"),
    
    # Problem solving
    ("[CHAT] User: My code has an error\nAssistant: I can help! Please share: the error message, the code that's failing, and what you expected to happen. This will help me diagnose the issue.\n\n",
     "[CHAT] User: How do I debug?\nAssistant: Debugging tips: read error messages carefully, use print() statements to check values, test small sections separately, and check for typos in variable names.\n\n",
     "[CHAT] User: I'm stuck on a problem\nAssistant: Let's break it down: What are you trying to achieve? What have you tried so far? What specific part is confusing? I'm here to help!\n\n"),
]

# ============================================
# CODE EXAMPLES
# ============================================

CODE_EXAMPLES = [
    # Python basics
    """# Python: Print Hello World
print("Hello, World!")
""",
    
    """# Python: Variables and types
name = "Alice"
age = 25
height = 5.6
is_student = True
""",
    
    """# Python: Lists
fruits = ["apple", "banana", "cherry"]
fruits.append("orange")
print(fruits[0])
print(len(fruits))
""",
    
    """# Python: Dictionaries
person = {
    "name": "Bob",
    "age": 30,
    "city": "New York"
}
print(person["name"])
""",
    
    """# Python: For loop
for i in range(5):
    print(f"Count: {i}")
""",
    
    """# Python: While loop
count = 0
while count < 5:
    print(count)
    count += 1
""",
    
    """# Python: Function
def add_numbers(a, b):
    return a + b

result = add_numbers(5, 3)
print(result)
""",
    
    """# Python: Class
class Dog:
    def __init__(self, name):
        self.name = name
    
    def bark(self):
        print(f"{self.name} says Woof!")

my_dog = Dog("Buddy")
my_dog.bark()
""",
    
    """# Python: List comprehension
numbers = [1, 2, 3, 4, 5]
squared = [x**2 for x in numbers]
print(squared)
""",
    
    """# Python: File reading
with open('file.txt', 'r') as f:
    content = f.read()
    print(content)
""",
]

# ============================================
# MINECRAFT KNOWLEDGE
# ============================================

MINECRAFT_FACTS = [
    "Diamonds spawn most commonly at Y-level -59 in Minecraft 1.18 and later.",
    "A full set of diamond armor requires 24 diamonds.",
    "Netherite is stronger than diamond and doesn't burn in lava.",
    "To make a Nether portal, you need at least 10 obsidian blocks.",
    "Creepers explode when they get within 3 blocks of a player.",
    "You need 3 iron ingots and 4 wooden planks to craft an anvil.",
    "Enchanting tables require 15 bookshelves for maximum level enchantments.",
    "Ender pearls can teleport you but cause 5 damage when used.",
    "The Wither boss requires 4 soul sand and 3 wither skeleton skulls to summon.",
    "Redstone dust can carry power up to 15 blocks without a repeater.",
    "Villagers will breed when given enough food and there are available beds.",
    "Iron golems spawn naturally in villages with at least 10 villagers.",
    "The Elytra wings are found only in End Ship treasure rooms.",
    "Shulker boxes retain their items when broken, making them portable storage.",
    "Beacon pyramids require iron, gold, emerald, or diamond blocks.",
    "Tridents are obtained as rare drops from drowned mobs.",
    "Mending enchantment repairs items using experience orbs.",
    "Fortune III can yield up to 4 diamonds per diamond ore block.",
    "Soul speed enchantment only works on soul sand and soul soil.",
    "Ancient debris spawns in the Nether at Y-levels 8-22.",
]

MINECRAFT_TUTORIALS = [
    """How to build a simple house in Minecraft:
1. Gather wood by punching trees
2. Craft planks and sticks from wood
3. Build walls 5x5 blocks, 4 blocks high
4. Add a roof with stairs or slabs
5. Place door, windows, and torches
6. Add bed and crafting table inside""",

    """Efficient mining strategy:
1. Mine straight down to Y-level -59
2. Create a 2x1 tunnel
3. Mine branch tunnels every 3 blocks
4. Each branch tunnel should be 2 blocks high
5. Bring torches, food, and spare pickaxes
6. Watch for lava and water""",

    """How to defeat the Ender Dragon:
1. Gather supplies: bow, arrows, diamond armor
2. Bring ender pearls and golden apples
3. Destroy end crystals first (shoot or climb)
4. Shoot dragon when flying or perched
5. Drink potions for extra strength
6. Collect dragon egg after victory""",

    """Starting a farm:
1. Find flat land near water
2. Hoe dirt blocks (right-click with hoe)
3. Plant seeds within 4 blocks of water
4. Add torches or other light sources
5. Wait for crops to grow (wheat turns golden)
6. Harvest and replant""",
]

# ============================================
# GENERATOR FUNCTIONS
# ============================================

def generate_conversations(num_samples=1000):
    """Generate synthetic conversations."""
    conversations = []
    for _ in range(num_samples):
        template_group = random.choice(CONVERSATION_TEMPLATES)
        conversations.append(random.choice(template_group))
    return ''.join(conversations)

def generate_code_examples(num_samples=500):
    """Generate code examples with explanations."""
    examples = []
    for _ in range(num_samples):
        code = random.choice(CODE_EXAMPLES)
        examples.append(f"[CODE] {code}\n")
    return '\n'.join(examples)

def generate_minecraft_knowledge(num_samples=500):
    """Generate Minecraft facts and tutorials."""
    knowledge = []
    for _ in range(num_samples):
        item = random.choice(MINECRAFT_FACTS + MINECRAFT_TUTORIALS)
        knowledge.append(f"[MINECRAFT] {item}\n\n")
    return ''.join(knowledge)

# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  Expanding Training Datasets")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
    
    # Generate conversation data
    print("Generating conversation data...")
    conv_data = generate_conversations(2000)
    with open('conversation/synthetic_conversations.txt', 'w') as f:
        f.write(conv_data)
    print(f"  ✓ Created: synthetic_conversations.txt ({len(conv_data):,} chars)")
    
    # Generate code examples
    print("Generating code examples...")
    code_data = generate_code_examples(1000)
    with open('coding/synthetic_code.txt', 'w') as f:
        f.write(code_data)
    print(f"  ✓ Created: synthetic_code.txt ({len(code_data):,} chars)")
    
    # Generate Minecraft knowledge
    print("Generating Minecraft knowledge...")
    mc_data = generate_minecraft_knowledge(1000)
    with open('minecraft/synthetic_knowledge.txt', 'w') as f:
        f.write(mc_data)
    print(f"  ✓ Created: synthetic_knowledge.txt ({len(mc_data):,} chars)")
    
    # Calculate totals
    total_chars = len(conv_data) + len(code_data) + len(mc_data)
    total_mb = total_chars / (1024 * 1024)
    
    print(f"\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"  Total new data: {total_chars:,} chars (~{total_mb:.1f} MB)")
    print(f"  Ready to train!")
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
