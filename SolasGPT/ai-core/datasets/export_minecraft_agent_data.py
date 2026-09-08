#!/usr/bin/env python3
"""Export Minecraft agent gameplay data to training dataset."""

import os
import json
from pathlib import Path

def load_agent_memory():
    """Load mc_agent_memory.json from turbowarp-ai-backend."""
    memory_path = Path(__file__).parent.parent.parent / 'turbowarp-ai-backend' / 'mc_agent_memory.json'
    if not memory_path.exists():
        print(f"⚠️ Agent memory not found at {memory_path}")
        return {}
    
    with open(memory_path, 'r') as f:
        return json.load(f)

def load_observations():
    """Load observation files from /tmp/solasai-observations."""
    obs_dir = Path('/tmp/solasai-observations')
    if not obs_dir.exists():
        print(f"⚠️ Observations directory not found at {obs_dir}")
        return []
    
    all_observations = []
    for obs_file in obs_dir.glob('*-observations.json'):
        try:
            with open(obs_file, 'r') as f:
                data = json.load(f)
                all_observations.extend(data)
        except Exception as e:
            print(f"⚠️ Failed to load {obs_file}: {e}")
    
    return all_observations

def format_action(action):
    """Format action dictionary to readable text."""
    parts = []
    if action.get('forward'): parts.append('move forward')
    if action.get('back'): parts.append('move back')
    if action.get('left'): parts.append('strafe left')
    if action.get('right'): parts.append('strafe right')
    if action.get('jump'): parts.append('jump')
    if action.get('sprint'): parts.append('sprint')
    if action.get('sneak'): parts.append('sneak')
    if action.get('attack'): parts.append('attack')
    if action.get('use'): parts.append('use item')
    
    return ', '.join(parts) if parts else 'idle'

def format_observation(obs):
    """Format observation to training example."""
    action = obs.get('actionType', 'unknown')
    block = obs.get('blockName', '')
    item = obs.get('itemName', '')
    player = obs.get('playerName', 'player')
    
    if action == 'place' and block:
        return f"{player} placed {block}"
    elif action == 'break' and block:
        return f"{player} broke {block}"
    elif action == 'use' and item:
        return f"{player} used {item}"
    elif action == 'craft' and item:
        return f"{player} crafted {item}"
    else:
        return f"{player} performed {action}"

def export_agent_dataset():
    """Export agent data to training dataset."""
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  Exporting Minecraft Agent Gameplay Data")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
    
    # Load agent memory
    agent_memory = load_agent_memory()
    sessions = agent_memory.get('sessions', {})
    
    # Load observations
    observations = load_observations()
    
    output_lines = []
    
    # Export sessions with objectives and actions
    print(f"Processing {len(sessions)} sessions...")
    for session_id, session_data in sessions.items():
        objective = session_data.get('objective', '')
        mode = session_data.get('mode', 'general')
        last_action = session_data.get('lastAction', {})
        
        if objective:
            # Create training examples for objectives
            output_lines.append(f"[MINECRAFT_AGENT] User: {objective}")
            
            # Describe the action taken
            action_desc = format_action(last_action)
            output_lines.append(f"Agent: {action_desc}")
            output_lines.append("")
    
    # Export observations from imitation learning
    print(f"Processing {len(observations)} observations...")
    for obs in observations[:500]:  # Limit to most recent 500
        obs_text = format_observation(obs)
        output_lines.append(f"[MINECRAFT_AGENT] Observation: {obs_text}")
        output_lines.append("")
    
    # Add common gameplay patterns
    print("Adding common gameplay patterns...")
    patterns = [
        ("[MINECRAFT_AGENT] User: collect wood", "Agent: move forward, look for trees, break log blocks"),
        ("[MINECRAFT_AGENT] User: mine stone", "Agent: equip pickaxe, move forward, break stone blocks"),
        ("[MINECRAFT_AGENT] User: attack enemy", "Agent: equip sword, move forward, attack, strafe left"),
        ("[MINECRAFT_AGENT] User: build shelter", "Agent: equip blocks, place block, move back, place block"),
        ("[MINECRAFT_AGENT] User: find diamonds", "Agent: mine down to Y=11, branch mine, break diamond ore"),
        ("[MINECRAFT_AGENT] User: farm crops", "Agent: use hoe on dirt, plant seeds, wait, harvest crops"),
        ("[MINECRAFT_AGENT] User: explore cave", "Agent: place torches, move forward carefully, mine ores"),
        ("[MINECRAFT_AGENT] User: fight zombie", "Agent: equip sword, sprint forward, attack, back away, attack"),
        ("[MINECRAFT_AGENT] User: cook food", "Agent: place furnace, use furnace, add fuel, wait for cooking"),
        ("[MINECRAFT_AGENT] User: craft tools", "Agent: open crafting table, select recipe, craft item"),
    ]
    
    for prompt, response in patterns:
        output_lines.append(prompt)
        output_lines.append(response)
        output_lines.append("")
    
    # Write to file
    output_path = Path(__file__).parent / 'minecraft' / 'agent_gameplay.txt'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(output_lines))
    
    print(f"\n✓ Exported {len(output_lines)} lines to {output_path}")
    print(f"  Sessions: {len(sessions)}")
    print(f"  Observations: {min(len(observations), 500)}")
    print(f"  Patterns: {len(patterns)}")
    
    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  Done! Add to training with --dataset all")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

if __name__ == '__main__':
    export_agent_dataset()
