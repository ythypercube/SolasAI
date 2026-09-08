#!/usr/bin/env python3
"""
Fix context markers in dataset files based on content analysis.
Re-categorizes Q&A pairs using keyword detection.
"""

import os
import re
from pathlib import Path

# Keyword sets for context detection
MINECRAFT_KEYWORDS = {
    'minecraft', 'diamond', 'craft', 'mine', 'block', 'mob', 'creeper',
    'ender', 'dragon', 'farm', 'redstone', 'pickaxe', 'ore', 'spawn',
    'biome', 'nether', 'village', 'enchant', 'potion', 'survival',
    'sword', 'armor', 'food', 'hunger', 'elytra', 'portal', 'obsidian',
    'furnace', 'smelt', 'bucket', 'lava', 'water', 'torch', 'bed', 'chest',
    'hostile', 'zombie', 'skeleton', 'spider', 'enderman', 'stronghold'
}

# More restrictive - only mark as CODE if asking about writing/debugging code
CODE_PATTERNS = {
    'write a', 'write code', 'write function', 'create a function',
    'how to code', 'how do i write', 'syntax for', 'example code',
    'python code', 'javascript code', 'code to', 'function to',
    'write script', 'debug', 'fix error', 'syntax error',
    'import statement', 'for loop', 'while loop', 'if statement',
    'how to implement', 'algorithm for', 'code example'
}

def detect_context_from_content(text: str) -> str:
    """Detect context marker based on text content."""
    text_lower = text.lower()
    
    # Count keyword matches
    minecraft_score = sum(1 for kw in MINECRAFT_KEYWORDS if kw in text_lower)
    code_match = any(pattern in text_lower for pattern in CODE_PATTERNS)
    
    # Determine context
    if minecraft_score > 0:
        return '[MINECRAFT]'
    elif code_match:
        return '[CODE]'
    else:
        return '[CHAT]'

def fix_file_markers(filepath: Path) -> tuple[int, int]:
    """Fix context markers in a single file. Returns (total, changed) counts."""
    lines = []
    total_pairs = 0
    changed_pairs = 0
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split into lines
    raw_lines = content.split('\n')
    
    i = 0
    while i < len(raw_lines):
        line = raw_lines[i].strip()
        
        # Check if this is a User line
        if ' User: ' in line or line.startswith('User: '):
            # Extract current marker and user message
            user_idx = line.find('User: ')
            if user_idx >= 0:
                current_marker = '[CHAT]'  # default
                if line.startswith('['):
                    end_bracket = line.find(']')
                    if end_bracket > 0:
                        current_marker = line[:end_bracket + 1]
                
                user_message = line[user_idx + 6:].strip()
                
                # Look ahead for Assistant response
                assistant_line = ''
                if i + 1 < len(raw_lines):
                    next_line = raw_lines[i + 1].strip()
                    if next_line.startswith('Assistant: '):
                        assistant_line = next_line[11:].strip()
                
                # Detect correct context from content
                combined_text = user_message + ' ' + assistant_line
                correct_marker = detect_context_from_content(combined_text)
                
                # Update marker if changed
                if correct_marker != current_marker:
                    changed_pairs += 1
                
                total_pairs += 1
                
                # Reconstruct line with correct marker
                new_line = f"{correct_marker} User: {user_message}"
                lines.append(new_line)
                i += 1
                continue
        
        # Keep other lines as-is
        lines.append(raw_lines[i])
        i += 1
    
    # Write back to file
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    return total_pairs, changed_pairs

def main():
    datasets_dir = Path(__file__).parent
    
    print("Fixing context markers in dataset files...")
    print("=" * 60)
    
    total_files = 0
    total_pairs = 0
    total_changed = 0
    
    # Process all subdirectories
    for subdir_name in ['conversation', 'minecraft', 'coding']:
        subdir = datasets_dir / subdir_name
        if not subdir.exists():
            continue
        
        print(f"\n{subdir_name}/:")
        
        for filepath in sorted(subdir.glob('*.txt')):
            if filepath.name == 'feedback_log.jsonl':
                continue
            
            pairs, changed = fix_file_markers(filepath)
            total_files += 1
            total_pairs += pairs
            total_changed += changed
            
            status = f"  {filepath.name}: {pairs} pairs, {changed} changed"
            if changed > 0:
                status += " ✓"
            print(status)
    
    print(f"\n{'=' * 60}")
    print(f"Summary:")
    print(f"  Files processed: {total_files}")
    print(f"  Total Q&A pairs: {total_pairs}")
    print(f"  Markers changed: {total_changed}")
    print(f"  Accuracy: {(total_pairs - total_changed) / max(1, total_pairs) * 100:.1f}%")

if __name__ == '__main__':
    main()
