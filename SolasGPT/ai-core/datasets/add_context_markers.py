#!/usr/bin/env python3
"""Add context markers to existing dataset files."""

import os

def add_markers_to_file(filepath, context_marker):
    """Add context marker before User: lines in conversation files."""
    
    if not os.path.exists(filepath):
        print(f"  ⚠️ Skipping {filepath} (not found)")
        return
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if not content.strip():
        print(f"  ⚠️ Skipping {filepath} (empty)")
        return
    
    # Check if markers already exist
    if '[CHAT]' in content or '[CODE]' in content or '[MINECRAFT]' in content:
        print(f"  ✓ Skipping {filepath} (markers already present)")
        return
    
    # Simple approach: replace "User:" with "[MARKER] User:"
    updated_content = content.replace('\nUser:', f'\n{context_marker} User:')
    
    # Handle first line if it starts with "User:"
    if updated_content.startswith('User:'):
        updated_content = f'{context_marker} {updated_content}'
    
    # Write back
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(updated_content)
    
    print(f"  ✓ Updated {filepath}")

def main():
    base_dir = os.path.dirname(__file__)
    
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  Adding Context Markers to Datasets")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
    
    # Conversation files
    print("Conversation datasets:")
    add_markers_to_file(os.path.join(base_dir, 'conversation/general_chat.txt'), '[CHAT]')
    add_markers_to_file(os.path.join(base_dir, 'conversation/expanded_dialogue.txt'), '[CHAT]')
    
    # Coding files (simple format)
    print("\nCoding datasets:")
    code_file = os.path.join(base_dir, 'coding/code_examples.txt')
    if os.path.exists(code_file):
        with open(code_file, 'r') as f:
            code_content = f.read()
        if code_content.strip() and '[CODE]' not in code_content:
            # Add [CODE] at the beginning
            with open(code_file, 'w') as f:
                f.write(f'[CODE] {code_content}')
            print(f"  ✓ Updated {code_file}")
        else:
            print(f"  ✓ Skipping {code_file}")
    
    # Minecraft files - these are typically structured as facts/tips per line or paragraph
    print("\nMinecraft datasets:")
    mc_file = os.path.join(base_dir, 'minecraft/gameplay_knowledge.txt')
    if os.path.exists(mc_file):
        with open(mc_file, 'r') as f:
            lines = f.readlines()
        
        if lines and '[MINECRAFT]' not in ''.join(lines):
            # Add [MINECRAFT] before lines that start new content (non-empty, not already marked)
            new_lines = []
            for i, line in enumerate(lines):
                if line.strip() and (i == 0 or lines[i-1].strip() == ''):
                    # Start of new section
                    new_lines.append(f'[MINECRAFT] {line}')
                else:
                    new_lines.append(line)
            
            with open(mc_file, 'w') as f:
                f.writelines(new_lines)
            print(f"  ✓ Updated {mc_file}")
        else:
            print(f"  ✓ Skipping {mc_file}")
    
    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  Done! Ready to retrain.")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

if __name__ == '__main__':
    main()
