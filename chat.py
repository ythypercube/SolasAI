#!/usr/bin/env python3
"""
Simple interactive chat interface for SolasGPT
"""
import requests
import sys

SERVER_URL = "http://localhost:8788"

def chat():
    print("╔═══════════════════════════════════════════════════════════╗")
    print("║              Welcome to SolasGPT Chat! 🤖                ║")
    print("╚═══════════════════════════════════════════════════════════╝")
    print()
    print("Commands:")
    print("  • Type your message and press Enter")
    print("  • Type 'quit' or 'exit' to exit")
    print("  • Type 'reset' to start a new conversation")
    print()
    
    session_id = "user-session"
    
    while True:
        try:
            # Get user input
            user_input = input("\n\033[1;36mYou:\033[0m ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("\n👋 Goodbye!")
                break
                
            if user_input.lower() == 'reset':
                response = requests.post(f"{SERVER_URL}/reset", json={"session_id": session_id})
                if response.status_code == 200:
                    print("\n✅ Conversation reset!")
                else:
                    print("\n❌ Reset failed")
                continue
            
            # Send to server
            response = requests.post(
                f"{SERVER_URL}/chat",
                json={"message": user_input, "session_id": session_id},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                reply = data.get('reply', data.get('response', 'No response'))
                print(f"\n\033[1;32mSolasGPT:\033[0m {reply}")
            else:
                print(f"\n❌ Error: {response.status_code}")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except requests.exceptions.ConnectionError:
            print("\n❌ Cannot connect to server. Make sure chat_server.py is running!")
            print("   Run: ./start-chat-server.sh")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    chat()
