import asyncio
from meshcore import MeshCore, EventType
import ollama
import time

MODEL = 'tinyllama'

def chat_with_ollama(prompt):
    """Sends a message to local Ollama and returns the response."""
    try:
        response = ollama.chat(model=MODEL, messages=[
            {
                'role': 'user',
                'content': prompt,
            },
        ])
        return response['message']['content']
    except Exception as e:
        return f"Error: {e}"

async def main():
    print("Connecting...")
    meshcore = await MeshCore.create_serial("/dev/ttyACM0")
    print("Connected!")
    
    result = await meshcore.commands.get_contacts()

    if result.type == EventType.ERROR:
        print(f"Error getting contacts: {result.payload}")
        return

    contacts = result.payload

    print(f"Found {len(contacts)} contacts")

    if not contacts:
        print("No contacts found")
        return

    contact = next(iter(contacts.items()))[1]

    print(f"Using contact: {contact}")
    
    async def on_message(event):
        try:
            if "text" in event.payload:
                print("\n=== NEW MESSAGE ===")
                recieved = event.payload['text']
                print(f"Message: {recieved}")
               
                message  = chat_with_ollama(recieved)
                print("ollama response:",message)
                result = await meshcore.commands.send_msg(
                    contact,
                    message
                )
                
                

        except Exception as e:
            print(f"Listener error: {e}")

    meshcore.subscribe(
        EventType.CONTACT_MSG_RECV,
        on_message
    )

    await meshcore.start_auto_message_fetching()

    print("\nListener started!")
    
    try:
        # Keep program running forever
        while True:
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        print("\nnooooooooooooooooooooooo")

    finally:
        await meshcore.stop_auto_message_fetching()
        await meshcore.disconnect()
        print("Disconnected.")
    
asyncio.run(main())
