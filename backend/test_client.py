import asyncio
import websockets
import json

async def connect_to_server():
    async with websockets.connect('ws://localhost:8765') as websocket:
        # Track status for each source
        source_status = {}
        
        while True:
            try:
                message = await websocket.recv()
                data = json.loads(message)
                
                if "status" in data:
                    print(f"Connection status: {data['status']}")
                else:
                    source = data.get('source')
                    status = data.get('table_status')
                    timestamp = data.get('timestamp')
                    
                    source_status[source] = status
                    print(f"Source {source} at {timestamp:.2f}:")
                    print(f"Table status: {status}")
                    print("-" * 40)
                    
            except websockets.exceptions.ConnectionClosed:
                print("Connection closed")
                break
            except Exception as e:
                print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(connect_to_server()) 