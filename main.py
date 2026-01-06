from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import base64
import wave
import asyncio
import os
from datetime import datetime
import json
import uuid

app = FastAPI()

# Enable CORS for browser access from anywhere
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2

# Store multiple recordings
recordings = {}  # {recording_id: {audio_data, transcripts, start_time, end_time}}
current_recording_id = None
active_websockets = set()

TRANSCRIPTION_CHUNK_SIZE = SAMPLE_RATE * 2 * 3  # 3 seconds

@app.post("/recording/start")
async def start_recording():
    """Start a new recording session"""
    global current_recording_id
    
    recording_id = str(uuid.uuid4())
    current_recording_id = recording_id
    
    recordings[recording_id] = {
        "audio_data": bytearray(),
        "transcription_buffer": bytearray(),
        "transcripts": [],
        "start_time": datetime.now().isoformat(),
        "end_time": None,
        "status": "recording"
    }
    
    await broadcast_status({
        "type": "recording_started",
        "recording_id": recording_id,
        "timestamp": recordings[recording_id]["start_time"]
    })
    
    return {"ok": True, "recording_id": recording_id}

@app.post("/recording/stop")
async def stop_recording():
    """Stop the current recording and finalize transcription"""
    global current_recording_id
    
    if not current_recording_id:
        return {"ok": False, "error": "No active recording"}
    
    recording = recordings[current_recording_id]
    recording["end_time"] = datetime.now().isoformat()
    recording["status"] = "completed"
    
    # Transcribe any remaining audio in buffer
    if len(recording["transcription_buffer"]) > 0:
        await transcribe_chunk(current_recording_id, force=True)
    
    # Save final WAV file
    filename = f"recording_{current_recording_id[:8]}.wav"
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(recording["audio_data"])
    
    recording["filename"] = filename
    
    await broadcast_status({
        "type": "recording_stopped",
        "recording_id": current_recording_id,
        "timestamp": recording["end_time"],
        "duration": len(recording["audio_data"]) // (SAMPLE_WIDTH * SAMPLE_RATE)
    })
    
    stopped_id = current_recording_id
    current_recording_id = None
    
    return {"ok": True, "recording_id": stopped_id}

@app.post("/audio")
async def receive_audio(data: dict):
    """Receive audio chunks from ESP32"""
    global current_recording_id
    
    # Auto-start recording if not active
    if not current_recording_id:
        await start_recording()
    
    recording = recordings[current_recording_id]
    
    # Decode and store audio
    audio_chunk = base64.b64decode(data["audio"])
    recording["audio_data"].extend(audio_chunk)
    recording["transcription_buffer"].extend(audio_chunk)
    
    # Transcribe when buffer is full
    if len(recording["transcription_buffer"]) >= TRANSCRIPTION_CHUNK_SIZE:
        await transcribe_chunk(current_recording_id)
    
    # Broadcast update
    await broadcast_status({
        "type": "audio_update",
        "recording_id": current_recording_id,
        "samples": len(recording["audio_data"]) // SAMPLE_WIDTH,
        "duration": len(recording["audio_data"]) // (SAMPLE_WIDTH * SAMPLE_RATE)
    })
    
    return {"ok": True}

async def transcribe_chunk(recording_id: str, force: bool = False):
    """Transcribe accumulated audio buffer for a recording"""
    recording = recordings.get(recording_id)
    if not recording:
        return
    
    buffer = recording["transcription_buffer"]
    if len(buffer) == 0:
        return
    
    # Save chunk to temp file
    temp_file = f"temp_{recording_id}_{datetime.now().timestamp()}.wav"
    with wave.open(temp_file, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(buffer)
    
    try:
        import openai
        
        with open(temp_file, "rb") as audio_file:
            transcript = openai.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language="en"
            )
        
        # Store transcript
        transcript_entry = {
            "text": transcript.text,
            "timestamp": datetime.now().isoformat()
        }
        recording["transcripts"].append(transcript_entry)
        
        # Broadcast to connected clients
        await broadcast_status({
            "type": "transcription",
            "recording_id": recording_id,
            "text": transcript.text,
            "timestamp": transcript_entry["timestamp"]
        })
        
        # Clear buffer after successful transcription
        recording["transcription_buffer"] = bytearray()
        
    except Exception as e:
        print(f"Transcription error: {e}")
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)

@app.get("/recordings")
async def list_recordings():
    """Get all recordings with their transcripts"""
    result = []
    for rec_id, rec_data in recordings.items():
        result.append({
            "id": rec_id,
            "start_time": rec_data["start_time"],
            "end_time": rec_data["end_time"],
            "status": rec_data["status"],
            "duration": len(rec_data["audio_data"]) // (SAMPLE_WIDTH * SAMPLE_RATE),
            "transcripts": rec_data["transcripts"],
            "filename": rec_data.get("filename")
        })
    
    # Sort by start time, newest first
    result.sort(key=lambda x: x["start_time"], reverse=True)
    return result

@app.get("/recording/{recording_id}")
async def get_recording(recording_id: str):
    """Get a specific recording"""
    if recording_id not in recordings:
        return {"error": "Recording not found"}
    
    rec_data = recordings[recording_id]
    return {
        "id": recording_id,
        "start_time": rec_data["start_time"],
        "end_time": rec_data["end_time"],
        "status": rec_data["status"],
        "duration": len(rec_data["audio_data"]) // (SAMPLE_WIDTH * SAMPLE_RATE),
        "transcripts": rec_data["transcripts"],
        "filename": rec_data.get("filename")
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for live updates"""
    await websocket.accept()
    active_websockets.add(websocket)
    
    try:
        # Send current state
        await websocket.send_json({
            "type": "connected",
            "current_recording": current_recording_id,
            "total_recordings": len(recordings)
        })
        
        while True:
            await websocket.receive_text()
    except:
        pass
    finally:
        active_websockets.remove(websocket)

async def broadcast_status(message: dict):
    """Broadcast to all connected WebSocket clients"""
    disconnected = set()
    for ws in active_websockets:
        try:
            await ws.send_json(message)
        except:
            disconnected.add(ws)
    
    for ws in disconnected:
        active_websockets.discard(ws)

@app.get("/")
async def get_client():
    """Web interface for viewing recordings"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Voice Notes</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }
            .container {
                max-width: 900px;
                margin: 0 auto;
            }
            .header {
                background: white;
                padding: 30px;
                border-radius: 16px;
                margin-bottom: 20px;
                box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            }
            h1 {
                color: #333;
                font-size: 2em;
                margin-bottom: 10px;
            }
            .status {
                display: flex;
                align-items: center;
                gap: 10px;
                color: #666;
                font-size: 1.1em;
            }
            .recording-dot {
                width: 16px;
                height: 16px;
                background: #ff4444;
                border-radius: 50%;
                animation: pulse 1.5s infinite;
                display: none;
            }
            .recording-dot.active { display: block; }
            @keyframes pulse {
                0%, 100% { opacity: 1; transform: scale(1); }
                50% { opacity: 0.5; transform: scale(1.2); }
            }
            .recordings {
                display: flex;
                flex-direction: column;
                gap: 16px;
            }
            .recording-card {
                background: white;
                border-radius: 16px;
                padding: 24px;
                box-shadow: 0 8px 32px rgba(0,0,0,0.1);
                transition: transform 0.2s;
            }
            .recording-card:hover {
                transform: translateY(-2px);
                box-shadow: 0 12px 40px rgba(0,0,0,0.15);
            }
            .recording-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 16px;
                padding-bottom: 16px;
                border-bottom: 2px solid #f0f0f0;
            }
            .recording-title {
                font-size: 1.2em;
                font-weight: 600;
                color: #333;
            }
            .recording-meta {
                display: flex;
                gap: 16px;
                color: #666;
                font-size: 0.9em;
            }
            .badge {
                display: inline-block;
                padding: 4px 12px;
                border-radius: 12px;
                font-size: 0.85em;
                font-weight: 600;
            }
            .badge.recording {
                background: #ff4444;
                color: white;
            }
            .badge.completed {
                background: #4CAF50;
                color: white;
            }
            .transcript {
                background: #f9f9f9;
                padding: 16px;
                border-radius: 8px;
                margin: 8px 0;
                border-left: 4px solid #667eea;
            }
            .transcript-time {
                color: #999;
                font-size: 0.85em;
                margin-bottom: 8px;
            }
            .transcript-text {
                color: #333;
                line-height: 1.6;
            }
            .empty-state {
                background: white;
                border-radius: 16px;
                padding: 60px 40px;
                text-align: center;
                color: #999;
            }
            .empty-state svg {
                width: 80px;
                height: 80px;
                margin-bottom: 20px;
                opacity: 0.3;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🎤 Voice Notes</h1>
                <div class="status">
                    <div class="recording-dot" id="recordingDot"></div>
                    <span id="statusText">Ready</span>
                </div>
            </div>
            
            <div class="recordings" id="recordings">
                <div class="empty-state">
                    <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
                              d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z">
                        </path>
                    </svg>
                    <h2>No recordings yet</h2>
                    <p>Press the button on your device to start recording</p>
                </div>
            </div>
        </div>
        
        <script>
            const ws = new WebSocket('ws://' + window.location.host + '/ws');
            let currentRecordingId = null;
            
            ws.onopen = () => {
                console.log('Connected');
                loadRecordings();
            };
            
            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                console.log('Message:', data);
                
                if (data.type === 'recording_started') {
                    currentRecordingId = data.recording_id;
                    document.getElementById('recordingDot').classList.add('active');
                    document.getElementById('statusText').textContent = 'Recording...';
                    loadRecordings();
                }
                
                if (data.type === 'recording_stopped') {
                    currentRecordingId = null;
                    document.getElementById('recordingDot').classList.remove('active');
                    document.getElementById('statusText').textContent = 'Ready';
                    loadRecordings();
                }
                
                if (data.type === 'audio_update') {
                    updateRecordingDuration(data.recording_id, data.duration);
                }
                
                if (data.type === 'transcription') {
                    addTranscript(data.recording_id, data.text, data.timestamp);
                }
            };
            
            async function loadRecordings() {
                const response = await fetch('/recordings');
                const recordings = await response.json();
                
                const container = document.getElementById('recordings');
                
                if (recordings.length === 0) {
                    container.innerHTML = `
                        <div class="empty-state">
                            <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
                                      d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z">
                                </path>
                            </svg>
                            <h2>No recordings yet</h2>
                            <p>Press the button on your device to start recording</p>
                        </div>
                    `;
                    return;
                }
                
                container.innerHTML = recordings.map(rec => createRecordingCard(rec)).join('');
            }
            
            function createRecordingCard(rec) {
                const startTime = new Date(rec.start_time);
                const timeStr = startTime.toLocaleString();
                const duration = formatDuration(rec.duration);
                
                const transcripts = rec.transcripts.map(t => `
                    <div class="transcript">
                        <div class="transcript-time">${new Date(t.timestamp).toLocaleTimeString()}</div>
                        <div class="transcript-text">${t.text}</div>
                    </div>
                `).join('');
                
                return `
                    <div class="recording-card" id="recording-${rec.id}">
                        <div class="recording-header">
                            <div>
                                <div class="recording-title">${timeStr}</div>
                            </div>
                            <span class="badge ${rec.status}">${rec.status}</span>
                        </div>
                        <div class="recording-meta">
                            <span>⏱️ <span id="duration-${rec.id}">${duration}</span></span>
                            <span>💬 ${rec.transcripts.length} segments</span>
                        </div>
                        <div id="transcripts-${rec.id}">
                            ${transcripts || '<p style="color: #999; padding: 20px 0;">Waiting for transcription...</p>'}
                        </div>
                    </div>
                `;
            }
            
            function updateRecordingDuration(recordingId, duration) {
                const el = document.getElementById(`duration-${recordingId}`);
                if (el) {
                    el.textContent = formatDuration(duration);
                }
            }
            
            function addTranscript(recordingId, text, timestamp) {
                const container = document.getElementById(`transcripts-${recordingId}`);
                if (!container) return;
                
                const time = new Date(timestamp).toLocaleTimeString();
                const transcript = document.createElement('div');
                transcript.className = 'transcript';
                transcript.innerHTML = `
                    <div class="transcript-time">${time}</div>
                    <div class="transcript-text">${text}</div>
                `;
                
                // Remove "waiting" message if present
                const waiting = container.querySelector('p');
                if (waiting) waiting.remove();
                
                container.appendChild(transcript);
            }
            
            function formatDuration(seconds) {
                const mins = Math.floor(seconds / 60);
                const secs = seconds % 60;
                return `${mins}:${secs.toString().padStart(2, '0')}`;
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
