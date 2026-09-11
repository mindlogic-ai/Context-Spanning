"""The live loop around the engine.

  frame_stream.py       frame-clock loop for a wav file: utterance ASR, `<ret>` handling, span injection
  websocket_server.py   the same loop over a WebSocket, for the browser page in `web/`
  user_leveller.py      user-channel denoise + loudness levelling to the training level
  default_persona.py    the persona the demo starts from
"""
