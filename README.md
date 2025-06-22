# Obscurify
### Keeping your secrets ... secret
Obscurify aims to garuntee your security by automatically detecting and blurring sensitive information on your screen when sharing in a virtual meeting.

## What it does
Obscurify captures your desktop stream, finds anything sensitive, and live-blurs it before the pixels ever leave your machine.

Works on both text and images (faces, passwords, names, addresses, etc.).

Uses an agent (with Letta) to retrieve external personal information to hide

Lets users customize additional items they’d like to hide with a simple checkbox panel.

Uses multiple paths like regex, gemini, and opencv to identify sensitive information

Runs fast enough for real-time conferencing.* 

## To try yourself:
run the python file obscurify_fast.py with your own GEMINI_API_KEY and LETTA_TOKEN
