import json
import random
from fastapi import FastAPI, Request
import logging

logging.basicConfig(filename="./outputs/game.log", level=logging.INFO)

app = FastAPI()


@app.head("/api/v1/ping")
async def handle_ping():
    return ""


@app.post("/api/v1/command")
async def handle_command(request: Request):
    state = await request.json()
    logging.info(json.dumps(state))

    direction = random.choice(["U", "D", "L", "R", "N"])
    is_place_bomb = random.choice([True, False])
    return {"direction": direction, "is_place_bomb": is_place_bomb}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5000)
