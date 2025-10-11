import os
import json
import time
import numpy as np
import pygame
from Config.log_dir import FRAME_PATH, FRAME_META_PATH

DEFAULT_SIZE = (320, 180)  # placeholder before first frame
FRAME_CHECK_INTERVAL = 0.2  # seconds between file polls (~5Hz)

def load_meta():
    if not os.path.exists(FRAME_META_PATH):
        return None
    try:
        with open(FRAME_META_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return None

def load_frame():
    if not os.path.exists(FRAME_PATH):
        return None
    try:
        return np.load(FRAME_PATH)
    except Exception:
        return None

def main():
    os.environ["SDL_VIDEO_WINDOW_POS"] = "0,30"
    pygame.init()
    # No RESIZABLE flag — user cannot resize manually
    screen = pygame.display.set_mode(DEFAULT_SIZE, flags=0)
    pygame.display.set_caption("Live NN Output")
    clock = pygame.time.Clock()

    last_shape = DEFAULT_SIZE
    last_check = 0

    running = True
    while running:
        # Always process events so window stays responsive
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        now = time.time()
        if now - last_check >= FRAME_CHECK_INTERVAL:
            last_check = now
            meta = load_meta()
            if meta and meta.get("new_frame"):
                arr = load_frame()
                if arr is not None:
                    h, w = arr.shape[:2]
                    if (w, h) != last_shape:
                        screen = pygame.display.set_mode((w, h), flags=0)
                        last_shape = (w, h)

                    surface = pygame.surfarray.make_surface(np.transpose(arr, (1, 0, 2)))
                    screen.blit(surface, (0, 0))
                    pygame.display.flip()

                    # Mark as consumed
                    try:
                        with open(FRAME_META_PATH, "w") as f:
                            json.dump({"new_frame": False}, f)
                    except Exception:
                        pass

        clock.tick(60)  # keep event loop responsive at up to 60 FPS

    pygame.quit()

if __name__ == "__main__":
    main()
