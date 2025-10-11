import numpy as np
import pygame
from src.backend_cupy import to_cpu

def _prep_img(arr):
    """Ensure np.uint8 RGB for display."""
    img = to_cpu(arr)
    if img is None:
        return None
    if img.ndim == 2:
        img = np.stack([img]*3, axis=-1)
    elif img.ndim == 3 and img.shape[2] == 1:
        img = np.repeat(img, 3, axis=-1)
    if img.dtype != np.uint8:
        a = img.astype(np.float32)
        vmax = float(a.max()) if a.size else 1.0
        if vmax <= 1.0 + 1e-6:
            a *= 255.0
        img = np.clip(a, 0, 255).astype(np.uint8)
    return img

def final_viewer(image_list):
    """
    image_list: list of (label:str, array: HxWx{1,3}, uint8/float[0..1 or 0..255])
    Shows one image at a time; use LEFT/RIGHT or A/D to flip.
    """
    images = [(label, _prep_img(img)) for label, img in image_list if _prep_img(img) is not None]
    if not images:
        print("[final_viewer] No images to display")
        return

    pygame.init()
    idx = 0
    H, W = images[0][1].shape[:2]
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption(images[0][0])
    clock = pygame.time.Clock()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_RIGHT, pygame.K_d):
                    idx = (idx + 1) % len(images)
                elif event.key in (pygame.K_LEFT, pygame.K_a):
                    idx = (idx - 1) % len(images)
                elif event.key == pygame.K_ESCAPE:
                    running = False
                pygame.display.set_caption(images[idx][0])

        label, img = images[idx]
        surf = pygame.surfarray.make_surface(np.transpose(img, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()