
import requests
from io import BytesIO
from PIL import Image, ImageOps

# Example mapping (replace with actual logo URLs)
# You can use official Wikipedia logo URLs or any CDN
nfl_logos = {
    "ARI": "https://a.espncdn.com/i/teamlogos/nfl/500/ari.png",
    "ATL": "https://a.espncdn.com/i/teamlogos/nfl/500/atl.png",
    "BAL": "https://a.espncdn.com/i/teamlogos/nfl/500/bal.png",
    "BUF": "https://a.espncdn.com/i/teamlogos/nfl/500/buf.png",
    "CAR": "https://a.espncdn.com/i/teamlogos/nfl/500/car.png",
    "CHI": "https://a.espncdn.com/i/teamlogos/nfl/500/chi.png",
    "CIN": "https://a.espncdn.com/i/teamlogos/nfl/500/cin.png",
    "CLE": "https://a.espncdn.com/i/teamlogos/nfl/500/cle.png",
    "DAL": "https://a.espncdn.com/i/teamlogos/nfl/500/dal.png",
    "DEN": "https://a.espncdn.com/i/teamlogos/nfl/500/den.png",
    "DET": "https://a.espncdn.com/i/teamlogos/nfl/500/det.png",
    "GB":  "https://a.espncdn.com/i/teamlogos/nfl/500/gb.png",
    "HOU": "https://a.espncdn.com/i/teamlogos/nfl/500/hou.png",
    "IND": "https://a.espncdn.com/i/teamlogos/nfl/500/ind.png",
    "JAX": "https://a.espncdn.com/i/teamlogos/nfl/500/jax.png",
    "KC":  "https://a.espncdn.com/i/teamlogos/nfl/500/kc.png",
    "LV":  "https://a.espncdn.com/i/teamlogos/nfl/500/lv.png",
    "LAC": "https://a.espncdn.com/i/teamlogos/nfl/500/lac.png",
    "LAR": "https://a.espncdn.com/i/teamlogos/nfl/500/lar.png",
    "MIA": "https://a.espncdn.com/i/teamlogos/nfl/500/mia.png",
    "MIN": "https://a.espncdn.com/i/teamlogos/nfl/500/min.png",
    "NE":  "https://a.espncdn.com/i/teamlogos/nfl/500/ne.png",
    "NO":  "https://a.espncdn.com/i/teamlogos/nfl/500/no.png",
    "NYG": "https://a.espncdn.com/i/teamlogos/nfl/500/nyg.png",
    "NYJ": "https://a.espncdn.com/i/teamlogos/nfl/500/nyj.png",
    "PHI": "https://a.espncdn.com/i/teamlogos/nfl/500/phi.png",
    "PIT": "https://a.espncdn.com/i/teamlogos/nfl/500/pit.png",
    "SF":  "https://a.espncdn.com/i/teamlogos/nfl/500/sf.png",
    "SEA": "https://a.espncdn.com/i/teamlogos/nfl/500/sea.png",
    "TB":  "https://a.espncdn.com/i/teamlogos/nfl/500/tb.png",
    "TEN": "https://a.espncdn.com/i/teamlogos/nfl/500/ten.png",
    "WAS": "https://a.espncdn.com/i/teamlogos/nfl/500/wsh.png"
}

# Grid setup
rows, cols = 5, 7
tile_size = 200  # each square for a logo
canvas_width = cols * tile_size
canvas_height = rows * tile_size
canvas = Image.new("RGB", (canvas_width, canvas_height), "black")

logos = []

# Download and resize each logo, add black background
for team, url in nfl_logos.items():
    try:
        response = requests.get(url, timeout=10)
        logo = Image.open(BytesIO(response.content)).convert("RGBA")

        # Scale logo to fit nicely inside a square
        logo.thumbnail((tile_size - 20, tile_size - 20), Image.LANCZOS)

        # Create black square
        square = Image.new("RGB", (tile_size, tile_size), (13, 17, 22))

        # Center logo
        lx = (tile_size - logo.width) // 2
        ly = (tile_size - logo.height) // 2

        # Paste logo with transparency
        square.paste(logo, (lx, ly), logo)
        logos.append(square)
    except Exception as e:
        print(f"Failed to load {team}: {e}")

# Place logos into grid
x, y = 0, 0
for i, logo in enumerate(logos):
    # handle centered last row
    row = i // cols
    col = i % cols
    if row == rows - 1:  # last row
        num_last = len(logos) - (rows - 1) * cols
        offset_x = (canvas_width - num_last * tile_size) // 2
        x = offset_x + (col * tile_size)
    else:
        x = col * tile_size
    y = row * tile_size
    canvas.paste(logo, (x, y))

canvas.save("nfl_logos_grid_black.png")
print("Saved nfl_logos_grid_black.png")

