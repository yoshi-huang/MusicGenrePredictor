import os
import sys

# Must be set before importing torch or numba to avoid LLVM SVML symbol conflict
os.environ.setdefault("NUMBA_DISABLE_INTEL_SVML", "1")

# Ensure backend/ is on path so core.* modules are importable
_BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

import asyncio

import discord
from discord import ButtonStyle
from discord.ext import commands
from discord.ui import Button, View

import processFunc as F
from core.audio import GENRES
from core.config import paths, dcbot_cfg

# ---------------------------------------------------------------------------
# Bot setup
# ---------------------------------------------------------------------------

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix=dcbot_cfg["command_prefix"], intents=intents)
bot.help_command = None  # disable default help command


# ---------------------------------------------------------------------------
# TimeSlider view — paginated broken-barh windows
# ---------------------------------------------------------------------------

class TimeSlider(View):
    def __init__(self, image_paths: list[str], titles: list[str]):
        super().__init__(timeout=None)
        self.images = image_paths
        self.titles = titles
        self.index = 0

        self.prev_btn = Button(label="◀️ Prev", style=ButtonStyle.primary)
        self.next_btn = Button(label="Next ▶️", style=ButtonStyle.primary)
        self.prev_btn.callback = self._prev
        self.next_btn.callback = self._next
        self.add_item(self.prev_btn)
        self.add_item(self.next_btn)

    async def _prev(self, interaction: discord.Interaction):
        if self.index > 0:
            self.index -= 1
            await self._update(interaction)
        else:
            await interaction.response.defer()

    async def _next(self, interaction: discord.Interaction):
        if self.index < len(self.images) - 1:
            self.index += 1
            await self._update(interaction)
        else:
            await interaction.response.defer()

    async def _update(self, interaction: discord.Interaction):
        path = self.images[self.index]
        filename = os.path.basename(path)
        embed = discord.Embed(title=self.titles[self.index], color=discord.Color.red())
        embed.set_image(url=f"attachment://{filename}")
        await interaction.response.edit_message(
            embed=embed,
            attachments=[discord.File(path, filename=filename)],
            view=self,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_valid_youtube_url(url: str) -> bool:
    return "https://www.youtube.com/" in url or "https://youtu.be/" in url


def _get_broken_barh_paths() -> list[str]:
    folder = paths["dcbot_windows_dir"]
    return sorted(
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.startswith("BrokenBarh_") and f.endswith(".png")
    )


def _titles_from_paths(file_paths: list[str]) -> list[str]:
    titles = []
    for p in file_paths:
        stem = os.path.basename(p).replace("BrokenBarh_", "").replace(".png", "")
        start, end = stem.split("-")
        titles.append(f"{int(start)} – {int(end)} s")
    return titles


async def _run_analysis(url: str) -> tuple:
    """Run the blocking analysis in a thread pool and return (winner, lines, title)."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, F.run, url)


def _build_result_embeds(winner: int, lines: str, title: str) -> tuple[discord.Embed, discord.Embed]:
    spectrogram_path = paths["dcbot_spectrogram"] + ".png"
    bar_path = paths["dcbot_bar_chart"] + ".png"

    embed1 = discord.Embed(title="Music Mel Spectrogram", color=discord.Color.red())
    embed1.set_image(url="attachment://SpectrogramCanvas.png")

    embed2 = discord.Embed(
        title=title or "Music Genre Analysis Result",
        description=f"{lines}\nThe most likely genre is: **{GENRES[winner]}**",
        color=discord.Color.red(),
    )
    embed2.set_image(url="attachment://BarChartCanvas.png")
    embed2.set_footer(text="Powered by MusicGenrePredictor | 2025")

    return embed1, embed2, spectrogram_path, bar_path


def _audio_file_if_valid() -> list[discord.File]:
    audio_path = paths["audio_temp_file"]
    if os.path.exists(audio_path) and os.path.getsize(audio_path) < 25 * 1024 * 1024:
        return [discord.File(audio_path, filename="audio.mp3")]
    return []


# ---------------------------------------------------------------------------
# Slash command  /genre
# ---------------------------------------------------------------------------

@bot.tree.command(name="genre", description="Predict the music genre of a YouTube video 🎵")
async def genre_slash(interaction: discord.Interaction, url: str):
    if not _is_valid_youtube_url(url):
        await interaction.response.send_message(
            "Please provide a valid YouTube URL, e.g. `/genre https://www.youtube.com/...`",
            ephemeral=True,
        )
        return

    # 初始回應為私訊（ephemeral），僅作為「分析中」的進度提示
    await interaction.response.send_message("🔍 Analysing music genre…", ephemeral=True)

    try:
        winner, lines, title = await _run_analysis(url)
    except Exception as exc:
        await interaction.edit_original_response(content=f"❌ Analysis failed: {exc}")
        return

    embed1, embed2, spectrogram_path, bar_path = _build_result_embeds(winner, lines, title)

    if not all(os.path.exists(p) for p in [spectrogram_path, bar_path]):
        await interaction.edit_original_response(content="❌ Analysis images are missing.")
        return

    # 結果全部透過 followup 公開發送
    await interaction.followup.send(
        "🎶 Genre prediction result:",
        embeds=[embed1, embed2],
        files=[
            discord.File(spectrogram_path, filename="SpectrogramCanvas.png"),
            discord.File(bar_path, filename="BarChartCanvas.png"),
        ],
    )

    audio_files = _audio_file_if_valid()
    if audio_files:
        await interaction.followup.send(
            f"🎵 {title}" if title else "🎵 Audio:",
            files=audio_files,
        )

    window_paths = _get_broken_barh_paths()
    if window_paths:
        titles = _titles_from_paths(window_paths)
        view = TimeSlider(window_paths, titles)
        first = window_paths[0]
        embed3 = discord.Embed(title=titles[0], color=discord.Color.red())
        embed3.set_image(url=f"attachment://{os.path.basename(first)}")
        await interaction.followup.send(
            embed=embed3,
            view=view,
            file=discord.File(first, filename=os.path.basename(first)),
        )


# ---------------------------------------------------------------------------
# Prefix command  /genre  (legacy)
# ---------------------------------------------------------------------------

@bot.command(name="genre")
async def genre_prefix(ctx: commands.Context, url: str = None):
    if not url or not _is_valid_youtube_url(url):
        await ctx.send("Please provide a valid YouTube URL, e.g. `!genre https://www.youtube.com/...`")
        return

    status_msg = await ctx.send("URL received, analysing music genre…")

    try:
        winner, lines, title = await _run_analysis(url)
    except Exception as exc:
        await status_msg.edit(content=f"❌ Analysis failed: {exc}")
        return

    embed1, embed2, spectrogram_path, bar_path = _build_result_embeds(winner, lines, title)

    if not all(os.path.exists(p) for p in [spectrogram_path, bar_path]):
        await ctx.send("❌ Analysis images are missing.")
        return

    await ctx.send(
        "Time chart and genre prediction:",
        embeds=[embed1, embed2],
        files=[
            discord.File(spectrogram_path, filename="SpectrogramCanvas.png"),
            discord.File(bar_path, filename="BarChartCanvas.png"),
        ],
    )

    audio_files = _audio_file_if_valid()
    if audio_files:
        await ctx.send(f"🎵 {title}" if title else "🎵 Audio:", files=audio_files)

    window_paths = _get_broken_barh_paths()
    if window_paths:
        titles = _titles_from_paths(window_paths)
        view = TimeSlider(window_paths, titles)
        first = window_paths[0]
        embed3 = discord.Embed(title=titles[0], color=discord.Color.red())
        embed3.set_image(url=f"attachment://{os.path.basename(first)}")
        await ctx.send(
            embed=embed3,
            view=view,
            file=discord.File(first, filename=os.path.basename(first)),
        )


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

@bot.event
async def on_ready():
    slash = await bot.tree.sync()
    print(f"Synced {len(slash)} slash command(s)")
    print(f"Logged in as {bot.user}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def start(token: str):
    await bot.start(token)


if __name__ == "__main__":
    token = dcbot_cfg.get("token") or os.environ.get("DISCORD_BOT_TOKEN", "")
    if not token:
        raise RuntimeError("No Discord bot token found. Set dcbot.token in config.yaml or DISCORD_BOT_TOKEN env var.")
    asyncio.run(start(token))
