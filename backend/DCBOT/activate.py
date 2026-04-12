import os

# Must be set before importing torch or numba to avoid LLVM SVML symbol conflict
os.environ.setdefault("NUMBA_DISABLE_INTEL_SVML", "1")

import discord
from discord.ext import commands
from discord import ButtonStyle
from discord.ui import Button, View
import requests
import asyncio
import os

import DCBOT.processFunc as F
from core.audio import GENRES
from core.config import paths, dcbot_cfg

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix=dcbot_cfg["command_prefix"], intents=intents)

class TimeSlider(View):
    def __init__(self, image_paths, titles):
        super().__init__(timeout=None)
        self.images = image_paths
        self.titles = titles
        self.index = 0

        self.prev_btn = Button(label="Prev ◀️", style=ButtonStyle.primary)
        self.next_btn = Button(label="▶️ Next", style=ButtonStyle.primary)
        self.prev_btn.callback = self.prev
        self.next_btn.callback = self.next
        self.add_item(self.prev_btn)
        self.add_item(self.next_btn)

    async def prev(self, interaction):
        if self.index > 0:
            self.index -= 1
            await self.update_embed(interaction)

    async def next(self, interaction):
        if self.index < len(self.images) - 1:
            self.index += 1
            await self.update_embed(interaction)

    async def update_embed(self, interaction):
        embed = discord.Embed(title=self.titles[self.index])
        embed.set_image(url=f"attachment://{os.path.basename(self.images[self.index])}")
        await interaction.response.edit_message(
            embed=embed,
            attachments=[discord.File(self.images[self.index], filename=os.path.basename(self.images[self.index]))],
            view=self
        )

def get_broken_barh_paths():
    folder = paths["dcbot_windows_dir"]
    return [
        os.path.join(folder, f) for f in os.listdir(folder)
        if f.startswith("BrokenBarh_") and f.endswith(".png")
    ]

def generate_titles_from_filenames(file_paths):
    return [
        f"{int(os.path.basename(p).split('.')[0].replace('BrokenBarh_', '').split('-')[0])} - {int(os.path.basename(p).split('.')[0].replace('BrokenBarh_', '').split('-')[1])} s"
        for p in file_paths]

@bot.event
async def on_ready():
    print(f"Logged in as {bot.user}")

@bot.command(name='genre')
async def genre(ctx, url: str = None):
    if not ("https://www.youtube.com/" in url or "https://youtu.be/" in url):
        await ctx.send("Please provide a YouTube URL, e.g. `/genre https://www.youtube.com/...`")
        return

    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            await ctx.send("Unable to fetch the URL content.")
            return
    except Exception as e:
        await ctx.send(f"URL request error: {e}")
        return

    await ctx.send("URL received successfully, analyzing music genre...")

    def blocking_task():
        return F.run(url)

    loop = asyncio.get_event_loop()
    winner, lines = await loop.run_in_executor(None, blocking_task)

    spectrogram_path = paths["dcbot_spectrogram"] + ".png"
    bar_path         = paths["dcbot_bar_chart"]    + ".png"

    if not all(os.path.exists(p) for p in [spectrogram_path, bar_path]):
        await ctx.send("Analysis images are missing or paths are incorrect.")
        return

    embed1 = discord.Embed(title="Music Mel Spectrogram :", color=discord.Color.red())
    embed1.set_image(url="attachment://SpectrogramCanvas.png")

    embed2 = discord.Embed(
        title="Music Genre Analysis Result",
        description=f"{lines}\nThe most likely genre is: **{GENRES[winner]}**",
        color=discord.Color.red()
    )
    embed2.set_image(url="attachment://BarChartCanvas.png")
    embed2.set_footer(text="Powered by MusicGenrePredictor | 2025.6.29 ")

    await ctx.send(
        "Time chart and genre prediction:",
        embeds=[embed1, embed2],
        files=[
            discord.File(spectrogram_path, filename="SpectrogramCanvas.png"),
            discord.File(bar_path, filename="BarChartCanvas.png")
        ]
    )

    window_paths = get_broken_barh_paths()
    titles = generate_titles_from_filenames(window_paths)
    view = TimeSlider(window_paths, titles)
    embed3 = discord.Embed(title=titles[0], color=discord.Color.red())
    embed3.set_image(url=f"attachment://{os.path.basename(window_paths[0])}")
    await ctx.send(
        embed=embed3,
        view=view,
        file=discord.File(window_paths[0], filename=os.path.basename(window_paths[0]))
    )

token = dcbot_cfg.get("token") or os.environ.get("DISCORD_BOT_TOKEN", "")
bot.run(token)
