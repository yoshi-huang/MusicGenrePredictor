from setuptools import setup, find_packages

setup(
    name="yt-genre",
    version="1.0.0",
    description="CLI tool: predict music genre from a YouTube URL",
    py_modules=["yt_genre"],
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "yt-genre=yt_genre:main",
        ],
    },
)
