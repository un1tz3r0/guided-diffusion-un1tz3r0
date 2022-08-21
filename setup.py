from setuptools import setup

setup(
    name="guided-diffusion",
    py_modules=["guided_diffusion"],
    install_requires=[
      "blobfile>=1.0.5",
      "torch",
      "tqdm",
      # these are used by the anythingdiffusion clipfetch.py in scripts/...
      "aiofile==3.7.4",
      "aiohttp==3.8.1",
      "aiomultiprocess==0.9.0",
      "click==8.0.3",
      "clip-retrieval==2.34.2",
      "img2dataset==1.32.0"
    ],
)
