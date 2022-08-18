#!/usr/bin/python3

#%pip install click clip-retrieval img2dataset aiomultiprocess aiohttp aiofile

from clip_retrieval.clip_client import ClipClient, Modality
import json, pathlib, functools
import aiomultiprocess, asyncio, aiohttp, aiofile
from aiohttp import request, ClientTimeout
from aiomultiprocess import Pool
import click
from PIL import Image
from io import BytesIO

import mimetypes
mimetypes.init()

def pure_pil_alpha_to_color_v2(image, color=(255, 255, 255)):
		"""Alpha composite an RGBA Image with a specified color. Simpler, faster
		version than the solutions above.

		Source: http://stackoverflow.com/a/9459208/284318

		Keyword Arguments:
		image -- PIL RGBA Image object
		color -- Tuple r, g, b (default 255, 255, 255)
		"""
		image.load() # needed for split()
		background = Image.new('RGB', image.size, color)
		background.paste(image, mask=image.split()[3]) # 3 is the alpha channel
		return background

def fixrgb(im):
		''' convert images to 24bpp RGB, as expected by the StyleGAN3 train.py 
		script. Images with alpha transparency are composited over a black 
		background, and palette-indexed images are converted to direct color '''
		if im.mode == "P" or im.mode == "L":
				om = im.convert("RGB")
		elif im.mode == "RGBA":
				om = pure_pil_alpha_to_color_v2(im)
		elif im.mode == "RGB":
				om = im
		else:
				raise RuntimeError(f"fixrgb(): Unknown mode {im.mode}!")
		return om

def torgbpng(filename=None, data=None):
		''' convert any image file inplace or image data buffer to RGB, removing
		alpha and expanding indexed color palettes, and write it back replacing
		the original image file's contents, or return a buffer containing a 24bpp
		PNG image if no filename was given. '''
		if data==None:
				with open(filename, "rb") as fin:
						data = fin.read()
		im = Image.open(BytesIO(data))
		im = fixrgb(im)
		with BytesIO() as outbuf:
				im.save(outbuf, "png")
				if filename!=None:
						with open(filename, "wb") as fout:
								fout.write(bytes(outbuf.getbuffer()))
				else:
						return bytes(outbuf.getbuffer())
		return True


async def get(outdir, timeout, noconvert, result):
		try:
				if len(list(pathlib.Path(outdir).glob(f"{result['id']}.*"))) > 0:
						# there exists a file {outdir}/{id}.* already, skip downloading
						return (result['id'], None, None)
				#print(f"fetch id={result['id']} from: {result['url']}")
				async with request("GET", result['url'], timeout=ClientTimeout(total=timeout)) as response:
						content = await response.content.read()
						assert(len(content) > 0)
						contype = response.content_type
						assert(contype != None and contype.startswith("image/"))
						if noconvert:
								outfext=mimetypes.guess_extension(contype)
								assert(outfext != None)
						else:
								content = torgbpng(data=content)
								outfext = ".png"
						outfname=f"{outdir}/{result['id']}{outfext}"
						async with aiofile.async_open(outfname, "wb") as outf:
								#print(f"Writing {len(content)} bytes: {result['id']} <- {result['url']}")
								await outf.write(content)
						return (result['id'], outfname, None)
		except Exception as err:
				return (result['id'], None, str(err))

async def async_main(query_text,
				aesthetic_score=9.0,
				aesthetic_weight=0.5,
				num_images=5000,
				outdir="./dataset",
				paralellism=4,
				timeout=5.0,
				noconvert=False):

		client = ClipClient(
				url="https://knn5.laion.ai/knn-service",
				indice_name="laion5B",
				aesthetic_score=aesthetic_score,
				aesthetic_weight=aesthetic_weight,
				modality=Modality.IMAGE,
				num_images=num_images
		)

		#Query by text
		results = client.query(text=query_text)

		print(f"Query returned {len(results)} results...")

		pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)

		counter = 0
		interval = 50
		fetched = {}
		failed = {}
		skipped = set()

		async with Pool(paralellism) as pool:
				async for fileid, filename, failerr in pool.map(functools.partial(get, outdir, timeout, noconvert), results):
						#fileid = result['id']
						if failerr != None:
								failed[fileid] = failerr
								#print(f"Errored id={result['id']}: {failerr}\n\t(url={result['url']})")
						elif filename != None:
								fetched[fileid] = filename
								#print(f"Fetched id={result['id']}: filename={filename}")
						else:
								skipped.add(fileid)
								#print(f"Skipped id={result['id']}")
						counter = counter + 1
						if counter > interval:
							counter = 0
							print(f"Progress: {len(fetched)+len(failed)+len(skipped)}/{len(results)}: {len(fetched)} fetched, {len(failed)} failed, {len(skipped)} skipped", flush=True)
		print("\nDone!")

@click.command()
@click.option('--count', default=1000, help='number of images')
@click.option('--score', default=9.0, help='minimum aesthetic score')
@click.option('--weight', default=0.5, help='aesthetic score weight')
@click.option('--paralell', default=4, help='number of concurrent workers')
@click.option('--timeout', default=5.0, help='timeout in seconds')
@click.argument('query_text')
@click.argument('outdir')
def main(query_text, outdir, count, paralell, timeout, weight, score):
		asyncio.run(
				async_main(
						query_text=query_text,
						aesthetic_score=score,
						aesthetic_weight=weight,
						num_images=count,
						outdir=outdir,
						timeout=timeout,
						paralellism=paralell
				)
		)

if __name__ == '__main__':
		__spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
		main()
