from PIL import Image
from io import BytesIO

def pure_pil_alpha_to_color_v2(image, color=(255, 255, 255)):
		"""Alpha composite an RGBA Image with a specified color.

		Simpler, faster version than the solutions above.

		Source: http://stackoverflow.com/a/9459208/284318

		Keyword Arguments:
		image -- PIL RGBA Image object
		color -- Tuple r, g, b (default 255, 255, 255)

		"""
		image.load()  # needed for split()
		background = Image.new('RGB', image.size, color)
		background.paste(image, mask=image.split()[3])  # 3 is the alpha channel
		return background

def fixrgb(im):
		''' convert images to 24bpp RGB, as expected by the StyleGAN3 train.py script. Images with alpha transparency are composited over a black background, and palette-indexed images are converted to direct color '''
		if im.mode == "P" or im.mode == "L":
				om = im.convert("RGB")
		elif im.mode == "RGBA":
				om = pure_pil_alpha_to_color_v2(im)
		elif im.mode == "RGB":
				om = im
		else:
				raise RuntimeError(f"fixrgb(): Unknown mode {im.mode}!")
		return om

def fixrgbinplace(filename):
		''' convert a png file to RGB, removing alpha and expanding indexed color palettes, and write it back replacing the original image file's contents '''
		with open(filename, "rb") as fin:
				data = fin.read()
		im = Image.open(BytesIO(data))
		im = fixrgb(im)
		with BytesIO() as outbuf:
				im.save(outbuf, "png")
				with open(filename, "wb") as fout:
						fout.write(bytes(outbuf.getbuffer()))
		return True

def weightedchoice(keyweights):
		''' given a dictionary of keys and integer weights, choose a key with the probability of each key
		being chosen proportional to it's weight. '''
		from random import randint
		totalweight = int(sum(keyweights.values()))
		pos = randint(0, totalweight)
		accum = 0
		for key, weight in keyweights.items():
			accum = accum + weight
			if pos < accum:
				return key
		return None

def randomcrops(indir, outdir, outcount, outsize, weighting=1.0, withclasses=True, statusinterval=100):
		''' prepare a dataset for stylegan training from source images

			@param indir		the directory containing the source images
			@param outdir		the directory in which to save the numbered output images, will be created if it does not exist
			@param outcount	the number of output images to generate e.g. 50000
			@param outsize	the dimension of the square output images e.g. 256 for 256x256

		generates @param outcount randomly selected @param outsize x @param outsize crops of the images in @param indir.
		the images are weighted based on area (vertical resolution x horizontal resolution) and then cropped at random
		x,y offsets to ensure that all pixels of the input are represented evenly in the generated dataset. (@todo does
		this sample edge pixels fairly? i suspect not... hrmm.) (@todo also, some filtering and rejection might be good,
		for instance there are some images in the eboy.io pixelscapes database which have large regions of empty space
		due to their isometric projection. rejecting crops which are all or mostly one color could improve results.)
		'''

		from pathlib import Path
		from PIL import Image
		from random import randint, choice
		from io import BytesIO
		import math

		# read in source images
		sourceims = {}
		sourceweights = {}
		skipped = 0
		total = 0
		for infile in Path(indir).rglob("*"):
				total = total + 1
				try:
						with open(str(infile), "rb") as infh:
								data = infh.read()
						sourceim = Image.open(BytesIO(data))
				except Exception as err:
						print(f"Error opening {infile}: {err}")
						skipped = skipped + 1
						continue
				if sourceim.width < outsize or sourceim.height < outsize:
						print(f"Skipping {infile} because its resolution {sourceim.width}x{sourceim.height} is smaller than {outsize}x{outsize}!")
						skipped = skipped + 1
						continue
				sourceims[str(infile)]=fixrgb(sourceim)
				sourceweights[str(infile)] = int(sourceim.width * sourceim.height)
				#sourceweights[str(infile)] = int(math.sqrt((sourceim.width-outsize)**2 + (sourceim.height-outsize)**2))

		# show summary of source images and weights
		print()
		print(f"Skipped {skipped} of {total} images... weights for images based on area:")
		totalweight = sum(sourceweights.values())
		averageweight = totalweight/len(list(sourceweights.items()))
		scaledweights = {}
		for key, weight in sourceweights.items():
			scaledweights[key] = averageweight + (weight - averageweight) * weighting
		for key, weight in scaledweights.items():
			print(f"  {weight/totalweight*100.0: 3.2g} {key}")
		print()

		# create output directory if it does not exist
		if not Path(outdir).exists():
			print(f"Creating non-existant output directory: {outdir}")
			Path(outdir).mkdir(parents=True, exist_ok=False)
			print()
		else:
			print(f"WARNING! Output directory exists, make sure this is what you want!")
			print()

		# generate output images

		# build dicts of class number and numbered class output directory
		if withclasses:
			print(f"Creating classes for crops from each input image...")
			classnums = {}
			classdirs = {}
			for classnum, key in enumerate(scaledweights.keys()):
				classnums[key] = classnum
				classdir = Path(outdir) / f"{classnum:05d}"
				if not classdir.exists():
					classdir.mkdir(parents=True, exist_ok=False)
				else:
					print("WARNING! Class output subdirectory exists!")
				classdirs[key] = classdir

		print(f"Generating random crops...")
		for outnum in range(0, outcount):
				sourceimfile = weightedchoice(scaledweights)
				sourceim = sourceims[sourceimfile]
				sourcex = randint(0, sourceim.width - outsize)
				sourcey = randint(0, sourceim.height - outsize)
				outim = sourceim.crop((sourcex, sourcey, sourcex+outsize, sourcey+outsize))
				if not withclasses:
					outpath = Path(outdir) / f"{outnum:06d}.png"
				else:
					outpath = classdirs[sourceimfile] / f"{outnum:06d}.png"
				outim.save(str(outpath))
				if (outnum//statusinterval)*statusinterval == outnum:
					print(f"\rSaved crop {outnum}/{outcount} at {sourcex}x{sourcey} from {sourceimfile} to {str(outpath)}\x1b[K", end="")
		print("Done!")

if __name__ == "__main__":
	import argparse, os, pathlib, sys, re
	parser = argparse.ArgumentParser(description="Generate randomly cropped squares from a set of source images for training GANs")
	parser.add_argument("sourcedir", type=str, help="The path to a directory containing the input images to crop")
	parser.add_argument("destdir", type=str, help="The path to the output directory where the cropped images should be saved")
	parser.add_argument("--no-classes", dest="withclasses", action="store_false", required=False, help="Do not generate output class directories based on which of the source images each training image was cropped from. Helps GANs avoid mode collapse and train faster.")
	parser.add_argument("--classes", dest="withclasses", action="store_true", default=True, required=False, help="Generate output class directories based on which of the source images each training image was cropped from. Opposite of --no-classes, this is the default.")
	parser.add_argument("--weighting", type=float, default=0.5, help="How much weighting to apply based on source image pixel count. Set to 0.0 for equal number of samples from each source image regardless of whether some source images are larger than others. 1.0 means weight by width * height, so a source with 500kpixels will be samples 5x as frequently as one with 100kpix. Intermediate values interpolate between even distribution and pixel-count-proportional weighting.")
	parser.add_argument("--size", type=int, default=256, help="The size (width and height) of the square region to crop and output from the source images")
	parser.add_argument("--count", type=int, default=50000, help="The number of randomly cropped output images to generate from the source images")
	parser.add_argument("--status-interval", type=int, default=100, help="Print a progress message to stderr after each chunk of this many output images are written")
	args = parser.parse_args()
	randomcrops(args.sourcedir, args.destdir, args.count, args.size, weighting=args.weighting, withclasses=args.withclasses, statusinterval=args.status_interval)
