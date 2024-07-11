import os
import subprocess
from multiprocessing import Pool
from functools import partial

DIRECTORY_PATH = 'outputs/outdoor_reolink_02'
OUTPUT_DIR     = 'outputs/COMPRESSED_outdoor_reolink_02'
NUM_PROCESSES  = 20

def re_encode_movie(movie_path, output_dir):
	try:
		path, filename = os.path.split(movie_path)
		command = f"ffmpeg -i '{movie_path}' '{output_dir}'/'{filename}'"
		subprocess.run(command, shell=True, check=True)
		#print(command)
		print(f"Re-encoded: {movie_path}")
	except subprocess.CalledProcessError as e:
		print(f"Error re-encoding {movie_path}: {e}")

def re_encode_movies_in_directory(directory, output_dir, num_processes):
	movies = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.mp4')]
	with Pool(num_processes) as pool:
		pool.map(partial(re_encode_movie, output_dir=output_dir), movies)


re_encode_movies_in_directory(DIRECTORY_PATH, OUTPUT_DIR, NUM_PROCESSES)
